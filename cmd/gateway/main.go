package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sadik/media-stream-pipeline/internal/config"
	"github.com/sadik/media-stream-pipeline/internal/ipc"
	"github.com/sadik/media-stream-pipeline/internal/session"
	"github.com/sadik/media-stream-pipeline/internal/webrtc"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type OfferRequest struct {
	SDP string `json:"sdp"`
}

type AnswerResponse struct {
	SDP string `json:"sdp"`
}

type DirectPcmRequest struct {
	Samples []float32 `json:"samples"`
}

func main() {
	cfg := config.Load()
	log.Printf("[Gateway] Starting Media Stream WebRTC Gateway on port %s...", cfg.HTTPPort)

	// 1. Initialize POSIX Shared Memory
	shmHandle, err := ipc.CreateOrOpenShm(cfg.ShmPath)
	if err != nil {
		log.Fatalf("[Gateway] Failed to initialize POSIX shared memory: %v", err)
	}
	defer shmHandle.Close()
	log.Printf("[Gateway] Shared memory segment mapped successfully: %s", cfg.ShmPath)

	writer := ipc.NewRingBufferWriter(shmHandle)
	hub := session.NewHub()
	sessionHandler := webrtc.NewSessionHandler(hub, writer, cfg.SampleRate, cfg.FrameSamples)

	// 2. Start Background Telemetry Broadcaster
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	broadcaster := webrtc.NewTelemetryBroadcaster(writer, hub)
	go broadcaster.StartBroadcasting(ctx)

	// 3. HTTP Routes
	mux := http.NewServeMux()

	withCORS := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}
			next(w, r)
		}
	}

	// Health Check
	mux.HandleFunc("/health", withCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "healthy",
			"clients": hub.ClientCount(),
			"shm":     cfg.ShmPath,
		})
	}))

	// Latest Telemetry HTTP Polling Endpoint (Fallback)
	var latestSeq uint64 = 0
	mux.HandleFunc("/api/telemetry", withCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if pkt, ok := writer.ReadLatestProcessedPacket(&latestSeq); ok {
			transcriptLen := bytes.IndexByte(pkt.MlTranscript[:], 0)
			if transcriptLen < 0 {
				transcriptLen = len(pkt.MlTranscript)
			}
			transcript := string(pkt.MlTranscript[:transcriptLen])

			spectrogramSlice := make([]float32, ipc.SpectrogramBins)
			for i := 0; i < ipc.SpectrogramBins; i++ {
				spectrogramSlice[i] = pkt.Spectrogram[i]
			}

			_ = json.NewEncoder(w).Encode(webrtc.TelemetryPayload{
				Type:          "telemetry",
				Sequence:      pkt.SequenceNumber,
				TimestampNs:   pkt.TimestampNs,
				VadActive:     pkt.VadActive != 0,
				VadConfidence: pkt.VadConfidence,
				MlConfidence:  pkt.MlConfidence,
				Transcript:    transcript,
				Spectrogram:   spectrogramSlice,
			})
		} else {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "waiting"})
		}
	}))

	// WebRTC SDP Offer Negotiation (REST)
	mux.HandleFunc("/api/offer", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		var req OfferRequest
		if err := json.Unmarshal(body, &req); err != nil || req.SDP == "" {
			http.Error(w, "Missing or invalid sdp offer", http.StatusBadRequest)
			return
		}

		answerSDP, _, err := sessionHandler.HandleOffer(req.SDP)
		if err != nil {
			log.Printf("[Gateway] Failed to handle offer: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(AnswerResponse{SDP: answerSDP})
	}))

	// Direct PCM Ingestion
	mux.HandleFunc("/api/pcm", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req DirectPcmRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if len(req.Samples) > 0 {
			sessionHandler.PushDirectPcm(req.Samples)
		}
		w.WriteHeader(http.StatusAccepted)
	}))

	// WebSocket Streaming & Signaling Endpoint
	mux.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("[Gateway] WebSocket upgrade failed: %v", err)
			return
		}
		hub.AddWebSocket(conn)
		log.Printf("[Gateway] WebSocket client connected (Total: %d)", hub.ClientCount())

		defer func() {
			hub.RemoveWebSocket(conn)
			log.Printf("[Gateway] WebSocket client disconnected (Remaining: %d)", hub.ClientCount())
		}()

		for {
			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}

			if msgType, ok := msg["type"].(string); ok {
				if msgType == "offer" {
					if sdp, ok := msg["sdp"].(string); ok {
						answer, _, err := sessionHandler.HandleOffer(sdp)
						if err == nil {
							_ = conn.WriteJSON(map[string]interface{}{
								"type": "answer",
								"sdp":  answer,
							})
						}
					}
				} else if msgType == "pcm" {
					if samplesRaw, ok := msg["samples"].([]interface{}); ok {
						samples := make([]float32, len(samplesRaw))
						for i, v := range samplesRaw {
							if f, ok := v.(float64); ok {
								samples[i] = float32(f)
							}
						}
						sessionHandler.PushDirectPcm(samples)
					}
				}
			}
		}
	})

	// Static Web Frontend Hosting
	if _, err := os.Stat(cfg.WebStaticDir); err == nil {
		mux.Handle("/", http.FileServer(http.Dir(cfg.WebStaticDir)))
	}

	server := &http.Server{
		Addr:    ":" + cfg.HTTPPort,
		Handler: mux,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[Gateway] Server error: %v", err)
		}
	}()

	log.Printf("[Gateway] WebRTC Gateway running at http://localhost:%s", cfg.HTTPPort)
	<-stop

	log.Println("[Gateway] Shutting down gracefully...")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	_ = server.Shutdown(shutdownCtx)
	log.Println("[Gateway] Server stopped.")
}
