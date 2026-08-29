package main

import (
	"bytes"
	"context"
	"encoding/base64"
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

type DirectVideoRequest struct {
	Base64Data string `json:"data,omitempty"`
	Width      int    `json:"width,omitempty"`
	Height     int    `json:"height,omitempty"`
}

func main() {
	cfg := config.Load()
	log.Printf("[Gateway] Starting Dual-Track Media Stream WebRTC Gateway on port %s...", cfg.HTTPPort)

	// 1. Initialize POSIX Shared Memory for Audio
	audioShmHandle, err := ipc.CreateOrOpenShm(cfg.ShmPath)
	if err != nil {
		log.Fatalf("[Gateway] Failed to initialize Audio POSIX shared memory: %v", err)
	}
	defer audioShmHandle.Close()
	log.Printf("[Gateway] Audio shared memory segment mapped: %s", cfg.ShmPath)
	audioWriter := ipc.NewRingBufferWriter(audioShmHandle)

	// 2. Initialize POSIX Shared Memory for Video
	videoShmHandle, err := ipc.CreateOrOpenVideoShm(cfg.VideoShmPath)
	if err != nil {
		log.Fatalf("[Gateway] Failed to initialize Video POSIX shared memory: %v", err)
	}
	defer videoShmHandle.Close()
	log.Printf("[Gateway] Video shared memory segment mapped: %s", cfg.VideoShmPath)
	videoWriter := ipc.NewVideoRingBufferWriter(videoShmHandle)

	hub := session.NewHub()
	sessionHandler := webrtc.NewSessionHandler(
		hub,
		audioWriter,
		videoWriter,
		cfg.SampleRate,
		cfg.FrameSamples,
		cfg.VideoWidth,
		cfg.VideoHeight,
	)

	// 3. Start Background Telemetry Broadcaster (Audio + Video at 60 FPS)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	broadcaster := webrtc.NewTelemetryBroadcaster(audioWriter, videoWriter, hub)
	go broadcaster.StartBroadcasting(ctx)

	// 4. HTTP Routes
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
			"status":    "healthy",
			"clients":   hub.ClientCount(),
			"audio_shm": cfg.ShmPath,
			"video_shm": cfg.VideoShmPath,
		})
	}))

	// Latest Telemetry HTTP Polling Endpoint (Fallback)
	var latestAudioSeq uint64 = 0
	var latestVideoSeq uint64 = 0
	mux.HandleFunc("/api/telemetry", withCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		var detections []ipc.DetectionBox
		var videoSeq uint64 = 0
		var videoTs uint64 = 0
		var detCount uint32 = 0

		if vPkt, ok := videoWriter.ReadLatestProcessedPacket(&latestVideoSeq); ok {
			videoSeq = vPkt.SequenceNumber
			videoTs = vPkt.TimestampNs
			detCount = vPkt.DetectionCount
			cnt := int(vPkt.DetectionCount)
			if cnt > ipc.MaxDetections {
				cnt = ipc.MaxDetections
			}
			detections = make([]ipc.DetectionBox, cnt)
			for i := 0; i < cnt; i++ {
				detections[i] = vPkt.Detections[i]
			}
		}

		if pkt, ok := audioWriter.ReadLatestProcessedPacket(&latestAudioSeq); ok {
			transcriptLen := bytes.IndexByte(pkt.MlTranscript[:], 0)
			if transcriptLen < 0 {
				transcriptLen = len(pkt.MlTranscript)
			}
			transcript := string(pkt.MlTranscript[:transcriptLen])

			spectrogramSlice := make([]float32, ipc.SpectrogramBins)
			for i := 0; i < ipc.SpectrogramBins; i++ {
				spectrogramSlice[i] = pkt.Spectrogram[i]
			}

			currentNowNs := uint64(time.Now().UnixNano())
			var audioLatency float32 = 0.0
			if pkt.TimestampNs > 0 {
				if currentNowNs >= pkt.TimestampNs {
					diffMs := float32(currentNowNs-pkt.TimestampNs) / 1e6
					if diffMs < 500.0 {
						audioLatency = diffMs
					}
				} else {
					audioLatency = 1.0
				}
			}
			var visionLatency float32 = 0.0
			if videoTs > 0 {
				if currentNowNs >= videoTs {
					diffMs := float32(currentNowNs-videoTs) / 1e6
					if diffMs < 500.0 {
						visionLatency = diffMs
					}
				} else {
					visionLatency = 1.0
				}
			}

			_ = json.NewEncoder(w).Encode(webrtc.TelemetryPayload{
				Type:             "telemetry",
				Sequence:         pkt.SequenceNumber,
				SequenceNumber:   pkt.SequenceNumber,
				TimestampNs:      pkt.TimestampNs,
				VadActive:        pkt.VadActive != 0,
				VadConfidence:    pkt.VadConfidence,
				MlConfidence:     pkt.MlConfidence,
				MlTranscript:     transcript,
				Transcript:       transcript,
				Spectrogram:      spectrogramSlice,
				VideoSequence:    videoSeq,
				VideoTimestampNs: videoTs,
				DetectionCount:   detCount,
				DetectionsCount:  detCount,
				Detections:       detections,
				AudioLatencyMs:   audioLatency,
				VisionLatencyMs:  visionLatency,
			})
		} else if videoSeq > 0 {
			currentNowNs := uint64(time.Now().UnixNano())
			var visionLatency float32 = 0.0
			if videoTs > 0 {
				if currentNowNs >= videoTs {
					diffMs := float32(currentNowNs-videoTs) / 1e6
					if diffMs < 500.0 {
						visionLatency = diffMs
					}
				} else {
					visionLatency = 1.0
				}
			}

			_ = json.NewEncoder(w).Encode(webrtc.TelemetryPayload{
				Type:             "telemetry",
				VideoSequence:    videoSeq,
				VideoTimestampNs: videoTs,
				DetectionCount:   detCount,
				DetectionsCount:  detCount,
				Detections:       detections,
				AudioLatencyMs:   0.0,
				VisionLatencyMs:  visionLatency,
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

	// Direct PCM Ingestion (Audio)
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

	// Direct Video Frame Ingestion (Binary RGB24 or Base64 JSON)
	mux.HandleFunc("/api/video", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		contentType := r.Header.Get("Content-Type")
		if contentType == "application/octet-stream" {
			data, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			sessionHandler.PushDirectVideo(data, cfg.VideoWidth, cfg.VideoHeight)
		} else {
			var req DirectVideoRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if req.Base64Data != "" {
				decoded, err := base64.StdEncoding.DecodeString(req.Base64Data)
				if err == nil {
					wVal := req.Width
					if wVal == 0 {
						wVal = cfg.VideoWidth
					}
					hVal := req.Height
					if hVal == 0 {
						hVal = cfg.VideoHeight
					}
					sessionHandler.PushDirectVideo(decoded, wVal, hVal)
				}
			}
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
			messageType, message, err := conn.ReadMessage()
			if err != nil {
				break
			}

			if messageType == websocket.BinaryMessage {
				// Binary payload interpreted as direct RGB24 frame
				sessionHandler.PushDirectVideo(message, cfg.VideoWidth, cfg.VideoHeight)
			} else {
				var msg map[string]interface{}
				if err := json.Unmarshal(message, &msg); err == nil {
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
