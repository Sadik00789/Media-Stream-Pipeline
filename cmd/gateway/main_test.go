package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sadik/media-stream-pipeline/internal/config"
	"github.com/sadik/media-stream-pipeline/internal/ipc"
	"github.com/sadik/media-stream-pipeline/internal/session"
	"github.com/sadik/media-stream-pipeline/internal/webrtc"
)

func setupTestGateway(t *testing.T) (*httptest.Server, *ipc.ShmHandle, *ipc.VideoShmHandle, *session.Hub) {
	t.Helper()
	pid := os.Getpid()
	aName := fmt.Sprintf("/test_gw_audio_%d_%s", pid, t.Name())
	vName := fmt.Sprintf("/test_gw_video_%d_%s", pid, t.Name())
	aName = strings.ReplaceAll(aName, "/", "_")
	vName = strings.ReplaceAll(vName, "/", "_")

	aShm, err := ipc.CreateOrOpenShm("/" + aName)
	if err != nil {
		t.Fatalf("CreateOrOpenShm: %v", err)
	}
	vShm, err := ipc.CreateOrOpenVideoShm("/" + vName)
	if err != nil {
		_ = aShm.Unlink()
		t.Fatalf("CreateOrOpenVideoShm: %v", err)
	}

	cfg := &config.Config{
		HTTPPort:        "0",
		ShmPath:         "/" + aName,
		VideoShmPath:    "/" + vName,
		SampleRate:      16000,
		FrameSamples:    ipc.FrameSamples,
		SpectrogramBins: ipc.SpectrogramBins,
		VideoWidth:      ipc.VideoWidth,
		VideoHeight:     ipc.VideoHeight,
		WebStaticDir:    "nonexistent_dir",
	}

	hub := session.NewHub()
	aWriter := ipc.NewRingBufferWriter(aShm)
	vWriter := ipc.NewVideoRingBufferWriter(vShm)
	handler := webrtc.NewSessionHandler(hub, aWriter, vWriter, cfg.SampleRate, cfg.FrameSamples, cfg.VideoWidth, cfg.VideoHeight)

	mux := buildMux(cfg, hub, handler, aWriter, vWriter)
	srv := httptest.NewServer(mux)
	t.Cleanup(func() {
		srv.Close()
		_ = aShm.Unlink()
		_ = vShm.Unlink()
	})
	return srv, aShm, vShm, hub
}

func TestGatewayHealthAndCORSEndpoints(t *testing.T) {
	srv, _, _, _ := setupTestGateway(t)

	req, _ := http.NewRequest(http.MethodOptions, srv.URL+"/health", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("OPTIONS /health status = %d", resp.StatusCode)
	}
	if resp.Header.Get("Access-Control-Allow-Origin") != "*" {
		t.Fatal("missing CORS header")
	}

	resp, err = http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /health status = %d", resp.StatusCode)
	}
	var health map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatal(err)
	}
	if health["status"] != "healthy" {
		t.Fatalf("unexpected health status: %v", health)
	}
}

func TestGatewayPcmVideoAndStatelessTelemetry(t *testing.T) {
	srv, aShm, vShm, _ := setupTestGateway(t)

	// 1. Telemetry waiting before any processed frames
	resp, err := http.Get(srv.URL + "/api/telemetry")
	if err != nil {
		t.Fatal(err)
	}
	var waiting map[string]interface{}
	_ = json.NewDecoder(resp.Body).Decode(&waiting)
	_ = resp.Body.Close()
	if waiting["status"] != "waiting" {
		t.Fatalf("expected waiting telemetry, got %v", waiting)
	}

	// 2. Post PCM samples
	samples := make([]float32, ipc.FrameSamples)
	pcmBody, _ := json.Marshal(DirectPcmRequest{Samples: samples})
	resp, err = http.Post(srv.URL+"/api/pcm", "application/json", bytes.NewReader(pcmBody))
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("POST /api/pcm status = %d", resp.StatusCode)
	}
	if atomic.LoadUint64(&aShm.Ring.WriteHead) != 1 {
		t.Fatalf("expected Audio WriteHead 1, got %d", aShm.Ring.WriteHead)
	}

	// 3. Post binary and base64 video
	resp, err = http.Post(srv.URL+"/api/video", "application/octet-stream", bytes.NewReader(make([]byte, 128)))
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("POST binary /api/video status = %d", resp.StatusCode)
	}

	b64Body, _ := json.Marshal(DirectVideoRequest{Base64Data: base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4})})
	resp, err = http.Post(srv.URL+"/api/video", "application/json", bytes.NewReader(b64Body))
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if atomic.LoadUint64(&vShm.Ring.WriteHead) != 2 {
		t.Fatalf("expected Video WriteHead 2, got %d", vShm.Ring.WriteHead)
	}

	// 4. Mark processed and verify concurrent stateless /api/telemetry calls succeed
	atomic.StoreUint32(&aShm.Ring.Packets[0].Flags, ipc.FlagIngested|ipc.FlagDspDone|ipc.FlagMlDone)
	aShm.Ring.Packets[0].VadActive = 1
	atomic.StoreUint32(&vShm.Ring.Packets[1].Flags, ipc.FlagIngested|ipc.FlagMlDone)
	vShm.Ring.Packets[1].DetectionCount = 1

	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := http.Get(srv.URL + "/api/telemetry")
			if err != nil {
				t.Errorf("GET /api/telemetry failed: %v", err)
				return
			}
			defer func() { _ = r.Body.Close() }()
			var payload webrtc.TelemetryPayload
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Errorf("decode telemetry failed: %v", err)
				return
			}
			if payload.Sequence != 1 || payload.VideoSequence != 2 {
				t.Errorf("unexpected telemetry seqs: audio=%d video=%d", payload.Sequence, payload.VideoSequence)
			}
		}()
	}
	wg.Wait()
}

func TestGatewayOfferAndWebSocket(t *testing.T) {
	srv, aShm, vShm, _ := setupTestGateway(t)

	// Invalid offer returns 400 or 500
	resp, err := http.Post(srv.URL+"/api/offer", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty SDP, got %d", resp.StatusCode)
	}

	// WebSocket PCM + Binary Video
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("WS dial failed: %v", err)
	}
	defer func() { _ = conn.Close() }()

	if err := conn.WriteJSON(map[string]interface{}{
		"type":    "pcm",
		"samples": make([]float32, ipc.FrameSamples),
	}); err != nil {
		t.Fatal(err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, make([]byte, 256)); err != nil {
		t.Fatal(err)
	}

	// Wait briefly for server goroutine to read WS frames
	for i := 0; i < 50; i++ {
		if atomic.LoadUint64(&aShm.Ring.WriteHead) >= 1 && atomic.LoadUint64(&vShm.Ring.WriteHead) >= 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if atomic.LoadUint64(&aShm.Ring.WriteHead) < 1 || atomic.LoadUint64(&vShm.Ring.WriteHead) < 1 {
		t.Fatalf("expected WS frames ingested: audio=%d video=%d",
			atomic.LoadUint64(&aShm.Ring.WriteHead),
			atomic.LoadUint64(&vShm.Ring.WriteHead))
	}
}
