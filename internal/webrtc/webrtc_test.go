package webrtc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/pion/webrtc/v3"
	"github.com/sadik/media-stream-pipeline/internal/ipc"
	"github.com/sadik/media-stream-pipeline/internal/session"
)

func TestAudioAndVideoSinks(t *testing.T) {
	aName := fmt.Sprintf("/test_webrtc_audio_%d", os.Getpid())
	vName := fmt.Sprintf("/test_webrtc_video_%d", os.Getpid())

	aShm, err := ipc.CreateOrOpenShm(aName)
	if err != nil {
		t.Fatalf("CreateOrOpenShm: %v", err)
	}
	defer func() { _ = aShm.Unlink() }()

	vShm, err := ipc.CreateOrOpenVideoShm(vName)
	if err != nil {
		t.Fatalf("CreateOrOpenVideoShm: %v", err)
	}
	defer func() { _ = vShm.Unlink() }()

	aWriter := ipc.NewRingBufferWriter(aShm)
	vWriter := ipc.NewVideoRingBufferWriter(vShm)

	aSink := NewAudioSink(aWriter, 16000, ipc.FrameSamples)
	// Push fewer than 512 samples -> buffered, not written yet
	aSink.PushDirectPcm(make([]float32, 256))
	if atomic.LoadUint64(&aShm.Ring.WriteHead) != 0 {
		t.Fatalf("expected WriteHead 0 for partial frame")
	}
	// Push another 256 -> flushed
	aSink.PushDirectPcm(make([]float32, 256))
	if atomic.LoadUint64(&aShm.Ring.WriteHead) != 1 {
		t.Fatalf("expected WriteHead 1 after full frame, got %d", aShm.Ring.WriteHead)
	}

	vSink := NewVideoSink(vWriter, ipc.VideoWidth, ipc.VideoHeight)
	vSink.PushDirectVideo(make([]byte, 100), ipc.VideoWidth, ipc.VideoHeight)
	vSink.PushDirectVideo(make([]byte, ipc.VideoFrameSize), ipc.VideoWidth, ipc.VideoHeight)
	if atomic.LoadUint64(&vShm.Ring.WriteHead) != 2 {
		t.Fatalf("expected Video WriteHead 2, got %d", vShm.Ring.WriteHead)
	}
}

func TestSessionHandlerOfferCleanupAndAnswer(t *testing.T) {
	aName := fmt.Sprintf("/test_offer_audio_%d", os.Getpid())
	vName := fmt.Sprintf("/test_offer_video_%d", os.Getpid())
	aShm, err := ipc.CreateOrOpenShm(aName)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = aShm.Unlink() }()
	vShm, err := ipc.CreateOrOpenVideoShm(vName)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = vShm.Unlink() }()

	hub := session.NewHub()
	handler := NewSessionHandler(
		hub,
		ipc.NewRingBufferWriter(aShm),
		ipc.NewVideoRingBufferWriter(vShm),
		16000, ipc.FrameSamples, ipc.VideoWidth, ipc.VideoHeight,
	)

	// Invalid SDP must clean up peer in hub (no leak)
	if _, _, err := handler.HandleOffer("invalid-sdp"); err == nil {
		t.Fatal("expected error for invalid SDP")
	}
	if hub.ClientCount() != 0 {
		t.Fatalf("expected 0 peers after failed HandleOffer, got %d", hub.ClientCount())
	}

	// Valid SDP offer produced by a real Pion PeerConnection
	clientPC, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = clientPC.Close() }()
	_, _ = clientPC.CreateDataChannel("telemetry", nil)
	offer, err := clientPC.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}

	answerSDP, peer, err := handler.HandleOffer(offer.SDP)
	if err != nil {
		t.Fatalf("HandleOffer failed: %v", err)
	}
	if answerSDP == "" || peer == nil {
		t.Fatal("expected non-empty answer SDP and peer")
	}
	if hub.ClientCount() != 1 {
		t.Fatalf("expected 1 peer in hub, got %d", hub.ClientCount())
	}
	hub.RemovePeer(peer.ID)
}

func TestTelemetryBroadcaster(t *testing.T) {
	aName := fmt.Sprintf("/test_bcast_audio_%d", os.Getpid())
	vName := fmt.Sprintf("/test_bcast_video_%d", os.Getpid())
	aShm, _ := ipc.CreateOrOpenShm(aName)
	defer func() { _ = aShm.Unlink() }()
	vShm, _ := ipc.CreateOrOpenVideoShm(vName)
	defer func() { _ = vShm.Unlink() }()

	aWriter := ipc.NewRingBufferWriter(aShm)
	vWriter := ipc.NewVideoRingBufferWriter(vShm)
	hub := session.NewHub()

	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		hub.AddWebSocket(c)
		for {
			if _, _, err := c.ReadMessage(); err != nil {
				hub.RemoveWebSocket(c)
				return
			}
		}
	}))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	clientConn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = clientConn.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	broadcaster := NewTelemetryBroadcaster(aWriter, vWriter, hub)
	go broadcaster.StartBroadcasting(ctx)

	_, _ = aWriter.WriteAudioFrame(make([]float32, ipc.FrameSamples), 16000)
	atomic.StoreUint32(&aShm.Ring.Packets[0].Flags, ipc.FlagIngested|ipc.FlagDspDone|ipc.FlagMlDone)
	aShm.Ring.Packets[0].VadActive = 1
	copy(aShm.Ring.Packets[0].MlTranscript[:], "voice_detected")

	_ = clientConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, msg, err := clientConn.ReadMessage()
	if err != nil {
		t.Fatalf("failed to receive telemetry broadcast: %v", err)
	}

	var payload TelemetryPayload
	if err := json.Unmarshal(msg, &payload); err != nil {
		t.Fatalf("invalid telemetry JSON: %v", err)
	}
	if payload.Type != "telemetry" || payload.Sequence != 1 || !payload.VadActive || payload.Transcript != "voice_detected" {
		t.Fatalf("unexpected telemetry payload: %+v", payload)
	}
}

