package session

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/pion/webrtc/v3"
)

func TestPeerAndHubLifecycle(t *testing.T) {
	hub := NewHub()
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatalf("NewPeerConnection failed: %v", err)
	}

	peer := NewPeer("peer-1", pc)
	hub.AddPeer(peer)
	if hub.ClientCount() != 1 {
		t.Fatalf("expected ClientCount 1, got %d", hub.ClientCount())
	}
	if !peer.Active() {
		t.Fatal("expected peer to be active")
	}
	if err := peer.SendData([]byte("hello")); err != nil {
		t.Fatalf("SendData without DataChannel should return nil, got %v", err)
	}

	hub.BroadcastTelemetry([]byte(`{"type":"telemetry"}`))

	hub.RemovePeer("peer-1")
	if hub.ClientCount() != 0 {
		t.Fatalf("expected ClientCount 0, got %d", hub.ClientCount())
	}
	if peer.Active() {
		t.Fatal("expected peer to be inactive after RemovePeer")
	}
}

func TestHubConcurrentWebSocketWrites(t *testing.T) {
	hub := NewHub()
	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}

	connected := make(chan *websocket.Conn, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		hub.AddWebSocket(c)
		connected <- c
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
		t.Fatalf("WebSocket dial failed: %v", err)
	}
	defer func() { _ = clientConn.Close() }()

	serverConn := <-connected
	if hub.ClientCount() != 1 {
		t.Fatalf("expected 1 WS client, got %d", hub.ClientCount())
	}

	var wg sync.WaitGroup
	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				hub.BroadcastTelemetry([]byte(`{"type":"telemetry"}`))
				_ = hub.SendWebSocketJSON(serverConn, map[string]interface{}{"type": "ack", "id": id})
			}
		}(g)
	}
	wg.Wait()
}

