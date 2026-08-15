package session

import (
	"sync"

	"github.com/gorilla/websocket"
)

type Hub struct {
	peers      map[string]*Peer
	wsClients  map[*websocket.Conn]bool
	mu         sync.RWMutex
}

func NewHub() *Hub {
	return &Hub{
		peers:     make(map[string]*Peer),
		wsClients: make(map[*websocket.Conn]bool),
	}
}

func (h *Hub) AddPeer(peer *Peer) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.peers[peer.ID] = peer
}

func (h *Hub) RemovePeer(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if peer, exists := h.peers[id]; exists {
		peer.Close()
		delete(h.peers, id)
	}
}

func (h *Hub) AddWebSocket(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.wsClients[conn] = true
}

func (h *Hub) RemoveWebSocket(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.wsClients, conn)
	_ = conn.Close()
}

func (h *Hub) BroadcastTelemetry(data []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Broadcast to WebRTC DataChannels
	for _, peer := range h.peers {
		if peer.IsActive {
			_ = peer.SendData(data)
		}
	}

	// Broadcast to active WebSocket connections
	for conn := range h.wsClients {
		_ = conn.WriteMessage(websocket.TextMessage, data)
	}
}

func (h *Hub) ClientCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.peers) + len(h.wsClients)
}
