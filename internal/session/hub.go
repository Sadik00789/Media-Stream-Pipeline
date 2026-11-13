package session

import (
	"sync"

	"github.com/gorilla/websocket"
)

type wsEntry struct {
	conn *websocket.Conn
	mu   *sync.Mutex
}

type Hub struct {
	peers     map[string]*Peer
	wsClients map[*websocket.Conn]*sync.Mutex
	mu        sync.RWMutex
}

func NewHub() *Hub {
	return &Hub{
		peers:     make(map[string]*Peer),
		wsClients: make(map[*websocket.Conn]*sync.Mutex),
	}
}

func (h *Hub) AddPeer(peer *Peer) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.peers[peer.ID] = peer
}

func (h *Hub) RemovePeer(id string) {
	h.mu.Lock()
	peer, exists := h.peers[id]
	if exists {
		delete(h.peers, id)
	}
	h.mu.Unlock()
	if exists {
		peer.Close()
	}
}

func (h *Hub) AddWebSocket(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.wsClients[conn] = &sync.Mutex{}
}

func (h *Hub) RemoveWebSocket(conn *websocket.Conn) {
	h.mu.Lock()
	wsMu, exists := h.wsClients[conn]
	delete(h.wsClients, conn)
	h.mu.Unlock()
	if exists && wsMu != nil {
		wsMu.Lock()
		_ = conn.Close()
		wsMu.Unlock()
	} else {
		_ = conn.Close()
	}
}

func (h *Hub) SendWebSocketJSON(conn *websocket.Conn, v interface{}) error {
	h.mu.RLock()
	wsMu := h.wsClients[conn]
	h.mu.RUnlock()
	if wsMu == nil {
		return nil
	}
	wsMu.Lock()
	defer wsMu.Unlock()
	return conn.WriteJSON(v)
}

func (h *Hub) BroadcastTelemetry(data []byte) {
	h.mu.RLock()
	peers := make([]*Peer, 0, len(h.peers))
	for _, peer := range h.peers {
		peers = append(peers, peer)
	}
	wsList := make([]wsEntry, 0, len(h.wsClients))
	for conn, mu := range h.wsClients {
		wsList = append(wsList, wsEntry{conn: conn, mu: mu})
	}
	h.mu.RUnlock()

	// Broadcast to WebRTC DataChannels
	for _, peer := range peers {
		if peer.Active() {
			_ = peer.SendData(data)
		}
	}

	// Broadcast to active WebSocket connections with per-connection write serialization
	for _, entry := range wsList {
		entry.mu.Lock()
		_ = entry.conn.WriteMessage(websocket.TextMessage, data)
		entry.mu.Unlock()
	}
}

func (h *Hub) ClientCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.peers) + len(h.wsClients)
}
