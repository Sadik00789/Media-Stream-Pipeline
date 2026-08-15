package session

import (
	"sync"

	"github.com/pion/webrtc/v3"
)

type Peer struct {
	ID          string
	PC          *webrtc.PeerConnection
	DataChannel *webrtc.DataChannel
	IsActive    bool
	mu          sync.Mutex
}

func NewPeer(id string, pc *webrtc.PeerConnection) *Peer {
	return &Peer{
		ID:       id,
		PC:       pc,
		IsActive: true,
	}
}

func (p *Peer) SetDataChannel(dc *webrtc.DataChannel) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.DataChannel = dc
}

func (p *Peer) SendData(data []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.DataChannel != nil && p.DataChannel.ReadyState() == webrtc.DataChannelStateOpen {
		return p.DataChannel.Send(data)
	}
	return nil
}

func (p *Peer) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.IsActive = false
	if p.PC != nil {
		_ = p.PC.Close()
	}
}
