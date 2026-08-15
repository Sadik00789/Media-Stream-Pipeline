package webrtc

import (
	"encoding/json"
	"log"

	"github.com/google/uuid"
	"github.com/pion/webrtc/v3"
	"github.com/sadik/media-stream-pipeline/internal/ipc"
	"github.com/sadik/media-stream-pipeline/internal/session"
)

type ControlRPCMessage struct {
	Action     string  `json:"action"`
	Gain       float32 `json:"gain,omitempty"`
	Threshold  float32 `json:"threshold,omitempty"`
	WindowType string  `json:"window_type,omitempty"`
}

type SessionHandler struct {
	api       *webrtc.API
	hub       *session.Hub
	audioSink *AudioSink
	config    webrtc.Configuration
}

func NewSessionHandler(hub *session.Hub, writer *ipc.RingBufferWriter, sampleRate, frameSize int) *SessionHandler {
	mediaEngine := &webrtc.MediaEngine{}
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		log.Printf("[WebRTC] Error registering codecs: %v", err)
	}

	settingEngine := webrtc.SettingEngine{}
	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(mediaEngine),
		webrtc.WithSettingEngine(settingEngine),
	)

	config := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{URLs: []string{"stun:stun.l.google.com:19302"}},
		},
	}

	sink := NewAudioSink(writer, sampleRate, frameSize)

	return &SessionHandler{
		api:       api,
		hub:       hub,
		audioSink: sink,
		config:    config,
	}
}

// HandleOffer processes an SDP offer from a client and produces an SDP answer
func (s *SessionHandler) HandleOffer(sdpOffer string) (string, *session.Peer, error) {
	pc, err := s.api.NewPeerConnection(s.config)
	if err != nil {
		return "", nil, err
	}

	peerID := uuid.New().String()
	peer := session.NewPeer(peerID, pc)
	s.hub.AddPeer(peer)

	// Accept incoming audio tracks
	pc.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		if track.Kind() == webrtc.RTPCodecTypeAudio {
			go s.audioSink.IngestAudioTrack(track)
		}
	})

	// Handle DataChannel for real-time telemetry and bi-directional RPC
	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		log.Printf("[WebRTC] DataChannel '%s' opened for peer %s", dc.Label(), peerID)
		peer.SetDataChannel(dc)

		dc.OnOpen(func() {
			log.Printf("[WebRTC] DataChannel ready for peer %s", peerID)
		})

		dc.OnMessage(func(msg webrtc.DataChannelMessage) {
			var rpc ControlRPCMessage
			if err := json.Unmarshal(msg.Data, &rpc); err == nil {
				log.Printf("[WebRTC RPC] Received action '%s' from peer %s (Gain: %.2f)", rpc.Action, peerID, rpc.Gain)
			}
		})
	})

	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		log.Printf("[WebRTC] Peer %s connection state: %s", peerID, state.String())
		if state == webrtc.PeerConnectionStateFailed ||
			state == webrtc.PeerConnectionStateClosed ||
			state == webrtc.PeerConnectionStateDisconnected {
			s.hub.RemovePeer(peerID)
		}
	})

	offer := webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  sdpOffer,
	}

	if err := pc.SetRemoteDescription(offer); err != nil {
		peer.Close()
		return "", nil, err
	}

	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		peer.Close()
		return "", nil, err
	}

	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		peer.Close()
		return "", nil, err
	}

	<-gatherComplete

	return pc.LocalDescription().SDP, peer, nil
}

func (s *SessionHandler) PushDirectPcm(samples []float32) {
	s.audioSink.PushDirectPcm(samples)
}
