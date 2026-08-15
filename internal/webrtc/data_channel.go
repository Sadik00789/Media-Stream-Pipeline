package webrtc

import (
	"bytes"
	"context"
	"encoding/json"
	"time"

	"github.com/sadik/media-stream-pipeline/internal/ipc"
	"github.com/sadik/media-stream-pipeline/internal/session"
)

type TelemetryPayload struct {
	Type          string    `json:"type"`
	Sequence      uint64    `json:"seq"`
	TimestampNs   uint64    `json:"timestamp_ns"`
	VadActive     bool      `json:"vad_active"`
	VadConfidence float32   `json:"vad_confidence"`
	MlConfidence  float32   `json:"ml_confidence"`
	Transcript    string    `json:"transcript"`
	Spectrogram   []float32 `json:"spectrogram"`
}

type TelemetryBroadcaster struct {
	writer *ipc.RingBufferWriter
	hub    *session.Hub
}

func NewTelemetryBroadcaster(writer *ipc.RingBufferWriter, hub *session.Hub) *TelemetryBroadcaster {
	return &TelemetryBroadcaster{
		writer: writer,
		hub:    hub,
	}
}

// StartBroadcasting runs a 60 FPS broadcast loop transmitting completed DSP/ML frames to WebRTC & WebSocket clients
func (b *TelemetryBroadcaster) StartBroadcasting(ctx context.Context) {
	ticker := time.NewTicker(16 * time.Millisecond) // ~60 FPS
	defer ticker.Stop()

	var lastSeq uint64 = 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if b.hub.ClientCount() == 0 {
				continue
			}

			if pkt, ok := b.writer.ReadLatestProcessedPacket(&lastSeq); ok {
				// Convert C string transcript to Go string
				transcriptLen := bytes.IndexByte(pkt.MlTranscript[:], 0)
				if transcriptLen < 0 {
					transcriptLen = len(pkt.MlTranscript)
				}
				transcript := string(pkt.MlTranscript[:transcriptLen])

				spectrogramSlice := make([]float32, ipc.SpectrogramBins)
				for i := 0; i < ipc.SpectrogramBins; i++ {
					spectrogramSlice[i] = pkt.Spectrogram[i]
				}

				payload := TelemetryPayload{
					Type:          "telemetry",
					Sequence:      pkt.SequenceNumber,
					TimestampNs:   pkt.TimestampNs,
					VadActive:     pkt.VadActive != 0,
					VadConfidence: pkt.VadConfidence,
					MlConfidence:  pkt.MlConfidence,
					Transcript:    transcript,
					Spectrogram:   spectrogramSlice,
				}

				if data, err := json.Marshal(payload); err == nil {
					b.hub.BroadcastTelemetry(data)
				}
			}
		}
	}
}
