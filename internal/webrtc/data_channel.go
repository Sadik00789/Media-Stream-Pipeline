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
	Type             string             `json:"type"`
	Sequence         uint64             `json:"seq"`
	TimestampNs      uint64             `json:"timestamp_ns"`
	VadActive        bool               `json:"vad_active"`
	VadConfidence    float32            `json:"vad_confidence"`
	MlConfidence     float32            `json:"ml_confidence"`
	Transcript       string             `json:"transcript"`
	Spectrogram      []float32          `json:"spectrogram"`
	VideoSequence    uint64             `json:"video_seq,omitempty"`
	VideoTimestampNs uint64             `json:"video_timestamp_ns,omitempty"`
	DetectionCount   uint32             `json:"detection_count,omitempty"`
	Detections       []ipc.DetectionBox `json:"detections,omitempty"`
}

type TelemetryBroadcaster struct {
	audioWriter *ipc.RingBufferWriter
	videoWriter *ipc.VideoRingBufferWriter
	hub         *session.Hub
}

func NewTelemetryBroadcaster(
	audioWriter *ipc.RingBufferWriter,
	videoWriter *ipc.VideoRingBufferWriter,
	hub *session.Hub,
) *TelemetryBroadcaster {
	return &TelemetryBroadcaster{
		audioWriter: audioWriter,
		videoWriter: videoWriter,
		hub:         hub,
	}
}

// StartBroadcasting runs a 60 FPS broadcast loop transmitting completed Audio & Video frames to WebRTC & WebSocket clients
func (b *TelemetryBroadcaster) StartBroadcasting(ctx context.Context) {
	ticker := time.NewTicker(16 * time.Millisecond) // ~60 FPS
	defer ticker.Stop()

	var lastAudioSeq uint64 = 0
	var lastVideoSeq uint64 = 0

	var cachedDetections []ipc.DetectionBox
	var cachedVideoSeq uint64 = 0
	var cachedVideoTs uint64 = 0
	var cachedDetCount uint32 = 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if b.hub.ClientCount() == 0 {
				continue
			}

			// Check for new processed video frames
			hasNewVideo := false
			if b.videoWriter != nil {
				if vPkt, ok := b.videoWriter.ReadLatestProcessedPacket(&lastVideoSeq); ok {
					hasNewVideo = true
					cachedVideoSeq = vPkt.SequenceNumber
					cachedVideoTs = vPkt.TimestampNs
					cachedDetCount = vPkt.DetectionCount

					cnt := int(vPkt.DetectionCount)
					if cnt > ipc.MaxDetections {
						cnt = ipc.MaxDetections
					}
					cachedDetections = make([]ipc.DetectionBox, cnt)
					for i := 0; i < cnt; i++ {
						cachedDetections[i] = vPkt.Detections[i]
					}
				}
			}

			// Check for new processed audio frames
			if pkt, ok := b.audioWriter.ReadLatestProcessedPacket(&lastAudioSeq); ok {
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
					Type:             "telemetry",
					Sequence:         pkt.SequenceNumber,
					TimestampNs:      pkt.TimestampNs,
					VadActive:        pkt.VadActive != 0,
					VadConfidence:    pkt.VadConfidence,
					MlConfidence:     pkt.MlConfidence,
					Transcript:       transcript,
					Spectrogram:      spectrogramSlice,
					VideoSequence:    cachedVideoSeq,
					VideoTimestampNs: cachedVideoTs,
					DetectionCount:   cachedDetCount,
					Detections:       cachedDetections,
				}

				if data, err := json.Marshal(payload); err == nil {
					b.hub.BroadcastTelemetry(data)
				}
			} else if hasNewVideo {
				// Broadcast video-only update if no new audio packet in this 16ms tick
				payload := TelemetryPayload{
					Type:             "telemetry",
					VideoSequence:    cachedVideoSeq,
					VideoTimestampNs: cachedVideoTs,
					DetectionCount:   cachedDetCount,
					Detections:       cachedDetections,
				}

				if data, err := json.Marshal(payload); err == nil {
					b.hub.BroadcastTelemetry(data)
				}
			}
		}
	}
}
