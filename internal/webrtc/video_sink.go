package webrtc

import (
	"log"
	"sync"

	"github.com/pion/webrtc/v3"
	"github.com/sadik/media-stream-pipeline/internal/ipc"
)

type VideoSink struct {
	writer *ipc.VideoRingBufferWriter
	width  int
	height int
	buffer []byte
	mu     sync.Mutex
}

func NewVideoSink(writer *ipc.VideoRingBufferWriter, width, height int) *VideoSink {
	return &VideoSink{
		writer: writer,
		width:  width,
		height: height,
		buffer: make([]byte, width*height*ipc.VideoChannels),
	}
}

// IngestVideoTrack consumes RTP video packets from client
func (s *VideoSink) IngestVideoTrack(track *webrtc.TrackRemote) {
	log.Printf("[WebRTC] Video sink attached to track %s (codec: %s, clock: %d)", track.ID(), track.Codec().MimeType, track.Codec().ClockRate)

	expectedFrameBytes := s.width * s.height * ipc.VideoChannels
	var accumulated []byte

	for {
		rtpPacket, _, err := track.ReadRTP()
		if err != nil {
			log.Printf("[WebRTC] Video track %s read terminated: %v", track.ID(), err)
			return
		}

		payload := rtpPacket.Payload
		if len(payload) == 0 {
			continue
		}

		s.mu.Lock()
		accumulated = append(accumulated, payload...)

		// If accumulated payload reached a full frame size or packet marker
		if len(accumulated) >= expectedFrameBytes || (rtpPacket.Marker && len(accumulated) > 1000) {
			frameData := accumulated
			if len(frameData) > expectedFrameBytes {
				frameData = frameData[:expectedFrameBytes]
			}

			// If frameData is smaller than expected, pad or tile
			if len(frameData) < expectedFrameBytes {
				copy(s.buffer, frameData)
				frameData = s.buffer
			}

			_, _ = s.writer.WriteVideoFrame(frameData, s.width, s.height, ipc.VideoFormatRGB24)
			accumulated = accumulated[:0]
		}
		s.mu.Unlock()
	}
}

// PushDirectVideo allows direct insertion of raw RGB24 bytes (e.g. from Canvas or benchmarks)
func (s *VideoSink) PushDirectVideo(pixels []byte, width, height int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	expectedSize := s.width * s.height * ipc.VideoChannels
	if len(pixels) >= expectedSize {
		_, _ = s.writer.WriteVideoFrame(pixels[:expectedSize], s.width, s.height, ipc.VideoFormatRGB24)
	} else if len(pixels) > 0 {
		copy(s.buffer, pixels)
		_, _ = s.writer.WriteVideoFrame(s.buffer, s.width, s.height, ipc.VideoFormatRGB24)
	}
}
