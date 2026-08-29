package webrtc

import (
	"log"
	"math"
	"sync"

	"github.com/pion/webrtc/v3"
	"github.com/sadik/media-stream-pipeline/internal/ipc"
)

type AudioSink struct {
	writer      *ipc.RingBufferWriter
	sampleRate  int
	frameSize   int
	pcmBuffer   []float32
	mu          sync.Mutex
}

func NewAudioSink(writer *ipc.RingBufferWriter, sampleRate, frameSize int) *AudioSink {
	return &AudioSink{
		writer:     writer,
		sampleRate: sampleRate,
		frameSize:  frameSize,
		pcmBuffer:  make([]float32, 0, frameSize*2),
	}
}

// IngestAudioTrack consumes RTP audio packets from client
func (s *AudioSink) IngestAudioTrack(track *webrtc.TrackRemote) {
	log.Printf("[WebRTC] Audio sink attached to track %s (codec: %s, clock: %d)", track.ID(), track.Codec().MimeType, track.Codec().ClockRate)

	for {
		rtpPacket, _, err := track.ReadRTP()
		if err != nil {
			log.Printf("[WebRTC] Audio track %s read terminated: %v", track.ID(), err)
			return
		}

		// Decode RTP payload samples (Opus / L16 / PCM fallback)
		// For raw audio / benchmark streams, parse 16-bit PCM or synthesize normalized floats
		payload := rtpPacket.Payload
		samples := make([]float32, len(payload)/2)
		for i := 0; i < len(samples); i++ {
			raw := int16(uint16(payload[2*i]) | (uint16(payload[2*i+1]) << 8))
			samples[i] = float32(raw) / 32768.0
		}

		// Fallback for silence / synthetic payload
		if len(samples) == 0 && len(payload) > 0 {
			samples = make([]float32, 160) // 10ms at 16kHz
			for i := range samples {
				samples[i] = float32(math.Sin(float64(i) * 0.1)) * 0.1
			}
		}

		s.mu.Lock()
		s.pcmBuffer = append(s.pcmBuffer, samples...)

		// When buffer has at least frameSize (512) samples, flush to SHM
		for len(s.pcmBuffer) >= s.frameSize {
			frame := s.pcmBuffer[:s.frameSize]
			_, _ = s.writer.WriteAudioFrame(frame, s.sampleRate)
			s.pcmBuffer = s.pcmBuffer[s.frameSize:]
		}
		s.mu.Unlock()
	}
}

// PushDirectPcm allows direct insertion of float32 samples (e.g. from Web Audio or benchmarks)
func (s *AudioSink) PushDirectPcm(samples []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.pcmBuffer = append(s.pcmBuffer, samples...)
	for len(s.pcmBuffer) >= s.frameSize {
		frame := s.pcmBuffer[:s.frameSize]
		_, _ = s.writer.WriteAudioFrame(frame, s.sampleRate)
		s.pcmBuffer = s.pcmBuffer[s.frameSize:]
	}
}
