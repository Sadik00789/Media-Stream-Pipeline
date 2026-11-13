package ipc

import (
	"sync"
	"sync/atomic"
	"time"
)

type RingBufferWriter struct {
	handle *ShmHandle
	seq    uint64
	mu     sync.Mutex
}

func NewRingBufferWriter(handle *ShmHandle) *RingBufferWriter {
	return &RingBufferWriter{
		handle: handle,
		seq:    0,
	}
}

// WriteAudioFrame writes a 512-sample float32 PCM frame to the shared memory ring buffer.
func (w *RingBufferWriter) WriteAudioFrame(pcm []float32, sampleRate int) (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)

	// Check if writer wrapped around reader (CAS-protected overrun handling)
	for {
		rHead := atomic.LoadUint64(&ring.ReadHead)
		if wHead < rHead+uint64(RingCapacity) {
			break
		}
		newR := wHead - uint64(RingCapacity) + 1
		if atomic.CompareAndSwapUint64(&ring.ReadHead, rHead, newR) {
			atomic.AddUint64(&ring.DroppedFrames, newR-rHead)
			atomic.AddUint64(&ring.OverrunCount, 1)
			break
		}
	}

	slot := int(wHead % uint64(RingCapacity))
	packet := &ring.Packets[slot]

	// Clear Flags before mutating fields so readers never observe stale completion flags
	atomic.StoreUint32(&packet.Flags, 0)

	w.seq++
	packet.SequenceNumber = w.seq
	packet.TimestampNs = uint64(time.Now().UnixNano())
	packet.SampleRate = uint32(sampleRate)
	packet.Channels = 1
	packet.SampleCount = uint32(len(pcm))
	packet.VadActive = 0
	packet.VadConfidence = 0
	packet.MlConfidence = 0
	packet.MlTranscript[0] = 0

	// Fast copy into fixed float32 buffer
	copyLen := len(pcm)
	if copyLen > FrameSamples {
		copyLen = FrameSamples
	}
	for i := 0; i < copyLen; i++ {
		packet.PcmData[i] = pcm[i]
	}
	for i := copyLen; i < FrameSamples; i++ {
		packet.PcmData[i] = 0.0
	}

	atomic.StoreUint32(&packet.Flags, FlagIngested)
	// Atomically commit write cursor
	atomic.StoreUint64(&ring.WriteHead, wHead+1)
	return w.seq, nil
}

// ReadLatestProcessedPacket retrieves a safe copy of the most recently completed DSP/ML packet.
func (w *RingBufferWriter) ReadLatestProcessedPacket(lastSeq *uint64) (*AudioFramePacket, bool) {
	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)
	if wHead == 0 {
		return nil, false
	}

	// Look backwards from write head for completed ML or DSP frame
	for i := 0; i < RingCapacity && uint64(i) < wHead; i++ {
		idx := wHead - 1 - uint64(i)
		slot := int(idx % uint64(RingCapacity))
		pkt := &ring.Packets[slot]

		flags := atomic.LoadUint32(&pkt.Flags)
		seq := atomic.LoadUint64(&pkt.SequenceNumber)
		if (flags&(FlagMlDone|FlagDspDone)) != 0 && seq > *lastSeq {
			copyPkt := *pkt
			// Re-verify slot wasn't overwritten mid-copy
			if atomic.LoadUint32(&pkt.Flags) != 0 && atomic.LoadUint64(&pkt.SequenceNumber) == seq {
				*lastSeq = seq
				return &copyPkt, true
			}
		}
	}

	return nil, false
}
