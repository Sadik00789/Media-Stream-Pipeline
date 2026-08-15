package ipc

import (
	"sync/atomic"
	"time"
)

type RingBufferWriter struct {
	handle *ShmHandle
	seq    uint64
}

func NewRingBufferWriter(handle *ShmHandle) *RingBufferWriter {
	return &RingBufferWriter{
		handle: handle,
		seq:    0,
	}
}

// WriteAudioFrame writes a 512-sample float32 PCM frame to the shared memory ring buffer.
func (w *RingBufferWriter) WriteAudioFrame(pcm []float32, sampleRate int) (uint64, error) {
	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)
	rHead := atomic.LoadUint64(&ring.ReadHead)

	// Check if writer wrapped around reader (overrun handling)
	if wHead >= rHead+uint64(RingCapacity) {
		newR := wHead - uint64(RingCapacity) + 1
		dropped := newR - rHead
		atomic.AddUint64(&ring.DroppedFrames, dropped)
		atomic.AddUint64(&ring.OverrunCount, 1)
		atomic.StoreUint64(&ring.ReadHead, newR)
	}

	slot := int(wHead % uint64(RingCapacity))
	packet := &ring.Packets[slot]

	w.seq++
	packet.SequenceNumber = w.seq
	packet.TimestampNs = uint64(time.Now().UnixNano())
	packet.SampleRate = uint32(sampleRate)
	packet.Channels = 1
	packet.SampleCount = uint32(len(pcm))
	packet.Flags = FlagIngested

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

	// Atomically commit write cursor
	atomic.StoreUint64(&ring.WriteHead, wHead+1)
	return w.seq, nil
}

// ReadLatestProcessedPacket retrieves the most recently completed DSP/ML packet
func (w *RingBufferWriter) ReadLatestProcessedPacket(lastSeq *uint64) (*AudioFramePacket, bool) {
	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)
	if wHead == 0 {
		return nil, false
	}

	// Look backwards from write head for completed DSP frame
	for i := 0; i < RingCapacity && uint64(i) < wHead; i++ {
		idx := wHead - 1 - uint64(i)
		slot := int(idx % uint64(RingCapacity))
		pkt := &ring.Packets[slot]

		if (pkt.Flags&FlagDspDone) != 0 && pkt.SequenceNumber > *lastSeq {
			*lastSeq = pkt.SequenceNumber
			return pkt, true
		}
	}

	return nil, false
}
