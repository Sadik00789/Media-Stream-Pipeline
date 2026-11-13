package ipc

import (
	"sync"
	"sync/atomic"
	"time"
)

type VideoRingBufferWriter struct {
	handle *VideoShmHandle
	seq    uint64
	mu     sync.Mutex
}

func NewVideoRingBufferWriter(handle *VideoShmHandle) *VideoRingBufferWriter {
	return &VideoRingBufferWriter{
		handle: handle,
		seq:    0,
	}
}

// WriteVideoFrame writes a 640x480 RGB24 pixel buffer to shared memory ring buffer.
func (w *VideoRingBufferWriter) WriteVideoFrame(pixels []byte, width, height, format int) (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)

	// CAS-protected overrun handling
	for {
		rHead := atomic.LoadUint64(&ring.ReadHead)
		if wHead < rHead+uint64(VideoRingCapacity) {
			break
		}
		newR := wHead - uint64(VideoRingCapacity) + 1
		if atomic.CompareAndSwapUint64(&ring.ReadHead, rHead, newR) {
			atomic.AddUint64(&ring.DroppedFrames, newR-rHead)
			atomic.AddUint64(&ring.OverrunCount, 1)
			break
		}
	}

	slot := int(wHead % uint64(VideoRingCapacity))
	packet := &ring.Packets[slot]

	// Clear Flags before mutating fields so readers never observe stale completion flags
	atomic.StoreUint32(&packet.Flags, 0)

	w.seq++
	packet.SequenceNumber = w.seq
	packet.TimestampNs = uint64(time.Now().UnixNano())
	packet.Width = uint32(width)
	packet.Height = uint32(height)
	packet.Format = uint32(format)
	packet.DetectionCount = 0

	// Fast copy into fixed pixel buffer
	copyLen := len(pixels)
	if copyLen > VideoFrameSize {
		copyLen = VideoFrameSize
	}
	copy(packet.PixelData[:copyLen], pixels[:copyLen])
	if copyLen < VideoFrameSize {
		clear(packet.PixelData[copyLen:])
	}

	atomic.StoreUint32(&packet.Flags, FlagIngested)
	// Atomically commit write cursor
	atomic.StoreUint64(&ring.WriteHead, wHead+1)
	return w.seq, nil
}

// ReadLatestProcessedPacket retrieves a safe copy of the most recently completed Vision ML packet.
func (w *VideoRingBufferWriter) ReadLatestProcessedPacket(lastSeq *uint64) (*VideoFramePacket, bool) {
	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)
	if wHead == 0 {
		return nil, false
	}

	// Look backwards from write head for completed ML frame
	for i := 0; i < VideoRingCapacity && uint64(i) < wHead; i++ {
		idx := wHead - 1 - uint64(i)
		slot := int(idx % uint64(VideoRingCapacity))
		pkt := &ring.Packets[slot]

		flags := atomic.LoadUint32(&pkt.Flags)
		seq := atomic.LoadUint64(&pkt.SequenceNumber)
		if (flags&FlagMlDone) != 0 && seq > *lastSeq {
			copyPkt := *pkt
			if atomic.LoadUint32(&pkt.Flags) != 0 && atomic.LoadUint64(&pkt.SequenceNumber) == seq {
				*lastSeq = seq
				return &copyPkt, true
			}
		}
	}

	return nil, false
}

