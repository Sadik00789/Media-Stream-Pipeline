package ipc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	VideoWidth        = 640
	VideoHeight       = 480
	VideoChannels     = 3
	VideoFrameSize    = VideoWidth * VideoHeight * VideoChannels // 921600 bytes
	VideoRingCapacity = 8
	VideoShmMagic     = 0x56494445 // 'VIDE'
	VideoShmVersion   = 1

	VideoFormatRGB24   = 0
	VideoFormatYUV420P = 1
	MaxDetections      = 16
)

type DetectionBox struct {
	Ymin       float32 `json:"ymin"`
	Xmin       float32 `json:"xmin"`
	Ymax       float32 `json:"ymax"`
	Xmax       float32 `json:"xmax"`
	Confidence float32 `json:"confidence"`
	ClassID    uint32  `json:"class_id"`
}

type VideoFramePacket struct {
	SequenceNumber uint64
	TimestampNs    uint64
	Width          uint32
	Height         uint32
	Format         uint32
	DetectionCount uint32
	Flags          uint32
	Reserved       uint32
	Detections     [MaxDetections]DetectionBox
	PixelData      [VideoFrameSize]byte
}

type SharedMemoryVideoRingBuffer struct {
	Magic          uint32
	Version        uint32
	BufferCapacity uint32
	PacketSize     uint32
	WriteHead      uint64
	ReadHead       uint64
	DroppedFrames  uint64
	OverrunCount   uint64
	Packets        [VideoRingCapacity]VideoFramePacket
}

type VideoShmHandle struct {
	Name    string
	ShmPath string
	Fd      int
	Size    int
	Data    []byte
	Ring    *SharedMemoryVideoRingBuffer
}

func CreateOrOpenVideoShm(name string) (*VideoShmHandle, error) {
	cleanName := strings.TrimPrefix(name, "/")
	shmPath := filepath.Join("/dev/shm", cleanName)
	size := int(unsafe.Sizeof(SharedMemoryVideoRingBuffer{}))

	fd, err := unix.Open(shmPath, unix.O_RDWR|unix.O_CREAT, 0666)
	if err != nil {
		return nil, fmt.Errorf("open failed for %s: %w", shmPath, err)
	}

	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("ftruncate failed for %s: %w", shmPath, err)
	}

	data, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("mmap failed for %s: %w", shmPath, err)
	}

	ring := (*SharedMemoryVideoRingBuffer)(unsafe.Pointer(&data[0]))
	if ring.Magic != VideoShmMagic {
		ring.Magic = VideoShmMagic
		ring.Version = VideoShmVersion
		ring.BufferCapacity = VideoRingCapacity
		ring.PacketSize = uint32(unsafe.Sizeof(VideoFramePacket{}))
		ring.WriteHead = 0
		ring.ReadHead = 0
		ring.DroppedFrames = 0
		ring.OverrunCount = 0
	}

	return &VideoShmHandle{
		Name:    name,
		ShmPath: shmPath,
		Fd:      fd,
		Size:    size,
		Data:    data,
		Ring:    ring,
	}, nil
}

func (s *VideoShmHandle) Close() error {
	if s.Data != nil {
		_ = unix.Munmap(s.Data)
		s.Data = nil
	}
	if s.Fd >= 0 {
		_ = unix.Close(s.Fd)
		s.Fd = -1
	}
	return nil
}

func (s *VideoShmHandle) Unlink() error {
	_ = s.Close()
	if s.ShmPath != "" {
		return os.Remove(s.ShmPath)
	}
	cleanName := strings.TrimPrefix(s.Name, "/")
	return os.Remove(filepath.Join("/dev/shm", cleanName))
}

type VideoRingBufferWriter struct {
	handle *VideoShmHandle
	seq    uint64
}

func NewVideoRingBufferWriter(handle *VideoShmHandle) *VideoRingBufferWriter {
	return &VideoRingBufferWriter{
		handle: handle,
		seq:    0,
	}
}

// WriteVideoFrame writes a 640x480 RGB24 pixel buffer to shared memory ring buffer.
func (w *VideoRingBufferWriter) WriteVideoFrame(pixels []byte, width, height, format int) (uint64, error) {
	ring := w.handle.Ring
	wHead := atomic.LoadUint64(&ring.WriteHead)
	rHead := atomic.LoadUint64(&ring.ReadHead)

	// Overrun handling
	if wHead >= rHead+uint64(VideoRingCapacity) {
		newR := wHead - uint64(VideoRingCapacity) + 1
		dropped := newR - rHead
		atomic.AddUint64(&ring.DroppedFrames, dropped)
		atomic.AddUint64(&ring.OverrunCount, 1)
		atomic.StoreUint64(&ring.ReadHead, newR)
	}

	slot := int(wHead % uint64(VideoRingCapacity))
	packet := &ring.Packets[slot]

	w.seq++
	packet.SequenceNumber = w.seq
	packet.TimestampNs = uint64(time.Now().UnixNano())
	packet.Width = uint32(width)
	packet.Height = uint32(height)
	packet.Format = uint32(format)
	packet.DetectionCount = 0
	packet.Flags = FlagIngested

	// Fast copy into fixed pixel buffer
	copyLen := len(pixels)
	if copyLen > VideoFrameSize {
		copyLen = VideoFrameSize
	}
	copy(packet.PixelData[:copyLen], pixels[:copyLen])

	// Atomically commit write cursor
	atomic.StoreUint64(&ring.WriteHead, wHead+1)
	return w.seq, nil
}

// ReadLatestProcessedPacket retrieves the most recently completed Vision ML packet
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

		if (pkt.Flags&FlagMlDone) != 0 && pkt.SequenceNumber > *lastSeq {
			*lastSeq = pkt.SequenceNumber
			return pkt, true
		}
	}

	return nil, false
}
