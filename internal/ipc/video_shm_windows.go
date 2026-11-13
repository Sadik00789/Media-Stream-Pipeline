package ipc

import (
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

type VideoShmHandle struct {
	Name    string
	ShmPath string
	Size    int
	Data    []byte
	Ring    *SharedMemoryVideoRingBuffer
	mmap    *winMmap
}

func CreateOrOpenVideoShm(name string) (*VideoShmHandle, error) {
	size := int(unsafe.Sizeof(SharedMemoryVideoRingBuffer{}))
	m, err := openWinMmap(name, size)
	if err != nil {
		return nil, err
	}

	ring := (*SharedMemoryVideoRingBuffer)(unsafe.Pointer(m.viewAddr))
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
		ShmPath: m.shmPath,
		Size:    size,
		Data:    m.data,
		Ring:    ring,
		mmap:    m,
	}, nil
}

func (s *VideoShmHandle) Close() error {
	s.Data = nil
	s.Ring = nil
	if s.mmap != nil {
		return s.mmap.close()
	}
	return nil
}

func (s *VideoShmHandle) Unlink() error {
	_ = s.Close()
	if s.ShmPath != "" {
		return os.Remove(s.ShmPath)
	}
	cleanName := strings.TrimPrefix(s.Name, "/")
	return os.Remove(filepath.Join(os.TempDir(), cleanName))
}

