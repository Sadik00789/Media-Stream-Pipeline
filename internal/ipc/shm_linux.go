package ipc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"golang.org/x/sys/unix"
)

type ShmHandle struct {
	Name    string
	ShmPath string
	Fd      int
	Size    int
	Data    []byte
	Ring    *SharedMemoryRingBuffer
}

func CreateOrOpenShm(name string) (*ShmHandle, error) {
	cleanName := strings.TrimPrefix(name, "/")
	shmPath := filepath.Join("/dev/shm", cleanName)
	size := int(unsafe.Sizeof(SharedMemoryRingBuffer{}))

	fd, err := unix.Open(shmPath, unix.O_RDWR|unix.O_CREAT, 0666)
	if err != nil {
		return nil, fmt.Errorf("open failed for %s: %w", shmPath, err)
	}

	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("fstat failed for %s: %w", shmPath, err)
	}
	if st.Size < int64(size) {
		if err := unix.Ftruncate(fd, int64(size)); err != nil {
			_ = unix.Close(fd)
			return nil, fmt.Errorf("ftruncate failed for %s: %w", shmPath, err)
		}
	}

	data, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("mmap failed for %s: %w", shmPath, err)
	}

	ring := (*SharedMemoryRingBuffer)(unsafe.Pointer(&data[0]))
	if ring.Magic != ShmMagic {
		ring.Magic = ShmMagic
		ring.Version = ShmVersion
		ring.BufferCapacity = RingCapacity
		ring.PacketSize = uint32(unsafe.Sizeof(AudioFramePacket{}))
		ring.WriteHead = 0
		ring.ReadHead = 0
		ring.DroppedFrames = 0
		ring.OverrunCount = 0
	}

	return &ShmHandle{
		Name:    name,
		ShmPath: shmPath,
		Fd:      fd,
		Size:    size,
		Data:    data,
		Ring:    ring,
	}, nil
}

func (s *ShmHandle) Close() error {
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

func (s *ShmHandle) Unlink() error {
	_ = s.Close()
	if s.ShmPath != "" {
		return os.Remove(s.ShmPath)
	}
	cleanName := strings.TrimPrefix(s.Name, "/")
	return os.Remove(filepath.Join("/dev/shm", cleanName))
}
