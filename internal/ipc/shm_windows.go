package ipc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

type winMmap struct {
	shmPath   string
	file      *os.File
	mapHandle windows.Handle
	viewAddr  uintptr
	data      []byte
}

func openWinMmap(name string, size int) (*winMmap, error) {
	cleanName := strings.TrimPrefix(name, "/")
	shmPath := filepath.Join(os.TempDir(), cleanName)

	file, err := os.OpenFile(shmPath, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return nil, fmt.Errorf("open failed for %s: %w", shmPath, err)
	}

	fi, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("stat failed for %s: %w", shmPath, err)
	}
	if fi.Size() < int64(size) {
		if err := file.Truncate(int64(size)); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("truncate failed for %s: %w", shmPath, err)
		}
	}

	high := uint32(uint64(size) >> 32)
	low := uint32(uint64(size))
	mapHandle, err := windows.CreateFileMapping(windows.Handle(file.Fd()), nil, windows.PAGE_READWRITE, high, low, nil)
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("CreateFileMapping failed for %s: %w", shmPath, err)
	}

	viewAddr, err := windows.MapViewOfFile(mapHandle, windows.FILE_MAP_READ|windows.FILE_MAP_WRITE, 0, 0, uintptr(size))
	if err != nil {
		_ = windows.CloseHandle(mapHandle)
		_ = file.Close()
		return nil, fmt.Errorf("MapViewOfFile failed for %s: %w", shmPath, err)
	}

	// MapViewOfFile returns an OS unmanaged virtual address (not tracked by Go GC).
	basePtr := *(*unsafe.Pointer)(unsafe.Pointer(&viewAddr))
	data := unsafe.Slice((*byte)(basePtr), size)
	return &winMmap{
		shmPath:   shmPath,
		file:      file,
		mapHandle: mapHandle,
		viewAddr:  viewAddr,
		data:      data,
	}, nil
}

func (m *winMmap) close() error {
	if m.viewAddr != 0 {
		_ = windows.UnmapViewOfFile(m.viewAddr)
		m.viewAddr = 0
		m.data = nil
	}
	if m.mapHandle != 0 {
		_ = windows.CloseHandle(m.mapHandle)
		m.mapHandle = 0
	}
	if m.file != nil {
		_ = m.file.Close()
		m.file = nil
	}
	return nil
}

type ShmHandle struct {
	Name    string
	ShmPath string
	Size    int
	Data    []byte
	Ring    *SharedMemoryRingBuffer
	mmap    *winMmap
}

func CreateOrOpenShm(name string) (*ShmHandle, error) {
	size := int(unsafe.Sizeof(SharedMemoryRingBuffer{}))
	m, err := openWinMmap(name, size)
	if err != nil {
		return nil, err
	}

	ring := (*SharedMemoryRingBuffer)(unsafe.Pointer(&m.data[0]))
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
		ShmPath: m.shmPath,
		Size:    size,
		Data:    m.data,
		Ring:    ring,
		mmap:    m,
	}, nil
}

func (s *ShmHandle) Close() error {
	s.Data = nil
	s.Ring = nil
	if s.mmap != nil {
		return s.mmap.close()
	}
	return nil
}

func (s *ShmHandle) Unlink() error {
	_ = s.Close()
	if s.ShmPath != "" {
		return os.Remove(s.ShmPath)
	}
	cleanName := strings.TrimPrefix(s.Name, "/")
	return os.Remove(filepath.Join(os.TempDir(), cleanName))
}
