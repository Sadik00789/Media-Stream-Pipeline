package ipc

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestABIStructSizes(t *testing.T) {
	if got := unsafe.Sizeof(DetectionBox{}); got != 24 {
		t.Fatalf("DetectionBox size = %d, want 24", got)
	}
	if got := unsafe.Sizeof(AudioFramePacket{}); got != 3248 {
		t.Fatalf("AudioFramePacket size = %d, want 3248", got)
	}
	if got := unsafe.Sizeof(VideoFramePacket{}); got != 922024 {
		t.Fatalf("VideoFramePacket size = %d, want 922024", got)
	}
	if got := unsafe.Sizeof(SharedMemoryRingBuffer{}); got != 48+RingCapacity*3248 {
		t.Fatalf("SharedMemoryRingBuffer size = %d, want %d", got, 48+RingCapacity*3248)
	}
	if got := unsafe.Sizeof(SharedMemoryVideoRingBuffer{}); got != 48+VideoRingCapacity*922024 {
		t.Fatalf("SharedMemoryVideoRingBuffer size = %d, want %d", got, 48+VideoRingCapacity*922024)
	}
}

func TestAudioShmWriteReadAndOverrun(t *testing.T) {
	name := fmt.Sprintf("/test_go_audio_%d", os.Getpid())
	handle, err := CreateOrOpenShm(name)
	if err != nil {
		t.Fatalf("CreateOrOpenShm failed: %v", err)
	}
	defer func() { _ = handle.Unlink() }()

	writer := NewRingBufferWriter(handle)
	pcm := make([]float32, FrameSamples)
	pcm[0] = 0.5

	seq1, err := writer.WriteAudioFrame(pcm, 16000)
	if err != nil || seq1 != 1 {
		t.Fatalf("WriteAudioFrame seq=%d err=%v", seq1, err)
	}

	var lastSeq uint64
	if _, ok := writer.ReadLatestProcessedPacket(&lastSeq); ok {
		t.Fatal("expected no processed packet before worker sets FlagDspDone/FlagMlDone")
	}

	// Simulate worker completing slot 0
	atomic.StoreUint32(&handle.Ring.Packets[0].Flags, FlagIngested|FlagDspDone|FlagMlDone)
	handle.Ring.Packets[0].VadActive = 1
	handle.Ring.Packets[0].MlConfidence = 0.91

	pkt, ok := writer.ReadLatestProcessedPacket(&lastSeq)
	if !ok || pkt.SequenceNumber != 1 || pkt.VadActive != 1 {
		t.Fatalf("unexpected processed packet: ok=%v pkt=%+v", ok, pkt)
	}

	// Trigger overrun (> RingCapacity frames without advancing ReadHead)
	for i := 0; i < RingCapacity+10; i++ {
		if _, err := writer.WriteAudioFrame(pcm, 16000); err != nil {
			t.Fatalf("WriteAudioFrame overrun loop failed: %v", err)
		}
	}
	if atomic.LoadUint64(&handle.Ring.OverrunCount) == 0 {
		t.Fatal("expected OverrunCount > 0")
	}
	if atomic.LoadUint64(&handle.Ring.DroppedFrames) == 0 {
		t.Fatal("expected DroppedFrames > 0")
	}
}

func TestVideoShmWriteReadAndOverrun(t *testing.T) {
	name := fmt.Sprintf("/test_go_video_%d", os.Getpid())
	handle, err := CreateOrOpenVideoShm(name)
	if err != nil {
		t.Fatalf("CreateOrOpenVideoShm failed: %v", err)
	}
	defer func() { _ = handle.Unlink() }()

	writer := NewVideoRingBufferWriter(handle)
	pixels := make([]byte, VideoFrameSize)
	pixels[0] = 220

	seq1, err := writer.WriteVideoFrame(pixels, VideoWidth, VideoHeight, VideoFormatRGB24)
	if err != nil || seq1 != 1 {
		t.Fatalf("WriteVideoFrame seq=%d err=%v", seq1, err)
	}

	var lastSeq uint64
	if _, ok := writer.ReadLatestProcessedPacket(&lastSeq); ok {
		t.Fatal("expected no processed video packet before FlagMlDone")
	}

	handle.Ring.Packets[0].DetectionCount = 1
	handle.Ring.Packets[0].Detections[0] = DetectionBox{Ymin: 0.1, Xmin: 0.1, Ymax: 0.5, Xmax: 0.5, Confidence: 0.95}
	atomic.StoreUint32(&handle.Ring.Packets[0].Flags, FlagIngested|FlagMlDone)

	pkt, ok := writer.ReadLatestProcessedPacket(&lastSeq)
	if !ok || pkt.SequenceNumber != 1 || pkt.DetectionCount != 1 {
		t.Fatalf("unexpected processed video packet: ok=%v", ok)
	}

	// Overrun
	for i := 0; i < VideoRingCapacity+5; i++ {
		if _, err := writer.WriteVideoFrame(pixels, VideoWidth, VideoHeight, VideoFormatRGB24); err != nil {
			t.Fatalf("WriteVideoFrame overrun failed: %v", err)
		}
	}
	if atomic.LoadUint64(&handle.Ring.OverrunCount) == 0 {
		t.Fatal("expected video OverrunCount > 0")
	}
}

func TestConcurrentRingBufferWritersRaceFree(t *testing.T) {
	name := fmt.Sprintf("/test_go_race_%d", os.Getpid())
	handle, err := CreateOrOpenShm(name)
	if err != nil {
		t.Fatalf("CreateOrOpenShm failed: %v", err)
	}
	defer func() { _ = handle.Unlink() }()

	writer := NewRingBufferWriter(handle)
	pcm := make([]float32, FrameSamples)

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				_, _ = writer.WriteAudioFrame(pcm, 16000)
			}
		}()
	}
	wg.Wait()

	if atomic.LoadUint64(&handle.Ring.WriteHead) != 400 {
		t.Fatalf("expected WriteHead 400, got %d", atomic.LoadUint64(&handle.Ring.WriteHead))
	}
}
