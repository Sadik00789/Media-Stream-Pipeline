package main

import (
	"flag"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sadik/media-stream-pipeline/internal/ipc"
)

func main() {
	shmName := flag.String("shm", "/media_stream_ring", "POSIX Shared Memory Name")
	tracks := flag.Int("tracks", 50, "Number of concurrent synthetic audio tracks")
	durationSec := flag.Int("duration", 10, "Benchmark duration in seconds")
	sampleRate := flag.Int("rate", 16000, "Audio sample rate in Hz")
	flag.Parse()

	fmt.Printf("[Benchmark] Initializing high-load synthetic client...\n")
	fmt.Printf("  -> Shared Memory: %s\n", *shmName)
	fmt.Printf("  -> Concurrent Tracks: %d\n", *tracks)
	fmt.Printf("  -> Sample Rate: %d Hz (512 samples/frame = 32ms)\n", *sampleRate)

	shmHandle, err := ipc.CreateOrOpenShm(*shmName)
	if err != nil {
		fmt.Printf("[Error] Failed to attach to SHM: %v\n", err)
		return
	}
	defer shmHandle.Close()

	writer := ipc.NewRingBufferWriter(shmHandle)

	var totalFramesWritten uint64
	var wg sync.WaitGroup
	stopChan := make(chan struct{})

	// Sine wave generator (1000 Hz)
	syntheticFrame := make([]float32, ipc.FrameSamples)
	for i := range syntheticFrame {
		syntheticFrame[i] = float32(math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / float64(*sampleRate)))
	}

	start := time.Now()

	for t := 0; t < *tracks; t++ {
		wg.Add(1)
		go func(trackID int) {
			defer wg.Done()
			ticker := time.NewTicker(32 * time.Millisecond) // Frame period
			defer ticker.Stop()

			for {
				select {
				case <-stopChan:
					return
				case <-ticker.C:
					_, err := writer.WriteAudioFrame(syntheticFrame, *sampleRate)
					if err == nil {
						atomic.AddUint64(&totalFramesWritten, 1)
					}
				}
			}
		}(t)
	}

	// Run for duration
	time.Sleep(time.Duration(*durationSec) * time.Second)
	close(stopChan)
	wg.Wait()

	elapsed := time.Since(start).Seconds()
	written := atomic.LoadUint64(&totalFramesWritten)
	fps := float64(written) / elapsed

	fmt.Printf("\n=== Benchmark Results ===\n")
	fmt.Printf("Duration:         %.2f s\n", elapsed)
	fmt.Printf("Total Frames:     %d\n", written)
	fmt.Printf("Throughput:       %.2f frames/sec (%.2f audio streams equivalent)\n", fps, fps/31.25)
	fmt.Printf("Ingestion Load:   %.2f MB/sec\n", (float64(written)*float64(ipc.FrameSamples*4))/(1024*1024*elapsed))
}
