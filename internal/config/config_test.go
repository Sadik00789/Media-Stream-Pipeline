package config

import "testing"

func TestLoadDefaultsAndOverrides(t *testing.T) {
	t.Setenv("HTTP_PORT", "")
	t.Setenv("SHM_NAME", "")
	t.Setenv("VIDEO_SHM_NAME", "")
	t.Setenv("SAMPLE_RATE", "")
	t.Setenv("FRAME_SAMPLES", "")

	cfg := Load()
	if cfg.HTTPPort != "8080" {
		t.Fatalf("expected default HTTPPort 8080, got %s", cfg.HTTPPort)
	}
	if cfg.ShmPath != "/media_stream_ring" {
		t.Fatalf("expected default ShmPath /media_stream_ring, got %s", cfg.ShmPath)
	}
	if cfg.VideoShmPath != "/video_stream_ring" {
		t.Fatalf("expected default VideoShmPath /video_stream_ring, got %s", cfg.VideoShmPath)
	}
	if cfg.SampleRate != 16000 || cfg.FrameSamples != 512 || cfg.SpectrogramBins != 256 {
		t.Fatalf("unexpected audio defaults: %+v", cfg)
	}
	if cfg.VideoWidth != 640 || cfg.VideoHeight != 480 {
		t.Fatalf("unexpected video defaults: %+v", cfg)
	}

	t.Setenv("HTTP_PORT", "9090")
	t.Setenv("SAMPLE_RATE", "48000")
	t.Setenv("FRAME_SAMPLES", "not-an-int")

	cfg2 := Load()
	if cfg2.HTTPPort != "9090" {
		t.Fatalf("expected HTTPPort 9090, got %s", cfg2.HTTPPort)
	}
	if cfg2.SampleRate != 48000 {
		t.Fatalf("expected SampleRate 48000, got %d", cfg2.SampleRate)
	}
	if cfg2.FrameSamples != 512 {
		t.Fatalf("expected invalid int fallback 512, got %d", cfg2.FrameSamples)
	}
}
