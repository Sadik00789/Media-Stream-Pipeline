package config

import (
	"os"
	"strconv"
)

type Config struct {
	HTTPPort       string
	ShmPath        string
	SampleRate     int
	FrameSamples   int
	SpectrogramBins int
	WebStaticDir   string
}

func Load() *Config {
	cfg := &Config{
		HTTPPort:        getEnv("HTTP_PORT", "8080"),
		ShmPath:         getEnv("SHM_NAME", "/media_stream_ring"),
		SampleRate:      getEnvInt("SAMPLE_RATE", 16000),
		FrameSamples:    getEnvInt("FRAME_SAMPLES", 512),
		SpectrogramBins: getEnvInt("SPECTROGRAM_BINS", 256),
		WebStaticDir:    getEnv("WEB_STATIC_DIR", "../../web/dist"),
	}
	return cfg
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
}
