/**
 * @file ipc_protocol.h
 * @brief Cross-language binary protocol for ultra-low latency shared memory IPC.
 * 
 * Synchronized across C11, Go, and Rust.
 * All structs use strict byte-level packing (#pragma pack(push, 1))
 * with 8-byte/4-byte natural alignments.
 */

#ifndef IPC_PROTOCOL_H
#define IPC_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRAME_SAMPLES 512
#define SPECTROGRAM_BINS 256
#define SHM_RING_CAPACITY 64
#define SHM_NAME "/media_stream_ring"
#define SHM_MAGIC 0x4D535452 /* 'MSTR' */
#define SHM_VERSION 1

/* Video Stream IPC Constants */
#define VIDEO_WIDTH 640
#define VIDEO_HEIGHT 480
#define VIDEO_CHANNELS 3
#define VIDEO_FRAME_SIZE (VIDEO_WIDTH * VIDEO_HEIGHT * VIDEO_CHANNELS) /* 921600 bytes */
#define VIDEO_RING_CAPACITY 8
#define VIDEO_SHM_NAME "/video_stream_ring"
#define VIDEO_SHM_MAGIC 0x56494445 /* 'VIDE' */
#define VIDEO_SHM_VERSION 1

#define VIDEO_FORMAT_RGB24   0
#define VIDEO_FORMAT_YUV420P 1
#define MAX_DETECTIONS 16

/* Packet processing state flags */
#define FLAG_INGESTED   (1 << 0)  /* Written by WebRTC Gateway */
#define FLAG_DSP_DONE   (1 << 1)  /* DSP (FFT/VAD) processed */
#define FLAG_ML_DONE    (1 << 2)  /* ML inference processed */
#define FLAG_BROADCAST  (1 << 3)  /* Telemetry transmitted to client */

#pragma pack(push, 1)

/**
 * @struct DetectionBox
 * @brief Represents an object/face detection bounding box with confidence and class.
 * Total size: 24 bytes.
 */
typedef struct {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
    float confidence;
    uint32_t class_id;
} DetectionBox;

/**
 * @struct AudioFramePacket
 * @brief Represents a single 512-sample audio frame with associated DSP & ML metadata.
 * Total size: 3248 bytes.
 */
typedef struct {
    uint64_t sequence_number;            /* Offset 0   (8 bytes) */
    uint64_t timestamp_ns;               /* Offset 8   (8 bytes) */
    uint32_t sample_rate;                /* Offset 16  (4 bytes) */
    uint32_t channels;                   /* Offset 20  (4 bytes) */
    uint32_t sample_count;               /* Offset 24  (4 bytes) */
    uint32_t vad_active;                 /* Offset 28  (4 bytes) */
    float vad_confidence;                /* Offset 32  (4 bytes) */
    float ml_confidence;                 /* Offset 36  (4 bytes) */
    uint32_t flags;                      /* Offset 40  (4 bytes) */
    uint32_t reserved;                   /* Offset 44  (4 bytes) */
    float pcm_data[FRAME_SAMPLES];       /* Offset 48  (2048 bytes) */
    float spectrogram[SPECTROGRAM_BINS]; /* Offset 2096 (1024 bytes) */
    char ml_transcript[128];             /* Offset 3120 (128 bytes) */
} AudioFramePacket;

/**
 * @struct VideoFramePacket
 * @brief Represents a single 640x480 RGB24 video frame with vision detection metadata.
 * Total size: 32 + 16*24 + 921600 = 922016 bytes.
 */
typedef struct {
    uint64_t sequence_number;            /* Offset 0   (8 bytes) */
    uint64_t timestamp_ns;               /* Offset 8   (8 bytes) */
    uint32_t width;                      /* Offset 16  (4 bytes) */
    uint32_t height;                     /* Offset 20  (4 bytes) */
    uint32_t format;                     /* Offset 24  (4 bytes: 0=RGB24, 1=YUV420p) */
    uint32_t detection_count;            /* Offset 28  (4 bytes) */
    uint32_t flags;                      /* Offset 32  (4 bytes) */
    uint32_t reserved;                   /* Offset 36  (4 bytes) */
    DetectionBox detections[MAX_DETECTIONS]; /* Offset 40  (384 bytes) */
    uint8_t pixel_data[VIDEO_FRAME_SIZE];    /* Offset 424 (921600 bytes) */
} VideoFramePacket;

/**
 * @struct SharedMemoryRingBuffer
 * @brief Lock-free SPSC / MPMC audio ring buffer layout in POSIX shared memory (/dev/shm).
 * Total size: 48 + 64 * 3248 = 207920 bytes.
 */
typedef struct {
    uint32_t magic;                      /* Validation magic (SHM_MAGIC) */
    uint32_t version;                    /* Protocol schema version */
    uint32_t buffer_capacity;            /* Maximum frame slots in ring buffer (64) */
    uint32_t packet_size;                /* Byte size of AudioFramePacket (3248) */
    uint64_t write_head;                 /* Monotonically increasing write cursor */
    uint64_t read_head;                  /* Monotonically increasing read cursor */
    uint64_t dropped_frames;             /* Counter for overrun discarded frames */
    uint64_t overrun_count;              /* Number of overrun incidents */
    AudioFramePacket packets[SHM_RING_CAPACITY]; /* Contiguous packet slots */
} SharedMemoryRingBuffer;

/**
 * @struct SharedMemoryVideoRingBuffer
 * @brief Lock-free SPSC video ring buffer layout in POSIX shared memory (/dev/shm).
 * Total size: 48 + 8 * 922024 = ~7.37 MB.
 */
typedef struct {
    uint32_t magic;                      /* Validation magic (VIDEO_SHM_MAGIC) */
    uint32_t version;                    /* Protocol schema version */
    uint32_t buffer_capacity;            /* Maximum frame slots (8) */
    uint32_t packet_size;                /* Byte size of VideoFramePacket */
    uint64_t write_head;                 /* Monotonically increasing write cursor */
    uint64_t read_head;                  /* Monotonically increasing read cursor */
    uint64_t dropped_frames;             /* Overrun discarded frames */
    uint64_t overrun_count;              /* Overrun incident counter */
    VideoFramePacket packets[VIDEO_RING_CAPACITY];
} SharedMemoryVideoRingBuffer;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* IPC_PROTOCOL_H */