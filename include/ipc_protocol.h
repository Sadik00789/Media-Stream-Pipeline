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

/* Packet processing state flags */
#define FLAG_INGESTED   (1 << 0)  /* Written by WebRTC Gateway */
#define FLAG_DSP_DONE   (1 << 1)  /* DSP (FFT/VAD) processed */
#define FLAG_ML_DONE    (1 << 2)  /* ML inference processed */
#define FLAG_BROADCAST  (1 << 3)  /* Telemetry transmitted to client */

#pragma pack(push, 1)

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
 * @struct SharedMemoryRingBuffer
 * @brief Lock-free SPSC / MPMC ring buffer layout in POSIX shared memory (/dev/shm).
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

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* IPC_PROTOCOL_H */