/**
 * UI live metrics and streaming transcript token renderer.
 */

import { DetectionBox } from '../video/overlay';

export interface TelemetryData {
  type?: string;
  seq?: number;
  sequence_number?: number;
  sequenceNumber?: number;
  timestamp_ns?: number;
  timestampNs?: number;
  vad_active?: boolean;
  vadActive?: boolean;
  vad_confidence?: number;
  vadConfidence?: number;
  ml_confidence?: number;
  mlConfidence?: number;
  ml_transcript?: string;
  mlTranscript?: string;
  transcript?: string;
  spectrogram?: number[];
  video_seq?: number;
  video_timestamp_ns?: number;
  detection_count?: number;
  detections_count?: number;
  detections?: DetectionBox[];
  audio_latency_ms?: number;
  audioLatencyMs?: number;
  vision_latency_ms?: number;
  visionLatencyMs?: number;
}

export class MetricsRenderer {
  private vadBadge = document.getElementById('vadBadge');
  private vadText = document.getElementById('vadText');
  private vadConfValue = document.getElementById('vadConfValue');
  private vadProgress = document.getElementById('vadProgress');
  private mlConfValue = document.getElementById('mlConfValue');
  private mlProgress = document.getElementById('mlProgress');
  private latencyValue = document.getElementById('latencyValue');
  private videoLatencyValue = document.getElementById('videoLatencyValue');
  private detectionsValue = document.getElementById('detectionsValue');
  private transcriptBox = document.getElementById('transcriptBox');
  private lastTranscript = '';
  private tokenCount = 0;

  update(data: TelemetryData) {
    const vadActive = data.vad_active ?? data.vadActive;
    const vadConfidence = data.vad_confidence ?? data.vadConfidence;
    const mlConfidence = data.ml_confidence ?? data.mlConfidence;
    const transcript = data.ml_transcript ?? data.mlTranscript ?? data.transcript ?? '';
    const audioLatency = data.audio_latency_ms ?? data.audioLatencyMs;
    const videoLatency = data.vision_latency_ms ?? data.visionLatencyMs;

    // 1. VAD Indicator
    if (this.vadBadge && this.vadText && vadActive !== undefined) {
      if (vadActive) {
        this.vadBadge.classList.add('active');
        this.vadText.textContent = 'VOICE ACTIVE';
      } else {
        this.vadBadge.classList.remove('active');
        this.vadText.textContent = 'SILENCE';
      }
    }

    // 2. VAD Confidence Bar
    if (this.vadConfValue && this.vadProgress && vadConfidence !== undefined) {
      const pct = (vadConfidence * 100).toFixed(1);
      this.vadConfValue.textContent = `${pct}%`;
      this.vadProgress.style.width = `${Math.min(100, Math.max(0, vadConfidence * 100))}%`;
    }

    // 3. Audio ML Confidence Bar
    if (this.mlConfValue && this.mlProgress && mlConfidence !== undefined) {
      const pct = (mlConfidence * 100).toFixed(1);
      this.mlConfValue.textContent = `${pct}%`;
      this.mlProgress.style.width = `${Math.min(100, Math.max(0, mlConfidence * 100))}%`;
    }

    // 4. Audio Ingestion Latency Estimate
    if (this.latencyValue) {
      if (audioLatency !== undefined && audioLatency > 0) {
        this.latencyValue.textContent = `${audioLatency.toFixed(1)} ms`;
      } else if (data.timestamp_ns) {
        const nowNs = Date.now() * 1_000_000;
        const latencyMs = Math.max(0.1, (nowNs - data.timestamp_ns) / 1_000_000);
        this.latencyValue.textContent = `${latencyMs.toFixed(1)} ms`;
      }
    }

    // 5. Video Detection Count & Latency
    if (data.detections !== undefined) {
      if (this.detectionsValue) {
        const count = data.detection_count ?? data.detections_count ?? data.detections.length;
        this.detectionsValue.textContent = `${count} ${count === 1 ? 'Target' : 'Targets'}`;
      }
      if (this.videoLatencyValue) {
        if (videoLatency !== undefined && videoLatency > 0) {
          this.videoLatencyValue.textContent = `${videoLatency.toFixed(1)} ms`;
        } else if (data.video_timestamp_ns) {
          const nowNs = Date.now() * 1_000_000;
          const vLatencyMs = Math.max(0.1, (nowNs - data.video_timestamp_ns) / 1_000_000);
          this.videoLatencyValue.textContent = `${vLatencyMs.toFixed(1)} ms`;
        }
      }
    }

    // 6. Streaming ASR Transcript Tokens
    if (transcript && transcript.trim() !== '' && transcript !== this.lastTranscript) {
      this.appendToken(transcript.trim());
    }
  }

  appendToken(token: string) {
    this.lastTranscript = token;
    if (!this.transcriptBox) return;

    // Remove placeholder text on first token
    if (this.tokenCount === 0) {
      this.transcriptBox.innerHTML = '';
    }
    this.tokenCount++;

    const span = document.createElement('span');
    span.textContent = ` [${token}] `;
    span.style.color = '#38bdf8';
    span.style.fontWeight = '600';
    span.style.marginRight = '4px';
    span.style.display = 'inline-block';
    span.style.transition = 'all 0.2s ease';

    this.transcriptBox.appendChild(span);
    this.transcriptBox.scrollTop = this.transcriptBox.scrollHeight;
  }
}
