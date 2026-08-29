/**
 * UI live metrics and streaming transcript token renderer.
 */

import { DetectionBox } from '../video/overlay';

export interface TelemetryData {
  type?: string;
  seq?: number;
  timestamp_ns?: number;
  vad_active?: boolean;
  vad_confidence?: number;
  ml_confidence?: number;
  transcript?: string;
  spectrogram?: number[];
  video_seq?: number;
  video_timestamp_ns?: number;
  detection_count?: number;
  detections?: DetectionBox[];
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

  update(data: TelemetryData) {
    // 1. VAD Indicator
    if (this.vadBadge && this.vadText && data.vad_active !== undefined) {
      if (data.vad_active) {
        this.vadBadge.classList.add('active');
        this.vadText.textContent = 'VOICE ACTIVE';
      } else {
        this.vadBadge.classList.remove('active');
        this.vadText.textContent = 'SILENCE';
      }
    }

    // 2. VAD Confidence Bar
    if (this.vadConfValue && this.vadProgress && data.vad_confidence !== undefined) {
      const pct = (data.vad_confidence * 100).toFixed(1);
      this.vadConfValue.textContent = `${pct}%`;
      this.vadProgress.style.width = `${pct}%`;
    }

    // 3. Audio ML Confidence Bar
    if (this.mlConfValue && this.mlProgress && data.ml_confidence !== undefined) {
      const pct = (data.ml_confidence * 100).toFixed(1);
      this.mlConfValue.textContent = `${pct}%`;
      this.mlProgress.style.width = `${pct}%`;
    }

    // 4. Audio Ingestion Latency Estimate
    if (this.latencyValue && data.timestamp_ns) {
      const nowNs = Date.now() * 1_000_000;
      const latencyMs = Math.max(0.1, (nowNs - data.timestamp_ns) / 1_000_000);
      this.latencyValue.textContent = `${latencyMs.toFixed(1)} ms`;
    }

    // 5. Video Detection Count & Latency
    if (data.detections !== undefined) {
      if (this.detectionsValue) {
        const count = data.detection_count ?? data.detections.length;
        this.detectionsValue.textContent = `${count} ${count === 1 ? 'Target' : 'Targets'}`;
      }
      if (this.videoLatencyValue && data.video_timestamp_ns) {
        const nowNs = Date.now() * 1_000_000;
        const vLatencyMs = Math.max(0.1, (nowNs - data.video_timestamp_ns) / 1_000_000);
        this.videoLatencyValue.textContent = `${vLatencyMs.toFixed(1)} ms`;
      }
    }

    // 6. Streaming Transcript
    if (this.transcriptBox && data.transcript && data.transcript !== this.lastTranscript) {
      this.lastTranscript = data.transcript;
      const span = document.createElement('span');
      span.textContent = ` [${data.transcript}] `;
      span.style.color = '#60a5fa';
      this.transcriptBox.appendChild(span);
      this.transcriptBox.scrollTop = this.transcriptBox.scrollHeight;
    }
  }
}
