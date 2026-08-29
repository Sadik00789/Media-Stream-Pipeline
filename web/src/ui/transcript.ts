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
  private vadBadge = document.getElementById('vadBadge') || document.getElementById('vad-badge');
  private vadText = document.getElementById('vadText') || document.getElementById('vad-text');
  private vadConfValue = document.getElementById('vadConfValue') || document.getElementById('vad-confidence-val');
  private vadProgress = document.getElementById('vadProgress') || document.getElementById('vad-confidence-progress');
  private mlConfValue = document.getElementById('mlConfValue') || document.getElementById('ml-confidence-val') || document.getElementById('ml-confidence');
  private mlProgress = document.getElementById('mlProgress') || document.getElementById('ml-confidence-progress') || document.getElementById('ml-confidence-bar');
  private latencyValue = document.getElementById('latencyValue') || document.getElementById('latency-val');
  private videoLatencyValue = document.getElementById('videoLatencyValue') || document.getElementById('video-latency-val');
  private detectionsValue = document.getElementById('detectionsValue') || document.getElementById('detections-val');
  private transcriptBox = document.getElementById('transcriptBox') || document.getElementById('transcript-box') || document.getElementById('liveTranscript');
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

    // 4. Audio Ingestion Latency (Strict Server Monotonic Time)
    if (this.latencyValue) {
      if (audioLatency !== undefined && audioLatency > 0) {
        this.latencyValue.textContent = `${audioLatency.toFixed(1)} ms`;
      } else {
        this.latencyValue.textContent = `0.0 ms`;
      }
    }

    // 5. Video Detection Count & Latency (Strict Server Monotonic Time)
    if (data.detections !== undefined) {
      if (this.detectionsValue) {
        const count = data.detection_count ?? data.detections_count ?? data.detections.length;
        this.detectionsValue.textContent = `${count} ${count === 1 ? 'Target' : 'Targets'}`;
      }
      if (this.videoLatencyValue) {
        if (videoLatency !== undefined && videoLatency > 0) {
          this.videoLatencyValue.textContent = `${videoLatency.toFixed(1)} ms`;
        } else {
          this.videoLatencyValue.textContent = `0.0 ms`;
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

    if (this.tokenCount === 0 || this.transcriptBox.textContent?.includes('Connect audio')) {
      this.transcriptBox.innerHTML = '';
    }
    this.tokenCount++;

    const badge = document.createElement('span');
    badge.style.display = 'inline-block';
    badge.style.padding = '3px 8px';
    badge.style.margin = '2px 4px';
    badge.style.borderRadius = '4px';
    badge.style.background = 'rgba(6, 182, 212, 0.15)';
    badge.style.border = '1px solid rgba(6, 182, 212, 0.4)';
    badge.style.color = '#06b6d4';
    badge.style.fontFamily = 'monospace';
    badge.style.fontWeight = '600';
    badge.style.fontSize = '0.85rem';
    badge.textContent = token;

    this.transcriptBox.appendChild(badge);
    this.transcriptBox.scrollTop = this.transcriptBox.scrollHeight;
  }
}

// Alias for TranscriptController
export { MetricsRenderer as TranscriptController };
