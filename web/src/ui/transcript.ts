/**
 * UI live metrics and streaming transcript token renderer.
 */

export interface TelemetryData {
  type?: string;
  seq: number;
  timestamp_ns: number;
  vad_active: boolean;
  vad_confidence: number;
  ml_confidence: number;
  transcript: string;
  spectrogram: number[];
}

export class MetricsRenderer {
  private vadBadge = document.getElementById('vadBadge');
  private vadText = document.getElementById('vadText');
  private vadConfValue = document.getElementById('vadConfValue');
  private vadProgress = document.getElementById('vadProgress');
  private mlConfValue = document.getElementById('mlConfValue');
  private mlProgress = document.getElementById('mlProgress');
  private latencyValue = document.getElementById('latencyValue');
  private transcriptBox = document.getElementById('transcriptBox');
  private lastTranscript = '';

  update(data: TelemetryData) {
    // 1. VAD Indicator
    if (this.vadBadge && this.vadText) {
      if (data.vad_active) {
        this.vadBadge.classList.add('active');
        this.vadText.textContent = 'VOICE ACTIVE';
      } else {
        this.vadBadge.classList.remove('active');
        this.vadText.textContent = 'SILENCE';
      }
    }

    // 2. VAD Confidence Bar
    if (this.vadConfValue && this.vadProgress) {
      const pct = (data.vad_confidence * 100).toFixed(1);
      this.vadConfValue.textContent = `${pct}%`;
      this.vadProgress.style.width = `${pct}%`;
    }

    // 3. ML Confidence Bar
    if (this.mlConfValue && this.mlProgress) {
      const pct = (data.ml_confidence * 100).toFixed(1);
      this.mlConfValue.textContent = `${pct}%`;
      this.mlProgress.style.width = `${pct}%`;
    }

    // 4. Ingestion Latency Estimate
    if (this.latencyValue && data.timestamp_ns) {
      const nowNs = Date.now() * 1_000_000;
      const latencyMs = Math.max(0.1, (nowNs - data.timestamp_ns) / 1_000_000);
      this.latencyValue.textContent = `${latencyMs.toFixed(1)} ms`;
    }

    // 5. Streaming Transcript
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
