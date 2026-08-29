/**
 * High-performance 60 FPS HTML5 Canvas Decibel Spectrogram & HUD Waterfall Renderer.
 */

export class SpectrogramVisualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private history: Float32Array[] = [];
  private maxHistory = 160;
  private colorMap = 'plasma';
  private peakFreqDecay = 0;
  private peakBin = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Could not get 2D canvas context');
    this.ctx = context;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(300, rect.width * window.devicePixelRatio);
    this.canvas.height = Math.max(150, rect.height * window.devicePixelRatio);
  }

  setColorMap(name: string) {
    this.colorMap = name;
  }

  pushSpectrogram(bins: number[]) {
    const arr = new Float32Array(bins);
    this.history.push(arr);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    // Find peak frequency in latest frame
    let maxVal = 0;
    let maxIdx = 0;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > maxVal) {
        maxVal = arr[i];
        maxIdx = i;
      }
    }
    if (maxVal > 0.3) {
      this.peakBin = maxIdx;
      this.peakFreqDecay = 1.0;
    }
  }

  render() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    // Background fill
    ctx.fillStyle = '#080a0f';
    ctx.fillRect(0, 0, w, h);

    if (this.history.length === 0) {
      // Idle HUD grid
      this.drawReferenceGrid(w, h);
      return;
    }

    const waterfallHeight = h * 0.72;
    const bottomH = h * 0.28;
    const numBins = this.history[0].length;
    const columnWidth = w / this.maxHistory;
    const binHeight = waterfallHeight / numBins;

    // 1. Draw Spectrogram Waterfall (Top 72%)
    for (let col = 0; col < this.history.length; col++) {
      const frame = this.history[col];
      const x = col * columnWidth;

      for (let bin = 0; bin < numBins; bin++) {
        const val = frame[bin]; // already normalized dB [0.0, 1.0]
        if (val > 0.05) {
          const y = waterfallHeight - (bin + 1) * binHeight;
          ctx.fillStyle = this.getColormapColor(val);
          ctx.fillRect(x, y, columnWidth + 0.8, binHeight + 0.8);
        }
      }
    }

    // 2. Draw Reference Frequency Grid & Labels
    this.drawReferenceGrid(w, waterfallHeight);

    // 3. Draw Live FFT Power Bar Graph (Bottom 28%)
    const latest = this.history[this.history.length - 1];
    const barWidth = w / numBins;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
    ctx.fillRect(0, waterfallHeight, w, bottomH);

    // Grid divider line
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, waterfallHeight);
    ctx.lineTo(w, waterfallHeight);
    ctx.stroke();

    for (let bin = 0; bin < numBins; bin++) {
      const val = latest[bin];
      const barH = val * (bottomH - 12);
      const x = bin * barWidth;
      const y = h - barH;

      ctx.fillStyle = this.getColormapColor(val);
      ctx.fillRect(x, y, barWidth - 0.5, barH);
    }

    // 4. Draw Peak Frequency Indicator with Smooth Decay
    if (this.peakFreqDecay > 0.05) {
      const peakFreqHz = Math.round(this.peakBin * (16000 / 512));
      const peakX = this.peakBin * barWidth;

      ctx.strokeStyle = `rgba(239, 68, 68, ${this.peakFreqDecay})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(peakX, waterfallHeight);
      ctx.lineTo(peakX, h);
      ctx.stroke();

      ctx.fillStyle = `rgba(255, 255, 255, ${this.peakFreqDecay})`;
      ctx.font = `${Math.floor(11 * window.devicePixelRatio)}px monospace`;
      ctx.fillText(`Peak: ${peakFreqHz} Hz`, peakX + 4, waterfallHeight + 16 * window.devicePixelRatio);

      this.peakFreqDecay *= 0.94; // smooth decay
    }
  }

  private drawReferenceGrid(w: number, h: number) {
    const ctx = this.ctx;
    const freqMarkers = [
      { freq: 1000, label: '1 kHz' },
      { freq: 2000, label: '2 kHz' },
      { freq: 4000, label: '4 kHz' },
      { freq: 8000, label: '8 kHz' },
    ];

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
    ctx.font = `${Math.floor(10 * window.devicePixelRatio)}px sans-serif`;

    for (const marker of freqMarkers) {
      const bin = (marker.freq / 8000) * 256; // 8000 Hz Nyquist = Bin 256
      const y = h - (bin / 256) * h;

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();

      ctx.fillText(marker.label, w - 45 * window.devicePixelRatio, y - 4);
    }
  }

  private getColormapColor(v: number): string {
    const clamped = Math.max(0, Math.min(1, v));
    if (this.colorMap === 'viridis') {
      const r = Math.floor(clamped * 68 + (1 - clamped) * 34);
      const g = Math.floor(clamped * 200 + (1 - clamped) * 50);
      const b = Math.floor(clamped * 110 + (1 - clamped) * 90);
      return `rgb(${r},${g},${b})`;
    } else if (this.colorMap === 'fire') {
      const r = Math.floor(Math.min(255, clamped * 350));
      const g = Math.floor(Math.min(255, Math.max(0, (clamped - 0.25) * 300)));
      const b = Math.floor(Math.min(255, Math.max(0, (clamped - 0.65) * 400)));
      return `rgb(${r},${g},${b})`;
    } else {
      // Cyber Plasma
      const r = Math.floor(Math.sin(clamped * Math.PI * 0.8) * 210 + 35);
      const g = Math.floor(Math.pow(clamped, 2.2) * 190);
      const b = Math.floor(Math.cos(clamped * Math.PI * 0.5) * 220 + 35);
      return `rgb(${r},${g},${b})`;
    }
  }
}
