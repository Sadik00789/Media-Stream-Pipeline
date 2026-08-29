/**
 * High-performance HTML5 Canvas HUD Detection Overlay Renderer.
 */

export interface DetectionBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  confidence: number;
  class_id?: number;
}

export class DetectionOverlayRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private latestBoxes: DetectionBox[] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not get 2D canvas context for overlay');
    this.ctx = context;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(300, (rect.width || 640) * window.devicePixelRatio);
    this.canvas.height = Math.max(200, (rect.height || 480) * window.devicePixelRatio);
  }

  updateDetections(boxes: DetectionBox[]) {
    this.latestBoxes = boxes || [];
  }

  render() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, w, h);

    if (this.latestBoxes.length === 0) {
      // Draw idle crosshair
      this.drawIdleCrosshairs(w, h);
      return;
    }

    // Draw active detection boxes
    for (const box of this.latestBoxes) {
      const bx = box.xmin * w;
      const by = box.ymin * h;
      const bw = Math.max(10, (box.xmax - box.xmin) * w);
      const bh = Math.max(10, (box.ymax - box.ymin) * h);
      const isFace = (box.class_id ?? 0) === 0;

      const strokeColor = isFace ? '#06b6d4' : '#a855f7'; // Cyan for Face, Purple for Object
      const glowColor = isFace ? 'rgba(6, 182, 212, 0.4)' : 'rgba(168, 85, 247, 0.4)';

      ctx.save();
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 12 * window.devicePixelRatio;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2 * window.devicePixelRatio;

      // Draw bounding box rectangle
      ctx.strokeRect(bx, by, bw, bh);

      // Draw cyber corner brackets
      const cornerLen = Math.min(16 * window.devicePixelRatio, Math.min(bw, bh) * 0.25);
      ctx.lineWidth = 3 * window.devicePixelRatio;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(bx, by + cornerLen);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx + cornerLen, by);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(bx + bw - cornerLen, by);
      ctx.lineTo(bx + bw, by);
      ctx.lineTo(bx + bw, by + cornerLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(bx, by + bh - cornerLen);
      ctx.lineTo(bx, by + bh);
      ctx.lineTo(bx + cornerLen, by + bh);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(bx + bw - cornerLen, by + bh);
      ctx.lineTo(bx + bw, by + bh);
      ctx.lineTo(bx + bw, by + bh - cornerLen);
      ctx.stroke();

      // Label Tag
      const label = isFace
        ? `FACE ${(box.confidence * 100).toFixed(0)}%`
        : `TARGET ${(box.confidence * 100).toFixed(0)}%`;

      ctx.font = `bold ${Math.floor(11 * window.devicePixelRatio)}px monospace`;
      const textWidth = ctx.measureText(label).width;
      const tagH = 20 * window.devicePixelRatio;
      const tagW = textWidth + 12 * window.devicePixelRatio;

      ctx.fillStyle = strokeColor;
      ctx.fillRect(bx, Math.max(0, by - tagH), tagW, tagH);

      ctx.fillStyle = '#080a0f';
      ctx.fillText(label, bx + 6 * window.devicePixelRatio, Math.max(tagH - 5, by - 5 * window.devicePixelRatio));

      ctx.restore();
    }
  }

  private drawIdleCrosshairs(w: number, h: number) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    const cx = w / 2;
    const cy = h / 2;
    const r = 24 * window.devicePixelRatio;

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.moveTo(cx - r - 10, cy);
    ctx.lineTo(cx + r + 10, cy);
    ctx.moveTo(cx, cy - r - 10);
    ctx.lineTo(cx, cy + r + 10);
    ctx.stroke();

    ctx.restore();
  }
}
