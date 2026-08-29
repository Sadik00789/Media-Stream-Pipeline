/**
 * Web Video capture processor for 640x480 RGB24 streaming.
 */

export class VideoCaptureProcessor {
  private mediaStream: MediaStream | null = null;
  private videoEl: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private frameTimer: number | null = null;
  private onFrameCallback: ((pixels: Uint8Array) => void) | null = null;

  async start(
    videoElement: HTMLVideoElement,
    onFrame?: (pixels: Uint8Array) => void
  ): Promise<MediaStream> {
    this.videoEl = videoElement;
    this.onFrameCallback = onFrame || null;

    // 1. Request 640x480 @ 30 FPS video + audio
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
      },
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    this.videoEl.srcObject = this.mediaStream;
    await this.videoEl.play().catch(() => {});

    // 2. Setup Offscreen Canvas for RGB24 Frame Extraction
    this.canvas = document.createElement('canvas');
    this.canvas.width = 640;
    this.canvas.height = 480;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    if (this.onFrameCallback) {
      this.startFrameCaptureLoop();
    }

    return this.mediaStream;
  }

  private startFrameCaptureLoop() {
    const targetFps = 30;
    const intervalMs = Math.floor(1000 / targetFps);

    this.frameTimer = window.setInterval(() => {
      if (!this.videoEl || !this.ctx || !this.onFrameCallback) return;
      if (this.videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

      this.ctx.drawImage(this.videoEl, 0, 0, 640, 480);
      const imgData = this.ctx.getImageData(0, 0, 640, 480);
      const rgba = imgData.data;

      // Pack RGBA into contiguous RGB24 buffer (921,600 bytes)
      const rgb = new Uint8Array(640 * 480 * 3);
      for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
        rgb[j + 0] = rgba[i + 0];
        rgb[j + 1] = rgba[i + 1];
        rgb[j + 2] = rgba[i + 2];
      }

      this.onFrameCallback(rgb);
    }, intervalMs);
  }

  stop() {
    if (this.frameTimer) {
      clearInterval(this.frameTimer);
      this.frameTimer = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
      this.videoEl = null;
    }
    this.canvas = null;
    this.ctx = null;
  }
}
