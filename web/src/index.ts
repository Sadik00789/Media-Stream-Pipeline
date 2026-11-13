import { AudioCaptureProcessor } from './audio/processor';
import { SpectrogramVisualizer } from './audio/visualizer';
import { VideoCaptureProcessor } from './video/processor';
import { DetectionOverlayRenderer } from './video/overlay';
import { bindControls, ControlHandle } from './ui/controls';
import { MetricsRenderer, TelemetryData } from './ui/transcript';

class App {
  private audioProcessor = new AudioCaptureProcessor();
  private videoProcessor = new VideoCaptureProcessor();
  private visualizer: SpectrogramVisualizer;
  private overlay: DetectionOverlayRenderer;
  private metrics = new MetricsRenderer();
  private controls!: ControlHandle;

  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private ws: WebSocket | null = null;
  private isAudioStreaming = false;
  private isVideoStreaming = false;
  private toneInterval: number | null = null;
  private videoTestInterval: number | null = null;

  private isSendingAudio = false;
  private isSendingVideo = false;
  private isSendingTone = false;
  private isSendingSynthVideo = false;

  private statusBadge = document.getElementById('statusBadge');
  private statusText = document.getElementById('statusText');
  private videoElement = document.getElementById('cameraVideo') as HTMLVideoElement;

  constructor() {
    const specCanvas = document.getElementById('spectrogramCanvas') as HTMLCanvasElement;
    this.visualizer = new SpectrogramVisualizer(specCanvas);

    const overlayCanvas = document.getElementById('overlayCanvas') as HTMLCanvasElement;
    this.overlay = new DetectionOverlayRenderer(overlayCanvas);

    this.initControls();
    this.initWebSocket();
    this.initWebRTC();
    this.startRenderLoop();
  }

  private initControls() {
    this.controls = bindControls({
      onToggleMic: () => this.toggleMicrophone(),
      onToggleCamera: () => this.toggleCamera(),
      onInjectTone: () => this.toggleSyntheticTone(),
      onInjectTestVideo: () => this.toggleSyntheticVideo(),
      onGainChange: (val) => this.audioProcessor.setGain(val),
      onColorMapChange: (map) => this.visualizer.setColorMap(map),
    });
  }

  private initWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/ws`;

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      this.ws.onopen = () => {
        this.updateStatus('Telemetry Connected (60 FPS)', true);
      };

      this.ws.onmessage = (e) => {
        try {
          if (typeof e.data === 'string') {
            const payload: TelemetryData = JSON.parse(e.data);
            if (payload.type === 'telemetry') {
              this.handleTelemetry(payload);
            }
          }
        } catch {
          // ignore non-json
        }
      };

      this.ws.onclose = () => {
        if (!this.dc || this.dc.readyState !== 'open') {
          this.updateStatus('Disconnected (Reconnecting...)', false);
        }
        setTimeout(() => this.initWebSocket(), 2000);
      };
    } catch (err) {
      console.warn('[WebSocket] Connection error:', err);
    }
  }

  private async initWebRTC() {
    try {
      this.pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });

      this.dc = this.pc.createDataChannel('telemetry', { ordered: false });
      this.dc.onopen = () => {
        this.updateStatus('WebRTC DataChannel Connected (60 FPS)', true);
      };

      this.dc.onmessage = (e) => {
        try {
          const payload: TelemetryData = JSON.parse(e.data);
          if (payload.type === 'telemetry') {
            this.handleTelemetry(payload);
          }
        } catch {
          // ignore
        }
      };

      const offer = await this.pc.createOffer();
      await this.pc.setLocalDescription(offer);

      await new Promise<void>((resolve) => {
        if (this.pc!.iceGatheringState === 'complete') {
          resolve();
        } else {
          const checkState = () => {
            if (this.pc!.iceGatheringState === 'complete') {
              this.pc!.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          };
          this.pc!.addEventListener('icegatheringstatechange', checkState);
          setTimeout(resolve, 1000);
        }
      });

      const resp = await fetch('/api/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: this.pc.localDescription?.sdp || offer.sdp }),
      });

      if (resp.ok) {
        const { sdp: answerSDP } = await resp.json();
        await this.pc.setRemoteDescription({ type: 'answer', sdp: answerSDP });
      }
    } catch (err) {
      console.warn('[WebRTC] Handshake fallback to WebSocket:', err);
    }
  }

  private handleTelemetry(payload: TelemetryData) {
    if (payload.spectrogram && payload.spectrogram.length > 0) {
      this.visualizer.pushSpectrogram(payload.spectrogram);
    }
    if (payload.detections !== undefined) {
      this.overlay.updateDetections(payload.detections);
    }
    this.metrics.update(payload);
  }

  private async toggleMicrophone() {
    if (this.isAudioStreaming && !this.toneInterval) {
      this.audioProcessor.stop();
      this.isAudioStreaming = false;
      this.controls.setMicActive(false);
      this.updateStatus('Microphone Stopped', true);
    } else {
      if (this.toneInterval) {
        clearInterval(this.toneInterval);
        this.toneInterval = null;
        this.controls.setToneActive(false);
      }
      try {
        await this.audioProcessor.start((samples) => {
          if (this.isSendingAudio) return;
          this.isSendingAudio = true;

          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'pcm', samples: Array.from(samples) }));
            this.isSendingAudio = false;
          } else {
            fetch('/api/pcm', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ samples: Array.from(samples) }),
            }).finally(() => {
              this.isSendingAudio = false;
            });
          }
        });
        this.isAudioStreaming = true;
        this.controls.setMicActive(true);
        this.updateStatus('Microphone Live (16kHz PCM)', true);
      } catch (err) {
        this.isAudioStreaming = false;
        this.controls.setMicActive(false);
        console.warn('Microphone start error:', err);
      }
    }
  }

  private async toggleCamera() {
    if (this.isVideoStreaming && !this.videoTestInterval) {
      this.videoProcessor.stop();
      this.isVideoStreaming = false;
      this.controls.setCamActive(false);
      this.updateStatus('Camera Stopped', true);
    } else {
      if (this.videoTestInterval) {
        clearInterval(this.videoTestInterval);
        this.videoTestInterval = null;
        this.controls.setTestVideoActive(false);
      }
      try {
        await this.videoProcessor.start(this.videoElement, (pixels) => {
          if (this.isSendingVideo) return;
          this.isSendingVideo = true;

          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(pixels.buffer);
            this.isSendingVideo = false;
          } else {
            fetch('/api/video', {
              method: 'POST',
              headers: { 'Content-Type': 'application/octet-stream' },
              body: new Blob([pixels.buffer as ArrayBuffer]),
            }).finally(() => {
              this.isSendingVideo = false;
            });
          }
        });
        this.isVideoStreaming = true;
        this.controls.setCamActive(true);
        this.updateStatus('Camera Live (640x480 RGB24 @ 30 FPS)', true);
      } catch (err) {
        this.isVideoStreaming = false;
        this.controls.setCamActive(false);
        console.warn('Camera start error:', err);
      }
    }
  }

  private async toggleSyntheticTone() {
    if (this.toneInterval) {
      clearInterval(this.toneInterval);
      this.toneInterval = null;
      this.controls.setToneActive(false);
      this.updateStatus('Synthetic Tone Stopped', true);
      return;
    }

    if (this.isAudioStreaming) {
      this.audioProcessor.stop();
      this.isAudioStreaming = false;
      this.controls.setMicActive(false);
    }

    this.controls.setToneActive(true);
    this.updateStatus('Streaming 1kHz Test Signal...', true);
    const samples = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      samples[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / 16000);
    }

    this.toneInterval = window.setInterval(() => {
      if (this.isSendingTone) return;
      this.isSendingTone = true;

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'pcm', samples: Array.from(samples) }));
        this.isSendingTone = false;
      } else {
        fetch('/api/pcm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ samples: Array.from(samples) }),
        }).finally(() => {
          this.isSendingTone = false;
        });
      }
    }, 32);
  }

  private async toggleSyntheticVideo() {
    if (this.videoTestInterval) {
      clearInterval(this.videoTestInterval);
      this.videoTestInterval = null;
      this.controls.setTestVideoActive(false);
      this.updateStatus('Test Pattern Stopped', true);
      return;
    }

    if (this.isVideoStreaming) {
      this.videoProcessor.stop();
      this.isVideoStreaming = false;
      this.controls.setCamActive(false);
    }

    this.controls.setTestVideoActive(true);
    this.updateStatus('Streaming Synthetic Video Pattern (30 FPS)...', true);
    let frameNum = 0;

    this.videoTestInterval = window.setInterval(() => {
      if (this.isSendingSynthVideo) return;
      this.isSendingSynthVideo = true;

      frameNum++;
      const rgb = new Uint8Array(640 * 480 * 3);

      // Draw synthetic moving target box (simulates face target)
      const posX = 150 + Math.floor(Math.sin(frameNum * 0.1) * 80);
      const posY = 120 + Math.floor(Math.cos(frameNum * 0.1) * 50);

      for (let y = 0; y < 480; y++) {
        for (let x = 0; x < 640; x++) {
          const idx = (y * 640 + x) * 3;
          if (x >= posX && x < posX + 180 && y >= posY && y < posY + 180) {
            rgb[idx + 0] = 220; // R (skin tone)
            rgb[idx + 1] = 160; // G
            rgb[idx + 2] = 120; // B
          } else {
            rgb[idx + 0] = 15;
            rgb[idx + 1] = 20;
            rgb[idx + 2] = 30;
          }
        }
      }

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(rgb.buffer);
        this.isSendingSynthVideo = false;
      } else {
        fetch('/api/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: new Blob([rgb.buffer as ArrayBuffer]),
        }).finally(() => {
          this.isSendingSynthVideo = false;
        });
      }
    }, 33);
  }

  private updateStatus(text: string, connected: boolean) {
    if (this.statusText) this.statusText.textContent = text;
    if (this.statusBadge) {
      if (connected) {
        this.statusBadge.classList.add('connected');
      } else {
        this.statusBadge.classList.remove('connected');
      }
    }
  }

  private startRenderLoop() {
    const loop = () => {
      this.visualizer.render();
      this.overlay.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new App();
});