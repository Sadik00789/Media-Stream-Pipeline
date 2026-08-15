import { AudioCaptureProcessor } from './audio/processor';
import { SpectrogramVisualizer } from './audio/visualizer';
import { bindControls } from './ui/controls';
import { MetricsRenderer, TelemetryData } from './ui/transcript';

class App {
  private processor = new AudioCaptureProcessor();
  private visualizer: SpectrogramVisualizer;
  private metrics = new MetricsRenderer();
  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private ws: WebSocket | null = null;
  private isStreaming = false;
  private toneInterval: number | null = null;

  private statusBadge = document.getElementById('statusBadge');
  private statusText = document.getElementById('statusText');

  constructor() {
    const canvas = document.getElementById('spectrogramCanvas') as HTMLCanvasElement;
    this.visualizer = new SpectrogramVisualizer(canvas);

    this.initControls();
    this.initWebSocket();
    this.startRenderLoop();
  }

  private initControls() {
    bindControls({
      onToggleMic: () => this.toggleMicrophone(),
      onInjectTone: () => this.toggleSyntheticTone(),
      onGainChange: (val) => this.processor.setGain(val),
      onColorMapChange: (map) => this.visualizer.setColorMap(map),
    });
  }

  private initWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/ws`;

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.onopen = () => {
        if (!this.isStreaming) {
          this.updateStatus('Telemetry Stream Connected', true);
        }
      };

      this.ws.onmessage = (e) => {
        try {
          const payload: TelemetryData = JSON.parse(e.data);
          if (payload.type === 'telemetry') {
            this.handleTelemetry(payload);
          }
        } catch {
          // ignore non-json
        }
      };

      this.ws.onclose = () => {
        setTimeout(() => this.initWebSocket(), 2000);
      };
    } catch (err) {
      console.warn('[WebSocket] Connection error:', err);
    }
  }

  private handleTelemetry(payload: TelemetryData) {
    if (payload.spectrogram && payload.spectrogram.length > 0) {
      this.visualizer.pushSpectrogram(payload.spectrogram);
    }
    this.metrics.update(payload);
  }

  private async toggleMicrophone() {
    if (this.isStreaming && !this.toneInterval) {
      this.stopStream();
    } else {
      if (this.toneInterval) {
        clearInterval(this.toneInterval);
        this.toneInterval = null;
      }
      await this.startStream(true);
    }
  }

  private async toggleSyntheticTone() {
    if (this.toneInterval) {
      clearInterval(this.toneInterval);
      this.toneInterval = null;
      this.stopStream();
      return;
    }

    if (!this.isStreaming) {
      await this.startStream(false);
    }

    this.updateStatus('Streaming 1kHz Test Signal...', true);
    const samples = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      samples[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / 16000);
    }

    this.toneInterval = window.setInterval(() => {
      fetch('/api/pcm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples: Array.from(samples) }),
      }).catch(() => { });
    }, 32);
  }

  private async startStream(useMic: boolean) {
    try {
      this.updateStatus('Connecting...', false);

      this.pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });

      this.dc = this.pc.createDataChannel('telemetry', { ordered: false });
      this.dc.onopen = () => {
        this.updateStatus('Live Connected (WebRTC 60 FPS)', true);
      };

      this.dc.onmessage = (e) => {
        try {
          const payload: TelemetryData = JSON.parse(e.data);
          this.handleTelemetry(payload);
        } catch {
          // ignore
        }
      };

      if (useMic) {
        await this.processor.start((samples) => {
          // Continuously stream uncompressed PCM float32 samples directly to Go SHM
          fetch('/api/pcm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ samples: Array.from(samples) }),
          }).catch(() => { });
        });
      }

      const offer = await this.pc.createOffer();
      await this.pc.setLocalDescription(offer);

      // Await complete ICE gathering before posting offer
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
          setTimeout(resolve, 1200);
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

      this.isStreaming = true;
      if (!useMic) {
        this.updateStatus('Streaming Synthetic Tone (60 FPS)', true);
      } else {
        this.updateStatus('Microphone Live (60 FPS)', true);
      }
    } catch (err) {
      console.warn('WebRTC fallback to WebSocket:', err);
      this.isStreaming = true;
      this.updateStatus('Connected (WebSocket / HTTP)', true);
    }
  }

  private stopStream() {
    if (this.toneInterval) {
      clearInterval(this.toneInterval);
      this.toneInterval = null;
    }
    this.processor.stop();
    if (this.dc) {
      this.dc.close();
      this.dc = null;
    }
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }
    this.isStreaming = false;
    this.updateStatus('Disconnected', false);
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
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new App();
});