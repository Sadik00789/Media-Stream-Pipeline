/**
 * Web Audio API stream capture and PCM frame chunking.
 */

export class AudioCaptureProcessor {
  private audioCtx: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processorNode: ScriptProcessorNode | null = null;
  private gainNode: GainNode | null = null;
  private onSamplesCallback: ((samples: Float32Array) => void) | null = null;

  async start(onSamples: (samples: Float32Array) => void): Promise<MediaStream> {
    this.onSamplesCallback = onSamples;

    // 1. Initialize AudioContext at 16kHz
    this.audioCtx = new AudioContext({ sampleRate: 16000 });
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }

    // 2. Request mic access (allow browser to handle native hardware sampling rate)
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
    this.gainNode = this.audioCtx.createGain();
    this.gainNode.gain.value = 1.0;

    // 3. Buffer 512 samples per frame (32ms at 16kHz)
    this.processorNode = this.audioCtx.createScriptProcessor(512, 1, 1);
    this.processorNode.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      if (this.onSamplesCallback) {
        this.onSamplesCallback(new Float32Array(input));
      }
    };

    source.connect(this.gainNode);
    this.gainNode.connect(this.processorNode);
    this.processorNode.connect(this.audioCtx.destination);

    return this.mediaStream;
  }

  setGain(value: number) {
    if (this.gainNode) {
      this.gainNode.gain.value = value;
    }
  }

  stop() {
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = null;
    }
  }
}