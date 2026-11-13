import test from 'node:test';
import assert from 'node:assert/strict';

type Listener = (event?: unknown) => void;

class MockClassList {
  private classes = new Set<string>();

  add(...tokens: string[]) {
    for (const t of tokens) this.classes.add(t);
  }

  remove(...tokens: string[]) {
    for (const t of tokens) this.classes.delete(t);
  }

  toggle(token: string, force?: boolean): boolean {
    if (force === undefined) {
      if (this.classes.has(token)) {
        this.classes.delete(token);
        return false;
      }
      this.classes.add(token);
      return true;
    }
    if (force) {
      this.classes.add(token);
    } else {
      this.classes.delete(token);
    }
    return force;
  }

  contains(token: string): boolean {
    return this.classes.has(token);
  }
}

class MockElement {
  id: string;
  textContent: string | null = '';
  innerHTML = '';
  value = '';
  scrollTop = 0;
  scrollHeight = 120;
  style: Record<string, string> = {};
  classList = new MockClassList();
  children: MockElement[] = [];
  private listeners = new Map<string, Listener[]>();

  constructor(id = '') {
    this.id = id;
  }

  addEventListener(type: string, cb: Listener) {
    const existing = this.listeners.get(type) ?? [];
    existing.push(cb);
    this.listeners.set(type, existing);
  }

  dispatchEvent(type: string, event?: Record<string, unknown>) {
    const cbs = this.listeners.get(type) ?? [];
    for (const cb of cbs) {
      cb(event ?? { target: this });
    }
  }

  appendChild(child: MockElement) {
    this.children.push(child);
  }
}

interface CanvasOp {
  op: string;
  args: unknown[];
}

function createMockCanvas() {
  const ops: CanvasOp[] = [];
  const ctx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    font: '',
    shadowColor: '',
    shadowBlur: 0,
    fillRect: (...args: unknown[]) => ops.push({ op: 'fillRect', args }),
    clearRect: (...args: unknown[]) => ops.push({ op: 'clearRect', args }),
    strokeRect: (...args: unknown[]) => ops.push({ op: 'strokeRect', args }),
    fillText: (...args: unknown[]) => ops.push({ op: 'fillText', args }),
    beginPath: () => ops.push({ op: 'beginPath', args: [] }),
    moveTo: (...args: unknown[]) => ops.push({ op: 'moveTo', args }),
    lineTo: (...args: unknown[]) => ops.push({ op: 'lineTo', args }),
    arc: (...args: unknown[]) => ops.push({ op: 'arc', args }),
    stroke: () => ops.push({ op: 'stroke', args: [] }),
    save: () => ops.push({ op: 'save', args: [] }),
    restore: () => ops.push({ op: 'restore', args: [] }),
    measureText: (text: string) => ({ width: text.length * 8 }),
  };

  const canvas = {
    width: 0,
    height: 0,
    getBoundingClientRect: () => ({ width: 640, height: 480 }),
    getContext: (kind: string) => (kind === '2d' ? ctx : null),
  };

  return { canvas: canvas as unknown as HTMLCanvasElement, ops };
}

function setupMockDom(ids: string[]) {
  const elements = new Map<string, MockElement>();
  for (const id of ids) {
    elements.set(id, new MockElement(id));
  }

  const windowListeners = new Map<string, Listener[]>();

  (globalThis as Record<string, unknown>).window = {
    devicePixelRatio: 1,
    addEventListener: (type: string, cb: Listener) => {
      const existing = windowListeners.get(type) ?? [];
      existing.push(cb);
      windowListeners.set(type, existing);
    },
  };

  (globalThis as Record<string, unknown>).document = {
    getElementById: (id: string) => elements.get(id) ?? null,
    createElement: (tag: string) => new MockElement(tag),
  };

  return elements;
}

test('SpectrogramVisualizer initializes, caps history at 160, and renders peak grid', async () => {
  setupMockDom([]);
  const { SpectrogramVisualizer } = await import('../src/audio/visualizer');
  const { canvas, ops } = createMockCanvas();
  const visualizer = new SpectrogramVisualizer(canvas);

  assert.equal(canvas.width, 640);
  assert.equal(canvas.height, 480);

  // Render idle state (empty history -> draws background + reference grid)
  visualizer.render();
  assert.ok(ops.some((o) => o.op === 'fillText' && o.args[0] === '1 kHz'));

  // Push >160 frames to verify ring history eviction & peak detection
  ops.length = 0;
  visualizer.setColorMap('fire');
  for (let frame = 0; frame < 170; frame++) {
    const bins = new Array<number>(64).fill(0.02);
    bins[10] = 0.85;
    visualizer.pushSpectrogram(bins);
  }

  visualizer.setColorMap('viridis');
  visualizer.render();

  assert.ok(ops.some((o) => o.op === 'fillText' && String(o.args[0]).startsWith('Peak: ')));
  assert.ok(ops.some((o) => o.op === 'fillRect'));
});

test('DetectionOverlayRenderer renders idle crosshair and active FACE/TARGET bounding boxes', async () => {
  setupMockDom([]);
  const { DetectionOverlayRenderer } = await import('../src/video/overlay');
  const { canvas, ops } = createMockCanvas();
  const overlay = new DetectionOverlayRenderer(canvas);

  // Idle state draws crosshair arc
  overlay.render();
  assert.ok(ops.some((o) => o.op === 'arc'));

  // Active detections (Face + Target)
  ops.length = 0;
  overlay.updateDetections([
    { ymin: 0.2, xmin: 0.3, ymax: 0.6, xmax: 0.7, confidence: 0.92, class_id: 0 },
    { ymin: 0.1, xmin: 0.1, ymax: 0.3, xmax: 0.25, confidence: 0.75, class_id: 1 },
  ]);
  overlay.render();

  const labels = ops
    .filter((o) => o.op === 'fillText')
    .map((o) => String(o.args[0]));
  assert.ok(labels.includes('FACE 92%'));
  assert.ok(labels.includes('TARGET 75%'));
  assert.equal(ops.filter((o) => o.op === 'strokeRect').length, 2);
});

test('MetricsRenderer updates DOM gauges, detection counts, and deduplicates transcript tokens', async () => {
  const elements = setupMockDom([
    'vadBadge',
    'vadText',
    'vadConfValue',
    'vadProgress',
    'mlConfValue',
    'mlProgress',
    'latencyValue',
    'videoLatencyValue',
    'detectionsValue',
    'transcriptBox',
  ]);
  elements.get('transcriptBox')!.textContent = 'Connect audio to begin...';

  const { MetricsRenderer } = await import('../src/ui/transcript');
  const metrics = new MetricsRenderer();

  metrics.update({
    vad_active: true,
    vad_confidence: 0.875,
    ml_confidence: 0.64,
    audio_latency_ms: 12.34,
    vision_latency_ms: 8.5,
    detections: [{ ymin: 0.1, xmin: 0.1, ymax: 0.5, xmax: 0.5, confidence: 0.9, class_id: 0 }],
    ml_transcript: 'hello',
  });

  assert.equal(elements.get('vadBadge')!.classList.contains('active'), true);
  assert.equal(elements.get('vadText')!.textContent, 'VOICE ACTIVE');
  assert.equal(elements.get('vadConfValue')!.textContent, '87.5%');
  assert.equal(elements.get('vadProgress')!.style.width, '87.5%');
  assert.equal(elements.get('mlConfValue')!.textContent, '64.0%');
  assert.equal(elements.get('latencyValue')!.textContent, '12.3 ms');
  assert.equal(elements.get('videoLatencyValue')!.textContent, '8.5 ms');
  assert.equal(elements.get('detectionsValue')!.textContent, '1 Target');
  assert.equal(elements.get('transcriptBox')!.children.length, 1);
  assert.equal(elements.get('transcriptBox')!.children[0].textContent, 'hello');

  // Duplicate transcript token should not append a duplicate badge; silence updates VAD badge
  metrics.update({
    vad_active: false,
    ml_transcript: 'hello',
    detections: [],
  });
  assert.equal(elements.get('vadBadge')!.classList.contains('active'), false);
  assert.equal(elements.get('vadText')!.textContent, 'SILENCE');
  assert.equal(elements.get('detectionsValue')!.textContent, '0 Targets');
  assert.equal(elements.get('transcriptBox')!.children.length, 1);

  // New token appends
  metrics.update({ ml_transcript: 'world' });
  assert.equal(elements.get('transcriptBox')!.children.length, 2);
  assert.equal(elements.get('transcriptBox')!.children[1].textContent, 'world');
});

test('bindControls synchronizes button states and dispatches control callbacks', async () => {
  const elements = setupMockDom([
    'toggleMicBtn',
    'toggleMicText',
    'toggleCamBtn',
    'toggleCamText',
    'testToneBtn',
    'testVideoBtn',
    'gainSlider',
    'gainValue',
    'colorMapSelect',
  ]);

  const { bindControls } = await import('../src/ui/controls');

  const events: string[] = [];
  let lastGain = 0;
  let lastMap = '';

  const handle = bindControls({
    onToggleMic: () => events.push('mic'),
    onToggleCamera: () => events.push('cam'),
    onInjectTone: () => events.push('tone'),
    onInjectTestVideo: () => events.push('test-video'),
    onGainChange: (val) => {
      lastGain = val;
    },
    onColorMapChange: (map) => {
      lastMap = map;
    },
  });

  // Programmatic state updates via ControlHandle
  handle.setMicActive(true);
  assert.equal(elements.get('toggleMicBtn')!.classList.contains('danger'), true);
  assert.equal(elements.get('toggleMicText')!.textContent, 'Stop Microphone');
  handle.setMicActive(false);
  assert.equal(elements.get('toggleMicBtn')!.classList.contains('danger'), false);
  assert.equal(elements.get('toggleMicText')!.textContent, 'Start Microphone');

  handle.setCamActive(true);
  assert.equal(elements.get('toggleCamBtn')!.classList.contains('danger'), true);
  assert.equal(elements.get('toggleCamText')!.textContent, 'Stop Camera');

  handle.setToneActive(true);
  assert.equal(elements.get('testToneBtn')!.classList.contains('danger'), true);
  assert.equal(elements.get('testToneBtn')!.classList.contains('secondary'), false);
  handle.setToneActive(false);
  assert.equal(elements.get('testToneBtn')!.classList.contains('secondary'), true);

  handle.setTestVideoActive(true);
  assert.equal(elements.get('testVideoBtn')!.classList.contains('danger'), true);

  // User interaction events
  elements.get('toggleMicBtn')!.dispatchEvent('click');
  elements.get('toggleCamBtn')!.dispatchEvent('click');
  elements.get('testToneBtn')!.dispatchEvent('click');
  elements.get('testVideoBtn')!.dispatchEvent('click');
  assert.deepEqual(events, ['mic', 'cam', 'tone', 'test-video']);

  const gainSlider = elements.get('gainSlider')!;
  gainSlider.value = '2.5';
  gainSlider.dispatchEvent('input', { target: gainSlider });
  assert.equal(lastGain, 2.5);
  assert.equal(elements.get('gainValue')!.textContent, '2.5x');

  const colorMapSelect = elements.get('colorMapSelect')!;
  colorMapSelect.value = 'viridis';
  colorMapSelect.dispatchEvent('change', { target: colorMapSelect });
  assert.equal(lastMap, 'viridis');
});

