/**
 * UI Controls and interaction bindings.
 */

export interface ControlCallbacks {
  onToggleMic: () => void;
  onInjectTone: () => void;
  onGainChange: (val: number) => void;
  onColorMapChange: (map: string) => void;
}

export function bindControls(callbacks: ControlCallbacks) {
  const toggleBtn = document.getElementById('toggleMicBtn') as HTMLButtonElement;
  const toggleText = document.getElementById('toggleMicText') as HTMLElement;
  const testToneBtn = document.getElementById('testToneBtn') as HTMLButtonElement;
  const gainSlider = document.getElementById('gainSlider') as HTMLInputElement;
  const gainValue = document.getElementById('gainValue') as HTMLElement;
  const colorMapSelect = document.getElementById('colorMapSelect') as HTMLSelectElement;

  let isStreaming = false;

  toggleBtn?.addEventListener('click', () => {
    isStreaming = !isStreaming;
    if (isStreaming) {
      toggleBtn.classList.add('danger');
      toggleText.textContent = 'Stop Microphone';
    } else {
      toggleBtn.classList.remove('danger');
      toggleText.textContent = 'Start Microphone';
    }
    callbacks.onToggleMic();
  });

  testToneBtn?.addEventListener('click', () => {
    callbacks.onInjectTone();
  });

  gainSlider?.addEventListener('input', (e) => {
    const val = parseFloat((e.target as HTMLInputElement).value);
    gainValue.textContent = `${val.toFixed(1)}x`;
    callbacks.onGainChange(val);
  });

  colorMapSelect?.addEventListener('change', (e) => {
    const val = (e.target as HTMLSelectElement).value;
    callbacks.onColorMapChange(val);
  });
}
