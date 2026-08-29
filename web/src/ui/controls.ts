/**
 * UI Controls and interaction bindings.
 */

export interface ControlCallbacks {
  onToggleMic: () => void;
  onToggleCamera: () => void;
  onInjectTone: () => void;
  onInjectTestVideo: () => void;
  onGainChange: (val: number) => void;
  onColorMapChange: (map: string) => void;
}

export function bindControls(callbacks: ControlCallbacks) {
  const toggleMicBtn = document.getElementById('toggleMicBtn') as HTMLButtonElement;
  const toggleMicText = document.getElementById('toggleMicText') as HTMLElement;

  const toggleCamBtn = document.getElementById('toggleCamBtn') as HTMLButtonElement;
  const toggleCamText = document.getElementById('toggleCamText') as HTMLElement;

  const testToneBtn = document.getElementById('testToneBtn') as HTMLButtonElement;
  const testVideoBtn = document.getElementById('testVideoBtn') as HTMLButtonElement;

  const gainSlider = document.getElementById('gainSlider') as HTMLInputElement;
  const gainValue = document.getElementById('gainValue') as HTMLElement;
  const colorMapSelect = document.getElementById('colorMapSelect') as HTMLSelectElement;

  let isMicActive = false;
  let isCamActive = false;

  toggleMicBtn?.addEventListener('click', () => {
    isMicActive = !isMicActive;
    if (isMicActive) {
      toggleMicBtn.classList.add('danger');
      if (toggleMicText) toggleMicText.textContent = 'Stop Microphone';
    } else {
      toggleMicBtn.classList.remove('danger');
      if (toggleMicText) toggleMicText.textContent = 'Start Microphone';
    }
    callbacks.onToggleMic();
  });

  toggleCamBtn?.addEventListener('click', () => {
    isCamActive = !isCamActive;
    if (isCamActive) {
      toggleCamBtn.classList.add('danger');
      if (toggleCamText) toggleCamText.textContent = 'Stop Camera';
    } else {
      toggleCamBtn.classList.remove('danger');
      if (toggleCamText) toggleCamText.textContent = 'Start Camera';
    }
    callbacks.onToggleCamera();
  });

  testToneBtn?.addEventListener('click', () => {
    callbacks.onInjectTone();
  });

  testVideoBtn?.addEventListener('click', () => {
    callbacks.onInjectTestVideo();
  });

  gainSlider?.addEventListener('input', (e) => {
    const val = parseFloat((e.target as HTMLInputElement).value);
    if (gainValue) gainValue.textContent = `${val.toFixed(1)}x`;
    callbacks.onGainChange(val);
  });

  colorMapSelect?.addEventListener('change', (e) => {
    const val = (e.target as HTMLSelectElement).value;
    callbacks.onColorMapChange(val);
  });
}
