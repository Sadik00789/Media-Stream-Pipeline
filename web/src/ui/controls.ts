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

export interface ControlHandle {
  setMicActive: (active: boolean) => void;
  setCamActive: (active: boolean) => void;
  setToneActive: (active: boolean) => void;
  setTestVideoActive: (active: boolean) => void;
}

export function bindControls(callbacks: ControlCallbacks): ControlHandle {
  const toggleMicBtn = document.getElementById('toggleMicBtn') as HTMLButtonElement | null;
  const toggleMicText = document.getElementById('toggleMicText') as HTMLElement | null;

  const toggleCamBtn = document.getElementById('toggleCamBtn') as HTMLButtonElement | null;
  const toggleCamText = document.getElementById('toggleCamText') as HTMLElement | null;

  const testToneBtn = document.getElementById('testToneBtn') as HTMLButtonElement | null;
  const testVideoBtn = document.getElementById('testVideoBtn') as HTMLButtonElement | null;

  const gainSlider = document.getElementById('gainSlider') as HTMLInputElement | null;
  const gainValue = document.getElementById('gainValue') as HTMLElement | null;
  const colorMapSelect = document.getElementById('colorMapSelect') as HTMLSelectElement | null;

  const setMicActive = (active: boolean) => {
    if (toggleMicBtn) {
      toggleMicBtn.classList.toggle('danger', active);
    }
    if (toggleMicText) {
      toggleMicText.textContent = active ? 'Stop Microphone' : 'Start Microphone';
    }
  };

  const setCamActive = (active: boolean) => {
    if (toggleCamBtn) {
      toggleCamBtn.classList.toggle('danger', active);
    }
    if (toggleCamText) {
      toggleCamText.textContent = active ? 'Stop Camera' : 'Start Camera';
    }
  };

  const setToneActive = (active: boolean) => {
    if (testToneBtn) {
      testToneBtn.classList.toggle('danger', active);
      testToneBtn.classList.toggle('secondary', !active);
    }
  };

  const setTestVideoActive = (active: boolean) => {
    if (testVideoBtn) {
      testVideoBtn.classList.toggle('danger', active);
      testVideoBtn.classList.toggle('secondary', !active);
    }
  };

  toggleMicBtn?.addEventListener('click', () => {
    callbacks.onToggleMic();
  });

  toggleCamBtn?.addEventListener('click', () => {
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

  return {
    setMicActive,
    setCamActive,
    setToneActive,
    setTestVideoActive,
  };
}
