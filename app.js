const morseMap = {
  a: '.-',
  b: '-...',
  c: '-.-.',
  d: '-..',
  e: '.',
  f: '..-.',
  g: '--.',
  h: '....',
  i: '..',
  j: '.---',
  k: '-.-',
  l: '.-..',
  m: '--',
  n: '-.',
  o: '---',
  p: '.--.',
  q: '--.-',
  r: '.-.',
  s: '...',
  t: '-',
  u: '..-',
  v: '...-',
  w: '.--',
  x: '-..-',
  y: '-.--',
  z: '--..',
  0: '-----',
  1: '.----',
  2: '..---',
  3: '...--',
  4: '....-',
  5: '.....',
  6: '-....',
  7: '--...',
  8: '---..',
  9: '----.',
  ' ': '/',
};

function encodeToMorse(text) {
  return text
    .toLowerCase()
    .split('')
    .map((char) => morseMap[char] ?? char)
    .join(' ');
}

const input = document.getElementById('text-input');
const output = document.getElementById('morse-output');
const convertBtn = document.getElementById('convert-btn');
const clearBtn = document.getElementById('clear-btn');
const cameraBtn = document.getElementById('camera-btn');
const cameraStopBtn = document.getElementById('camera-stop-btn');
const cameraVideo = document.getElementById('camera-video');
const cameraStatus = document.getElementById('camera-status');

let cameraStream = null;

function updateOutput() {
  output.textContent = encodeToMorse(input.value || '');
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraStatus.textContent = 'Camera access is not supported in this browser.';
    return;
  }

  try {
    cameraStatus.textContent = 'Requesting camera access...';
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    });

    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    cameraStatus.textContent = 'Camera is live.';
  } catch (error) {
    cameraStatus.textContent = 'Camera access was blocked or unavailable.';
    console.error(error);
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  if (cameraVideo.srcObject) {
    cameraVideo.srcObject = null;
  }

  cameraStatus.textContent = 'Camera stopped.';
}

convertBtn.addEventListener('click', updateOutput);
clearBtn.addEventListener('click', () => {
  input.value = '';
  output.textContent = '';
});
cameraBtn.addEventListener('click', startCamera);
cameraStopBtn.addEventListener('click', stopCamera);
input.addEventListener('input', updateOutput);

updateOutput();
