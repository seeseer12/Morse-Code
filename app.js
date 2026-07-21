import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2';

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

const morseReverseMap = Object.fromEntries(
  Object.entries(morseMap).map(([key, value]) => [value, key])
);

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
const decodedOutput = document.getElementById('decoded-output');

let cameraStream = null;
let faceLandmarker = null;
let decoderRunning = false;
let animationFrameId = null;
let eyeClosed = false;
let blinkStartTime = 0;
let lastBlinkTime = 0;
let lastActivityTime = 0;
let currentPattern = '';
let decodedText = '';

function updateOutput() {
  output.textContent = encodeToMorse(input.value || '');
}

function resetDecoder() {
  currentPattern = '';
  decodedText = '';
  lastBlinkTime = 0;
  lastActivityTime = 0;
  eyeClosed = false;
  decodedOutput.textContent = 'Decoded text will appear here.';
}

function finalizeLetter(now) {
  if (!currentPattern) {
    return;
  }

  const decodedChar = morseReverseMap[currentPattern] || '?';
  decodedText += decodedChar;
  decodedOutput.textContent = decodedText;
  currentPattern = '';
  lastActivityTime = now;
}

function handleBlink(isDot, now) {
  lastBlinkTime = now;
  lastActivityTime = now;
  currentPattern += isDot ? '.' : '-';
  decodedOutput.textContent = `${decodedText}${currentPattern}`;
  cameraStatus.textContent = isDot ? 'Detected dot' : 'Detected dash';
}

function computeEar(landmarks, width, height) {
  const points = landmarks.map((landmark) => [landmark.x * width, landmark.y * height]);
  const leftEye = [points[1], points[2], points[3], points[4], points[5], points[6]];
  const rightEye = [points[7], points[8], points[9], points[10], points[11], points[12]];

  const earForEye = (eye) => {
    const verticalA = Math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1]);
    const verticalB = Math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1]);
    const horizontal = Math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1]);
    return horizontal > 0 ? (verticalA + verticalB) / (2 * horizontal) : 0;
  };

  return (earForEye(leftEye) + earForEye(rightEye)) / 2;
}

async function loadModel() {
  if (faceLandmarker) {
    return faceLandmarker;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm');
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    return faceLandmarker;
  } catch (error) {
    cameraStatus.textContent = 'The browser model could not be loaded. Please try again in a secure context.';
    console.error(error);
    throw error;
  }
}

async function startDecoder() {
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
    await loadModel();

    resetDecoder();
    decoderRunning = true;
    cameraStatus.textContent = 'Decoder is running. Blink to spell Morse.';

    const processFrame = () => {
      if (!decoderRunning) {
        return;
      }

      const now = performance.now();
      if (cameraVideo.readyState >= 2 && faceLandmarker) {
        const result = faceLandmarker.detectForVideo(cameraVideo, now);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
          const landmarks = result.faceLandmarks[0];
          const ear = computeEar(landmarks, cameraVideo.videoWidth, cameraVideo.videoHeight);
          if (ear < 0.22) {
            if (!eyeClosed) {
              eyeClosed = true;
              blinkStartTime = now;
            }
          } else if (eyeClosed) {
            const blinkDuration = now - blinkStartTime;
            const isDot = blinkDuration <= 350;
            handleBlink(isDot, now);
            eyeClosed = false;
          }
        }
      }

      if (currentPattern && now - lastBlinkTime > 1200) {
        finalizeLetter(now);
      }

      if (!currentPattern && decodedText && now - lastActivityTime > 2500) {
        decodedText += ' ';
        decodedOutput.textContent = decodedText;
        lastActivityTime = now;
      }

      animationFrameId = window.requestAnimationFrame(processFrame);
    };

    if (animationFrameId) {
      window.cancelAnimationFrame(animationFrameId);
    }

    animationFrameId = window.requestAnimationFrame(processFrame);
  } catch (error) {
    cameraStatus.textContent = 'Camera access was blocked or unavailable.';
    console.error(error);
  }
}

function stopDecoder() {
  decoderRunning = false;
  if (animationFrameId) {
    window.cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  if (cameraVideo.srcObject) {
    cameraVideo.srcObject = null;
  }

  eyeClosed = false;
  cameraStatus.textContent = 'Decoder stopped.';
}

convertBtn.addEventListener('click', updateOutput);
clearBtn.addEventListener('click', () => {
  input.value = '';
  output.textContent = '';
});
cameraBtn.addEventListener('click', startDecoder);
cameraStopBtn.addEventListener('click', stopDecoder);
input.addEventListener('input', updateOutput);

updateOutput();
