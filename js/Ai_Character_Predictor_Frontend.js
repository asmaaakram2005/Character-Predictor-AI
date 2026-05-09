const body = document.body;

// =========================
// Theme Switcher
// =========================
function changeTheme(theme) {
  body.setAttribute('data-theme', theme);
}

// =========================
// Model Selection
// =========================
const modelCards = document.querySelectorAll('.model-card');

let selectedModel = 'cnn';

modelCards.forEach(card => {
  card.addEventListener('click', () => {
    modelCards.forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    selectedModel = card.dataset.model;
  });
});

// =========================
// Canvas
// =========================
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');

let drawing = false;

// =========================
// Canvas Init
// =========================
function initCanvas() {

  // Black background
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Drawing style
  ctx.lineWidth = 34;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // White text
  ctx.strokeStyle = 'white';

  // Glow
  ctx.shadowBlur = 10;
  ctx.shadowColor = 'white';
}

initCanvas();

// =========================
// Clear Canvas
// =========================
function clearCanvas() {

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = brushSize.value;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.strokeStyle = 'white';

  uploadedFileBlob = null;
}

// =========================
// Brush Size
// =========================
const brushSize = document.getElementById('brushSize');

brushSize.min = 10;
brushSize.max = 50;
brushSize.value = 34;

brushSize.addEventListener('input', () => {
  ctx.lineWidth = brushSize.value;
});

// =========================
// Drawing Logic
// =========================
canvas.addEventListener('mousedown', (e) => {

  drawing = true;

  uploadedFileBlob = null;

  draw(e);
});

canvas.addEventListener('mouseup', () => {

  drawing = false;

  ctx.beginPath();
});

canvas.addEventListener('mouseleave', () => {

  drawing = false;

  ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

function draw(event) {

  if (!drawing) return;

  const rect = canvas.getBoundingClientRect();

  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  ctx.lineTo(x, y);

  ctx.stroke();

  ctx.beginPath();

  ctx.moveTo(x, y);
}

canvas.addEventListener("touchstart", function (e) {
    if (e.target == canvas) {
        e.preventDefault(); 
        var touch = e.touches[0];
        var rect = canvas.getBoundingClientRect();
        var mouseEvent = new MouseEvent("mousedown", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }
}, false);

canvas.addEventListener("touchmove", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }
}, false);

canvas.addEventListener("touchend", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
        var mouseEvent = new MouseEvent("mouseup", {});
        canvas.dispatchEvent(mouseEvent);
    }
}, false);

// =========================
// Upload System
// =========================
const fileInput = document.getElementById('fileInput');

const uploadBtn = document.getElementById('uploadBtn');

const clearImageBtn = document.getElementById('clearImageBtn');

const imagePreview = document.getElementById('imagePreview');

const uploadPlaceholder = document.getElementById('uploadPlaceholder');

let uploadedFileBlob = null;

// Upload click
uploadBtn.addEventListener('click', () => {
  fileInput.click();
});

// File selected
fileInput.addEventListener('change', (e) => {

  const file = e.target.files[0];

  if (!file) return;

  uploadedFileBlob = file;

  const reader = new FileReader();

  reader.onload = function(event) {

    imagePreview.src = event.target.result;

    imagePreview.classList.remove('hidden');

    uploadPlaceholder.classList.add('hidden');
  };

  reader.readAsDataURL(file);
});

// Clear uploaded image
clearImageBtn.addEventListener('click', () => {

  uploadedFileBlob = null;

  fileInput.value = '';

  imagePreview.src = '';

  imagePreview.classList.add('hidden');

  uploadPlaceholder.classList.remove('hidden');
});

// =========================
// Prediction
// =========================
const predictBtn = document.getElementById('predictBtn');

const loading = document.getElementById('loading');

const resultCard = document.getElementById('resultCard');

const singleResult = document.getElementById('singleResult');

const compareResult = document.getElementById('compareResult');

predictBtn.addEventListener('click', async () => {

  loading.classList.remove('hidden');

  resultCard.classList.add('hidden');

  // Animation
  predictBtn.innerHTML = "Analyzing... 🤖";

  // Upload image
  if (uploadedFileBlob) {

    sendData(uploadedFileBlob);

  } else {

    // Canvas image
    canvas.toBlob((blob) => {

      if (blob) {

        sendData(blob);
      }

    }, 'image/png');
  }
});

// =========================
// Send To API
// =========================
async function sendData(blob) {

  const formData = new FormData();

  formData.append('file', blob, 'character.png');

  formData.append('model_type', selectedModel);

  try {

    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {

      throw new Error('Server Error');
    }

    const data = await response.json();

    loading.classList.add('hidden');

    resultCard.classList.remove('hidden');

    predictBtn.innerHTML = "Analyze Character ✨";

    // =========================
    // Processed Preview
    // =========================
    if (data.processed_image) {

      const apiPreview = document.getElementById('apiPreview');

      apiPreview.src = data.processed_image;
    }

    // =========================
    // Compare Mode
    // =========================
    if (selectedModel === 'compare') {

      compareResult.classList.remove('hidden');

      singleResult.innerHTML = '';

      document.getElementById('annPrediction').textContent =
        data.ann.prediction;

      document.getElementById('annConfidence').textContent =
        data.ann.confidence + '%';

      document.getElementById('cnnPrediction').textContent =
        data.cnn.prediction;

      document.getElementById('cnnConfidence').textContent =
        data.cnn.confidence + '%';
    }

    // =========================
    // Single Model
    // =========================
    else {

      compareResult.classList.add('hidden');

      let top5HTML = '';

      // Show top5 only if confidence < 70
      if (
        data.confidence < 70 &&
        data.top_5 &&
        data.top_5.length > 0
      ) {

        top5HTML = `
          <div class="probability-container">

            <p style="
              text-align:center;
              margin-bottom:15px;
              font-weight:600;
              opacity:0.7;
            ">
              🤔 AI isn't fully sure...
            </p>
        `;

        data.top_5.forEach((item, index) => {

          const barColor =
            index === 0
              ? 'var(--accent)'
              : 'rgba(255,255,255,0.25)';

          top5HTML += `
            <div class="prob-bar-wrapper">

              <div class="prob-info">
                <span>${item.label}</span>
                <span>${item.confidence}%</span>
              </div>

              <div class="prob-bar-bg">

                <div 
                  class="prob-bar-fill"
                  style="
                    width:${item.confidence}%;
                    background:${barColor};
                  "
                ></div>

              </div>

            </div>
          `;
        });

        top5HTML += `</div>`;
      }

      // Final result
      singleResult.innerHTML = `

        <div 
          class="result-item"
          style="
            text-align:center;
            animation:popIn 0.5s ease;
          "
        >

          <h4 style="
            margin-bottom:10px;
            opacity:0.7;
          ">
            Best Match
          </h4>

          <div 
            class="pred-val"
            style="
              font-size:5rem;
              font-weight:bold;
              color:var(--accent);
              line-height:1;
              margin-bottom:15px;
              text-shadow:0 0 20px var(--accent);
            "
          >
            ${data.prediction}
          </div>

          <div style="
            display:inline-block;
            padding:8px 18px;
            background:var(--accent);
            color:white;
            border-radius:30px;
            font-weight:600;
            margin-bottom:15px;
          ">
            Confidence: ${data.confidence}%
          </div>

          ${top5HTML}

        </div>
      `;
    }

  } catch (error) {

    console.error(error);

    alert("Failed to connect to AI server 😭");

    loading.classList.add('hidden');

    predictBtn.innerHTML = "Analyze Character ✨";
  }
}
