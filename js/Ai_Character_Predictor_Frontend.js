const body = document.body;

// تغيير الثيم
function changeTheme(theme) {
  body.setAttribute('data-theme', theme);
}

// اختيار الموديل (ANN, CNN, Compare)
const modelCards = document.querySelectorAll('.model-card');
let selectedModel = 'ann';

modelCards.forEach(card => {
  card.addEventListener('click', () => {
    modelCards.forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    selectedModel = card.dataset.model;
  });
});

// --- تعريف متغيرات الكانفاس الأساسية (لا تحذفيها) ---
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// إعدادات لوحة الرسم المعدلة لـ TMNIST
function initCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  ctx.lineWidth = 24; 
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'black';
  
  ctx.shadowBlur = 1;
  ctx.shadowColor = 'black';
}
initCanvas();

// مسح الكانفاس (تم دمجها في دالة واحدة)
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // إعادة ضبط الخصائص
  ctx.lineWidth = document.getElementById('brushSize')?.value || 24;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
}

// --- منطق رفع الصور ---
const fileInput = document.getElementById('uploadInput'); // تأكدي أن الـ ID مطابق للـ HTML
const uploadBtn = document.querySelector('.upload-btn');
const imagePreview = document.getElementById('imagePreview');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
let uploadedFileBlob = null;

if (uploadBtn && fileInput) {
  uploadBtn.addEventListener('click', () => fileInput.click());
}

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadedFileBlob = file;
      const reader = new FileReader();
      reader.onload = (event) => {
        if (imagePreview) {
          imagePreview.src = event.target.result;
          imagePreview.classList.remove('hidden');
          if (uploadPlaceholder) uploadPlaceholder.classList.add('hidden');
        }
      };
      reader.readAsDataURL(file);
    }
  });
}

// --- منطق الرسم بالماوس (تأكدي من وجود هذه الأسطر) ---
canvas.addEventListener('mousedown', (e) => { 
  drawing = true; 
  uploadedFileBlob = null; 
});

canvas.addEventListener('mouseup', () => { 
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

const brushSize = document.getElementById('brushSize');
if (brushSize) {
  brushSize.min = 10;
  brushSize.max = 40;
  brushSize.value = 24;
  brushSize.addEventListener('input', () => {
    ctx.lineWidth = brushSize.value;
  });
}

// --- إرسال البيانات للـ API ---
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const singleResult = document.getElementById('singleResult');
const compareResult = document.getElementById('compareResult');

predictBtn.addEventListener('click', async () => {
  loading.classList.remove('hidden');
  resultCard.classList.add('hidden');

  if (uploadedFileBlob) {
    sendData(uploadedFileBlob);
  } else {
    canvas.toBlob((blob) => {
      if (blob) {
        sendData(blob);
      }
    }, 'image/png');
  }
});

async function sendData(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'character.png');
  formData.append('model_type', selectedModel);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('خطأ في الاتصال بالسيرفر');

    const data = await response.json();
    loading.classList.add('hidden');
    resultCard.classList.remove('hidden');

    if (data.processed_image) {
      const apiPreview = document.getElementById('apiPreview');
      if (apiPreview) apiPreview.src = data.processed_image;
    }

    if (selectedModel === 'compare') {
      compareResult.classList.remove('hidden');
      singleResult.innerHTML = '';
      document.getElementById('annPrediction').textContent = data.ann.prediction;
      document.getElementById('annConfidence').textContent = data.ann.confidence + '%';
      document.getElementById('cnnPrediction').textContent = data.cnn.prediction;
      document.getElementById('cnnConfidence').textContent = data.cnn.confidence + '%';
    } else {
      compareResult.classList.add('hidden');
      
      // 1. بناء قائمة الاحتمالات الـ 5 (Top 5)
      // ملاحظة: تأكدي أن السيرفر يرسل قائمة باسم top_5
      let top5HTML = '';
      if (data.top_5 && data.top_5.length > 0) {
        top5HTML = `
          <div style="margin-top: 20px; text-align: left; background: rgba(0,0,0,0.05); padding: 15px; border-radius: 15px;">
            <p style="font-size: 0.85rem; margin-bottom: 12px; font-weight: 600; opacity: 0.7; text-align: center;">
              🎯 Top 5 Possibilities
            </p>
        `;

        data.top_5.forEach((item, index) => {
          // اللون يكون أغمق للنتيجة الأولى وأفتح للباقي
          const barColor = index === 0 ? 'var(--accent)' : 'rgba(0,0,0,0.2)';
          const fontWeight = index === 0 ? 'bold' : 'normal';

          top5HTML += `
            <div style="margin-bottom: 10px;">
              <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; font-weight: ${fontWeight};">
                <span>${item.label}</span>
                <span>${item.confidence}%</span>
              </div>
              <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.4); border-radius: 10px; overflow: hidden;">
                <div style="width: ${item.confidence}%; height: 100%; background: ${barColor}; border-radius: 10px; transition: width 0.8s ease-in-out;"></div>
              </div>
            </div>
          `;
        });
        top5HTML += `</div>`;
      }

      // 2. عرض النتيجة النهائية مع الاحتمالات
      singleResult.innerHTML = `
        <div class="result-item" style="text-align: center; animation: popIn 0.5s ease;">
          <h4 style="margin-bottom:10px; opacity: 0.7;">Best Match</h4>
          <div class="pred-val" style="font-size: 4rem; font-weight: bold; color: var(--accent); line-height: 1; margin-bottom: 10px;">
            ${data.prediction}
          </div>
          <div style="display: inline-block; padding: 5px 15px; background: var(--accent); color: white; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
            Confidence: ${data.confidence}%
          </div>
          
          ${top5HTML}
          
          ${data.confidence < 60 ? `
            <p style="margin-top: 15px; font-size: 0.8rem; color: #e74c3c; font-weight: 500;">
              ⚠️ AI is a bit uncertain, check the other possibilities!
            </p>
          ` : ''}
        </div>
      `;
    }
    
  } catch (error) {
    console.error('Error:', error);
    alert('فشل الاتصال بسيرفر الذكاء الاصطناعي.');
    loading.classList.add('hidden');
  }
}