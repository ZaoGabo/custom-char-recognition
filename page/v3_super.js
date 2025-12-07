// v3_super.js - Carga de imágenes y reconocimiento de texto completo
document.addEventListener('DOMContentLoaded', function () {
  // Referencias a elementos del DOM
  const browseBtn = document.getElementById('browseFilesBtn');
  const fileInput = document.getElementById('fileInput');
  const previewContainer = document.getElementById('previewContainer');
  const recognizeBtn = document.getElementById('recognizeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const progressBarContainer = document.getElementById('progressBarContainer');
  const progressBar = document.getElementById('progressBar');
  const recognizedTextContainer = document.getElementById('recognizedTextContainer');
  const copyBtn = document.getElementById('copyBtn');
  const downloadBtn = document.getElementById('downloadBtn');

  let currentImage = null;

  // Abrir selector de archivos
  browseBtn?.addEventListener('click', () => fileInput?.click());

  // Manejar selección de archivo y previsualización
  fileInput?.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;

    previewContainer.innerHTML = '';

    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.className = 'max-h-96 rounded shadow';
      img.onload = () => {
        currentImage = img;
        previewContainer.appendChild(img);
      };
    } else {
      previewContainer.innerHTML = '<p class="text-red-500">Solo se permiten archivos de imagen</p>';
    }
  });

  // Reconocer texto desde imagen usando TrOCR
  recognizeBtn?.addEventListener('click', async () => {
    if (!currentImage) {
      recognizedTextContainer.innerHTML = '<p class="text-red-500">Por favor, sube una imagen primero</p>';
      return;
    }

    try {
      // Mostrar barra de progreso
      progressBarContainer.style.display = 'block';
      progressBar.style.width = '30%';
      recognizedTextContainer.innerHTML = '<p class="text-gray-500 dark:text-gray-400">Procesando imagen con TrOCR...</p>';

      // Crear canvas para obtener la imagen en base64
      // Mantenemos la resolución original para el modelo Transformer
      const canvas = document.createElement('canvas');
      canvas.width = currentImage.naturalWidth;
      canvas.height = currentImage.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(currentImage, 0, 0);

      // Obtener imagen en formato Base64
      const imageBase64 = canvas.toDataURL('image/png');

      progressBar.style.width = '60%';

      // Llamar API TrOCR (endpoint v4)
      const response = await fetch('/api/v4/predict_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error API: ${response.status}`);
      }

      const result = await response.json();

      progressBar.style.width = '100%';

      // Mostrar resultados con un pequeño delay para suavidad
      setTimeout(() => {
        progressBarContainer.style.display = 'none';

        // Definir colores según nivel de confianza
        const confidenceColor = result.confidence > 0.8 ? 'bg-green-500' : result.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500';
        const confidenceTextColor = result.confidence > 0.8 ? 'text-green-400' : result.confidence > 0.5 ? 'text-yellow-400' : 'text-red-400';

        const resultHTML = `
          <div class="space-y-4">
            <div class="p-6 bg-slate-900 rounded-lg">
              <h3 class="text-sm font-semibold text-slate-400 mb-2">Texto Reconocido:</h3>
              <p class="text-2xl font-bold text-white break-words">${result.text || '(Sin texto detectado)'}</p>
            </div>
            <div class="space-y-2">
              <div class="flex justify-between text-sm">
                <span class="text-slate-300">Confianza:</span>
                <span class="font-semibold ${confidenceTextColor}">${(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div class="w-full bg-slate-700 rounded-full h-2">
                <div class="${confidenceColor} h-2 rounded-full transition-all duration-300" style="width: ${result.confidence * 100}%"></div>
              </div>
            </div>
            <div class="text-xs text-slate-500 text-center">
              Procesado con TrOCR (Transformer OCR)
            </div>
          </div>
        `;

        recognizedTextContainer.innerHTML = resultHTML;
      }, 500);

    } catch (error) {
      console.error('Error de reconocimiento:', error);
      progressBarContainer.style.display = 'none';
      recognizedTextContainer.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
    }
  });

  // Limpiar todo (botón limpiar)
  clearBtn?.addEventListener('click', () => {
    fileInput.value = '';
    currentImage = null;
    previewContainer.innerHTML = '<p class="text-gray-500 dark:text-gray-400">La previsualización aparecerá aquí.</p>';
    recognizedTextContainer.innerHTML = '<p class="text-gray-500 dark:text-gray-400">El texto reconocido aparecerá aquí.</p>';
    progressBarContainer.style.display = 'none';
  });

  // Copiar texto al portapapeles
  copyBtn?.addEventListener('click', () => {
    const text = recognizedTextContainer.textContent;
    if (text) {
      navigator.clipboard.writeText(text);
      // Feedback visual temporal
      const originalHTML = copyBtn.innerHTML;
      copyBtn.innerHTML = '<span class="material-symbols-outlined text-xl text-green-500">check</span>';
      setTimeout(() => (copyBtn.innerHTML = originalHTML), 1000);
    }
  });

  // Descargar texto como archivo .txt
  downloadBtn?.addEventListener('click', () => {
    const text = recognizedTextContainer.textContent;
    if (text) {
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'texto_reconocido.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  });
});
