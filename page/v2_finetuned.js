// v2_finetuned.js - Lógica del Canvas y API v2
document.addEventListener('DOMContentLoaded', function () {
  // Configuración del Canvas
  const canvasContainer = document.querySelector('.flex.flex-col.items-center.justify-center');
  const canvas = document.createElement('canvas');
  canvas.width = 280;
  canvas.height = 280;
  canvas.className = 'border-2 border-slate-700 rounded-lg bg-black cursor-crosshair';
  canvas.id = 'drawingCanvas';

  // Reemplazar placeholder con el canvas real
  canvasContainer.innerHTML = '';
  canvasContainer.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  let isDrawing = false;
  let isErasing = false;

  // Funciones de dibujo
  function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    if (isErasing) {
      ctx.strokeStyle = 'black';
    } else {
      ctx.strokeStyle = 'white';
    }

    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function stopDrawing() {
    isDrawing = false;
  }

  // Eventos de ratón
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing);

  // Eventos táctiles para móviles
  canvas.addEventListener('touchstart', startDrawing);
  canvas.addEventListener('touchmove', draw);
  canvas.addEventListener('touchend', stopDrawing);

  // Manejadores de botones
  const deleteBtn = document.getElementById('deleteBtn');
  const eraserBtn = document.getElementById('eraserBtn');
  const editBtn = document.getElementById('editBtn');
  const lineWeightRange = document.getElementById('lineWeightRange');
  const recognizeBtn = document.getElementById('recognizeBtn');
  const resultContainer = document.getElementById('resultContainer');
  const confidenceScore = document.getElementById('confidenceScore');
  const confidenceBar = document.getElementById('confidenceBar');
  const altPredictions = document.getElementById('altPredictions');

  // Limpiar lienzo
  deleteBtn?.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultContainer.innerHTML = '<p class="text-slate-400">Dibuja un caracter para reconocerlo</p>';
    document.getElementById('confidenceContainer').style.display = 'none';
    document.getElementById('altPredictionsContainer').style.display = 'none';
  });

  // Activar borrador
  eraserBtn?.addEventListener('click', () => {
    isErasing = !isErasing;
    if (isErasing) {
      eraserBtn.classList.add('bg-primary/20', 'ring-1', 'ring-primary');
      editBtn.classList.remove('bg-primary/20', 'ring-1', 'ring-primary');
    } else {
      eraserBtn.classList.remove('bg-primary/20', 'ring-1', 'ring-primary');
    }
  });

  // Activar modo dibujo (lápiz)
  editBtn?.addEventListener('click', () => {
    isErasing = false;
    eraserBtn.classList.remove('bg-primary/20', 'ring-1', 'ring-primary');
    editBtn.classList.add('bg-primary/20', 'ring-1', 'ring-primary');
  });

  // Ajustar grosor del trazo
  lineWeightRange?.addEventListener('input', (e) => {
    ctx.lineWidth = parseInt(e.target.value) * 3;
  });

  /**
   * Preprocesar imagen del canvas EXACTAMENTE como Streamlit/EMNIST
   * 
   * Pasos:
   * 1. Convertir a escala de grises
   * 2. Encontrar bounding box del contenido (recorte inteligente)
   * 3. Recortar al bounding box
   * 4. Escalar preservando aspect ratio a 20x20 max
   * 5. Centrar en canvas 28x28 con fondo negro
   * 6. Normalizar a [0, 1]
   * 7. Limpiar ruido (valores < 0.05 = 0)
   */
  function preprocessImage(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;

    // 1. Convertir a escala de grises (usar canal rojo ya que dibujamos blanco 0-255)
    const grayscale = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      grayscale[i] = data[i * 4];
    }

    // 2. Encontrar bounding box con umbral
    const threshold = 30;
    let minX = width, maxX = 0, minY = height, maxY = 0;
    let hasContent = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (grayscale[y * width + x] > threshold) {
          hasContent = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Si no hay contenido, devolver array vacío
    if (!hasContent) {
      return new Float32Array(784).fill(0);
    }

    // 3. Recortar al bounding box
    const cropWidth = maxX - minX + 1;
    const cropHeight = maxY - minY + 1;
    const cropped = new Float32Array(cropWidth * cropHeight);

    for (let y = 0; y < cropHeight; y++) {
      for (let x = 0; x < cropWidth; x++) {
        cropped[y * cropWidth + x] = grayscale[(y + minY) * width + (x + minX)];
      }
    }

    // 4. Escalar preservando aspect ratio a 20x20 max
    const targetSize = 20;
    let newWidth, newHeight;

    if (cropHeight > cropWidth) {
      newHeight = targetSize;
      newWidth = Math.max(1, Math.round(cropWidth * targetSize / cropHeight));
    } else {
      newWidth = targetSize;
      newHeight = Math.max(1, Math.round(cropHeight * targetSize / cropWidth));
    }

    // Crear canvas temporal para resize con interpolación de alta calidad
    const tempCanvas1 = document.createElement('canvas');
    tempCanvas1.width = cropWidth;
    tempCanvas1.height = cropHeight;
    const tempCtx1 = tempCanvas1.getContext('2d');
    const tempImageData = tempCtx1.createImageData(cropWidth, cropHeight);

    for (let i = 0; i < cropped.length; i++) {
      const val = Math.round(cropped[i]);
      tempImageData.data[i * 4] = val;
      tempImageData.data[i * 4 + 1] = val;
      tempImageData.data[i * 4 + 2] = val;
      tempImageData.data[i * 4 + 3] = 255;
    }
    tempCtx1.putImageData(tempImageData, 0, 0);

    // Resize con suavizado
    const tempCanvas2 = document.createElement('canvas');
    tempCanvas2.width = newWidth;
    tempCanvas2.height = newHeight;
    const tempCtx2 = tempCanvas2.getContext('2d');
    tempCtx2.imageSmoothingEnabled = true;
    tempCtx2.imageSmoothingQuality = 'high';
    tempCtx2.drawImage(tempCanvas1, 0, 0, newWidth, newHeight);

    const scaledData = tempCtx2.getImageData(0, 0, newWidth, newHeight);

    // 5. Centrar en imagen 28x28 (fondo negro)
    const final = new Float32Array(28 * 28).fill(0);
    const offsetX = Math.floor((28 - newWidth) / 2);
    const offsetY = Math.floor((28 - newHeight) / 2);

    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcIdx = (y * newWidth + x) * 4;
        const dstIdx = (y + offsetY) * 28 + (x + offsetX);
        // 6. Normalizar a [0, 1]
        let value = scaledData.data[srcIdx] / 255.0;
        // 7. Limpiar ruido (valores < 0.05 = 0)
        if (value < 0.05) value = 0;
        final[dstIdx] = value;
      }
    }

    return final;
  }

  // Función de reconocimiento principal
  recognizeBtn?.addEventListener('click', async () => {
    try {
      // Obtener datos del canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Preprocesar con el algoritmo avanzado
      const processed = preprocessImage(imageData);

      // Verificar si hay contenido dibujado
      const sum = processed.reduce((a, b) => a + b, 0);
      if (sum < 0.1) {
        resultContainer.innerHTML = '<p class="text-yellow-400">Dibuja algo primero</p>';
        return;
      }

      // Mostrar estado de carga
      resultContainer.innerHTML = '<p class="text-slate-400">Reconociendo...</p>';

      // Llamada a la API
      const response = await fetch('/api/v2/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: Array.from(processed) })
      });

      if (!response.ok) {
        throw new Error(`Error API: ${response.status}`);
      }

      const result = await response.json();

      // Mostrar resultados
      resultContainer.innerHTML = `
        <p class="text-7xl font-black text-white">${result.character}</p>
        <p class="text-sm font-medium text-slate-400">Caracter Reconocido</p>
      `;

      // Mostrar contenedor de confianza
      const confidenceContainer = document.getElementById('confidenceContainer');
      confidenceContainer.style.display = 'block';

      const conf = (result.confidence * 100).toFixed(1);
      confidenceScore.textContent = `${conf}%`;
      confidenceBar.style.width = `${conf}%`;
      confidenceBar.className = `h-2 rounded-full ${conf > 80 ? 'bg-green-500' : conf > 50 ? 'bg-yellow-500' : 'bg-red-500'}`;

      // Mostrar predicciones alternativas
      const altContainer = document.getElementById('altPredictionsContainer');
      altContainer.style.display = 'block';
      altPredictions.innerHTML = result.top5.slice(1, 4).map(pred => `
        <li class="flex justify-between items-center text-sm">
          <span class="font-mono text-slate-300">${pred.character}</span>
          <span class="text-slate-400">${(pred.probability * 100).toFixed(1)}%</span>
        </li>
      `).join('');

    } catch (error) {
      console.error('Error de reconocimiento:', error);
      resultContainer.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    }
  });
});
