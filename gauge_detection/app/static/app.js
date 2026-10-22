/**
 * GaugeVision AI - Industrial Gauge Detection & Reading Application
 * Interactive Canvas, Layers, Radial Meter, and API Integrations
 */

// Application State
const state = {
  currentImage: null,          // Image object
  currentImageBytes: null,     // ArrayBuffer or Blob
  currentFilename: null,       // String
  currentSampleId: null,       // String or null
  currentFile: null,           // File object for uploaded images
  detectionResult: null,       // DetectionResult JSON
  selectedGaugeIndex: 0,       // Active gauge in multi-gauge scenario
  hoveredGaugeIndex: null,     // Currently hovered gauge
  zoom: 1.0,                   // Canvas zoom level
  pan: { x: 0, y: 0 },         // Canvas pan offset
  isPanning: false,
  startPan: { x: 0, y: 0 },
  layers: {
    gauges: true,
    needles: true,
    numbers: true,
    hud: true,
    angles: true,
  },
  config: null,
  cameraStream: null,
  sampleCache: {},             // In-memory cache for instant demo sample display
  promptTemplates: [],         // Real-world industrial prompt templates catalog
  selectedTemplateId: 'default',
  isPromptEdited: false,
};

// DOM Elements
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');
const canvasWrapper = document.getElementById('canvasWrapper');
const canvasEmpty = document.getElementById('canvasEmpty');
const canvasLoader = document.getElementById('canvasLoader');
const loaderText = document.getElementById('loaderText');

// Init application
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  setupCanvasInteractions();
  await loadConfig();
  await loadPromptTemplates();
  await loadSamples();

  // Auto-run first sample for instant WOW experience
  const firstSampleCard = document.querySelector('.sample-card');
  if (firstSampleCard) {
    firstSampleCard.click();
  }
});


/* ============================================================
   CONFIG & SAMPLES LOADING
   ============================================================ */
async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    state.config = await res.json();
    updateProviderBadge();
  } catch (err) {
    console.error('Failed to load config:', err);
  }
}

function updateProviderBadge() {
  if (!state.config) return;
  const badgeText = document.getElementById('providerBadgeText');
  const provider = (state.config.provider || 'gemini').toUpperCase();
  
  if (provider === 'HYBRID') {
    const model = state.config.hybrid?.litellm_model || state.config.litellm?.model || 'Nemotron-VL';
    badgeText.textContent = `⚡ HYBRID (YOLO + ${model.split('-')[0]})`;
  } else if (provider === 'GEMINI') {
    const model = state.config.gemini?.model || 'gemini-flash-lite-latest';
    const cfg = state.config.gemini?.configured ? '' : ' (No Key)';
    badgeText.textContent = `GEMINI (${model})${cfg}`;
  } else if (provider === 'OPENAI') {
    badgeText.textContent = `OPENAI (${state.config.openai?.model || 'GPT-4O'})`;
  } else if (provider === 'LITELLM') {
    const model = state.config.litellm?.model || 'Proxy';
    badgeText.textContent = `LITELLM (${model})`;
  } else {
    badgeText.textContent = `AI VISION (${provider})`;
  }
}

/* ============================================================
   PROMPT TEMPLATES & GAUGE CATALOG
   ============================================================ */
function getActivePrompt() {
  const textarea = document.getElementById('promptEditorTextarea');
  return textarea ? textarea.value.trim() : null;
}

function getActiveGaugeType() {
  const select = document.getElementById('gaugeTypeSelect');
  return select ? select.value : 'default';
}

async function loadPromptTemplates() {
  try {
    const res = await fetch('/api/prompts/templates');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const templates = await res.json();
    state.promptTemplates = templates;
    renderTemplateSelector();
  } catch (err) {
    console.error('Failed to load prompt templates:', err);
  }
}

function renderTemplateSelector() {
  const select = document.getElementById('gaugeTypeSelect');
  if (!select || !state.promptTemplates.length) return;

  select.innerHTML = '';
  
  // Group templates by category
  const categories = {};
  state.promptTemplates.forEach(t => {
    const cat = t.category || 'Other';
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(t);
  });

  Object.keys(categories).forEach(cat => {
    const group = document.createElement('optgroup');
    group.label = cat.toUpperCase();
    categories[cat].forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.id;
      opt.textContent = t.name;
      if (t.id === state.selectedTemplateId) opt.selected = true;
      group.appendChild(opt);
    });
    select.appendChild(group);
  });

  applySelectedTemplate(state.selectedTemplateId || 'default', true);
}

function applySelectedTemplate(templateId, resetEditor = true) {
  const tpl = state.promptTemplates.find(t => t.id === templateId) || state.promptTemplates[0];
  if (!tpl) return;

  state.selectedTemplateId = tpl.id;

  // Update meta information
  const descEl = document.getElementById('promptDescText');
  if (descEl) descEl.textContent = tpl.description || '';

  const shapeEl = document.getElementById('metaShapeVal');
  if (shapeEl) shapeEl.textContent = tpl.shape || 'Standard';

  const indEl = document.getElementById('metaIndicatorVal');
  if (indEl) indEl.textContent = tpl.indicator_type || 'Pointer / Digits';

  const badgeAuto = document.getElementById('badgeAutoMode');
  if (badgeAuto) {
    if (tpl.id === 'default') {
      badgeAuto.textContent = 'AUTO';
      badgeAuto.style.background = '#254032';
      badgeAuto.style.borderColor = '#3b5a45';
      badgeAuto.style.color = '#bddeae';
    } else {
      badgeAuto.textContent = 'MANUAL';
      badgeAuto.style.background = '#21372c';
      badgeAuto.style.borderColor = '#3e5843';
      badgeAuto.style.color = '#d5f0cd';
    }
  }

  const textarea = document.getElementById('promptEditorTextarea');
  if (textarea && resetEditor) {
    textarea.value = tpl.prompt || '';
    updatePromptCharCount();
    setPromptStatusBadge(false);
    state.isPromptEdited = false;
  }
}

function updatePromptCharCount() {
  const textarea = document.getElementById('promptEditorTextarea');
  const countEl = document.getElementById('promptCharCount');
  if (textarea && countEl) {
    countEl.textContent = textarea.value.length;
  }
}

function setPromptStatusBadge(isEdited) {
  const badge = document.getElementById('promptStatusBadge');
  if (!badge) return;
  if (isEdited) {
    badge.textContent = 'Customized';
    badge.className = 'status-edited';
  } else {
    badge.textContent = 'Standard';
    badge.className = 'status-standard';
  }
}

async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();
    const grid = document.getElementById('samplesGrid');
    grid.innerHTML = '';

    samples.forEach((sample, idx) => {
      // Pre-cache detection payload if returned by server
      if (sample.detection_data) {
        state.sampleCache[sample.id] = sample.detection_data;
      }

      const card = document.createElement('div');
      card.className = 'sample-card';
      card.dataset.sampleId = sample.id;
      card.innerHTML = `
        <img class="sample-thumb" src="${sample.thumbnail_url}" alt="${sample.title}">
        <div class="sample-info">
          <div class="sample-title">${sample.title}</div>
          <div class="sample-type">${sample.expected_type.toUpperCase()}</div>
        </div>
      `;
      card.addEventListener('click', () => runSampleDetection(sample, card));
      grid.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load samples:', err);
  }
}

/* ============================================================
   DETECTION WORKFLOWS
   ============================================================ */
async function runSampleDetection(sample, cardElement) {
  // Update active state in UI
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  if (cardElement) cardElement.classList.add('active');

  state.currentSampleId = sample.id;
  state.currentFilename = sample.filename;
  state.currentFile = null;

  // FAST PATH: Instant display using precomputed cache (NO loading spinner, zero wait!)
  const cached = state.sampleCache[sample.id] || sample.detection_data;
  if (cached) {
    await loadImageFromUrl(sample.thumbnail_url);
    handleDetectionSuccess(cached, true);
    return;
  }

  // Fallback to fetch if not yet loaded in cache
  showLoader(`SCANNING SAMPLE: ${sample.title.toUpperCase()}`, 'Loading pre-computed detection data...');

  try {
    // 1. Load image onto canvas
    await loadImageFromUrl(sample.thumbnail_url);

    // 2. Call detection API
    const formData = new FormData();
    formData.append('sample_id', sample.id);

    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(`Detection failed with HTTP ${res.status}`);
    const data = await res.json();
    state.sampleCache[sample.id] = data;
    handleDetectionSuccess(data, false);

  } catch (err) {
    showToast(`Detection error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    hideLoader();
  }
}

async function runUploadedImageDetection(file) {
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  state.currentSampleId = null;
  state.currentFilename = file.name;
  state.currentFile = file;

  showLoader(`ANALYZING: ${file.name.toUpperCase()}`);

  try {
    // Read and load image
    const reader = new FileReader();
    const loadPromise = new Promise((resolve) => {
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          state.currentImage = img;
          resolve();
        };
        img.src = e.target.result;
      };
    });
    reader.readAsDataURL(file);
    await loadPromise;

    // Send to API with active prompt configuration
    const formData = new FormData();
    formData.append('file', file);
    
    const customPrompt = getActivePrompt();
    const gaugeType = getActiveGaugeType();
    if (customPrompt) formData.append('custom_prompt', customPrompt);
    if (gaugeType) formData.append('gauge_type', gaugeType);

    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(`Detection API returned error HTTP ${res.status}`);
    const data = await res.json();
    handleDetectionSuccess(data);

  } catch (err) {
    showToast(`Image processing error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    hideLoader();
  }
}

async function reanalyzeCurrentImage() {
  if (!state.currentImage && !state.currentFile && !state.currentSampleId) {
    showToast('No gauge image loaded. Please select a sample or upload a photo first.', 'warning');
    return;
  }

  // Refresh latest runtime configuration from backend
  try {
    await loadConfig();
  } catch (err) {
    console.warn('Config refresh warning:', err);
  }

  const activeProvider = state.config?.provider || 'gemini';
  const providerLabel = activeProvider.toUpperCase();

  const btns = [document.getElementById('btnReanalyze'), document.getElementById('btnReanalyzeSide')].filter(Boolean);
  btns.forEach(b => {
    b.disabled = true;
    b.classList.add('loading');
  });

  const customPrompt = getActivePrompt();
  const gaugeType = getActiveGaugeType();

  showLoader(`RE-ANALYZING WITH ${providerLabel}...`, `Running live AI inference using model: ${getModelDisplayName(activeProvider)}`);

  try {
    let result = null;

    if (state.currentSampleId) {
      // Re-run live AI model on sample image, explicitly passing provider_override, force_live, custom_prompt, gauge_type
      const formData = new FormData();
      formData.append('sample_id', state.currentSampleId);
      formData.append('provider_override', activeProvider);
      formData.append('force_live', 'true');
      if (customPrompt) formData.append('custom_prompt', customPrompt);
      if (gaugeType) formData.append('gauge_type', gaugeType);

      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Detection failed with status HTTP ${res.status}`);
      }
      result = await res.json();
    } else if (state.currentFile) {
      // Re-run live AI model on user uploaded file with active prompt
      const formData = new FormData();
      formData.append('file', state.currentFile);
      formData.append('provider_override', activeProvider);
      formData.append('force_live', 'true');
      if (customPrompt) formData.append('custom_prompt', customPrompt);
      if (gaugeType) formData.append('gauge_type', gaugeType);

      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Detection failed with status HTTP ${res.status}`);
      }
      result = await res.json();
    } else if (state.currentImage) {
      // Fallback for webcam snapshots or canvas states
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = state.currentImage.naturalWidth || state.currentImage.width;
      tempCanvas.height = state.currentImage.naturalHeight || state.currentImage.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(state.currentImage, 0, 0);
      const b64 = tempCanvas.toDataURL('image/jpeg', 0.95);

      const res = await fetch('/api/detect/json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: b64,
          provider: activeProvider,
          force_live: true,
          filename: state.currentFilename || 'gauge_reanalysis.jpg',
          custom_prompt: customPrompt,
          gauge_type: gaugeType
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Detection failed with status HTTP ${res.status}`);
      }
      result = await res.json();
    }

    if (result) {
      handleDetectionSuccess(result, false, true);
      showToast(`Re-analysis complete! Model: ${result.model || providerLabel} (${result.execution_time_ms}ms)`, 'success');
    }
  } catch (err) {
    showToast(`Re-analysis error: ${err.message}`, 'error');
    console.error('Re-analyze error:', err);
  } finally {
    hideLoader();
    btns.forEach(b => {
      b.disabled = false;
      b.classList.remove('loading');
    });
  }
}

function getModelDisplayName(provider) {
  if (!state.config) return provider.toUpperCase();
  if (provider === 'gemini') return state.config.gemini?.model || 'gemini-2.5-flash';
  if (provider === 'openai') return state.config.openai?.model || 'gpt-4o-mini';
  if (provider === 'litellm') return state.config.litellm?.model || 'Proxy Model';
  return 'Mock Demo';
}

function handleDetectionSuccess(result, isInstant = false, preserveZoom = false) {
  state.detectionResult = result;
  state.selectedGaugeIndex = 0;

  // Update Footer Stats (Model, Latency, Total Tokens, Resolution, Gauges, Provider)
  const modelEl = document.getElementById('statModel');
  if (modelEl) {
    modelEl.textContent = result.model || '-';
    modelEl.title = `Model: ${result.model || 'Default'}`;
  }

  const latencyEl = document.getElementById('statExecutionTime');
  if (latencyEl) {
    const lat = result.execution_time_ms;
    latencyEl.textContent = isInstant ? '0.5 ms (Cached)' : (lat >= 1000 ? `${(lat / 1000).toFixed(2)} s` : `${lat} ms`);
  }

  const tokensEl = document.getElementById('statTotalTokens');
  if (tokensEl) {
    if (result.total_tokens !== null && result.total_tokens !== undefined) {
      tokensEl.textContent = `${result.total_tokens.toLocaleString()}`;
      const p = result.prompt_tokens || 0;
      const c = result.completion_tokens || 0;
      tokensEl.title = `Prompt: ${p.toLocaleString()} | Completion: ${c.toLocaleString()}`;
    } else {
      tokensEl.textContent = isInstant ? 'Cached' : 'N/A';
      tokensEl.title = 'Token count not reported';
    }
  }

  document.getElementById('statResolution').textContent = `${result.image_width} x ${result.image_height} px`;
  document.getElementById('statGaugeCount').textContent = result.gauge_count;
  document.getElementById('statProvider').textContent = result.provider.toUpperCase();

  // Show multi-gauge tabs if >1 gauge
  updateGaugeTabs(result.gauges);

  // Update details panel
  updateGaugeDetailsPanel();

  // Update JSON Viewer
  document.getElementById('jsonCode').textContent = JSON.stringify(result, null, 2);

  // Render canvas
  if (!preserveZoom) {
    fitImageToCanvas();
  }
  renderCanvas();

  if (isInstant) {
    showToast(`Instant demo loaded: ${result.gauge_count} gauge(s) ready!`, 'info');
  } else {
    showToast(`Successfully detected ${result.gauge_count} gauge(s)! (${result.execution_time_ms}ms)`, 'success');
  }
}

/* ============================================================
   CANVAS RENDERING ENGINE
   ============================================================ */
function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      state.currentImage = img;
      fitImageToCanvas();
      renderCanvas();
      resolve(img);
    };
    img.onerror = reject;
    img.src = url;
  });
}

function fitImageToCanvas() {
  if (!state.currentImage) return;

  const wrapperRect = canvasWrapper.getBoundingClientRect();
  const maxW = wrapperRect.width - 40;
  const maxH = wrapperRect.height - 40;

  const scale = Math.min(maxW / state.currentImage.width, maxH / state.currentImage.height, 1.8);
  state.zoom = Math.max(scale, 0.4);
  state.pan = { x: 0, y: 0 };
  updateZoomDisplay();
}

function updateZoomDisplay() {
  document.getElementById('zoomLevel').textContent = `${Math.round(state.zoom * 100)}%`;
}

function renderCanvas() {
  if (!state.currentImage) {
    canvasEmpty.style.display = 'block';
    canvas.style.display = 'none';
    return;
  }

  canvasEmpty.style.display = 'none';
  canvas.style.display = 'block';

  const img = state.currentImage;
  const dpr = window.devicePixelRatio || 1;

  // Set physical canvas size
  const displayWidth = Math.round(img.width * state.zoom);
  const displayHeight = Math.round(img.height * state.zoom);

  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;

  ctx.save();
  ctx.scale(dpr, dpr);

  // Clear background
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  // Draw base image
  ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

  // If no detection result yet, return
  if (!state.detectionResult || !state.detectionResult.gauges) {
    ctx.restore();
    return;
  }

  const gauges = state.detectionResult.gauges;

  // Render Overlay Layers
  gauges.forEach((gauge, gIdx) => {
    const isSelected = gIdx === state.selectedGaugeIndex;
    const isHovered = gIdx === state.hoveredGaugeIndex;
    const isDimmed = (state.hoveredGaugeIndex !== null && !isHovered) || 
                     (state.hoveredGaugeIndex === null && state.selectedGaugeIndex !== null && !isSelected && gauges.length > 1);

    ctx.save();
    if (isDimmed) {
      ctx.globalAlpha = 0.35;
    }

    // 1. Gauge Bounding Box & Dial Center
    if (state.layers.gauges) {
      drawGaugeBox(gauge, displayWidth, displayHeight, isSelected || isHovered, gIdx);
    }

    // 2. Scale Numbers
    if (state.layers.numbers && gauge.numbers) {
      drawScaleNumbers(gauge.numbers, displayWidth, displayHeight, isSelected || isHovered);
    }

    // 3. Needles & Angle Arcs
    if (gauge.needles) {
      gauge.needles.forEach(needle => {
        if (state.layers.angles && needle.angle_degrees !== null) {
          drawAngleArc(gauge.dial_center, needle, displayWidth, displayHeight);
        }
        if (state.layers.needles) {
          drawNeedle(needle, displayWidth, displayHeight, isSelected || isHovered);
        }
      });
    }

    // 4. Reading HUD Tag
    if (state.layers.hud && gauge.reading) {
      drawReadingHud(gauge, displayWidth, displayHeight, isSelected || isHovered);
    }

    ctx.restore();
  });

  ctx.restore();
}

/* Helper Drawing Functions */
function normToPix(x, y, w, h) {
  return {
    px: (x / 1000.0) * w,
    py: (y / 1000.0) * h
  };
}

function drawGaugeBox(gauge, w, h, isHighlighted, index) {
  const y1 = (gauge.bbox.ymin / 1000.0) * h;
  const x1 = (gauge.bbox.xmin / 1000.0) * w;
  const y2 = (gauge.bbox.ymax / 1000.0) * h;
  const x2 = (gauge.bbox.xmax / 1000.0) * w;
  const bw = x2 - x1;
  const bh = y2 - y1;

  // Box border
  ctx.strokeStyle = isHighlighted ? '#b9e5b4' : 'rgba(185, 229, 180, 0.65)';
  ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
  ctx.strokeRect(x1, y1, bw, bh);

  // Subtle glass background if highlighted
  if (isHighlighted) {
    ctx.fillStyle = 'rgba(185, 229, 180, 0.1)';
    ctx.fillRect(x1, y1, bw, bh);
  }

  // Label badge at top-left of box
  const labelText = `[#${index + 1}] ${gauge.label}`;
  ctx.font = '600 11px "Inter", sans-serif';
  const textWidth = ctx.measureText(labelText).width;

  ctx.fillStyle = isHighlighted ? '#b9e5b4' : 'rgba(19, 35, 31, 0.9)';
  ctx.fillRect(x1, y1 - 20, textWidth + 12, 20);

  ctx.fillStyle = isHighlighted ? '#173d29' : '#d5f0cd';
  ctx.fillText(labelText, x1 + 6, y1 - 6);

  // Dial Center Crosshair
  const c = normToPix(gauge.dial_center.x, gauge.dial_center.y, w, h);
  ctx.strokeStyle = isHighlighted ? '#d5f0cd' : 'rgba(244, 247, 245, 0.8)';
  ctx.lineWidth = 1.5;
  
  // Center concentric rings
  ctx.beginPath();
  ctx.arc(c.px, c.py, 4, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(c.px, c.py, 8, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(185, 229, 180, 0.5)';
  ctx.stroke();
}

function drawNeedle(needle, w, h, isHighlighted) {
  const p = normToPix(needle.pivot_point.x, needle.pivot_point.y, w, h);
  const t = normToPix(needle.tip_point.x, needle.tip_point.y, w, h);

  const isSecondary = needle.type === 'secondary_pointer';
  const color = isSecondary ? '#c7aa67' : '#ff9282';

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = isHighlighted ? 3 : 2;

  // Needle Vector Line
  ctx.beginPath();
  ctx.moveTo(p.px, p.py);
  ctx.lineTo(t.px, t.py);
  ctx.stroke();

  // Pivot dot
  ctx.beginPath();
  ctx.arc(p.px, p.py, 5, 0, Math.PI * 2);
  ctx.fill();

  // Tip glowing dot
  ctx.beginPath();
  ctx.arc(t.px, t.py, isHighlighted ? 6 : 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#f4f7f5';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.restore();
}

function drawScaleNumbers(numbers, w, h, isHighlighted) {
  ctx.save();
  ctx.font = '600 10px "JetBrains Mono", monospace';

  numbers.forEach(num => {
    const y1 = (num.bbox.ymin / 1000.0) * h;
    const x1 = (num.bbox.xmin / 1000.0) * w;
    const y2 = (num.bbox.ymax / 1000.0) * h;
    const x2 = (num.bbox.xmax / 1000.0) * w;

    // Small boundary box
    ctx.strokeStyle = 'rgba(140, 183, 132, 0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Number text tag
    ctx.fillStyle = '#8cb784';
    ctx.fillText(num.value, x1, Math.max(y1 - 3, 10));
  });

  ctx.restore();
}

function drawAngleArc(pivot, needle, w, h) {
  const p = normToPix(pivot.x, pivot.y, w, h);
  const t = normToPix(needle.tip_point.x, needle.tip_point.y, w, h);

  const radius = Math.sqrt((t.px - p.px) ** 2 + (t.py - p.py) ** 2) * 0.45;
  const angleRad = Math.atan2(t.py - p.py, t.px - p.px);

  ctx.save();
  ctx.strokeStyle = 'rgba(148, 168, 157, 0.5)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);

  ctx.beginPath();
  ctx.arc(p.px, p.py, radius, 0, angleRad, angleRad < 0);
  ctx.stroke();

  ctx.restore();
}

function drawReadingHud(gauge, w, h, isHighlighted) {
  const x1 = (gauge.bbox.xmin / 1000.0) * w;
  const y2 = (gauge.bbox.ymax / 1000.0) * h;

  const readingText = `${gauge.reading.formatted}`;
  ctx.font = '700 13px "JetBrains Mono", monospace';
  const textW = ctx.measureText(readingText).width;

  const badgeW = textW + 20;
  const badgeH = 26;
  const badgeX = x1;
  const badgeY = Math.min(y2 + 8, h - badgeH - 5);

  ctx.save();
  // Pill background
  ctx.fillStyle = 'rgba(16, 26, 23, 0.92)';
  ctx.strokeStyle = isHighlighted ? '#b9e5b4' : 'rgba(185, 229, 180, 0.5)';
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 6);
  ctx.fill();
  ctx.stroke();

  // Glow dot
  ctx.fillStyle = '#b9e5b4';
  ctx.beginPath();
  ctx.arc(badgeX + 10, badgeY + badgeH / 2, 4, 0, Math.PI * 2);
  ctx.fill();

  // Reading Text
  ctx.fillStyle = '#f4f7f5';
  ctx.fillText(readingText, badgeX + 20, badgeY + 17);

  ctx.restore();
}

/* ============================================================
   RIGHT PANEL DETAILS & ANALYTICS
   ============================================================ */
function updateGaugeTabs(gauges) {
  const tabsBar = document.getElementById('gaugeTabsBar');
  const tabsList = document.getElementById('gaugeTabsList');
  tabsList.innerHTML = '';

  if (!gauges || gauges.length <= 1) {
    tabsBar.style.display = 'none';
    return;
  }

  tabsBar.style.display = 'flex';

  gauges.forEach((gauge, idx) => {
    const pill = document.createElement('button');
    pill.className = `gauge-tab-pill ${idx === state.selectedGaugeIndex ? 'active' : ''}`;
    pill.textContent = `[#${idx + 1}] ${gauge.reading.formatted}`;
    pill.addEventListener('click', () => {
      state.selectedGaugeIndex = idx;
      document.querySelectorAll('.gauge-tab-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      updateGaugeDetailsPanel();
      renderCanvas();
    });
    tabsList.appendChild(pill);
  });
}

function updateGaugeDetailsPanel() {
  const noSelect = document.getElementById('noSelectionMessage');
  const details = document.getElementById('analysisDetails');

  if (!state.detectionResult || !state.detectionResult.gauges || state.detectionResult.gauges.length === 0) {
    noSelect.style.display = 'block';
    details.style.display = 'none';
    return;
  }

  noSelect.style.display = 'none';
  details.style.display = 'flex';
  details.style.flexDirection = 'column';

  const gauge = state.detectionResult.gauges[state.selectedGaugeIndex] || state.detectionResult.gauges[0];

  // Title & Type
  document.getElementById('gaugeTitle').textContent = gauge.label;
  document.getElementById('gaugeTypeBadge').textContent = gauge.gauge_type.toUpperCase();

  // Thumbnail Crop
  const thumbImg = document.getElementById('gaugeThumb');
  if (gauge.crop_base64) {
    thumbImg.src = `data:image/jpeg;base64,${gauge.crop_base64}`;
    document.getElementById('gaugeThumbWrapper').style.display = 'block';
  } else {
    document.getElementById('gaugeThumbWrapper').style.display = 'none';
  }

  // Hero Reading
  document.getElementById('readingValue').textContent = gauge.reading.value;
  document.getElementById('readingUnit').textContent = gauge.reading.unit || '';
  document.getElementById('confidenceBadge').textContent = `Confidence: ${Math.round(gauge.reading.confidence * 100)}%`;
  
  if (gauge.reading.min_range !== null && gauge.reading.max_range !== null) {
    document.getElementById('rangeBadge').textContent = `Scale: ${gauge.reading.min_range} - ${gauge.reading.max_range}`;
  } else {
    document.getElementById('rangeBadge').textContent = '';
  }

  // Update SVG Analog Meter Visualizer
  updateAnalogVisualizer(gauge.reading);

  // Needles List (Vertical stack, clear labels)
  const needlesList = document.getElementById('needlesList');
  needlesList.innerHTML = '';
  gauge.needles.forEach((n, i) => {
    const item = document.createElement('div');
    item.className = 'needle-item';
    const angleText = n.angle_degrees !== null ? `${n.angle_degrees}°` : 'Interpolated';
    const typeLabel = n.type ? n.type.replace('_', ' ') : 'pointer';
    item.innerHTML = `
      <div class="needle-row">
        <span class="needle-key">Pointer #${i+1} (${typeLabel}):</span>
        <span class="needle-val" style="color: ${n.color === 'bronze' ? '#f59e0b' : '#ef4444'}">${n.color.toUpperCase()}</span>
      </div>
      <div class="needle-row">
        <span class="needle-key">Deflection Angle:</span>
        <span class="needle-val">${angleText}</span>
      </div>
      <div class="needle-row">
        <span class="needle-key">Tip Coordinates:</span>
        <span class="needle-val">(${Math.round(n.tip_point.x)}, ${Math.round(n.tip_point.y)})</span>
      </div>
    `;
    needlesList.appendChild(item);
  });

  // Scale Numbers Chips
  const numChips = document.getElementById('numbersChips');
  numChips.innerHTML = '';
  if (gauge.numbers && gauge.numbers.length > 0) {
    gauge.numbers.forEach(num => {
      const chip = document.createElement('span');
      chip.className = 'number-chip';
      chip.textContent = num.value;
      numChips.appendChild(chip);
    });
  } else {
    numChips.innerHTML = '<span class="text-muted" style="font-size:0.75rem">No distinct scale numbers detected</span>';
  }

  // Reasoning
  document.getElementById('readingReasoning').textContent = gauge.reading.reasoning || 'No detailed reasoning available.';
}

function updateAnalogVisualizer(reading) {
  if (!reading) return;
  const container = document.getElementById('radialDialSimulator');
  if (container && (container.style.display === 'none' || window.getComputedStyle(container).display === 'none')) {
    return;
  }
  const min = reading.min_range !== null ? reading.min_range : 0;
  const max = reading.max_range !== null ? reading.max_range : (reading.value > 100 ? 600 : 100);
  const val = Math.max(min, Math.min(reading.value, max));

  document.getElementById('svgMinText').textContent = min;
  document.getElementById('svgMaxText').textContent = max;

  // Fraction from 0 to 1
  const span = max - min || 1;
  const ratio = (val - min) / span;

  // Arc length is ~235.6px (semi-circle radius 75)
  const fullArc = 235.6;
  const offset = fullArc * (1 - ratio);
  const arcEl = document.getElementById('svgProgressArc');
  if (arcEl) {
    arcEl.style.transition = 'stroke-dashoffset 0.8s ease-out';
    arcEl.style.strokeDashoffset = offset;
  }

  // Needle angle: from 180° (left, ratio=0) to 0° (right, ratio=1)
  const angleDeg = 180 - (ratio * 180);
  const angleRad = (angleDeg * Math.PI) / 180;
  const needleLength = 65;
  const cx = 100, cy = 100;
  const tx = cx - needleLength * Math.cos(angleRad);
  const ty = cy - needleLength * Math.sin(angleRad);

  const needleEl = document.getElementById('svgNeedleLine');
  if (needleEl) {
    needleEl.style.transition = 'all 0.8s ease-out';
    needleEl.setAttribute('x2', tx);
    needleEl.setAttribute('y2', ty);
  }
}

/* ============================================================
   EVENT LISTENERS & INTERACTION
   ============================================================ */
function setupEventListeners() {
  // Layer Toggles
  ['Gauges', 'Needles', 'Numbers', 'HUD', 'Angles'].forEach(layer => {
    const chk = document.getElementById(`toggle${layer}`);
    if (chk) {
      chk.addEventListener('change', (e) => {
        state.layers[layer.toLowerCase()] = e.target.checked;
        renderCanvas();
      });
    }
  });

  // Zoom buttons
  document.getElementById('btnZoomIn').addEventListener('click', () => {
    state.zoom = Math.min(state.zoom * 1.25, 4.0);
    updateZoomDisplay();
    renderCanvas();
  });
  document.getElementById('btnZoomOut').addEventListener('click', () => {
    state.zoom = Math.max(state.zoom / 1.25, 0.3);
    updateZoomDisplay();
    renderCanvas();
  });
  document.getElementById('btnZoomFit').addEventListener('click', () => {
    fitImageToCanvas();
    renderCanvas();
  });

  // Re-Analyze Buttons (Top toolbar & Sidebar)
  const btnReanalyze = document.getElementById('btnReanalyze');
  if (btnReanalyze) {
    btnReanalyze.addEventListener('click', () => reanalyzeCurrentImage());
  }

  const btnReanalyzeSide = document.getElementById('btnReanalyzeSide');
  if (btnReanalyzeSide) {
    btnReanalyzeSide.addEventListener('click', () => reanalyzeCurrentImage());
  }

  // Prompt Configuration & Template Selection
  const gaugeTypeSelect = document.getElementById('gaugeTypeSelect');
  if (gaugeTypeSelect) {
    gaugeTypeSelect.addEventListener('change', (e) => {
      applySelectedTemplate(e.target.value, true);
    });
  }

  const promptTextarea = document.getElementById('promptEditorTextarea');
  if (promptTextarea) {
    promptTextarea.addEventListener('input', () => {
      updatePromptCharCount();
      state.isPromptEdited = true;
      setPromptStatusBadge(true);
    });
  }

  const btnResetPrompt = document.getElementById('btnResetPrompt');
  if (btnResetPrompt) {
    btnResetPrompt.addEventListener('click', () => {
      applySelectedTemplate(state.selectedTemplateId, true);
      showToast('Prompt reset to template standard', 'info');
    });
  }

  const btnToggleExpand = document.getElementById('btnTogglePromptExpand');
  const promptSection = document.getElementById('promptConfigSection');
  if (btnToggleExpand && promptSection) {
    btnToggleExpand.addEventListener('click', () => {
      promptSection.classList.toggle('is-expanded');
    });
  }

  // Dropzone upload
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnTrigger = document.getElementById('btnTriggerUpload');

  btnTrigger.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      runUploadedImageDetection(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      runUploadedImageDetection(e.dataTransfer.files[0]);
    }
  });

  // Tabs on Right Panel
  document.querySelectorAll('.panel-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.panel-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  // Copy JSON
  document.getElementById('btnCopyJson').addEventListener('click', () => {
    const code = document.getElementById('jsonCode').textContent;
    navigator.clipboard.writeText(code).then(() => {
      showToast('JSON copied to clipboard!', 'success');
    });
  });

  // Download Annotated Image
  document.getElementById('btnDownloadAnnotated').addEventListener('click', async () => {
    if (!state.currentImage) return;
    try {
      showToast('Rendering annotated image...', 'info');
      // Create a temporary link from current canvas
      const link = document.createElement('a');
      link.download = `gauge_detection_${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
      showToast('Image downloaded successfully!', 'success');
    } catch (err) {
      showToast(`Download error: ${err.message}`, 'error');
    }
  });

  // Settings Modal
  setupSettingsModal();

  // Camera Modal
  setupCameraModal();

  // Drag-to-Resize Panels
  setupPanelResizer();
  setupLeftPanelResizer();
}

/* ============================================================
   PANEL RESIZING ENGINE (DRAG TO STRETCH / SHRINK)
   ============================================================ */
function setupPanelResizer() {
  const resizer = document.getElementById('panelResizer');
  if (!resizer) return;

  // Restore saved width from localStorage if exists
  const savedWidth = localStorage.getItem('gaugevision_right_width');
  if (savedWidth) {
    const w = parseInt(savedWidth, 10);
    if (!isNaN(w) && w >= 260 && w <= window.innerWidth - 350) {
      document.documentElement.style.setProperty('--right-panel-width', `${w}px`);
    }
  }

  let isDragging = false;
  let startX = 0;
  let startWidth = 0;

  const onMouseDown = (e) => {
    isDragging = true;
    startX = e.clientX;
    const currentWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--right-panel-width'), 10) || 440;
    startWidth = currentWidth;
    document.body.classList.add('is-resizing');

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    e.preventDefault();
  };

  const onMouseMove = (e) => {
    if (!isDragging) return;
    const delta = startX - e.clientX;
    let newWidth = startWidth + delta;

    const minWidth = 260;
    const maxWidth = Math.max(minWidth, Math.min(window.innerWidth - 420, 950));
    newWidth = Math.max(minWidth, Math.min(newWidth, maxWidth));

    document.documentElement.style.setProperty('--right-panel-width', `${newWidth}px`);
  };

  const onMouseUp = () => {
    if (!isDragging) return;
    isDragging = false;
    document.body.classList.remove('is-resizing');
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);

    const currentWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--right-panel-width'), 10);
    if (currentWidth) {
      localStorage.setItem('gaugevision_right_width', currentWidth);
    }

    if (state.currentImage) {
      renderCanvas();
    }
  };

  // Double click to reset to default 440px
  resizer.addEventListener('dblclick', () => {
    const defaultWidth = 440;
    document.documentElement.style.setProperty('--right-panel-width', `${defaultWidth}px`);
    localStorage.setItem('gaugevision_right_width', defaultWidth);
    if (state.currentImage) {
      renderCanvas();
    }
    showToast('Right panel width reset to default (440px)', 'info');
  });

  resizer.addEventListener('mousedown', onMouseDown);

  // Responsive window resize event
  window.addEventListener('resize', () => {
    const currentWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--right-panel-width'), 10) || 440;
    const maxWidth = Math.max(260, window.innerWidth - 420);
    if (currentWidth > maxWidth) {
      document.documentElement.style.setProperty('--right-panel-width', `${maxWidth}px`);
    }
    if (state.currentImage) {
      renderCanvas();
    }
  });
}

function setupLeftPanelResizer() {
  const resizer = document.getElementById('leftPanelResizer');
  if (!resizer) return;

  // Restore saved width from localStorage if exists
  const savedWidth = localStorage.getItem('gaugevision_left_width');
  if (savedWidth) {
    const w = parseInt(savedWidth, 10);
    if (!isNaN(w) && w >= 240 && w <= 750) {
      document.documentElement.style.setProperty('--left-panel-width', `${w}px`);
    }
  }

  let isDragging = false;
  let startX = 0;
  let startWidth = 0;

  const onMouseDown = (e) => {
    isDragging = true;
    startX = e.clientX;
    const currentWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--left-panel-width'), 10) || 290;
    startWidth = currentWidth;
    document.body.classList.add('is-resizing');

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    e.preventDefault();
  };

  const onMouseMove = (e) => {
    if (!isDragging) return;
    const delta = e.clientX - startX;
    let newWidth = startWidth + delta;

    const minWidth = 240;
    const maxWidth = Math.max(minWidth, Math.min(window.innerWidth - 450, 750));
    newWidth = Math.max(minWidth, Math.min(newWidth, maxWidth));

    document.documentElement.style.setProperty('--left-panel-width', `${newWidth}px`);
  };

  const onMouseUp = () => {
    if (!isDragging) return;
    isDragging = false;
    document.body.classList.remove('is-resizing');
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);

    const currentWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--left-panel-width'), 10);
    if (currentWidth) {
      localStorage.setItem('gaugevision_left_width', currentWidth);
    }

    if (state.currentImage) {
      renderCanvas();
    }
  };

  // Double click to reset to default 290px
  resizer.addEventListener('dblclick', () => {
    const defaultWidth = 290;
    document.documentElement.style.setProperty('--left-panel-width', `${defaultWidth}px`);
    localStorage.setItem('gaugevision_left_width', defaultWidth);
    if (state.currentImage) {
      renderCanvas();
    }
    showToast('Left sidebar width reset to default (290px)', 'info');
  });

  resizer.addEventListener('mousedown', onMouseDown);

  // Responsive window resize event
  window.addEventListener('resize', () => {
    const currentWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--left-panel-width'), 10) || 290;
    const maxWidth = Math.max(240, window.innerWidth - 450);
    if (currentWidth > maxWidth) {
      document.documentElement.style.setProperty('--left-panel-width', `${maxWidth}px`);
    }
    if (state.currentImage) {
      renderCanvas();
    }
  });
}

function setupCanvasInteractions() {
  // 1. Mouse wheel zoom on image / canvasWrapper
  canvasWrapper.addEventListener('wheel', (e) => {
    if (!state.currentImage) return;

    // Prevent default document scrolling
    e.preventDefault();

    // Wheel delta: deltaY < 0 is scroll up (zoom in), deltaY > 0 is scroll down (zoom out)
    const zoomMultiplier = e.deltaY < 0 ? 1.15 : (1 / 1.15);
    const newZoom = Math.max(0.2, Math.min(state.zoom * zoomMultiplier, 5.0));

    if (Math.abs(newZoom - state.zoom) > 0.001) {
      state.zoom = newZoom;
      updateZoomDisplay();
      renderCanvas();
    }
  }, { passive: false });

  // 2. Click and drag to pan viewport when zoomed
  let isPanning = false;
  let panStartX = 0;
  let panStartY = 0;
  let scrollStartX = 0;
  let scrollStartY = 0;

  canvasWrapper.addEventListener('mousedown', (e) => {
    if (!state.currentImage) return;
    // Allow panning with middle click (button 1) or left click (button 0 if not clicking a gauge)
    if (e.button === 1 || (e.button === 0 && state.hoveredGaugeIndex === null)) {
      isPanning = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      scrollStartX = canvasWrapper.scrollLeft;
      scrollStartY = canvasWrapper.scrollTop;
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    const dx = e.clientX - panStartX;
    const dy = e.clientY - panStartY;
    canvasWrapper.scrollLeft = scrollStartX - dx;
    canvasWrapper.scrollTop = scrollStartY - dy;
  });

  window.addEventListener('mouseup', () => {
    isPanning = false;
  });

  // 3. Mouse hover over canvas to detect hovered gauge
  canvas.addEventListener('mousemove', (e) => {
    if (isPanning) return;
    if (!state.detectionResult || !state.detectionResult.gauges) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const normX = (x / rect.width) * 1000;
    const normY = (y / rect.height) * 1000;

    let foundIdx = null;
    state.detectionResult.gauges.forEach((g, idx) => {
      if (normX >= g.bbox.xmin && normX <= g.bbox.xmax &&
          normY >= g.bbox.ymin && normY <= g.bbox.ymax) {
        foundIdx = idx;
      }
    });

    if (foundIdx !== state.hoveredGaugeIndex) {
      state.hoveredGaugeIndex = foundIdx;
      renderCanvas();
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (state.hoveredGaugeIndex !== null) {
      state.hoveredGaugeIndex = null;
      renderCanvas();
    }
  });

  canvas.addEventListener('click', () => {
    if (isPanning) return;
    if (state.hoveredGaugeIndex !== null) {
      state.selectedGaugeIndex = state.hoveredGaugeIndex;
      updateGaugeTabs(state.detectionResult.gauges);
      updateGaugeDetailsPanel();
      renderCanvas();
    }
  });
}

/* ============================================================
   SETTINGS MODAL & PROVIDER SWITCHING
   ============================================================ */
/* ============================================================
   SETTINGS MODAL & DYNAMIC MODEL CATALOG
   ============================================================ */
const modelCache = {
  hybrid: null,
  gemini: null,
  openai: null,
  litellm: null
};

function setupSettingsModal() {
  const modal = document.getElementById('settingsModal');
  const btnOpen = document.getElementById('btnOpenSettings');
  const btnClose = document.getElementById('btnCloseSettings');
  const btnCancel = document.getElementById('btnCancelSettings');
  const btnSave = document.getElementById('btnSaveSettings');
  const badge = document.getElementById('providerBadge');

  // Helper for model targets
  const getProviderTarget = (provider) => {
    if (provider === 'hybrid') {
      return {
        provider: 'hybrid',
        select: document.getElementById('selectHybridModel'),
        custom: document.getElementById('inputHybridModelCustom'),
        status: document.getElementById('statusHybrid'),
        btn: document.getElementById('btnRefreshHybrid'),
        getPayload: () => ({
          provider: 'hybrid',
          base_url: (document.getElementById('inputHybridBaseUrl') ? document.getElementById('inputHybridBaseUrl').value.trim() : '') || 'http://172.18.60.121:4000',
          api_key: document.getElementById('inputHybridKey') ? document.getElementById('inputHybridKey').value.trim() : ''
        }),
        getCurrentModel: () => state.config?.hybrid?.litellm_model || state.config?.litellm?.model || 'Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121'
      };
    } else if (provider === 'gemini') {
      return {
        provider: 'gemini',
        select: document.getElementById('selectGeminiModel'),
        custom: document.getElementById('inputGeminiModelCustom'),
        status: document.getElementById('statusGemini'),
        btn: document.getElementById('btnRefreshGemini'),
        getPayload: () => ({
          provider: 'gemini',
          api_key: document.getElementById('inputGeminiKey').value.trim()
        }),
        getCurrentModel: () => state.config?.gemini?.model || 'gemini-flash-lite-latest'
      };
    } else if (provider === 'openai') {
      return {
        provider: 'openai',
        select: document.getElementById('selectOpenaiModel'),
        custom: document.getElementById('inputOpenaiModelCustom'),
        status: document.getElementById('statusOpenai'),
        btn: document.getElementById('btnRefreshOpenai'),
        getPayload: () => ({
          provider: 'openai',
          api_key: document.getElementById('inputOpenaiKey').value.trim(),
          base_url: document.getElementById('inputOpenaiBaseUrl').value.trim()
        }),
        getCurrentModel: () => state.config?.openai?.model || 'gpt-4o-mini'
      };
    } else if (provider === 'litellm') {
      return {
        provider: 'litellm',
        select: document.getElementById('selectLitellmModel'),
        custom: document.getElementById('inputLitellmModelCustom'),
        status: document.getElementById('statusLitellm'),
        btn: document.getElementById('btnRefreshLitellm'),
        getPayload: () => ({
          provider: 'litellm',
          base_url: document.getElementById('inputLitellmBaseUrl').value.trim(),
          api_key: document.getElementById('inputLitellmKey').value.trim()
        }),
        getCurrentModel: () => state.config?.litellm?.model || 'Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121'
      };
    }
    return null;
  };

  const populateSelectOptions = (target, models, currentModel) => {
    const select = target.select;
    const customInput = target.custom;
    const prevVal = select.value && select.value !== '__custom__' ? select.value : currentModel;

    select.innerHTML = '';

    const hasCurrent = models.includes(prevVal);
    if (prevVal && !hasCurrent && prevVal !== '__custom__') {
      const opt = document.createElement('option');
      opt.value = prevVal;
      opt.textContent = `${prevVal} (Active)`;
      select.appendChild(opt);
    }

    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    });

    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = '+ Enter Custom Model Name...';
    select.appendChild(customOpt);

    if (prevVal) {
      select.value = prevVal;
      if (customInput) customInput.style.display = 'none';
    } else if (models.length > 0) {
      select.value = models[0];
      if (customInput) customInput.style.display = 'none';
    }
  };

  const fetchAndPopulateModels = async (provider, forceRefresh = false) => {
    const target = getProviderTarget(provider);
    if (!target || !target.select) return;

    const payload = target.getPayload();
    const cacheKey = `${provider}_${payload.api_key || ''}_${payload.base_url || ''}`;

    if (!forceRefresh && modelCache[cacheKey]) {
      populateSelectOptions(target, modelCache[cacheKey], target.getCurrentModel());
      return;
    }

    if (target.status) {
      target.status.textContent = 'Loading models...';
      target.status.className = 'indicator-text loading';
    }
    if (target.btn) {
      target.btn.classList.add('loading');
    }

    try {
      const res = await fetch('/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      modelCache[cacheKey] = data.models;

      populateSelectOptions(target, data.models, target.getCurrentModel());

      if (target.status) {
        if (data.error) {
          target.status.textContent = 'Fallback catalog';
          target.status.className = 'indicator-text error';
          target.status.title = data.error;
        } else {
          target.status.textContent = `✓ ${data.models.length} available`;
          target.status.className = 'indicator-text success';
          target.status.title = `Found ${data.models.length} models from ${provider.toUpperCase()}`;
        }
      }
    } catch (err) {
      console.warn(`Failed to fetch models for ${provider}:`, err);
      if (target.status) {
        target.status.textContent = 'Offline list';
        target.status.className = 'indicator-text error';
        target.status.title = err.message;
      }
    } finally {
      if (target.btn) {
        target.btn.classList.remove('loading');
      }
    }
  };

  // Setup custom option toggling
  ['hybrid', 'gemini', 'openai', 'litellm'].forEach(p => {
    const target = getProviderTarget(p);
    if (target && target.select && target.custom) {
      target.select.addEventListener('change', (e) => {
        if (e.target.value === '__custom__') {
          target.custom.style.display = 'block';
          target.custom.focus();
        } else {
          target.custom.style.display = 'none';
        }
      });
    }
  });

  // Setup auto-fetch on typing key or base URL
  let debounceTimers = {};
  const bindAutoFetch = (inputEl, provider) => {
    if (!inputEl) return;
    inputEl.addEventListener('input', () => {
      clearTimeout(debounceTimers[provider]);
      debounceTimers[provider] = setTimeout(() => {
        fetchAndPopulateModels(provider, true);
      }, 600);
    });
    inputEl.addEventListener('change', () => {
      fetchAndPopulateModels(provider, true);
    });
  };

  bindAutoFetch(document.getElementById('inputHybridBaseUrl'), 'hybrid');
  bindAutoFetch(document.getElementById('inputHybridKey'), 'hybrid');
  bindAutoFetch(document.getElementById('inputGeminiKey'), 'gemini');
  bindAutoFetch(document.getElementById('inputOpenaiKey'), 'openai');
  bindAutoFetch(document.getElementById('inputOpenaiBaseUrl'), 'openai');
  bindAutoFetch(document.getElementById('inputLitellmBaseUrl'), 'litellm');
  bindAutoFetch(document.getElementById('inputLitellmKey'), 'litellm');

  // Manual refresh buttons
  const btnRefH = document.getElementById('btnRefreshHybrid');
  if (btnRefH) btnRefH.addEventListener('click', () => fetchAndPopulateModels('hybrid', true));
  const btnRefG = document.getElementById('btnRefreshGemini');
  if (btnRefG) btnRefG.addEventListener('click', () => fetchAndPopulateModels('gemini', true));
  const btnRefO = document.getElementById('btnRefreshOpenai');
  if (btnRefO) btnRefO.addEventListener('click', () => fetchAndPopulateModels('openai', true));
  const btnRefL = document.getElementById('btnRefreshLitellm');
  if (btnRefL) btnRefL.addEventListener('click', () => fetchAndPopulateModels('litellm', true));

  const openModal = () => {
    if (state.config) {
      let activeP = state.config.provider;
      if (!activeP || activeP === 'mock') activeP = 'gemini';

      const radio = document.querySelector(`input[name="aiProvider"][value="${activeP}"]`) || document.querySelector('input[name="aiProvider"]');
      if (radio) {
        radio.checked = true;
        toggleProviderSettingsVisibility(radio.value);
      }

      if (state.config.openai) {
        document.getElementById('inputOpenaiBaseUrl').value = state.config.openai.base_url || 'https://api.openai.com/v1';
      }
      if (state.config.litellm) {
        document.getElementById('inputLitellmBaseUrl').value = state.config.litellm.base_url || 'http://172.18.60.121:4000';
      }
      const hybridBase = document.getElementById('inputHybridBaseUrl');
      if (hybridBase) {
        hybridBase.value = state.config.litellm?.base_url || 'http://172.18.60.121:4000';
      }

      // Pre-fetch models for the active provider and others
      fetchAndPopulateModels(radio ? radio.value : 'gemini');
    }
    modal.style.display = 'flex';
  };

  const closeModal = () => modal.style.display = 'none';

  btnOpen.addEventListener('click', openModal);
  badge.addEventListener('click', openModal);
  btnClose.addEventListener('click', closeModal);
  btnCancel.addEventListener('click', closeModal);

  document.querySelectorAll('input[name="aiProvider"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      toggleProviderSettingsVisibility(e.target.value);
      fetchAndPopulateModels(e.target.value);
    });
  });

  btnSave.addEventListener('click', async () => {
    const radio = document.querySelector('input[name="aiProvider"]:checked');
    const selectedProvider = radio ? radio.value : 'gemini';
    const payload = { provider: selectedProvider };

    if (selectedProvider === 'hybrid') {
      const key = document.getElementById('inputHybridKey')?.value.trim();
      if (key) payload.litellm_api_key = key;
      const baseUrl = document.getElementById('inputHybridBaseUrl')?.value.trim();
      if (baseUrl) payload.litellm_base_url = baseUrl;

      const sel = document.getElementById('selectHybridModel');
      const custom = document.getElementById('inputHybridModelCustom');
      payload.litellm_model = (sel.value === '__custom__' ? custom.value.trim() : sel.value) || 'Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121';
    } else if (selectedProvider === 'gemini') {
      const key = document.getElementById('inputGeminiKey').value.trim();
      if (key) payload.gemini_api_key = key;

      const sel = document.getElementById('selectGeminiModel');
      const custom = document.getElementById('inputGeminiModelCustom');
      payload.gemini_model = (sel.value === '__custom__' ? custom.value.trim() : sel.value) || 'gemini-flash-lite-latest';
    } else if (selectedProvider === 'openai') {
      const key = document.getElementById('inputOpenaiKey').value.trim();
      if (key) payload.openai_api_key = key;
      payload.openai_base_url = document.getElementById('inputOpenaiBaseUrl').value.trim();

      const sel = document.getElementById('selectOpenaiModel');
      const custom = document.getElementById('inputOpenaiModelCustom');
      payload.openai_model = (sel.value === '__custom__' ? custom.value.trim() : sel.value) || 'gpt-4o-mini';
    } else if (selectedProvider === 'litellm') {
      const key = document.getElementById('inputLitellmKey').value.trim();
      if (key) payload.litellm_api_key = key;
      payload.litellm_base_url = document.getElementById('inputLitellmBaseUrl').value.trim();

      const sel = document.getElementById('selectLitellmModel');
      const custom = document.getElementById('inputLitellmModelCustom');
      payload.litellm_model = (sel.value === '__custom__' ? custom.value.trim() : sel.value) || 'Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD-A30.0.121';
    }

    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Failed to save configuration');
      await loadConfig();
      closeModal();
      const currentModelName = payload[`${selectedProvider}_model`] || payload.litellm_model || '';
      showToast(`AI Provider updated to: ${selectedProvider.toUpperCase()} (${currentModelName}). Click 'Re-Analyze' to run on current image!`, 'success');
    } catch (err) {
      showToast(`Configuration error: ${err.message}`, 'error');
    }
  });
}

function toggleProviderSettingsVisibility(provider) {
  const gHybrid = document.getElementById('groupHybrid');
  if (gHybrid) gHybrid.style.display = provider === 'hybrid' ? 'block' : 'none';
  document.getElementById('groupGemini').style.display = provider === 'gemini' ? 'block' : 'none';
  document.getElementById('groupOpenai').style.display = provider === 'openai' ? 'block' : 'none';
  document.getElementById('groupLitellm').style.display = provider === 'litellm' ? 'block' : 'none';
}

/* ============================================================
   CAMERA / WEBCAM INTEGRATION
   ============================================================ */
function setupCameraModal() {
  const modal = document.getElementById('cameraModal');
  const btnOpen = document.getElementById('btnOpenCamera');
  const btnClose = document.getElementById('btnCloseCamera');
  const btnStop = document.getElementById('btnStopCamera');
  const btnCapture = document.getElementById('btnCapturePhoto');
  const video = document.getElementById('cameraVideo');

  btnOpen.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      state.cameraStream = stream;
      video.srcObject = stream;
      modal.style.display = 'flex';
    } catch (err) {
      showToast(`Cannot access webcam: ${err.message}`, 'error');
    }
  });

  const stopCamera = () => {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(t => t.stop());
      state.cameraStream = null;
    }
    modal.style.display = 'none';
  };

  btnClose.addEventListener('click', stopCamera);
  btnStop.addEventListener('click', stopCamera);

  btnCapture.addEventListener('click', async () => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth || 640;
    tempCanvas.height = video.videoHeight || 480;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    const b64 = tempCanvas.toDataURL('image/jpeg', 0.9);
    stopCamera();

    showLoader('PROCESSING WEBCAM CAPTURE...');

    try {
      const img = new Image();
      img.onload = async () => {
        state.currentImage = img;
        fitImageToCanvas();
        renderCanvas();

          const res = await fetch('/api/detect/json', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_base64: b64,
              filename: `webcam_${Date.now()}.jpg`,
              custom_prompt: getActivePrompt(),
              gauge_type: getActiveGaugeType()
            })
          });

        if (!res.ok) throw new Error(`Detection API error ${res.status}`);
        const data = await res.json();
        handleDetectionSuccess(data);
      };
      img.src = b64;
    } catch (err) {
      showToast(`Webcam processing error: ${err.message}`, 'error');
    } finally {
      hideLoader();
    }
  });
}

/* ============================================================
   UI HELPERS & TOASTS
   ============================================================ */
function showLoader(text, subtext) {
  loaderText.textContent = text || 'AI DETECTING GAUGES & PREDICTING READINGS...';
  const subEl = document.querySelector('.loader-sub');
  if (subEl) {
    subEl.textContent = subtext || 'Locating pointers, scale markings, and angles...';
  }
  canvasLoader.style.display = 'flex';
}

function hideLoader() {
  canvasLoader.style.display = 'none';
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
