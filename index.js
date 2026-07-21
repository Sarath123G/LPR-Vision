// API CONFIGURATION
const HOST = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
    ? `${window.location.protocol}//${window.location.host}` 
    : 'http://127.0.0.1:8080';

const API_URLs = {
    checkHealth: `${HOST}/api/health`,
    processImage: `${HOST}/api/process-image`,
    processFrame: `${HOST}/api/process-frame`,
    uploadVideo: `${HOST}/api/upload-video`,
    whitelist: `${HOST}/api/whitelist`,
    history: `${HOST}/api/history`,
    clearHistory: `${HOST}/api/clear-history`
};

// GLOBAL STATES
let activeMode = 'upload'; // 'upload' or 'webcam'
let webcamStream = null;
let webcamInterval = null;
let isWebcamProcessing = false;
let currentWhitelists = [];
let detectionHistory = [];
let activeFilter = 'all';
let searchKeyword = '';
let processedStepsData = {}; // Store base64 data for step details modal
let lastSpeechPlate = "";
let lastSpeechTime = 0;

// DOM SELECTION
const elements = {
    // Status
    statusBackendText: document.getElementById('status-backend-text'),
    statusBackendDot: document.querySelector('#status-backend .status-dot'),
    
    // Sidebar Sliders
    engineSelect: document.getElementById('engine-select'),
    blurKernel: document.getElementById('blur-kernel'),
    sobelKernel: document.getElementById('sobel-kernel'),
    morphW: document.getElementById('morph-w'),
    morphH: document.getElementById('morph-h'),
    areaMin: document.getElementById('area-min'),
    areaMax: document.getElementById('area-max'),
    ratioMin: document.getElementById('ratio-min'),
    ratioMax: document.getElementById('ratio-max'),
    btnResetParams: document.getElementById('btn-reset-params'),
    
    // Slider values labels
    valBlurKernel: document.getElementById('val-blur-kernel'),
    valSobelKernel: document.getElementById('val-sobel-kernel'),
    valMorphKernel: document.getElementById('val-morph-kernel'),
    valAreaBounds: document.getElementById('val-area-bounds'),
    valRatioBounds: document.getElementById('val-ratio-bounds'),
    
    // Whitelist
    whitelistForm: document.getElementById('whitelist-form'),
    whitelistInput: document.getElementById('whitelist-input'),
    whitelistList: document.getElementById('whitelist-list'),
    
    // Tabs
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    navBtnPipeline: document.getElementById('nav-btn-pipeline'),
    
    // Detector Media
    modeBtnUpload: document.getElementById('mode-btn-upload'),
    modeBtnWebcam: document.getElementById('mode-btn-webcam'),
    modeBtnSample: document.getElementById('mode-btn-sample'),
    uploadZone: document.getElementById('upload-zone'),
    mediaInput: document.getElementById('media-input'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    imageContainer: document.getElementById('image-container'),
    imagePreview: document.getElementById('image-preview'),
    imageCanvas: document.getElementById('image-canvas'),
    videoContainer: document.getElementById('video-container'),
    videoPreview: document.getElementById('video-preview'),
    webcamContainer: document.getElementById('webcam-container'),
    webcamFeed: document.getElementById('webcam-feed'),
    webcamCanvas: document.getElementById('webcam-canvas'),
    webcamControlsPanel: document.getElementById('webcam-controls-panel'),
    btnStopWebcam: document.getElementById('btn-stop-webcam'),
    
    // HUD
    hudPlateText: document.getElementById('hud-plate-text'),
    hudVerdict: document.getElementById('hud-verdict'),
    hudConfidence: document.getElementById('hud-confidence'),
    hudCoords: document.getElementById('hud-coords'),
    hudPlateCrop: document.getElementById('hud-plate-crop'),
    chkTts: document.getElementById('chk-tts'),
    chkAlarm: document.getElementById('chk-alarm'),
    
    // Pipeline Step Images
    step1Img: document.getElementById('step-1-img'),
    step2Img: document.getElementById('step-2-img'),
    step3Img: document.getElementById('step-3-img'),
    step4Img: document.getElementById('step-4-img'),
    step5Img: document.getElementById('step-5-img'),
    step6Img: document.getElementById('step-6-img'),
    step7Img: document.getElementById('step-7-img'),
    step8Img: document.getElementById('step-8-img'),
    charSegmentsPanel: document.getElementById('char-segments-panel'),
    charsListRow: document.getElementById('chars-list-row'),
    pipelineCards: document.querySelectorAll('.pipeline-card'),
    
    // Audit History
    btnExportCsv: document.getElementById('btn-export-csv'),
    btnClearHistory: document.getElementById('btn-clear-history'),
    filterSearchInput: document.getElementById('filter-search-input'),
    filterButtons: document.querySelectorAll('.filter-btn'),
    auditTableBody: document.getElementById('audit-table-body'),
    
    // Modal
    stepModal: document.getElementById('step-modal'),
    modalOverlay: document.getElementById('modal-overlay'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    modalStepTitle: document.getElementById('modal-step-title'),
    modalStepImg: document.getElementById('modal-step-img'),
    modalStepMathDesc: document.getElementById('modal-step-math-desc'),
    modalStepCode: document.getElementById('modal-step-code')
};

// ALGORITHM STEP DETAILS FOR MODAL
const PIPELINE_STEP_DETAILS = {
    1: {
        title: "Step 1: Original Frame",
        desc: "The input frame loaded by the system. This serves as the raw RGB input for the processing pipeline.",
        code: "img = cv2.imread(image_path)\n# or\nret, img = cap.read()"
    },
    2: {
        title: "Step 2: Grayscale & Gaussian Blur",
        desc: "First, the frame is converted to grayscale to reduce dimensionality. Next, a Gaussian filter is applied to blur the image, smoothing out high-frequency speckle noise and texture details that could disrupt edge detection.",
        code: "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nblurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)"
    },
    3: {
        title: "Step 3: Vertical Edges (Sobel X)",
        desc: "A horizontal Sobel filter (derivative in X direction) is run to capture strong vertical edge transitions. License plates usually have high horizontal frequency and vertical edges (due to the boundary and vertical characters), which makes Sobel X ideal for highlighting them.",
        code: "sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=sobel_kernel)"
    },
    4: {
        title: "Step 4: Otsu Binarization",
        desc: "Otsu's thresholding automatically computes an optimal threshold value to convert the vertical edge image into a binary (black and white) mask. It works by minimizing the intra-class variance of the black and white pixel intensities.",
        code: "_, threshold_img = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
    },
    5: {
        title: "Step 5: Morphological Closing",
        desc: "A closing operation (dilation followed by erosion) using a horizontal rectangular structuring element (e.g., 22x3) is performed. This closes the gaps between the vertical characters on the plate, clustering them together to form a solid white rectangular block on a black background.",
        code: "element = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, morph_h))\nmorph_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, element)"
    },
    6: {
        title: "Step 6: Plate Contours candidates",
        desc: "The system finds all external boundaries (contours) of white objects in the morphologically closed image. These contours are then validated against geometric constraints, specifically area limits and width-to-height aspect ratios typical of standard rectangular license plates.",
        code: "contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)\n# Draw contours:\ncv2.drawContours(img_copy, contours, -1, (0, 153, 255), 2)"
    },
    7: {
        title: "Step 7: Cleaned cropped Plate",
        desc: "Once a valid bounding contour is located, that region is cropped out. We apply adaptive local thresholding to isolate characters. Contours are run again on the cropped plate, identifying the largest nested contour representing the plate body, and correcting rotation/shearing if necessary.",
        code: "# Crop plate\nplate = img[y:y+h, x:x+w]\n# Adaptive thresholding to clean\ngray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)\nthresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)"
    },
    8: {
        title: "Step 8: Character Segmentation",
        desc: "Connected Component Analysis (CCA) labels separate blobs in the plate. We filter out blobs that don't match typical character proportions (height, aspect ratio, solidity) to remove plates' screws, margins, and speckles. The remaining blobs are sorted from left to right, providing clean cropped letters for the OCR engine.",
        code: "from skimage import measure\nlabels = measure.label(thresh_plate, background=0)\n# Filter and sort by x coordinates\nfor label in np.unique(labels):\n    # Filter aspect ratio, solidity, height\n    # Segment out character"
    }
};

// AUDIO ALARM SYNTHESIZER
function playAlarmChime(isAuthorized) {
    if (!elements.chkAlarm.checked) return;
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        if (isAuthorized) {
            // Authorized chime: soft positive double-beep
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
            gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.15);
            
            osc.start();
            osc.stop(audioCtx.currentTime + 0.2);
            
            setTimeout(() => {
                const osc2 = audioCtx.createOscillator();
                const gain2 = audioCtx.createGain();
                osc2.connect(gain2);
                gain2.connect(audioCtx.destination);
                
                osc2.type = 'sine';
                osc2.frequency.setValueAtTime(1046.5, audioCtx.currentTime); // C6
                gain2.gain.setValueAtTime(0.1, audioCtx.currentTime);
                gain2.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
                osc2.start();
                osc2.stop(audioCtx.currentTime + 0.3);
            }, 100);
        } else {
            // Unauthorized alarm: low buzz sirens
            const osc1 = audioCtx.createOscillator();
            const osc2 = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            
            osc1.connect(gainNode);
            osc2.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            
            osc1.type = 'sawtooth';
            osc2.type = 'square';
            
            osc1.frequency.setValueAtTime(220, audioCtx.currentTime); // A3
            osc2.frequency.setValueAtTime(223, audioCtx.currentTime); // Slightly detuned for buzz
            
            gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.15, audioCtx.currentTime + 0.1);
            gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.45);
            
            // Frequency drop modulation
            osc1.frequency.linearRampToValueAtTime(110, audioCtx.currentTime + 0.4);
            osc2.frequency.linearRampToValueAtTime(112, audioCtx.currentTime + 0.4);
            
            osc1.start();
            osc2.start();
            osc1.stop(audioCtx.currentTime + 0.5);
            osc2.stop(audioCtx.currentTime + 0.5);
        }
    } catch (e) {
        console.error("Audio Web Synth error: ", e);
    }
}

// SPEECH SYNTHESIS (TTS)
function speakPlate(text, status) {
    if (!elements.chkTts.checked) return;
    const now = Date.now();
    // Prevent spamming the same plate speech in rapid succession
    if (text === lastSpeechPlate && (now - lastSpeechTime) < 5000) return;
    
    lastSpeechPlate = text;
    lastSpeechTime = now;
    
    // Format reading (e.g. DL 3C AG 1 2 3 4)
    // Separate letters and numbers to make speech clear
    const spacedText = text.split("").join(" ");
    const phrase = `Detected plate, ${spacedText}. Status, ${status}.`;
    
    const utterance = new SpeechSynthesisUtterance(phrase);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    
    // Choose a standard English voice if available
    const voices = window.speechSynthesis.getVoices();
    const enVoice = voices.find(voice => voice.lang.startsWith('en'));
    if (enVoice) utterance.voice = enVoice;
    
    window.speechSynthesis.speak(utterance);
}

// INTRUDER ALARM DETECT ROUTER
function processIntruderAlerts(plateText, status) {
    // Play chime
    playAlarmChime(status === 'Authorized');
    // Speak plate
    speakPlate(plateText, status);
}

// HEALTH CHECK
async function checkHealth() {
    try {
        const res = await fetch(API_URLs.checkHealth);
        if (res.ok) {
            elements.statusBackendText.textContent = "Connected";
            elements.statusBackendDot.className = "status-dot active";
        } else {
            throw new Error();
        }
    } catch (e) {
        elements.statusBackendText.textContent = "Offline";
        elements.statusBackendDot.className = "status-dot inactive";
    }
}

// GET SIDEBAR PARAMETERS
function getParams() {
    return {
        engine: elements.engineSelect.value,
        blur_kernel: parseInt(elements.blurKernel.value),
        sobel_kernel: parseInt(elements.sobelKernel.value),
        morph_w: parseInt(elements.morphW.value),
        morph_h: parseInt(elements.morphH.value),
        min_area: parseInt(elements.areaMin.value),
        max_area: parseInt(elements.areaMax.value),
        min_ratio: parseFloat(elements.ratioMin.value),
        max_ratio: parseFloat(elements.ratioMax.value)
    };
}

// UPDATE SLIDER LABELS
function updateSliderLabels() {
    elements.valBlurKernel.textContent = `${elements.blurKernel.value}x${elements.blurKernel.value}`;
    elements.valSobelKernel.textContent = elements.sobelKernel.value;
    elements.valMorphKernel.textContent = `${elements.morphW.value}x${elements.morphH.value}`;
    
    const areaMinK = elements.areaMin.value >= 1000 ? `${elements.areaMin.value / 1000}k` : elements.areaMin.value;
    const areaMaxK = elements.areaMax.value >= 1000 ? `${elements.areaMax.value / 1000}k` : elements.areaMax.value;
    elements.valAreaBounds.textContent = `${areaMinK} - ${areaMaxK}`;
    
    elements.valRatioBounds.textContent = `${parseFloat(elements.ratioMin.value).toFixed(1)} - ${parseFloat(elements.ratioMax.value).toFixed(1)}`;
}

// RESET SLIDER PARAMETERS
function resetParameters() {
    elements.engineSelect.value = "classic";
    elements.blurKernel.value = 7;
    elements.sobelKernel.value = 3;
    elements.morphW.value = 22;
    elements.morphH.value = 3;
    elements.areaMin.value = 500;
    elements.areaMax.value = 30000;
    elements.ratioMin.value = 1.5;
    elements.ratioMax.value = 7.0;
    
    updateSliderLabels();
}

// SYNC SLIDER INPUT EVENT LISTENERS
[elements.blurKernel, elements.sobelKernel, elements.morphW, elements.morphH, 
 elements.areaMin, elements.areaMax, elements.ratioMin, elements.ratioMax].forEach(slider => {
    slider.addEventListener('input', updateSliderLabels);
});

elements.btnResetParams.addEventListener('click', resetParameters);

// WHITELIST MANAGEMENT
async function fetchWhitelist() {
    try {
        const res = await fetch(API_URLs.whitelist);
        if (res.ok) {
            currentWhitelists = await res.json();
            renderWhitelist();
        }
    } catch (e) {
        console.error("Error fetching whitelist", e);
    }
}

function renderWhitelist() {
    elements.whitelistList.innerHTML = '';
    if (currentWhitelists.length === 0) {
        elements.whitelistList.innerHTML = '<li class="whitelist-item" style="color: var(--text-light); font-weight: normal; justify-content: center;">List Empty</li>';
        return;
    }
    
    currentWhitelists.forEach(plate => {
        const li = document.createElement('li');
        li.className = 'whitelist-item';
        li.innerHTML = `
            <span>${plate}</span>
            <button class="btn-whitelist-del" data-plate="${plate}"><i class="fa-solid fa-trash-can"></i></button>
        `;
        elements.whitelistList.appendChild(li);
    });

    // Add delete listeners
    document.querySelectorAll('.btn-whitelist-del').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const plate = e.currentTarget.getAttribute('data-plate');
            await deleteFromWhitelist(plate);
        });
    });
}

async function addToWhitelist(plate) {
    if (!plate) return;
    try {
        const res = await fetch(API_URLs.whitelist, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plate })
        });
        if (res.ok) {
            currentWhitelists = await res.json();
            renderWhitelist();
            elements.whitelistInput.value = '';
            // Refresh history table statuses
            fetchHistory();
        }
    } catch (e) {
        console.error("Error adding whitelist item", e);
    }
}

async function deleteFromWhitelist(plate) {
    try {
        const res = await fetch(`${API_URLs.whitelist}/${plate}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            currentWhitelists = await res.json();
            renderWhitelist();
            // Refresh history table statuses
            fetchHistory();
        }
    } catch (e) {
        console.error("Error deleting whitelist item", e);
    }
}

elements.whitelistForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const val = elements.whitelistInput.value.trim().toUpperCase();
    addToWhitelist(val);
});

// HISTORY MANAGEMENT
async function fetchHistory() {
    try {
        const res = await fetch(API_URLs.history);
        if (res.ok) {
            detectionHistory = await res.json();
            renderHistoryTable();
        }
    } catch (e) {
        console.error("Error fetching history log", e);
    }
}

function renderHistoryTable() {
    const tbody = elements.auditTableBody;
    tbody.innerHTML = '';
    
    // Filter history records locally
    const filteredHistory = detectionHistory.filter(record => {
        // Filter by Status Tab
        if (activeFilter !== 'all' && record.status !== activeFilter) return false;
        
        // Filter by Text Search Box
        if (searchKeyword) {
            const keyword = searchKeyword.toLowerCase();
            return record.plate.toLowerCase().includes(keyword) || 
                   record.timestamp.includes(keyword) || 
                   record.status.toLowerCase().includes(keyword);
        }
        return true;
    });

    if (filteredHistory.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-state-row">
                <td colspan="5" class="table-empty-state">
                    <i class="fa-solid fa-box-open"></i>
                    <p>No matching records found.</p>
                </td>
            </tr>
        `;
        return;
    }

    filteredHistory.forEach(record => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${record.timestamp}</td>
            <td class="plate-crop-td"><img src="${HOST}${record.image_url}" alt="Crop"></td>
            <td class="plate-text-td">${record.plate}</td>
            <td>${record.confidence}%</td>
            <td><span class="status-badge ${record.status.toLowerCase()}">${record.status}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

// CLEAR HISTORY
elements.btnClearHistory.addEventListener('click', async () => {
    if (!confirm("Are you sure you want to delete all historical logs? This action is permanent.")) return;
    try {
        const res = await fetch(API_URLs.clearHistory, { method: 'POST' });
        if (res.ok) {
            detectionHistory = await res.json();
            renderHistoryTable();
            // Clear HUD display too
            resetHUD();
        }
    } catch (e) {
        console.error("Error clearing logs", e);
    }
});

// CSV EXPORT
elements.btnExportCsv.addEventListener('click', () => {
    if (detectionHistory.length === 0) {
        alert("No logs to export!");
        return;
    }
    
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "ID,Timestamp,Plate Text,Confidence Score,Status\n";
    
    detectionHistory.forEach(record => {
        csvContent += `"${record.id}","${record.timestamp}","${record.plate}",${record.confidence},"${record.status}"\n`;
    });
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `lpr_audit_logs_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// HUD & PREVIEWS RESET
function resetHUD() {
    elements.hudPlateText.textContent = "------";
    elements.hudVerdict.textContent = "NO PLATE";
    elements.hudVerdict.className = "stat-val status-badge unauthorized";
    elements.hudConfidence.textContent = "0.0%";
    elements.hudCoords.textContent = "N/A";
    elements.hudPlateCrop.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='30' viewBox='0 0 100 30'><rect width='100' height='30' fill='%23e2e8f0'/><text x='50%' y='60%' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='10' fill='%2394a3b8'>No plate</text></svg>";
}

// UPDATE HUD STATS
function updateHUD(plateData) {
    if (!plateData) {
        resetHUD();
        return;
    }
    elements.hudPlateText.textContent = plateData.text;
    elements.hudVerdict.textContent = plateData.status;
    elements.hudVerdict.className = `stat-val status-badge ${plateData.status.toLowerCase()}`;
    elements.hudConfidence.textContent = `${Math.round(plateData.confidence * 100)}%`;
    
    const [x, y, w, h] = plateData.box;
    elements.hudCoords.textContent = `[${x}, ${y}, ${w}x${h}]`;
    
    if (plateData.cropped_url) {
        elements.hudPlateCrop.src = HOST + plateData.cropped_url;
    }
}

// TABS NAVIGATION CONTROLS
elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        elements.tabButtons.forEach(b => b.classList.remove('active'));
        elements.tabPanels.forEach(p => p.classList.remove('active'));
        
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// SEARCH & FILTER KEYBOARD EVENTS
elements.filterSearchInput.addEventListener('input', (e) => {
    searchKeyword = e.target.value;
    renderHistoryTable();
});

elements.filterButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
        elements.filterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.getAttribute('data-filter');
        renderHistoryTable();
    });
});

// MEDIA DETECTOR MODES (UPLOAD vs WEBCAM)
elements.modeBtnUpload.addEventListener('click', () => {
    switchMode('upload');
});

elements.modeBtnWebcam.addEventListener('click', () => {
    switchMode('webcam');
});

// TRY SAMPLE VIDEO EVENT LISTENER
elements.modeBtnSample.addEventListener('click', async () => {
    elements.imageContainer.classList.add('hidden');
    elements.videoContainer.classList.remove('hidden');
    elements.uploadZone.classList.add('hidden');
    stopWebcamStream();
    
    elements.loadingOverlay.classList.remove('hidden');
    elements.loadingText.innerHTML = `
        <div style="text-align: center;">
            <p style="margin-bottom: 8px;">Processing Sample Video...</p>
            <small style="color: var(--text-muted); font-size: 12px; font-weight: normal;">
                (Analyzing frame-by-frame on CPU backend)
            </small>
        </div>
    `;
    
    const params = getParams();
    const formData = new FormData();
    formData.append("engine", params.engine);
    formData.append("min_area", params.min_area);
    formData.append("max_area", params.max_area);
    formData.append("min_ratio", params.min_ratio);
    formData.append("max_ratio", params.max_ratio);
    formData.append("blur_kernel", params.blur_kernel);
    formData.append("sobel_kernel", params.sobel_kernel);
    formData.append("morph_w", params.morph_w);
    formData.append("morph_h", params.morph_h);
    
    try {
        const res = await fetch(`${HOST}/api/process-sample`, {
            method: 'POST',
            body: formData
        });
        
        if (res.ok) {
            const data = await res.json();
            elements.loadingOverlay.classList.add('hidden');
            
            if (data.success) {
                const player = elements.videoPreview;
                player.src = HOST + data.processed_video_url;
                player.load();
                player.play();
                
                if (data.plates_detected.length > 0) {
                    const latest = data.plates_detected[data.plates_detected.length - 1];
                    updateHUD({
                        text: latest.text,
                        status: latest.status,
                        confidence: latest.confidence,
                        box: [0, 0, 0, 0]
                    });
                    processIntruderAlerts(latest.text, latest.status);
                }
            } else {
                alert("Failed to process sample video.");
            }
            fetchHistory();
        } else {
            throw new Error();
        }
    } catch (e) {
        elements.loadingOverlay.classList.add('hidden');
        alert("Error processing sample video on server.");
    }
});

function switchMode(mode) {
    activeMode = mode;
    if (mode === 'upload') {
        elements.modeBtnUpload.classList.add('active');
        elements.modeBtnWebcam.classList.remove('active');
        
        elements.uploadZone.classList.remove('hidden');
        elements.webcamContainer.classList.add('hidden');
        elements.webcamControlsPanel.classList.add('hidden');
        
        // Terminate webcam
        stopWebcamStream();
    } else {
        elements.modeBtnUpload.classList.remove('active');
        elements.modeBtnWebcam.classList.add('active');
        
        elements.uploadZone.classList.add('hidden');
        elements.imageContainer.classList.add('hidden');
        elements.videoContainer.classList.add('hidden');
        elements.webcamContainer.classList.remove('hidden');
        elements.webcamControlsPanel.classList.remove('hidden');
        
        // Start webcam
        startWebcamStream();
    }
}

// DRAG AND DROP FILE UPLOAD
const dropZone = elements.uploadZone;

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary)";
        dropZone.style.backgroundColor = "var(--primary-light)";
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        dropZone.style.backgroundColor = "#ffffff";
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        handleUploadedFile(files[0]);
    }
});

elements.mediaInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleUploadedFile(e.target.files[0]);
    }
});

// ROUTE UPLOADED FILE TYPE
function handleUploadedFile(file) {
    if (!file || file.size === 0) {
        console.warn("Ignoring empty file upload (possible browser auto-fill/reload restoration).");
        return;
    }
    if (file.type.startsWith('image/')) {
        elements.imageContainer.classList.remove('hidden');
        elements.videoContainer.classList.add('hidden');
        elements.uploadZone.classList.add('hidden');
        
        // Show local preview
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.imagePreview.src = e.target.result;
            elements.imagePreview.onload = () => {
                // Clear any old canvas annotations
                const canvas = elements.imageCanvas;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Process image on backend
                sendImageToBackend(file);
            };
        };
        reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
        elements.imageContainer.classList.add('hidden');
        elements.videoContainer.classList.remove('hidden');
        elements.uploadZone.classList.add('hidden');
        
        sendVideoToBackend(file);
    } else {
        alert("Unsupported file format! Please upload an image or video.");
    }
}

// SEND IMAGE TO FastAPI BACKEND
async function sendImageToBackend(file) {
    elements.loadingOverlay.classList.remove('hidden');
    elements.loadingText.textContent = "Processing Image & Extracting License Plate...";
    
    const params = getParams();
    const formData = new FormData();
    formData.append("image", file);
    formData.append("engine", params.engine);
    formData.append("min_area", params.min_area);
    formData.append("max_area", params.max_area);
    formData.append("min_ratio", params.min_ratio);
    formData.append("max_ratio", params.max_ratio);
    formData.append("blur_kernel", params.blur_kernel);
    formData.append("sobel_kernel", params.sobel_kernel);
    formData.append("morph_w", params.morph_w);
    formData.append("morph_h", params.morph_h);
    
    try {
        const res = await fetch(API_URLs.processImage, {
            method: 'POST',
            body: formData
        });
        
        if (res.ok) {
            const data = await res.json();
            elements.loadingOverlay.classList.add('hidden');
            
            // 1. Update HUD
            if (data.success && data.plates.length > 0) {
                const detected = data.plates[0];
                updateHUD(detected);
                drawBoundingBoxOnImage(detected.box, detected.text);
                
                // Sound and speech alerts
                processIntruderAlerts(detected.text, detected.status);
            } else {
                updateHUD(null);
                alert("No license plates detected under current parameters.");
            }
            
            // 2. Refresh History table
            fetchHistory();
            
            // 3. Load Visual Pipeline steps images
            if (data.steps) {
                processedStepsData = data.steps;
                loadPipelineSteps(data.steps, data.segmented_characters);
            }
        } else {
            throw new Error();
        }
    } catch (e) {
        elements.loadingOverlay.classList.add('hidden');
        alert("Error connecting to backend server.");
    }
}

// DRAW BOX ON PREVIEW IMAGE
function drawBoundingBoxOnImage(box, text) {
    const img = elements.imagePreview;
    const canvas = elements.imageCanvas;
    const ctx = canvas.getContext('2d');
    
    // Sync canvas resolution with rendered dimension of image (bounding coordinates scaling)
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Original resolution bounds
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    // Scaling ratio
    const scaleX = canvas.width / naturalWidth;
    const scaleY = canvas.height / naturalHeight;
    
    const [x, y, w, h] = box;
    
    // Scale box
    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = w * scaleX;
    const drawH = h * scaleY;
    
    // Draw box border outline
    ctx.strokeStyle = "#0099ff";
    ctx.lineWidth = 3;
    ctx.strokeRect(drawX, drawY, drawW, drawH);
    
    // Label tag background
    ctx.fillStyle = "#0099ff";
    ctx.font = "bold 12px Inter";
    const tagText = text;
    const textWidth = ctx.measureText(tagText).width;
    
    ctx.fillRect(drawX - 1.5, drawY - 20, textWidth + 12, 20);
    
    // Draw text inside label tag
    ctx.fillStyle = "#ffffff";
    ctx.fillText(tagText, drawX + 6, drawY - 6);
}

// UPLOAD VIDEO PROCESSING
async function sendVideoToBackend(file) {
    elements.loadingOverlay.classList.remove('hidden');
    elements.loadingText.innerHTML = `
        <div style="text-align: center;">
            <p style="margin-bottom: 8px;">Analyzing Video and Running LPR...</p>
            <small style="color: var(--text-muted); font-size: 12px; font-weight: normal;">
                (Frame-by-frame OCR processing may take a few seconds)
            </small>
        </div>
    `;
    
    const params = getParams();
    const formData = new FormData();
    formData.append("video", file);
    formData.append("engine", params.engine);
    formData.append("min_area", params.min_area);
    formData.append("max_area", params.max_area);
    formData.append("min_ratio", params.min_ratio);
    formData.append("max_ratio", params.max_ratio);
    formData.append("blur_kernel", params.blur_kernel);
    formData.append("sobel_kernel", params.sobel_kernel);
    formData.append("morph_w", params.morph_w);
    formData.append("morph_h", params.morph_h);
    
    try {
        const res = await fetch(API_URLs.uploadVideo, {
            method: 'POST',
            body: formData
        });
        
        if (res.ok) {
            const data = await res.json();
            elements.loadingOverlay.classList.add('hidden');
            
            if (data.success) {
                // Set output processed video player source
                const player = elements.videoPreview;
                player.src = HOST + data.processed_video_url;
                player.load();
                player.play();
                
                // Show alerts for any detected vehicle
                if (data.plates_detected.length > 0) {
                    const latest = data.plates_detected[data.plates_detected.length - 1];
                    updateHUD({
                        text: latest.text,
                        status: latest.status,
                        confidence: latest.confidence,
                        box: [0, 0, 0, 0] // No coordinate overlay needed for pre-processed video output
                    });
                    processIntruderAlerts(latest.text, latest.status);
                }
            } else {
                alert("Failed to process video file.");
            }
            
            // Refresh history table
            fetchHistory();
        } else {
            throw new Error();
        }
    } catch (e) {
        elements.loadingOverlay.classList.add('hidden');
        alert("Error connecting to server for video processing.");
    }
}

// WEBCAM LIVE STREAMING
async function startWebcamStream() {
    resetHUD();
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'environment' }
        });
        
        elements.webcamFeed.srcObject = webcamStream;
        
        // Grab frames at regular intervals (approx 3 frames per second on CPU backend to maintain real-time performance)
        webcamInterval = setInterval(captureWebcamFrameAndProcess, 450);
        
    } catch (e) {
        alert("Cannot open webcam. Please verify camera permissions.");
        switchMode('upload');
    }
}

function stopWebcamStream() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }
    isWebcamProcessing = false;
}

elements.btnStopWebcam.addEventListener('click', () => {
    switchMode('upload');
});

// WEBCAM FRAME GRABBER & RENDER LOOP
async function captureWebcamFrameAndProcess() {
    if (isWebcamProcessing || !webcamStream) return;
    
    isWebcamProcessing = true;
    const video = elements.webcamFeed;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Compress frame and grab base64 string
    const base64Frame = canvas.toDataURL('image/jpeg', 0.7);
    const params = getParams();
    
    try {
        const res = await fetch(API_URLs.processFrame, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Frame,
                engine: params.engine,
                min_area: params.min_area,
                max_area: params.max_area,
                min_ratio: params.min_ratio,
                max_ratio: params.max_ratio,
                blur_kernel: params.blur_kernel,
                sobel_kernel: params.sobel_kernel,
                morph_w: params.morph_w,
                morph_h: params.morph_h
            })
        });
        
        if (res.ok) {
            const data = await res.json();
            
            // Clear prior webcam annotations
            const webCanvas = elements.webcamCanvas;
            const webCtx = webCanvas.getContext('2d');
            webCanvas.width = video.clientWidth;
            webCanvas.height = video.clientHeight;
            webCtx.clearRect(0, 0, webCanvas.width, webCanvas.height);
            
            if (data.success && data.plates.length > 0) {
                const detected = data.plates[0];
                updateHUD(detected);
                
                // Draw box on live canvas
                drawBoundingBoxOnWebcamCanvas(detected.box, detected.text);
                
                // Sound and speech alerts
                processIntruderAlerts(detected.text, detected.status);
                
                // Refresh audit database log
                fetchHistory();
            } else {
                // Optional: Soft fade out HUD if nothing found for 2-3 seconds
            }
        }
    } catch (e) {
        console.error("Webcam frame processor request error: ", e);
    } finally {
        isWebcamProcessing = false;
    }
}

// DRAW BOX ON LIVE WEBCAM HUD CANVAS
function drawBoundingBoxOnWebcamCanvas(box, text) {
    const video = elements.webcamFeed;
    const canvas = elements.webcamCanvas;
    const ctx = canvas.getContext('2d');
    
    // Scale coordinate factors
    const naturalWidth = video.videoWidth || 640;
    const naturalHeight = video.videoHeight || 480;
    const scaleX = canvas.width / naturalWidth;
    const scaleY = canvas.height / naturalHeight;
    
    const [x, y, w, h] = box;
    
    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = w * scaleX;
    const drawH = h * scaleY;
    
    // Border
    ctx.strokeStyle = "#10b981"; // Use green for webcam tracking box
    ctx.lineWidth = 3;
    ctx.strokeRect(drawX, drawY, drawW, drawH);
    
    // Label tag
    ctx.fillStyle = "#10b981";
    ctx.font = "bold 12px Inter";
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(drawX - 1.5, drawY - 20, textWidth + 12, 20);
    
    // Text label
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, drawX + 6, drawY - 6);
}

// LOAD PIPELINE STEPS IN TABS
function loadPipelineSteps(steps, chars) {
    elements.step1Img.src = steps["1_original"] || "";
    elements.step2Img.src = steps["2_grayscale"] || "";
    elements.step3Img.src = steps["3_sobelx"] || "";
    elements.step4Img.src = steps["4_threshold"] || "";
    elements.step5Img.src = steps["5_morphology"] || "";
    elements.step6Img.src = steps["6_contours"] || "";
    elements.step7Img.src = steps["7_clean_plate"] || "";
    elements.step8Img.src = steps["8_character_mask"] || "";

    // Load segmented single character pills
    const charsRow = elements.charsListRow;
    charsRow.innerHTML = '';
    
    if (chars && chars.length > 0) {
        elements.charSegmentsPanel.classList.remove('hidden');
        chars.forEach(charB64 => {
            const div = document.createElement('div');
            div.className = 'char-crop-box';
            div.innerHTML = `<img src="${charB64}" alt="Character crop">`;
            charsRow.appendChild(div);
        });
    } else {
        elements.charSegmentsPanel.classList.add('hidden');
    }
}

// INTERACTIVE PIPELINE CARDS DETAILED MODAL VIEW
elements.pipelineCards.forEach(card => {
    card.addEventListener('click', () => {
        const stepNum = parseInt(card.getAttribute('data-step'));
        const stepInfo = PIPELINE_STEP_DETAILS[stepNum];
        
        // Find base64 image from memory data
        let imageSrc = "";
        if (stepNum === 1) imageSrc = processedStepsData["1_original"];
        else if (stepNum === 2) imageSrc = processedStepsData["2_grayscale"];
        else if (stepNum === 3) imageSrc = processedStepsData["3_sobelx"];
        else if (stepNum === 4) imageSrc = processedStepsData["4_threshold"];
        else if (stepNum === 5) imageSrc = processedStepsData["5_morphology"];
        else if (stepNum === 6) imageSrc = processedStepsData["6_contours"];
        else if (stepNum === 7) imageSrc = processedStepsData["7_clean_plate"];
        else if (stepNum === 8) imageSrc = processedStepsData["8_character_mask"];

        if (!imageSrc) {
            alert("No processed CV pipeline loaded. Upload an image in the 'Live Detector' tab first!");
            return;
        }

        elements.modalStepTitle.textContent = stepInfo.title;
        elements.modalStepImg.src = imageSrc;
        elements.modalStepMathDesc.textContent = stepInfo.desc;
        elements.modalStepCode.textContent = stepInfo.code;
        
        elements.stepModal.classList.remove('hidden');
    });
});

// CLOSE MODAL
function closeModal() {
    elements.stepModal.classList.add('hidden');
}
elements.btnCloseModal.addEventListener('click', closeModal);
elements.modalOverlay.addEventListener('click', closeModal);

// INIT APP
async function init() {
    resetParameters();
    await checkHealth();
    await fetchWhitelist();
    await fetchHistory();
    elements.loadingOverlay.classList.add('hidden');
}

// Run health checks every 10 seconds to detect server connection drops
setInterval(checkHealth, 10000);

// Initialize application
init();
