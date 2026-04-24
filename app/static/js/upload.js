(function () {
    const dropZone    = document.getElementById('dropZone');
    const fileInput   = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const previewImg  = document.getElementById('previewImg');
    const previewName = document.getElementById('previewName');
    const btnProcess  = document.getElementById('btnProcess');
    const btnText     = document.getElementById('btnText');
    const btnSpinner  = document.getElementById('btnSpinner');
    const statusMsg   = document.getElementById('statusMsg');
    const progressWrap= document.getElementById('progressWrap');
    const progressBar = document.getElementById('progressBar');
    const progressLbl = document.getElementById('progressLabel');
    const resultsSection = document.getElementById('resultsSection');

    let selectedFile = null;

    // ── Drag & drop ──────────────────────────────────────────────
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover',  e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () =>
        dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        const allowed = ['image/jpeg', 'image/png', 'image/webp'];
        if (!allowed.includes(file.type)) {
            showStatus('Formato no permitido. Usa JPG, PNG o WEBP.', true);
            return;
        }
        if (file.size > 16 * 1024 * 1024) {
            showStatus('El archivo supera los 16 MB.', true);
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src        = e.target.result;
            previewName.textContent =
                file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
            previewArea.style.display = 'block';
            btnProcess.disabled = false;
            showStatus('');
        };
        reader.readAsDataURL(file);
    }

    // ── Envío al API ──────────────────────────────────────────────
    btnProcess.addEventListener('click', async () => {
        if (!selectedFile) return;
        setLoading(true);
        setProgress(10, 'Enviando imagen...');
        resultsSection.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            setProgress(30, 'Detectando rostro...');
            const res  = await fetch('/api/process', {
                method: 'POST', body: formData
            });
            setProgress(70, 'Generando resultados...');
            const data = await res.json();

            if (!res.ok) {
                showStatus(data.error || 'Error al procesar', true);
                setProgress(0, '');
                return;
            }

            setProgress(95, 'Renderizando...');
            renderResults(data);
            setProgress(100, 'Listo');
            showStatus('');

            setTimeout(() => {
                progressWrap.style.display = 'none';
            }, 800);

        } catch (err) {
            showStatus('Error de conexión con el servidor.', true);
            setProgress(0, '');
        } finally {
            setLoading(false);
        }
    });

    // ── Renderizar resultados ─────────────────────────────────────
    function renderResults(data) {
        // Detección
        setImg('imgDetection', data.detection.image_b64);
        setMetrics('metricsDetection', [
            ['Confianza', (data.detection.confidence * 100).toFixed(0) + '%'],
            ['BBox X',    data.detection.bbox?.x ?? '-'],
            ['BBox Y',    data.detection.bbox?.y ?? '-'],
            ['Ancho',     data.detection.bbox?.w ?? '-'],
            ['Alto',      data.detection.bbox?.h ?? '-'],
        ]);

        // Landmarks
        setImg('imgLandmarks', data.landmarks.image_b64);
        setMetrics('metricsLandmarks', [
            ['Puntos detectados', data.landmarks.count],
        ]);

        // Normal Map
        setImg('imgNormalMap',     data.normal_map.map_b64);
        setImg('imgNormalOverlay', data.normal_map.overlay_b64);

        // Reconstrucción 3D
        setImg('imgPointcloud', data.reconstruction.pointcloud_b64);
        const mesh = data.reconstruction.mesh;
        setMetrics('metricsMesh', [
            ['Vértices', mesh.vertex_count],
            ['Caras',    mesh.face_count],
        ]);

        // Lanzar visor 3D Three.js
        if (window.initViewer3D) {
            window.initViewer3D(
                mesh.vertices,
                mesh.faces,
                data.detection.image_b64
            );
        }

        // Pose
        if (data.pose.success) {
            setImg('imgPoseAxes',     data.pose.axes_b64);
            setImg('imgPoseEllipses', data.pose.ellipses_b64);
            const a = data.pose.euler_angles;
            setMetrics('metricsPose', [
                ['Yaw',   a.yaw   + '°'],
                ['Pitch', a.pitch + '°'],
                ['Roll',  a.roll  + '°'],
            ]);
        }

        // Mostrar sección y activar primera tab
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        activateTab('detection');
    }

    // ── Tabs ──────────────────────────────────────────────────────
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () =>
            activateTab(tab.dataset.tab));
    });

    function activateTab(name) {
        document.querySelectorAll('.tab').forEach(t =>
            t.classList.toggle('active', t.dataset.tab === name));
        document.querySelectorAll('.tab-panel').forEach(p =>
            p.classList.toggle('active', p.id === 'tab-' + name));
    }

    // ── Helpers ───────────────────────────────────────────────────
    function setImg(id, b64) {
        const el = document.getElementById(id);
        if (el && b64) el.src = 'data:image/jpeg;base64,' + b64;
    }

    function setMetrics(id, pairs) {
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = pairs.map(([k, v]) =>
            `<div class="metric-chip">${k}<span>${v}</span></div>`
        ).join('');
    }

    function setLoading(on) {
        btnProcess.disabled    = on;
        btnText.textContent    = on ? 'Procesando...' : 'Procesar Imagen';
        btnSpinner.style.display = on ? 'inline-block' : 'none';
        progressWrap.style.display = on ? 'block' : 'none';
    }

    function setProgress(pct, label) {
        progressBar.style.width    = pct + '%';
        progressLbl.textContent    = label;
    }

    function showStatus(msg, isError = false) {
        statusMsg.textContent = msg;
        statusMsg.className   = 'status-msg' + (isError ? ' error' : '');
    }
})();