// upload.js — maneja drag & drop y envío al API
(function () {
    const dropZone   = document.getElementById('dropZone');
    const fileInput  = document.getElementById('fileInput');
    const previewArea= document.getElementById('previewArea');
    const previewImg = document.getElementById('previewImg');
    const previewName= document.getElementById('previewName');
    const btnProcess = document.getElementById('btnProcess');
    const btnText    = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const statusMsg  = document.getElementById('statusMsg');

    let selectedFile = null;

    // Clic en drop zone abre selector
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag events
    dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
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
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
            previewArea.style.display = 'block';
            btnProcess.disabled = false;
            showStatus('');
        };
        reader.readAsDataURL(file);
    }

    btnProcess.addEventListener('click', async () => {
        if (!selectedFile) return;
        setLoading(true);
        showStatus('Procesando imagen...');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res  = await fetch('/api/process', { method: 'POST', body: formData });
            const data = await res.json();
            if (res.ok) {
                showStatus('✓ ' + (data.message || 'Procesado correctamente'));
                // TODO FASE 6+: redirigir a resultado
            } else {
                showStatus(data.error || 'Error al procesar', true);
            }
        } catch (err) {
            showStatus('Error de conexión con el servidor', true);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(loading) {
        btnProcess.disabled = loading;
        btnText.textContent  = loading ? 'Procesando...' : 'Procesar Imagen';
        btnSpinner.style.display = loading ? 'inline-block' : 'none';
    }

    function showStatus(msg, isError = false) {
        statusMsg.textContent = msg;
        statusMsg.className = 'status-msg' + (isError ? ' error' : '');
    }
})();