// viewer3d.js — visor 3D interactivo con Three.js (CDN)
(function () {

    function loadThree(callback) {
        if (window.THREE) { callback(); return; }
        const s = document.createElement('script');
        s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
        s.onload = callback;
        document.head.appendChild(s);
    }

    window.initViewer3D = function (vertices, faces, imgB64) {
        loadThree(() => buildScene(vertices, faces, imgB64));
    };

    function buildScene(vertices, faces, imgB64) {
        const canvas   = document.getElementById('canvas3d');
        if (!canvas) return;

        // Limpiar escena anterior si existía
        if (canvas._renderer) {
            canvas._renderer.dispose();
        }

        const W = canvas.clientWidth  || 400;
        const H = canvas.clientHeight || 300;

        // ── Renderer ──────────────────────────────────────────────
        const renderer = new THREE.WebGLRenderer({
            canvas, antialias: true, alpha: true
        });
        renderer.setSize(W, H);
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setClearColor(0x161921, 1);
        canvas._renderer = renderer;

        // ── Escena y cámara ───────────────────────────────────────
        const scene  = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 100);
        camera.position.set(0, 0, 3);

        // ── Iluminación ───────────────────────────────────────────
        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dirLight = new THREE.DirectionalLight(0x5c7cff, 0.8);
        dirLight.position.set(1, 2, 3);
        scene.add(dirLight);
        const dirLight2 = new THREE.DirectionalLight(0x3ecf8e, 0.4);
        dirLight2.position.set(-2, -1, 1);
        scene.add(dirLight2);

        // ── Geometría desde landmarks ─────────────────────────────
        const geometry = new THREE.BufferGeometry();

        const posArray = new Float32Array(vertices.length * 3);
        vertices.forEach((v, i) => {
            posArray[i * 3]     = v[0];
            posArray[i * 3 + 1] = v[1];
            posArray[i * 3 + 2] = v[2] || 0;
        });
        geometry.setAttribute('position',
            new THREE.BufferAttribute(posArray, 3));

        const idxArray = new Uint32Array(faces.length * 3);
        faces.forEach((f, i) => {
            idxArray[i * 3]     = f[0];
            idxArray[i * 3 + 1] = f[1];
            idxArray[i * 3 + 2] = f[2];
        });
        geometry.setIndex(new THREE.BufferAttribute(idxArray, 1));
        geometry.computeVertexNormals();

        // ── Material wireframe + sólido ───────────────────────────
        const matSolid = new THREE.MeshPhongMaterial({
            color:       0x9ba3c4,
            shininess:   40,
            transparent: true,
            opacity:     0.85,
            side:        THREE.DoubleSide
        });
        const matWire = new THREE.MeshBasicMaterial({
            color:       0x5c7cff,
            wireframe:   true,
            transparent: true,
            opacity:     0.25
        });

        const meshSolid = new THREE.Mesh(geometry, matSolid);
        const meshWire  = new THREE.Mesh(geometry, matWire);

        const group = new THREE.Group();
        group.add(meshSolid);
        group.add(meshWire);
        scene.add(group);

        // ── Puntos ────────────────────────────────────────────────
        const ptGeo = new THREE.BufferGeometry();
        ptGeo.setAttribute('position',
            new THREE.BufferAttribute(posArray.slice(), 3));
        const ptMat = new THREE.PointsMaterial({
            color: 0x3ecf8e, size: 0.015
        });
        scene.add(new THREE.Points(ptGeo, ptMat));

        // ── Controles de órbita simples ───────────────────────────
        let isDragging = false;
        let prevMouse  = { x: 0, y: 0 };
        let rotX = 0, rotY = 0, zoom = 3;

        canvas.addEventListener('mousedown', e => {
            isDragging = true;
            prevMouse  = { x: e.clientX, y: e.clientY };
        });
        window.addEventListener('mouseup',   () => { isDragging = false; });
        window.addEventListener('mousemove', e => {
            if (!isDragging) return;
            const dx = e.clientX - prevMouse.x;
            const dy = e.clientY - prevMouse.y;
            rotY += dx * 0.008;
            rotX += dy * 0.008;
            rotX  = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotX));
            prevMouse = { x: e.clientX, y: e.clientY };
        });
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            zoom += e.deltaY * 0.005;
            zoom  = Math.max(1.0, Math.min(6.0, zoom));
        }, { passive: false });

        // Touch
        let lastTouch = null;
        canvas.addEventListener('touchstart', e => {
            lastTouch = { x: e.touches[0].clientX,
                          y: e.touches[0].clientY };
        });
        canvas.addEventListener('touchmove', e => {
            e.preventDefault();
            if (!lastTouch) return;
            const dx = e.touches[0].clientX - lastTouch.x;
            const dy = e.touches[0].clientY - lastTouch.y;
            rotY += dx * 0.008;
            rotX += dy * 0.008;
            lastTouch = { x: e.touches[0].clientX,
                          y: e.touches[0].clientY };
        }, { passive: false });

        // ── Loop de animación ─────────────────────────────────────
        function animate() {
            requestAnimationFrame(animate);
            group.rotation.x = rotX;
            group.rotation.y = rotY;
            camera.position.z = zoom;
            renderer.render(scene, camera);
        }
        animate();
    }
})();