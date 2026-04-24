from flask import render_template, jsonify, request
from . import main_bp, api_bp


class UploadController:
    """Controlador principal — maneja subida y procesamiento de imágenes."""

    @staticmethod
    def index():
        """Renderiza la página principal."""
        return render_template("index.html")

    @staticmethod
    def health():
        """Endpoint de salud de la API."""
        return jsonify({
            "status": "ok",
            "message": "Face3D Web API funcionando correctamente",
            "version": "1.0.0"
        })

    @staticmethod
    def process():
        """Placeholder del endpoint de procesamiento (FASE 2+)."""
        if "file" not in request.files:
            return jsonify({"error": "No se encontró archivo en la petición"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        # TODO: implementar pipeline completo en FASE 2+
        return jsonify({
            "status": "placeholder",
            "message": "Pipeline de procesamiento se implementa en FASE 2",
            "filename": file.filename
        }), 200


# ── Rutas web ──────────────────────────────────────────────────────
@main_bp.route("/")
def index():
    return UploadController.index()


# ── Rutas API ──────────────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    return UploadController.health()


@api_bp.route("/process", methods=["POST"])
def process():
    return UploadController.process()