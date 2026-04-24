import traceback
from flask import render_template, jsonify, request, current_app
from pathlib import Path

from . import main_bp, api_bp
from app.services.file_storage import FileStorageService
from app.services.serializer   import ResponseSerializer
from app.models.face_detector       import FaceDetector
from app.models.landmark_detector   import LandmarkDetector
from app.models.normal_map_generator import NormalMapGenerator
from app.models.face_reconstructor  import FaceReconstructor3D
from app.models.pose_estimator      import PoseEstimator


class UploadController:
    """
    Controlador principal — orquesta el pipeline completo de
    detección, reconstrucción y serialización de resultados.
    """

    @staticmethod
    def _get_storage() -> FileStorageService:
        return FileStorageService(
            upload_dir=Path(current_app.config["UPLOAD_FOLDER"]),
            output_dir=Path(current_app.config.get(
                "OUTPUT_FOLDER", "outputs"))
        )

    @staticmethod
    def index():
        return render_template("index.html")

    @staticmethod
    def health():
        return jsonify({
            "status":  "ok",
            "message": "Face3D Web API funcionando correctamente",
            "version": "1.0.0"
        })

    @staticmethod
    def process():
        # ── Validar archivo ───────────────────────────────────────
        if "file" not in request.files:
            return jsonify(
                {"error": "No se encontró archivo en la petición"}
            ), 400

        file    = request.files["file"]
        storage = UploadController._get_storage()
        valid, err_msg = storage.validate_file(file)

        if not valid:
            return jsonify({"error": err_msg}), 400

        try:
            # ── Leer imagen ───────────────────────────────────────
            image, _ = storage.save_and_read(file)

            # ── Pipeline de Visión Artificial ─────────────────────
            # 1. Detección
            detector = FaceDetector()
            r_det    = detector.process(image)

            if not r_det["found"]:
                return jsonify({
                    "error": "No se detectó ningún rostro en la imagen. "
                             "Usa una foto frontal con buena iluminación."
                }), 422

            # 2. Landmarks
            lm_det = LandmarkDetector()
            r_lm   = lm_det.process(image)

            if not r_lm["found"]:
                return jsonify({
                    "error": "No se pudieron detectar los landmarks faciales."
                }), 422

            # 3. Normal Map
            nmg  = NormalMapGenerator(output_size=256)
            r_nm = nmg.generate(r_lm["landmarks_3d"], image)

            # 4. Reconstrucción 3D
            rec   = FaceReconstructor3D()
            r_rec = rec.reconstruct(r_lm["landmarks_3d"], image)

            # 5. Pose
            pose   = PoseEstimator()
            r_pose = pose.estimate(
                r_lm["pose_points_2d"],
                r_lm["face_3d_model"],
                image
            )

            # ── Serializar respuesta ──────────────────────────────
            response = ResponseSerializer.build_response(
                detection=r_det,
                landmarks=r_lm,
                normal_map=r_nm,
                reconstruction=r_rec,
                pose=r_pose,
                original_image=image
            )
            return jsonify(response), 200

        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "error": f"Error interno al procesar la imagen: {str(e)}"
            }), 500


# ── Rutas web ─────────────────────────────────────────────────────
@main_bp.route("/")
def index():
    return UploadController.index()


# ── Rutas API ─────────────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    return UploadController.health()


@api_bp.route("/process", methods=["POST"])
def process():
    return UploadController.process()