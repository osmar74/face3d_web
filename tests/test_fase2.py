
# Test FASE 2 — Ejecutar desde face3d_web\ con:
#    python tests/test_fase2.py ruta\a\foto.jpg

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.face_detector   import FaceDetector
from app.models.landmark_detector import LandmarkDetector


def test_pipeline(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: No se pudo leer la imagen: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"\nImagen cargada: {w}x{h} px")

    # ── Test 1: FaceDetector ─────────────────────────────────────
    print("\n[1/2] FaceDetector...")
    detector = FaceDetector(min_confidence=0.5)
    result_det = detector.process(image)

    if result_det["found"]:
        bbox = result_det["bbox"]
        print(f"  Rostro detectado — BBox: x={bbox['x']} y={bbox['y']} "
              f"w={bbox['w']} h={bbox['h']} "
              f"confianza={result_det.get('confidence', 0):.0%}")
        cv2.imwrite("outputs/test_deteccion.jpg", result_det["draw_image"])
        cv2.imwrite("outputs/test_crop.jpg",      result_det["face_crop"])
        print("  Guardado: outputs/test_deteccion.jpg")
        print("  Guardado: outputs/test_crop.jpg")
    else:
        print("  No se detectó ningún rostro.")
        return
    detector.close()

    # ── Test 2: LandmarkDetector ─────────────────────────────────
    print("\n[2/2] LandmarkDetector (FaceMesh 468 puntos)...")
    lm_detector = LandmarkDetector()
    result_lm   = lm_detector.process(image)

    if result_lm["found"]:
        pts2d = result_lm["landmarks_2d"]
        pts3d = result_lm["landmarks_3d"]
        print(f"  Landmarks 2D: {pts2d.shape}  — ejemplo pt[1]: {pts2d[1].round(1)}")
        print(f"  Landmarks 3D: {pts3d.shape}  — ejemplo pt[1]: {pts3d[1].round(1)}")
        print(f"  Puntos pose (6 pts): {result_lm['pose_points_2d'].round(1)}")
        cv2.imwrite("outputs/test_landmarks.jpg", result_lm["draw_image"])
        print("  Guardado: outputs/test_landmarks.jpg")
    else:
        print("  No se detectaron landmarks.")
    lm_detector.close()

    print("\nFASE 2 — Tests completados correctamente.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tests/test_fase2.py ruta\\a\\foto.jpg")
        sys.exit(1)
    test_pipeline(sys.argv[1])