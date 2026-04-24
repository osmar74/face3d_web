# test_fase3.py — ejecutar desde face3d_web\ con:
# python tests/test_fase3.py ruta\a\foto.jpg
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.face_detector       import FaceDetector
from app.models.landmark_detector   import LandmarkDetector
from app.models.normal_map_generator import NormalMapGenerator
from app.models.face_reconstructor  import FaceReconstructor3D
from app.models.pose_estimator      import PoseEstimator


def test_fase3(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: No se pudo leer la imagen: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"\nImagen: {w}x{h} px")

    # ── 1. Detección ─────────────────────────────────────────────
    print("\n[1/4] FaceDetector...")
    det    = FaceDetector()
    r_det  = det.process(image)
    if not r_det["found"]:
        print("  ERROR: No se detectó rostro.")
        return
    print(f"  OK — bbox: {r_det['bbox']}")

    # ── 2. Landmarks ─────────────────────────────────────────────
    print("\n[2/4] LandmarkDetector...")
    lm     = LandmarkDetector()
    r_lm   = lm.process(image)
    if not r_lm["found"]:
        print("  ERROR: No se detectaron landmarks.")
        return
    pts3d = r_lm["landmarks_3d"]
    print(f"  OK — landmarks: {pts3d.shape}")

    # ── 3. Normal Map ────────────────────────────────────────────
    print("\n[3/4] NormalMapGenerator...")
    nmg    = NormalMapGenerator(output_size=256)
    r_nm   = nmg.generate(pts3d, image)
    cv2.imwrite("outputs/test_normal_map.jpg",
                r_nm["normal_map_bgr"])
    cv2.imwrite("outputs/test_normal_overlay.jpg",
                r_nm["normal_map_overlay"])
    print("  OK — guardado: outputs/test_normal_map.jpg")
    print("       guardado: outputs/test_normal_overlay.jpg")

    # ── 4. Reconstrucción 3D ─────────────────────────────────────
    print("\n[4/4] FaceReconstructor3D + PoseEstimator...")
    rec    = FaceReconstructor3D()
    r_rec  = rec.reconstruct(pts3d, image)
    cv2.imwrite("outputs/test_pointcloud.jpg",
                r_rec["pointcloud_img_bgr"])
    mesh   = r_rec["mesh_data"]
    print(f"  OK — vértices: {mesh['vertex_count']}  "
          f"caras: {mesh['face_count']}")
    print("       guardado: outputs/test_pointcloud.jpg")

    # ── 5. Pose ───────────────────────────────────────────────────
    pose   = PoseEstimator()
    r_pose = pose.estimate(
        r_lm["pose_points_2d"],
        r_lm["face_3d_model"],
        image
    )
    if r_pose["success"]:
        ang = r_pose["euler_angles"]
        print(f"  OK — Yaw:{ang['yaw']:.1f}°  "
              f"Pitch:{ang['pitch']:.1f}°  "
              f"Roll:{ang['roll']:.1f}°")
        cv2.imwrite("outputs/test_pose_axes.jpg",
                    r_pose["pose_image_bgr"])
        cv2.imwrite("outputs/test_pose_ellipses.jpg",
                    r_pose["pose_3d_img_bgr"])
        print("       guardado: outputs/test_pose_axes.jpg")
        print("       guardado: outputs/test_pose_ellipses.jpg")
    else:
        print("  ERROR: solvePnP no convergió.")

    print("\nFASE 3 completada correctamente.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tests/test_fase3.py ruta\\a\\foto.jpg")
        sys.exit(1)
    test_fase3(sys.argv[1])