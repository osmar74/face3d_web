import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from .base_pipeline import BasePipeline

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / \
             "models_weights" / "face_landmarker.task"

# Índices de 6 puntos clave para solvePnP
POSE_LANDMARK_IDS = [1, 33, 263, 61, 291, 199]

# Modelo 3D genérico de cara (mm, sistema OpenCV)
FACE_3D_MODEL_POINTS = np.array([
    [0.0,     0.0,     0.0  ],   # Nariz          (1)
    [-225.0,  170.0, -135.0 ],   # Ojo izquierdo  (33)
    [ 225.0,  170.0, -135.0 ],   # Ojo derecho    (263)
    [-150.0, -150.0, -125.0 ],   # Labio izq.     (61)
    [ 150.0, -150.0, -125.0 ],   # Labio der.     (291)
    [0.0,   -330.0,  -65.0  ],   # Barbilla       (199)
], dtype=np.float64)

# Conexiones del contorno facial para dibujar (subset de 468 pts)
FACE_OVAL_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
    103, 67, 109, 10
]


class LandmarkDetector(BasePipeline):
    """
    Detecta 478 landmarks faciales 3D usando MediaPipe Tasks API (v0.10+).
    Extrae puntos clave para estimación de pose con solvePnP.
    """

    def __init__(self, min_confidence: float = 0.5):
        base_options = mp_python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, image: np.ndarray) -> dict:
        """
        Detecta landmarks faciales en la imagen.
        Returns:
            dict con keys: found, landmarks_2d, landmarks_3d,
                           pose_points_2d, face_3d_model,
                           draw_image, image_size
        """
        self.validate_input(image)
        h, w = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results   = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return {
                "found":          False,
                "landmarks_2d":   None,
                "landmarks_3d":   None,
                "pose_points_2d": None,
                "draw_image":     image.copy()
            }

        face_lm = results.face_landmarks[0]   # lista de NormalizedLandmark

        # Arrays de coordenadas
        landmarks_2d = np.array(
            [[lm.x * w, lm.y * h] for lm in face_lm],
            dtype=np.float64
        )
        landmarks_3d = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face_lm],
            dtype=np.float64
        )

        # Puntos 2D para solvePnP (solo 6 puntos clave)
        pose_points_2d = landmarks_2d[POSE_LANDMARK_IDS]

        # Dibujar landmarks sobre copia
        draw_image = image.copy()
        self._draw_landmarks(draw_image, landmarks_2d)

        return {
            "found":          True,
            "landmarks_2d":   landmarks_2d,
            "landmarks_3d":   landmarks_3d,
            "pose_points_2d": pose_points_2d,
            "face_3d_model":  FACE_3D_MODEL_POINTS,
            "draw_image":     draw_image,
            "image_size":     (w, h)
        }

    def _draw_landmarks(self, image: np.ndarray,
                        landmarks_2d: np.ndarray) -> None:
        """Dibuja puntos y contorno facial manualmente."""
        n_pts = len(landmarks_2d)

        # Puntos individuales
        for pt in landmarks_2d:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(image, (x, y), 1, (62, 207, 142), -1)

        # Contorno facial
        for i in range(len(FACE_OVAL_IDS) - 1):
            idx_a = FACE_OVAL_IDS[i]
            idx_b = FACE_OVAL_IDS[i + 1]
            if idx_a < n_pts and idx_b < n_pts:
                pt_a = tuple(landmarks_2d[idx_a].astype(int))
                pt_b = tuple(landmarks_2d[idx_b].astype(int))
                cv2.line(image, pt_a, pt_b, (92, 124, 255), 1)

        # Resaltar los 6 puntos de pose
        for idx in POSE_LANDMARK_IDS:
            if idx < n_pts:
                pt = tuple(landmarks_2d[idx].astype(int))
                cv2.circle(image, pt, 4, (245, 158, 11), -1)

    def close(self):
        pass

    def __del__(self):
        pass