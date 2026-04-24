import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from .base_pipeline import BasePipeline

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / \
             "models_weights" / "blaze_face_short_range.tflite"


class FaceDetector(BasePipeline):
    """
    Detecta el rostro en una imagen usando MediaPipe Tasks API (v0.10+).
    Retorna bounding box y región recortada del rostro.
    """

    def __init__(self, min_confidence: float = 0.5):
        self._min_confidence = min_confidence
        base_options = mp_python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def process(self, image: np.ndarray) -> dict:
        """
        Detecta el primer rostro encontrado en la imagen.
        Returns:
            dict con keys: found, bbox, face_crop, draw_image, confidence
        """
        self.validate_input(image)
        h, w = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results   = self._detector.detect(mp_image)

        if not results.detections:
            return {
                "found":      False,
                "bbox":       None,
                "face_crop":  None,
                "draw_image": image.copy()
            }

        # Primera detección
        detection = results.detections[0]
        box = detection.bounding_box          # ya en píxeles

        x  = max(box.origin_x, 0)
        y  = max(box.origin_y, 0)
        bw = box.width
        bh = box.height
        x2 = min(x + bw, w)
        y2 = min(y + bh, h)

        conf = detection.categories[0].score if detection.categories else 0.0
        bbox = {"x": x, "y": y, "w": bw, "h": bh, "x2": x2, "y2": y2}

        face_crop  = self._crop_with_margin(image, bbox, margin=0.20)
        draw_image = image.copy()

        # Dibujar bounding box
        cv2.rectangle(draw_image, (x, y), (x2, y2), (92, 124, 255), 2)
        label = f"Rostro: {conf:.0%}"
        cv2.putText(draw_image, label, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (92, 124, 255), 2)

        return {
            "found":      True,
            "bbox":       bbox,
            "face_crop":  face_crop,
            "draw_image": draw_image,
            "confidence": float(conf)
        }

    def _crop_with_margin(self, image: np.ndarray,
                          bbox: dict, margin: float = 0.20) -> np.ndarray:
        h, w = image.shape[:2]
        mx = int(bbox["w"] * margin)
        my = int(bbox["h"] * margin)
        x1 = max(bbox["x"] - mx, 0)
        y1 = max(bbox["y"] - my, 0)
        x2 = min(bbox["x2"] + mx, w)
        y2 = min(bbox["y2"] + my, h)
        return image[y1:y2, x1:x2].copy()

    def close(self):
        pass

    def __del__(self):
        pass