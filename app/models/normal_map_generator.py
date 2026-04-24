import cv2
import numpy as np
from .base_pipeline import BasePipeline


class NormalMapGenerator(BasePipeline):
    """
    Genera un mapa de normales de superficie del rostro a partir
    de los landmarks 3D. Cada normal se codifica como color RGB
    donde R=X, G=Y, B=Z (igual a la imagen 2 de referencia).
    """

    def __init__(self, output_size: int = 256):
        self._output_size = output_size

    def process(self, image: np.ndarray) -> dict:
        """No aplica — usar generate() directamente."""
        self.validate_input(image)
        return {"image": image}

    def generate(self, landmarks_3d: np.ndarray,
                 image: np.ndarray) -> dict:
        """
        Genera el normal map a partir de landmarks 3D.
        Args:
            landmarks_3d : array (N, 3) con coordenadas XYZ
            image        : imagen original BGR
        Returns:
            dict con keys: normal_map_bgr, normal_map_overlay,
                           normals_array
        """
        h, w = image.shape[:2]
        size  = self._output_size

        # Normalizar landmarks al rango [-1, 1]
        pts = landmarks_3d.copy().astype(np.float64)
        for i in range(3):
            col_min, col_max = pts[:, i].min(), pts[:, i].max()
            rng = col_max - col_min if col_max != col_min else 1.0
            pts[:, i] = 2.0 * (pts[:, i] - col_min) / rng - 1.0

        # Calcular normal por cada landmark usando vecinos
        normals = self._compute_normals(pts)

        # Crear imagen de normal map (RGB)
        normal_map = np.zeros((size, size, 3), dtype=np.uint8)
        pts_px = landmarks_3d[:, :2].copy()
        pts_px[:, 0] = (pts_px[:, 0] / w) * (size - 1)
        pts_px[:, 1] = (pts_px[:, 1] / h) * (size - 1)
        pts_px = pts_px.astype(int)
        pts_px = np.clip(pts_px, 0, size - 1)

        for i, (px, py) in enumerate(pts_px):
            r = int((normals[i, 0] * 0.5 + 0.5) * 255)
            g = int((normals[i, 1] * 0.5 + 0.5) * 255)
            b = int((normals[i, 2] * 0.5 + 0.5) * 255)
            cv2.circle(normal_map, (px, py), 2, (b, g, r), -1)

        # Suavizar el mapa
        normal_map = cv2.GaussianBlur(normal_map, (5, 5), 0)

        # Rellenar huecos con inpaint
        gray  = cv2.cvtColor(normal_map, cv2.COLOR_BGR2GRAY)
        mask  = (gray == 0).astype(np.uint8) * 255
        if mask.any():
            normal_map = cv2.inpaint(normal_map, mask, 3,
                                     cv2.INPAINT_TELEA)

        # Overlay sobre imagen original redimensionada
        img_resized = cv2.resize(image, (size, size))
        overlay     = cv2.addWeighted(img_resized, 0.35,
                                      normal_map,   0.65, 0)

        return {
            "normal_map_bgr":     normal_map,
            "normal_map_overlay": overlay,
            "normals_array":      normals
        }

    def _compute_normals(self, pts: np.ndarray) -> np.ndarray:
        """Calcula normales aproximadas por cada punto usando
        diferencias finitas con sus vecinos más cercanos."""
        from scipy.spatial import KDTree

        tree   = KDTree(pts[:, :2])
        normals = np.zeros((len(pts), 3), dtype=np.float64)

        for i, pt in enumerate(pts):
            _, idxs = tree.query(pt[:2], k=min(6, len(pts)))
            neighbors = pts[idxs[1:]]          # excluir el propio punto
            if len(neighbors) < 2:
                normals[i] = [0.0, 0.0, 1.0]
                continue
            # Vectores desde el punto a sus vecinos
            v1 = neighbors[0] - pt
            v2 = neighbors[1] - pt
            n  = np.cross(v1, v2)
            nm = np.linalg.norm(n)
            normals[i] = n / nm if nm > 1e-8 else np.array([0.0, 0.0, 1.0])
            # Asegurar que la normal apunte hacia la cámara (Z positivo)
            if normals[i, 2] < 0:
                normals[i] = -normals[i]

        return normals