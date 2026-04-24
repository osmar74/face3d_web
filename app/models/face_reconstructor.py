import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # sin ventana de GUI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from io import BytesIO
from .base_pipeline import BasePipeline


class FaceReconstructor3D(BasePipeline):
    """
    Construye la malla 3D del rostro y genera visualizaciones:
    - Nube de puntos 3D (como imagen 3 de referencia)
    - Datos del mesh para Three.js
    """

    def process(self, image: np.ndarray) -> dict:
        self.validate_input(image)
        return {"image": image}

    def reconstruct(self, landmarks_3d: np.ndarray,
                    image: np.ndarray) -> dict:
        """
        Genera la nube de puntos 3D y el mesh del rostro.
        Args:
            landmarks_3d : array (N, 3)
            image        : imagen original BGR
        Returns:
            dict con keys: pointcloud_img_bgr, mesh_data,
                           landmarks_normalized
        """
        h, w = image.shape[:2]

        # Normalizar puntos al rango [0, 1]
        pts = landmarks_3d.copy().astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        depth_range = pts[:, 2].max() - pts[:, 2].min()
        if depth_range > 1e-8:
            pts[:, 2] = (pts[:, 2] - pts[:, 2].min()) / depth_range

        # Generar imagen de nube de puntos (matplotlib 3D)
        pointcloud_img = self._render_pointcloud(pts, image)

        # Datos del mesh para Three.js
        mesh_data = self._build_mesh_data(landmarks_3d, w, h)

        return {
            "pointcloud_img_bgr":  pointcloud_img,
            "mesh_data":           mesh_data,
            "landmarks_normalized": pts.tolist()
        }

    def _render_pointcloud(self, pts_norm: np.ndarray,
                           image: np.ndarray) -> np.ndarray:
        """Renderiza nube de puntos 3D como imagen (estilo imagen 3)."""
        fig = plt.figure(figsize=(6, 6), facecolor="#0d0f14")
        ax  = fig.add_subplot(111, projection="3d",
                              facecolor="#0d0f14")

        xs = pts_norm[:, 0]
        ys = pts_norm[:, 1]
        zs = pts_norm[:, 2]

        # Color por profundidad (igual que imagen de referencia)
        colors = plt.cm.rainbow(zs)

        ax.scatter(xs, ys, zs,
                   c=colors, s=1.5, alpha=0.8, linewidths=0)

        # Imagen original como textura del plano base
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (64, 64))
        ax.imshow(img_rgb,
                  extent=[0, 1, 0, 1],
                  origin="upper",
                  aspect="auto",
                  zorder=0,
                  alpha=0.6,
                  transform=ax.transData)

        # Estilo oscuro
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#2e3350")

        ax.tick_params(colors="#5a6080", labelsize=7)
        ax.set_xlabel("x", color="#5a6080", fontsize=8)
        ax.set_ylabel("y", color="#5a6080", fontsize=8)
        ax.set_zlabel("z", color="#5a6080", fontsize=8)
        ax.set_title("Nube de puntos 3D", color="#9ba3c4",
                     fontsize=10, pad=10)
        ax.view_init(elev=20, azim=-60)

        # Convertir figura a imagen BGR
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        plt.close(fig)
        return img_bgr

    def _build_mesh_data(self, landmarks_3d: np.ndarray,
                         img_w: int, img_h: int) -> dict:
        """
        Construye datos del mesh en formato compatible con Three.js.
        Usa Delaunay 2D proyectado sobre los landmarks.
        """
        from scipy.spatial import Delaunay

        pts2d = landmarks_3d[:, :2].copy()
        pts2d[:, 0] = (pts2d[:, 0] / img_w) * 2 - 1   # [-1, 1]
        pts2d[:, 1] = -((pts2d[:, 1] / img_h) * 2 - 1) # invertir Y

        depth = landmarks_3d[:, 2].copy()
        d_min, d_max = depth.min(), depth.max()
        d_rng = d_max - d_min if d_max != d_min else 1.0
        depth_norm = ((depth - d_min) / d_rng) * 0.5  # Z escalado

        tri = Delaunay(pts2d)

        vertices = np.column_stack([pts2d, depth_norm]).tolist()
        faces    = tri.simplices.tolist()

        return {
            "vertices": vertices,
            "faces":    faces,
            "vertex_count": len(vertices),
            "face_count":   len(faces)
        }