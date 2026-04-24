import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from io import BytesIO
from .base_pipeline import BasePipeline


class PoseEstimator(BasePipeline):
    """
    Estima la pose de la cabeza usando solvePnP (OpenCV).
    Genera imagen con elipses de orientación (estilo imagen 4).
    """

    def process(self, image: np.ndarray) -> dict:
        self.validate_input(image)
        return {"image": image}

    def estimate(self, pose_points_2d: np.ndarray,
                 face_3d_model:    np.ndarray,
                 image:            np.ndarray) -> dict:
        """
        Estima yaw, pitch, roll y genera visualización con elipses.
        Args:
            pose_points_2d : array (6, 2) puntos 2D en imagen
            face_3d_model  : array (6, 3) modelo 3D de referencia
            image          : imagen BGR original
        Returns:
            dict con keys: success, rotation_vec, translation_vec,
                           euler_angles, pose_image_bgr,
                           pose_3d_img_bgr
        """
        self.validate_input(image)
        h, w = image.shape[:2]

        # Matriz de cámara aproximada (cámara pinhole)
        focal   = w
        cx, cy  = w / 2.0, h / 2.0
        cam_mat = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            face_3d_model,
            pose_points_2d.astype(np.float64),
            cam_mat,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"success": False}

        # Convertir a ángulos de Euler (yaw, pitch, roll)
        rmat, _ = cv2.Rodrigues(rvec)
        euler   = self._rotation_matrix_to_euler(rmat)
        yaw, pitch, roll = euler

        # Imagen con ejes de pose sobre el rostro
        pose_img  = self._draw_pose_axes(image.copy(),
                                         rvec, tvec,
                                         cam_mat, dist_coeffs)

        # Imagen 3D con elipses de orientación (estilo imagen 4)
        pose_3d   = self._draw_3d_ellipses(image, rvec, tvec)

        return {
            "success":         True,
            "rotation_vec":    rvec.flatten().tolist(),
            "translation_vec": tvec.flatten().tolist(),
            "euler_angles": {
                "yaw":   round(float(yaw),   2),
                "pitch": round(float(pitch), 2),
                "roll":  round(float(roll),  2)
            },
            "pose_image_bgr":  pose_img,
            "pose_3d_img_bgr": pose_3d
        }

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> tuple:
        """Convierte matriz de rotación 3x3 a ángulos Euler en grados."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2( R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2( R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0

        return (np.degrees(y),   # yaw
                np.degrees(x),   # pitch
                np.degrees(z))   # roll

    def _draw_pose_axes(self, image: np.ndarray,
                        rvec, tvec, cam_mat,
                        dist_coeffs) -> np.ndarray:
        """Dibuja ejes XYZ de orientación sobre el rostro."""
        axis_len = 80.0
        axis_pts = np.float32([
            [axis_len, 0,        0       ],
            [0,        axis_len, 0       ],
            [0,        0,       -axis_len],
            [0,        0,        0       ]
        ])
        img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec,
                                        cam_mat, dist_coeffs)
        img_pts = img_pts.reshape(-1, 2).astype(int)
        origin  = img_pts[3]

        cv2.arrowedLine(image, origin, img_pts[0], (0,   0,   220), 3)  # X rojo
        cv2.arrowedLine(image, origin, img_pts[1], (0,   220, 0  ), 3)  # Y verde
        cv2.arrowedLine(image, origin, img_pts[2], (220, 0,   0  ), 3)  # Z azul

        h, w = image.shape[:2]
        txt  = f"Yaw:{self._get_angle(rvec)[0]:.0f} " \
               f"Pitch:{self._get_angle(rvec)[1]:.0f} " \
               f"Roll:{self._get_angle(rvec)[2]:.0f}"
        cv2.putText(image, txt, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (92, 124, 255), 1)
        return image

    def _get_angle(self, rvec: np.ndarray) -> tuple:
        rmat, _ = cv2.Rodrigues(rvec)
        return self._rotation_matrix_to_euler(rmat)

    def _draw_3d_ellipses(self, image: np.ndarray,
                           rvec: np.ndarray,
                           tvec: np.ndarray) -> np.ndarray:
        """
        Genera visualización 3D con elipses de orientación
        (estilo imagen 4 de referencia).
        """
        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_matrix_to_euler(rmat)

        fig = plt.figure(figsize=(8, 4), facecolor="#1c2030")
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for idx, (az, el, title) in enumerate([(-60, 15, "Vista 1"),
                                                (-30, 20, "Vista 2")]):
            ax = fig.add_subplot(1, 2, idx + 1,
                                 projection="3d",
                                 facecolor="#1c2030")

            # Renderizar imagen del rostro como textura
            img_s = cv2.resize(img_rgb, (80, 80))
            h_s, w_s = img_s.shape[:2]
            x_range = np.linspace(-1, 1, w_s)
            y_range = np.linspace(-1, 1, h_s)
            Xg, Yg  = np.meshgrid(x_range, y_range)
            Zg      = np.zeros_like(Xg)
            ax.plot_surface(Xg, Yg, Zg,
                            rstride=1, cstride=1,
                            facecolors=img_s / 255.0,
                            shade=False, alpha=0.85)

            # Dibujar elipses de orientación
            theta = np.linspace(0, 2 * np.pi, 60)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            scale = 1.6

            # Elipse Yaw (plano XZ)
            r_yaw  = np.radians(yaw)
            xe = scale * cos_t
            ze = scale * sin_t
            ye = np.zeros_like(theta)
            ax.plot(xe, ye, ze, color="#5c7cff",
                    linewidth=1.2, alpha=0.85)

            # Elipse Pitch (plano YZ)
            ye2 = scale * cos_t
            ze2 = scale * sin_t
            xe2 = np.zeros_like(theta)
            ax.plot(xe2, ye2, ze2, color="#3ecf8e",
                    linewidth=1.2, alpha=0.85)

            # Elipse Roll (plano XY)
            xe3 = scale * cos_t
            ye3 = scale * sin_t
            ze3 = np.zeros_like(theta)
            ax.plot(xe3, ye3, ze3, color="#f59e0b",
                    linewidth=1.2, alpha=0.85)

            # Estilo
            for pane in [ax.xaxis.pane,
                         ax.yaxis.pane,
                         ax.zaxis.pane]:
                pane.fill      = False
                pane.set_edgecolor("#2e3350")

            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.tick_params(colors="#5a6080", labelsize=6)
            ax.set_title(f"Y:{yaw:.0f}° P:{pitch:.0f}° R:{roll:.0f}°",
                         color="#9ba3c4", fontsize=8)
            ax.view_init(elev=el, azim=az + yaw * 0.3)

        plt.tight_layout(pad=0.5)
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        plt.close(fig)
        return img_bgr