import cv2
import base64
import numpy as np


class ResponseSerializer:
    """
    Convierte resultados del pipeline a formato JSON
    serializable para el frontend.
    """

    @staticmethod
    def image_to_base64(image_bgr: np.ndarray,
                        fmt: str = ".jpg") -> str:
        """Convierte imagen BGR a string base64."""
        success, buffer = cv2.imencode(
            fmt, image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not success:
            raise ValueError("No se pudo codificar la imagen a base64")
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def build_response(detection: dict,
                       landmarks: dict,
                       normal_map: dict,
                       reconstruction: dict,
                       pose: dict,
                       original_image: np.ndarray) -> dict:
        """
        Construye el JSON completo de respuesta para el frontend.
        """
        ser = ResponseSerializer

        response = {
            "status": "ok",
            "detection": {
                "found":      detection.get("found", False),
                "confidence": detection.get("confidence", 0),
                "bbox":       detection.get("bbox"),
                "image_b64":  ser.image_to_base64(
                                  detection["draw_image"])
            },
            "landmarks": {
                "count":     len(landmarks["landmarks_2d"])
                             if landmarks.get("landmarks_2d") is not None
                             else 0,
                "image_b64": ser.image_to_base64(
                                 landmarks["draw_image"])
            },
            "normal_map": {
                "map_b64":     ser.image_to_base64(
                                   normal_map["normal_map_bgr"]),
                "overlay_b64": ser.image_to_base64(
                                   normal_map["normal_map_overlay"])
            },
            "reconstruction": {
                "pointcloud_b64": ser.image_to_base64(
                                      reconstruction["pointcloud_img_bgr"]),
                "mesh": {
                    "vertex_count": reconstruction["mesh_data"]["vertex_count"],
                    "face_count":   reconstruction["mesh_data"]["face_count"],
                    "vertices":     reconstruction["mesh_data"]["vertices"],
                    "faces":        reconstruction["mesh_data"]["faces"]
                }
            },
            "pose": {
                "success":      pose.get("success", False),
                "euler_angles": pose.get("euler_angles", {}),
                "axes_b64":     ser.image_to_base64(
                                    pose["pose_image_bgr"])
                                if pose.get("success") else None,
                "ellipses_b64": ser.image_to_base64(
                                    pose["pose_3d_img_bgr"])
                                if pose.get("success") else None,
            }
        }
        return response