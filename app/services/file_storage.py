import cv2
import uuid
import numpy as np
from pathlib import Path
from werkzeug.datastructures import FileStorage


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_DIMENSION      = 1024   # px — redimensiona si es mayor


class FileStorageService:
    """
    Gestiona lectura, validación y guardado de imágenes.
    """

    def __init__(self, upload_dir: Path, output_dir: Path):
        self._upload_dir = Path(upload_dir)
        self._output_dir = Path(output_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validación ────────────────────────────────────────────────

    def is_allowed(self, filename: str) -> bool:
        ext = Path(filename).suffix.lstrip(".").lower()
        return ext in ALLOWED_EXTENSIONS

    def validate_file(self, file: FileStorage) -> tuple[bool, str]:
        """
        Valida el archivo subido.
        Returns:
            (True, "") si es válido
            (False, mensaje_error) si no lo es
        """
        if not file or file.filename == "":
            return False, "No se seleccionó ningún archivo"
        if not self.is_allowed(file.filename):
            return False, (
                f"Extensión no permitida. "
                f"Usa: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        return True, ""

    # ── Lectura ───────────────────────────────────────────────────

    def save_and_read(self, file: FileStorage) -> tuple[np.ndarray, Path]:
        """
        Guarda el archivo en disco y lo retorna como numpy array BGR.
        Returns:
            (imagen_bgr, ruta_guardada)
        """
        ext      = Path(file.filename).suffix.lower()
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = self._upload_dir / filename
        file.save(str(save_path))

        image = cv2.imread(str(save_path))
        if image is None:
            raise ValueError(
                f"No se pudo decodificar la imagen: {filename}"
            )

        # Redimensionar si excede el máximo
        image = self._resize_if_needed(image)
        return image, save_path

    def read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"No se encontró: {path}")
        return self._resize_if_needed(img)

    # ── Guardado de resultados ────────────────────────────────────

    def save_result(self, image_bgr: np.ndarray,
                    name: str) -> Path:
        """Guarda una imagen de resultado en outputs/."""
        out_path = self._output_dir / name
        cv2.imwrite(str(out_path), image_bgr)
        return out_path

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resize_if_needed(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= MAX_DIMENSION:
            return image
        scale = MAX_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h),
                          interpolation=cv2.INTER_AREA)