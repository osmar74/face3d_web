from abc import ABC, abstractmethod
import numpy as np


class BasePipeline(ABC):
    """Clase base abstracta para todos los modelos del pipeline."""

    @abstractmethod
    def process(self, image: np.ndarray) -> dict:
        """
        Procesa una imagen y retorna resultados.
        Args:
            image: imagen BGR como numpy array (OpenCV format)
        Returns:
            dict con los resultados del procesamiento
        """
        pass

    @staticmethod
    def validate_input(image: np.ndarray) -> bool:
        """Valida que la imagen sea un numpy array BGR válido."""
        if image is None:
            raise ValueError("La imagen no puede ser None")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Se esperaba numpy.ndarray, se recibió {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Se esperaba imagen BGR (H,W,3), forma recibida: {image.shape}")
        if image.size == 0:
            raise ValueError("La imagen está vacía")
        return True