import os
from pathlib import Path


class Config:
    """Configuración central de la aplicación."""

    # Rutas base
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER: Path = BASE_DIR / "uploads"
    OUTPUT_FOLDER: Path = BASE_DIR / "outputs"
    MODELS_FOLDER: Path = BASE_DIR / "models_weights"

    # Flask
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    # Imágenes permitidas
    ALLOWED_EXTENSIONS: set = {"png", "jpg", "jpeg", "webp"}

    # Parámetros de procesamiento
    IMAGE_MAX_DIMENSION: int = 1024   # redimensiona si es mayor
    FACE_MIN_CONFIDENCE: float = 0.5

    @classmethod
    def init_app(cls, app) -> None:
        """Crea carpetas necesarias si no existen."""
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.MODELS_FOLDER.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG: bool = True


class ProductionConfig(Config):
    DEBUG: bool = False


config_by_name: dict = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}