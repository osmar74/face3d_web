import os
from pathlib import Path


class Config:
    BASE_DIR:             Path = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER:        str  = str(Path(__file__).resolve().parent.parent / "uploads")
    OUTPUT_FOLDER:        str  = str(Path(__file__).resolve().parent.parent / "outputs")
    MODELS_FOLDER:        Path = Path(__file__).resolve().parent.parent / "models_weights"
    SECRET_KEY:           str  = os.getenv("SECRET_KEY", "dev-secret-key")
    MAX_CONTENT_LENGTH:   int  = int(os.getenv("MAX_CONTENT_LENGTH",
                                               16 * 1024 * 1024))
    ALLOWED_EXTENSIONS:   set  = {"png", "jpg", "jpeg", "webp"}
    IMAGE_MAX_DIMENSION:  int  = 1024
    FACE_MIN_CONFIDENCE:  float = 0.5

    @classmethod
    def init_app(cls, app) -> None:
        Path(cls.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        cls.MODELS_FOLDER.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG: bool = True


class ProductionConfig(Config):
    DEBUG: bool = False


config_by_name: dict = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "default":     DevelopmentConfig,
}