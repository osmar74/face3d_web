from flask import Flask
from .config import config_by_name, Config


def create_app(config_name: str = "default") -> Flask:
    """Factory function — crea e inicializa la app Flask."""
    app = Flask(__name__)

    # Cargar configuración
    cfg = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(cfg)
    cfg.init_app(app)

    # Registrar blueprints
    from .controllers import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app