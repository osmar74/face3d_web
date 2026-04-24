from flask import Blueprint

# Blueprint para páginas web (vistas)
main_bp = Blueprint("main", __name__)

# Blueprint para la API REST
api_bp = Blueprint("api", __name__)

# Importar rutas después de crear los blueprints
from . import upload_controller  # noqa: E402, F401