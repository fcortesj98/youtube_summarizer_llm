"""Entry point: python app.py"""

from ytbot.config import settings
from ytbot.ui import build_interface

if __name__ == "__main__":
    build_interface().launch(server_name=settings.server_name, server_port=settings.server_port)
