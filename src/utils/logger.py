"""
Logger configurado con loguru para logging estructurado.
Reemplaza los print() dispersos por logging profesional.
"""
from loguru import logger
import sys
import os
from pathlib import Path

def setup_logger(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Configura el logger global de la aplicación.
    
    Args:
        log_level: Nivel de log (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directorio donde guardar logs
    
    Returns:
        Logger configurado
    """
    # Remover configuración por defecto
    logger.remove()
    
    # Console output con colores
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File output (rotativo)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / "app_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info(f"Logger configurado - Nivel: {log_level}")
    return logger

# Crear instancia global
app_logger = setup_logger(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs")
)
