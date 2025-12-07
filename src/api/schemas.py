"""
Esquemas de Datos (Pydantic Models)

Define las estructuras de datos para validación de entrada/salida en la API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Esquema para solicitud de predicción de caracter único."""
    image: List[float] = Field(
        ..., 
        description="Array aplanado de la imagen. Valores normalizados [0, 1].",
        example=[0.0] * 784
    )
    width: Optional[int] = Field(None, description="Ancho de la imagen (requerido si no es cuadrada, opcional)")
    height: Optional[int] = Field(None, description="Alto de la imagen (requerido si no es cuadrada, opcional)")


class Top5Prediction(BaseModel):
    """Detalle de una predicción individual en el top 5."""
    character: str = Field(..., description="Caracter predicho")
    probability: float = Field(..., description="Probabilidad (0-1)")


class PredictionResponse(BaseModel):
    """Respuesta estándar para predicción de caracter."""
    character: str = Field(..., description="Caracter con mayor probabilidad")
    confidence: float = Field(..., description="Nivel de confianza global (0-1)")
    top5: List[Top5Prediction] = Field(..., description="Lista de las 5 mejores predicciones")


class PredictionTextResponse(BaseModel):
    """Respuesta para predicción de texto (segmentado). Obsoleto pero mantenido por compatibilidad."""
    text: str = Field(..., description="Texto completo predicho")
    segments: List[PredictionResponse] = Field(default=[], description="Detalle por cada segmento/caracter")


class TrOCRTextResponse(BaseModel):
    """Respuesta para predicción de texto completo con TrOCR."""
    text: str = Field(..., description="Texto reconocido de la imagen")
    confidence: float = Field(..., description="Confianza promedio del reconocimiento")


class ImageBase64Request(BaseModel):
    """Solicitud genérica para imágenes en base64."""
    image_base64: str = Field(..., description="String de imagen en Base64")
