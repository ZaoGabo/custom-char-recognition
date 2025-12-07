"""
Mapeo de Etiquetas (Label Map)

Gestiona la conversión entre índices numéricos (usados por el modelo)
y etiquetas legibles (caracteres).
"""
import json
from typing import List, Optional, Dict
from .config import CUSTOM_LABELS


class LabelMap:
    """Clase para mapear etiquetas a índices y viceversa."""
    
    def __init__(self, labels: Optional[List[str]] = None):
        """
        Inicializa el mapeo.
        
        Args:
            labels: Lista de etiquetas. Si es None, usa A-Z por defecto.
                   IMPORTANTE: El orden debe coincidir con el entrenamiento del modelo.
        """
        if labels is None:
            # Etiquetas por defecto (A-Z) para casos de prueba
            labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Mantener orden original estrictamente
        self.labels = list(labels)
        self.label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label: Dict[int, str] = {idx: label for idx, label in enumerate(self.labels)}
        
    def get_index(self, label: str) -> int:
        """Retorna el índice numérico para una etiqueta dada."""
        return self.label_to_index.get(label, -1)
    
    def get_label(self, index: int) -> str:
        """Retorna la etiqueta legible para un índice dado."""
        return self.index_to_label.get(index, "Unknown")
    
    def get_num_classes(self) -> int:
        """Retorna el número total de clases."""
        return len(self.labels)
    
    def save(self, filepath: str) -> None:
        """Guarda el mapeo actual en un archivo JSON."""
        data = {
            'labels': self.labels,
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'LabelMap':
        """Carga un mapeo desde un archivo JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data['labels'])
    
    def __str__(self) -> str:
        return f"LabelMap({self.get_num_classes()} clases)"
    
    def __repr__(self) -> str:
        return self.__str__()


# Instancia global por defecto usando etiquetas configuradas
DEFAULT_LABEL_MAP = LabelMap(CUSTOM_LABELS)

def label_to_index(label: str) -> int:
    """Helper para obtener índice usando el mapa por defecto."""
    return DEFAULT_LABEL_MAP.get_index(label)
