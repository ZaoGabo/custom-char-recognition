"""
Mapeo de etiquetas a índices y viceversa.
"""

import json

class LabelMap:
    def __init__(self, labels=None):
        """
        Inicializar el mapeo de etiquetas.
        
        Args:
            labels (list): Lista de etiquetas únicas (orden importa!)
        """
        if labels is None:
            # Etiquetas por defecto (A-Z)
            labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # NO ordenar - mantener el orden original que coincide con el modelo
        self.labels = list(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.labels)}
        
    def get_index(self, label):
        """Obtener índice de una etiqueta."""
        return self.label_to_index.get(label, -1)
    
    def get_label(self, index):
        """Obtener etiqueta de un índice."""
        return self.index_to_label.get(index, "Unknown")
    
    def get_num_classes(self):
        """Obtener número total de clases."""
        return len(self.labels)
    
    def save(self, filepath):
        """Guardar mapeo en archivo JSON."""
        data = {
            'labels': self.labels,
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Cargar mapeo desde archivo JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        label_map = cls(data['labels'])
        return label_map
    
    def __str__(self):
        return f"LabelMap({self.get_num_classes()} classes: {self.labels})"
    
    def __repr__(self):
        return self.__str__()

# Instancia global por defecto
from .config import CUSTOM_LABELS
DEFAULT_LABEL_MAP = LabelMap(CUSTOM_LABELS)

def label_to_index(label):
    """Función auxiliar para obtener el índice de una etiqueta usando el mapa por defecto."""
    return DEFAULT_LABEL_MAP.get_index(label)

