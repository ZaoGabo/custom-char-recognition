import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.label_map import DEFAULT_LABEL_MAP
from src.trainer import (
    cargar_modelo_guardado,
    entrenar_modelo,
    evaluar_modelo,
    preparar_datos,
)


def _formatear_metricas(metricas: Dict[str, Dict[str, float]]) -> str:
    lineas = []
    for split, valores in metricas.items():
        loss = valores.get('loss', float('nan'))
        acc = valores.get('accuracy', float('nan'))
        lineas.append(f"  {split:5s} | loss={loss:.4f} | acc={acc:.4f}")
    return '\n'.join(lineas)


def _analizar_confusiones(modelo, limite_por_split: int = 5):
    num_clases = DEFAULT_LABEL_MAP.get_num_classes()
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(aplicar_augmentacion=False)

    resultados = {}
    for nombre, X_split, y_split in (
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test),
    ):
        preds = modelo.predecir(X_split)
        matriz = np.zeros((num_clases, num_clases), dtype=np.int32)
        for real, pred in zip(y_split, preds):
            matriz[real, pred] += 1

        per_class = []
        for idx in range(num_clases):
            total = matriz[idx].sum()
            aciertos = matriz[idx, idx]
            if total == 0:
                continue
            precision_clase = aciertos / total
            matriz[idx, idx] = 0
            peor_idx = int(np.argmax(matriz[idx]))
            peor_conteo = matriz[idx, peor_idx]
            matriz[idx, idx] = aciertos
            etiqueta_real = DEFAULT_LABEL_MAP.get_label(idx)
            etiqueta_conf = DEFAULT_LABEL_MAP.get_label(peor_idx)
            per_class.append({
                'label': etiqueta_real,
                'accuracy': precision_clase,
                'worst_confusion': etiqueta_conf,
                'confusions': int(peor_conteo),
                'support': int(total),
            })

        per_class.sort(key=lambda item: item['accuracy'])
        resultados[nombre] = per_class[:limite_por_split]

    return resultados


def main():
    parser = argparse.ArgumentParser(description="Pipeline unificado de entrenamiento y evaluacion")
    parser.add_argument('--force', action='store_true', help='Reentrenar aunque exista un modelo guardado')
    parser.add_argument('--skip-train', action='store_true', help='Omitir entrenamiento y usar el modelo existente')
    parser.add_argument('--verbose', action='store_true', help='Mostrar progreso detallado durante el entrenamiento')
    parser.add_argument('--save-metrics', type=Path, help='Guardar metricas en archivo JSON')
    parser.add_argument('--confusion-report', action='store_true', help='Mostrar analisis de confusiones por clase')
    parser.add_argument('--limit', type=int, default=5, help='Numero de clases con menor precision a mostrar por split')
    args = parser.parse_args()

    if args.skip_train:
        modelo = cargar_modelo_guardado()
        metricas_entrenamiento = None
    else:
        modelo, metricas_entrenamiento = entrenar_modelo(force=args.force, verbose=args.verbose)

    metricas_evaluacion = evaluar_modelo(modelo)

    print('\nMetricas de evaluacion (sin augmentacion):')
    print(_formatear_metricas(metricas_evaluacion))

    if metricas_entrenamiento is not None:
        print('\nMetricas registradas durante el entrenamiento:')
        print(_formatear_metricas(metricas_entrenamiento))

    if args.save_metrics:
        datos = {
            'evaluation': metricas_evaluacion,
            'training': metricas_entrenamiento,
        }
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(json.dumps(datos, indent=2), encoding='utf-8')
        print(f"\nMetricas guardadas en {args.save_metrics}")

    if args.confusion_report:
        analisis = _analizar_confusiones(modelo, args.limit)
        print('\nClases con menor precision (top confusiones):')
        for split, elementos in analisis.items():
            print(f"  [{split}]")
            if not elementos:
                print('    sin datos')
                continue
            for item in elementos:
                label = item['label']
                acc = item['accuracy']
                worst = item['worst_confusion']
                confs = item['confusions']
                soporte = item['support']
                print(
                    f"    {label}: acc={acc:.3f} | peor={worst} ({confs} conf.) | soporte={soporte}"
                )


if __name__ == '__main__':
    main()
