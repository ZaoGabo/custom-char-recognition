"""Flujo central de entrenamiento para el modelo CNN v2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from ..cnn_model_v2 import crear_modelo_cnn_v2
from ..config import CUSTOM_LABELS, DATA_CONFIG, NETWORK_CONFIG, PATHS
from ..data_loader import DataLoader, _leer_imagen_gris
from ..label_map import DEFAULT_LABEL_MAP, LabelMap
from ..utils import normalize_image

LOGGER = logging.getLogger(__name__)


def _one_hot(labels: np.ndarray, num_clases: int) -> np.ndarray:
    """Codificar ``labels`` en formato one-hot."""
    if num_clases <= 0:
        raise ValueError("num_clases debe ser positivo")
    vector = np.zeros((labels.shape[0], num_clases), dtype=np.float32)
    vector[np.arange(labels.shape[0]), labels] = 1.0
    return vector


def _actualizar_tasa_aprendizaje(
    epoca: int,
    base_lr: Optional[float] = None,
    scheduler_config: Optional[Dict[str, float]] = None,
) -> float:
    """Calcular la tasa de aprendizaje para ``epoca`` usando configuracion step-decay."""
    if epoca < 0:
        raise ValueError("La epoca no puede ser negativa")

    config = scheduler_config or NETWORK_CONFIG.get("lr_scheduler_config", {})
    tasa_base = base_lr if base_lr is not None else NETWORK_CONFIG.get("tasa_aprendizaje", 0.001)

    if not config:
        return tasa_base

    tipo = config.get("tipo", "step_decay")
    if tipo != "step_decay":
        return tasa_base

    tasa_decaimiento = float(config.get("tasa_decaimento", 0.1))
    epocas_decaimiento = int(config.get("epocas_decaimento", 50))
    pasos = epoca // max(epocas_decaimiento, 1)
    return tasa_base * (tasa_decaimiento ** pasos)


def _select_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Seleccionar dispositivo de ejecucion."""
    has_mps = hasattr(torch.backends, "mps") and getattr(torch.backends.mps, "is_available", lambda: False)()

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        device = device.lower()
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA no esta disponible en este entorno")
            return torch.device("cuda")
        if device in {"cpu", "mps"}:
            if device == "mps" and not has_mps:
                raise RuntimeError("MPS no esta disponible")
            return torch.device(device)
        raise ValueError(f"Dispositivo no reconocido: {device}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _cargar_dataset(
    rutas: Sequence[str],
    etiquetas: Sequence[int],
    tamano_imagen: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Leer imagenes desde ``rutas`` y devolver tensores normalizados."""
    if not rutas:
        raise ValueError("No se proporcionaron rutas de imagen para cargar")

    imagenes: List[np.ndarray] = []
    for ruta in rutas:
        imagen = _leer_imagen_gris(ruta, tamano_imagen)
        imagenes.append(imagen)

    matriz = np.stack(imagenes, axis=0)
    matriz = normalize_image(matriz)
    matriz = matriz.reshape(matriz.shape[0], 1, tamano_imagen[0], tamano_imagen[1])
    return matriz.astype(np.float32), np.asarray(etiquetas, dtype=np.int64)


def _crear_torch_dataloader(
    datos: np.ndarray,
    etiquetas: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> TorchDataLoader:
    """Crear ``DataLoader`` de PyTorch a partir de arrays de NumPy."""
    tensores_X = torch.from_numpy(datos)
    tensores_y = torch.from_numpy(etiquetas)
    dataset = TensorDataset(tensores_X, tensores_y)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _has_image_files(path: Path) -> bool:
    if not path.exists():
        return False
    for child in path.iterdir():
        if child.name.startswith('.'):
            continue
        if child.is_dir() and any(child.glob('*.png')):
            return True
        if child.is_file() and child.suffix.lower() == '.png':
            return True
    return False


def _train_loop(
    model: torch.nn.Module,
    train_loader: TorchDataLoader,
    val_loader: Optional[TorchDataLoader],
    device: torch.device,
    epocas: int,
    base_lr: float,
    scheduler_config: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[Dict[str, Optional[float]]]:
    """Realizar el bucle de entrenamiento principal."""
    criterio = torch.nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=NETWORK_CONFIG.get("lambda_l2", 0.0))

    scheduler = None
    if scheduler_config and scheduler_config.get("tipo") == "step_decay":
        paso = int(scheduler_config.get("epocas_decaimento", 50))
        gamma = float(scheduler_config.get("tasa_decaimento", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizador, step_size=max(paso, 1), gamma=gamma)

    historia: List[Dict[str, Optional[float]]] = []

    for epoca in range(epocas):
        model.train()
        acumulado_loss = 0.0
        total_muestras = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizador.zero_grad()
            logits = model(batch_X)
            loss = criterio(logits, batch_y)
            loss.backward()
            optimizador.step()

            acumulado_loss += loss.item() * batch_X.size(0)
            total_muestras += batch_X.size(0)

        loss_train = acumulado_loss / max(total_muestras, 1)
        loss_val: Optional[float] = None
        acc_val: Optional[float] = None

        if val_loader is not None:
            model.eval()
            total_val = 0
            correctos = 0
            acumulado_val = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_X)
                    loss = criterio(logits, batch_y)
                    acumulado_val += loss.item() * batch_X.size(0)
                    predicciones = logits.argmax(dim=1)
                    correctos += (predicciones == batch_y).sum().item()
                    total_val += batch_X.size(0)
            loss_val = acumulado_val / max(total_val, 1)
            acc_val = correctos / max(total_val, 1)

        if scheduler is not None:
            scheduler.step()

        lr_actual = float(optimizador.param_groups[0]["lr"])
        historia.append(
            {
                "epoch": epoca + 1,
                "loss_train": loss_train,
                "loss_val": loss_val,
                "acc_val": acc_val,
                "lr": lr_actual,
            }
        )

        if verbose:
            mensaje = f"Epoca {epoca + 1}/{epocas} - loss: {loss_train:.4f}"
            if loss_val is not None and acc_val is not None:
                mensaje += f" - val_loss: {loss_val:.4f} - val_acc: {acc_val:.4f}"
            mensaje += f" - lr: {lr_actual:.6f}"
            LOGGER.info(mensaje)

    return historia


def _guardar_modelo(
    model: torch.nn.Module,
    output_dir: Path,
    label_map: LabelMap,
    history: List[Dict[str, Optional[float]]],
) -> Path:
    """Persistir el modelo entrenado y metadatos basicos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "cnn_model_v2.pt"
    torch.save({"state_dict": model.state_dict(), "labels": label_map.labels}, checkpoint_path)

    metadata_path = output_dir / "training_info.json"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "labels": label_map.labels,
                "num_classes": label_map.get_num_classes(),
                "history": history,
            },
            metadata_file,
            indent=2,
        )
    return checkpoint_path


def entrenar_modelo(
    force: bool = False,
    verbose: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    max_epochs: Optional[int] = None,
    model_dir_name: str = "cnn_modelo_v2",
    data_dir: Optional[Union[str, Path]] = None,
    generar_dataset: bool = False,
    max_muestras_por_clase: Optional[int] = None,
) -> Path:
    """Entrenar la CNN principal y guardar pesos y metadatos."""
    label_map = DEFAULT_LABEL_MAP if DEFAULT_LABEL_MAP.get_num_classes() == len(CUSTOM_LABELS) else LabelMap(CUSTOM_LABELS)
    modelos_root = Path(PATHS.get("modelos", "models"))
    output_dir = modelos_root / model_dir_name

    if output_dir.exists() and not force:
        LOGGER.info("El modelo ya existe en %s; usa --force para reentrenar", output_dir)
        return output_dir

    data_dir_path = Path(data_dir or PATHS.get("datos_procesados", "data/processed"))
    data_dir_path.mkdir(parents=True, exist_ok=True)

    if generar_dataset and not _has_image_files(data_dir_path):
        from ..scripts.descargar_emnist import descargar_emnist

        LOGGER.info("No hay datos procesados; descargando EMNIST a %s", data_dir_path)
        descargar_emnist(
            split="byclass",
            output_root=data_dir_path,
            download_root=Path(PATHS.get("datos_crudos", "data/raw")),
            max_per_class=max_muestras_por_clase,
            dry_run=False,
            skip_existing=False,
        )

    data_loader = DataLoader(str(data_dir_path), mapa_etiquetas=label_map)
    data_loader.cargar_desde_directorio()

    proporcion_entrenamiento = 0.85
    semilla = DATA_CONFIG.get("semilla", 42)
    train_rutas, val_rutas, train_labels, val_labels = data_loader.dividir_datos(
        proporcion_entrenamiento=proporcion_entrenamiento,
        semilla=semilla,
    )

    if not train_rutas:
        raise RuntimeError(f"No se encontraron muestras en {data_dir_path}")

    tamano_imagen = DATA_CONFIG.get("tamano_imagen", (28, 28))
    X_train, y_train = _cargar_dataset(train_rutas, train_labels, tamano_imagen)
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    if val_rutas:
        X_val, y_val = _cargar_dataset(val_rutas, val_labels, tamano_imagen)
    else:
        X_val, y_val = None, None

    batch_size = int(NETWORK_CONFIG.get("tamano_lote", 32))
    train_loader = _crear_torch_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _crear_torch_dataloader(X_val, y_val, batch_size, shuffle=False) if X_val is not None and y_val is not None else None

    device_obj = _select_device(device)
    model = crear_modelo_cnn_v2(
        num_classes=label_map.get_num_classes(),
        dropout_rate=float(NETWORK_CONFIG.get("dropout_rate", 0.5)),
    ).to(device_obj)

    epocas = max_epochs if max_epochs is not None else int(NETWORK_CONFIG.get("epocas", 50))
    base_lr = float(NETWORK_CONFIG.get("tasa_aprendizaje", 0.001))
    scheduler_config = NETWORK_CONFIG.get("lr_scheduler_config", {})

    historia = _train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_obj,
        epocas=epocas,
        base_lr=base_lr,
        scheduler_config=scheduler_config,
        verbose=verbose,
    )

    _guardar_modelo(model, output_dir, label_map, historia)
    return output_dir


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar la CNN v2 para reconocimiento de caracteres")
    parser.add_argument("--force", action="store_true", help="reentrenar aunque exista un modelo previo")
    parser.add_argument("--verbose", action="store_true", help="mostrar metricas por epoca")
    parser.add_argument("--epochs", type=int, default=None, help="numero maximo de epocas")
    parser.add_argument("--device", type=str, default=None, help="dispositivo (cpu, cuda, mps)")
    parser.add_argument("--data-dir", type=str, default=None, help="ruta alternativa con datos procesados")
    parser.add_argument("--model-dir-name", type=str, default="cnn_modelo_v2", help="nombre de la carpeta destino dentro de models/")
    parser.add_argument("--auto-download", action="store_true", help="descargar EMNIST si la carpeta de datos esta vacia")
    parser.add_argument("--max-per-class", type=int, default=None, help="limite de muestras por clase al autodescargar")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return entrenar_modelo(
        force=args.force,
        verbose=args.verbose,
        device=args.device,
        max_epochs=args.epochs,
        model_dir_name=args.model_dir_name,
        data_dir=args.data_dir,
        generar_dataset=args.auto_download,
        max_muestras_por_clase=args.max_per_class,
    )


if __name__ == "__main__":
    main()
