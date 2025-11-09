"""Entrenador robusto para la CNN v2."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from ..cnn_model_v2 import crear_modelo_cnn_v2
from ..config import (
    ADVANCED_AUGMENTATION_CONFIG,
    CUSTOM_LABELS,
    DATA_CONFIG,
    NETWORK_CONFIG,
    PATHS,
)
from ..data_loader import DataLoader
from ..label_map import DEFAULT_LABEL_MAP, LabelMap
from .pipeline import (
    _build_advanced_augmentations,
    _cargar_dataset,
    _crear_torch_dataloader,
    _guardar_modelo,
    _select_device,
    _CharacterDataset,
)

LOGGER = logging.getLogger(__name__)


class RecoverableTrainingError(RuntimeError):
    """Error recuperable que permite reintentar el entrenamiento."""


@dataclass
class RobustTrainerConfig:
    """Configuracion de alto nivel para ``RobustTrainer``."""

    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0
    max_checkpoints: int = 5
    resume: bool = True
    batch_size: Optional[int] = None
    scheduler_type: Optional[str] = None
    scheduler_min_lr: Optional[float] = None
    num_workers: int = 0


@dataclass
class TrainingResult:
    """Resultado del proceso de entrenamiento."""

    output_dir: Path
    best_model_path: Path
    history_path: Path
    epochs_trained: int
    best_metric_name: Optional[str]
    best_metric_value: Optional[float]
    history: List[Dict[str, Optional[float]]]


class RobustTrainer:
    """Entrenador resiliente con checkpoints rotativos y early stopping."""

    def __init__(
        self,
        *,
        model_dir_name: str = "cnn_modelo_v2_finetuned",
        data_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[RobustTrainerConfig] = None,
        label_map: Optional[LabelMap] = None,
        modelos_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or RobustTrainerConfig()
        self.model_dir_name = model_dir_name
        self.data_dir = Path(data_dir) if data_dir is not None else Path(PATHS.get("datos_procesados", "data/processed"))
        self.modelos_root = Path(modelos_root) if modelos_root is not None else Path(PATHS.get("modelos", "models"))
        self.output_dir = self.modelos_root / model_dir_name
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.history_path = self.output_dir / "history.json"
        self.device_request = device
        self.label_map = (
            label_map
            if label_map is not None
            else (DEFAULT_LABEL_MAP if DEFAULT_LABEL_MAP.get_num_classes() == len(CUSTOM_LABELS) else LabelMap(CUSTOM_LABELS))
        )
        self.metric_name: Optional[str] = None
        self.metric_mode: str = "max"

    # ------------------------------ API publica ------------------------------
    def train(
        self,
        *,
        force: bool = False,
        resume: Optional[bool] = None,
        max_epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> TrainingResult:
        """Ejecutar el entrenamiento completo y devolver los artefactos generados."""

        requested_resume = self.config.resume if resume is None else resume

        if self.output_dir.exists():
            if not force and not requested_resume:
                raise FileExistsError(
                    f"El modelo '{self.model_dir_name}' ya existe; usa force=True o resume=True para continuar"
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        device = _select_device(self.device_request)
        if verbose:
            LOGGER.setLevel(logging.INFO)

        train_loader, val_loader = self._prepare_dataloaders(device)
        model = self._build_model(device)

        base_lr = float(NETWORK_CONFIG.get("tasa_aprendizaje", 1e-3))
        weight_decay = float(NETWORK_CONFIG.get("lambda_l2", 0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        total_epochs = max_epochs if max_epochs is not None else int(NETWORK_CONFIG.get("epocas", 50))
        scheduler = self._build_scheduler(optimizer, total_epochs)

        criterion = torch.nn.CrossEntropyLoss()

        history: List[Dict[str, Optional[float]]] = []
        best_metric_value: Optional[float] = None
        best_metric_name: Optional[str] = None
        patience_counter = 0
        start_epoch = 0

        if requested_resume:
            resume_path = self.checkpoints_dir / "last.pth"
            if resume_path.exists():
                start_epoch, history, best_metric_name, best_metric_value, patience_counter = self._restore_checkpoint(
                    resume_path, model, optimizer, scheduler, device
                )
                LOGGER.info("Se reanuda entrenamiento desde la epoca %s", start_epoch)

        for epoch in range(start_epoch, total_epochs):
            try:
                metrics = self._run_epoch(
                    epoch=epoch,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    history=history,
                    verbose=verbose,
                )
            except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - depende del entorno
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RecoverableTrainingError("CUDA OOM durante entrenamiento") from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RecoverableTrainingError("OOM durante entrenamiento") from exc
                raise

            history.append(metrics)
            self._write_history(history)

            metric_key, metric_value, metric_mode = self._select_primary_metric(metrics)
            if self.metric_name is None:
                self.metric_name = metric_key
                self.metric_mode = metric_mode
            improvement = self._is_improvement(metric_value, best_metric_value)

            if improvement:
                best_metric_name = metric_key
                best_metric_value = metric_value
                patience_counter = 0
                best_model_path = _guardar_modelo(model, self.output_dir, self.label_map, history)
            else:
                patience_counter += 1
                best_model_path = self.output_dir / "cnn_model_v2.pt"

            self._save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                patience_counter=patience_counter,
            )

            if verbose:
                LOGGER.info(
                    "Epoca %s/%s - loss: %.4f - lr: %.6f%s",
                    epoch + 1,
                    total_epochs,
                    metrics.get("loss_train", 0.0) or 0.0,
                    float(optimizer.param_groups[0]["lr"]),
                    f" - {metric_key}: {metric_value:.4f}" if metric_value is not None else "",
                )

            if self.config.patience > 0 and patience_counter >= self.config.patience:
                LOGGER.info(
                    "Early stopping activado tras %s epocas sin mejora", self.config.patience
                )
                break

        if best_metric_name is None:
            best_metric_name = self.metric_name
        if best_metric_value is None:
            best_metric_value = history[-1].get(self.metric_name) if history else None
        if "best_model_path" not in locals():
            best_model_path = _guardar_modelo(model, self.output_dir, self.label_map, history)

        return TrainingResult(
            output_dir=self.output_dir,
            best_model_path=best_model_path,
            history_path=self.history_path,
            epochs_trained=len(history),
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            history=history,
        )

    # ------------------------------ helpers internos ------------------------------
    def _prepare_dataloaders(self, device: torch.device) -> Tuple[TorchDataLoader, Optional[TorchDataLoader]]:
        data_dir = self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        data_loader = DataLoader(str(data_dir), mapa_etiquetas=self.label_map)
        data_loader.cargar_desde_directorio()

        proporcion = DATA_CONFIG.get("proporcion_entrenamiento", 0.85)
        semilla = DATA_CONFIG.get("semilla", 42)
        train_rutas, val_rutas, train_labels, val_labels = data_loader.dividir_datos(
            proporcion_entrenamiento=proporcion,
            semilla=semilla,
        )

        if not train_rutas:
            raise RuntimeError(f"No se encontraron muestras en {data_dir}")

        tamano_imagen = DATA_CONFIG.get("tamano_imagen", (28, 28))
        advanced_transform = _build_advanced_augmentations(ADVANCED_AUGMENTATION_CONFIG, tamano_imagen)
        batch_size = self.config.batch_size or int(NETWORK_CONFIG.get("tamano_lote", 32))
        num_workers = max(self.config.num_workers, 0)
        pin_memory = device.type == "cuda"

        if advanced_transform is not None:
            train_dataset = _CharacterDataset(train_rutas, train_labels, tamano_imagen, transform=advanced_transform)
            val_dataset = _CharacterDataset(val_rutas, val_labels, tamano_imagen) if val_rutas else None
            train_loader = TorchDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = (
                TorchDataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                if val_dataset is not None
                else None
            )
        else:
            X_train, y_train = _cargar_dataset(train_rutas, train_labels, tamano_imagen)
            if val_rutas:
                X_val, y_val = _cargar_dataset(val_rutas, val_labels, tamano_imagen)
            else:
                X_val, y_val = None, None

            train_loader = _crear_torch_dataloader(X_train, y_train, batch_size, shuffle=True)
            val_loader = (
                _crear_torch_dataloader(X_val, y_val, batch_size, shuffle=False)
                if X_val is not None and y_val is not None
                else None
            )

        return train_loader, val_loader

    def _build_model(self, device: torch.device) -> torch.nn.Module:
        dropout = float(NETWORK_CONFIG.get("dropout_rate", 0.5))
        model = crear_modelo_cnn_v2(num_classes=self.label_map.get_num_classes(), dropout_rate=dropout)
        return model.to(device)

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = self.config.scheduler_type or NETWORK_CONFIG.get("lr_scheduler_config", {}).get("tipo", "step_decay")
        if scheduler_type == "cosine":
            min_lr = self.config.scheduler_min_lr
            if min_lr is None:
                min_lr = float(NETWORK_CONFIG.get("lr_scheduler_config", {}).get("lr_min", optimizer.param_groups[0]["lr"] * 0.01))
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_epochs, 1),
                eta_min=min_lr,
            )
        if scheduler_type == "step_decay":
            cfg = NETWORK_CONFIG.get("lr_scheduler_config", {})
            paso = int(cfg.get("epocas_decaimento", 50))
            gamma = float(cfg.get("tasa_decaimento", 0.1))
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(paso, 1), gamma=gamma)
        return None

    def _run_epoch(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        train_loader: TorchDataLoader,
        val_loader: Optional[TorchDataLoader],
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: torch.nn.Module,
        history: Sequence[Dict[str, Optional[float]]],
        verbose: bool,
    ) -> Dict[str, Optional[float]]:
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()

            clip_norm = self.config.gradient_clip_norm
            if clip_norm is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

            total_loss += float(loss.item()) * batch_X.size(0)
            total_samples += batch_X.size(0)

        loss_train = total_loss / max(total_samples, 1)

        loss_val: Optional[float] = None
        acc_val: Optional[float] = None

        if val_loader is not None:
            model.eval()
            total_val = 0
            acumulado_val = 0.0
            correctos = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    acumulado_val += float(loss.item()) * batch_X.size(0)
                    pred = logits.argmax(dim=1)
                    correctos += (pred == batch_y).sum().item()
                    total_val += batch_X.size(0)
            loss_val = acumulado_val / max(total_val, 1)
            acc_val = correctos / max(total_val, 1)

        if scheduler is not None:
            scheduler.step()

        metrics = {
            "epoch": epoch + 1,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "acc_val": acc_val,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if verbose:
            mensaje = f"Epoca {epoch + 1} - loss: {loss_train:.4f}"
            if loss_val is not None:
                mensaje += f" - val_loss: {loss_val:.4f}"
            if acc_val is not None:
                mensaje += f" - val_acc: {acc_val:.4f}"
            LOGGER.info(mensaje)

        return metrics

    def _select_primary_metric(
        self, metrics: Dict[str, Optional[float]]
    ) -> Tuple[str, Optional[float], str]:
        acc_val = metrics.get("acc_val")
        if acc_val is not None:
            return "acc_val", acc_val, "max"
        loss_val = metrics.get("loss_val")
        if loss_val is not None:
            return "loss_val", loss_val, "min"
        return "loss_train", metrics.get("loss_train"), "min"

    def _is_improvement(self, value: Optional[float], best: Optional[float]) -> bool:
        if value is None:
            return False
        if best is None:
            return True
        delta = max(self.config.min_delta, 0.0)
        if self.metric_mode == "max":
            return value > best + delta
        return value < best - delta

    def _save_checkpoint(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        history: Sequence[Dict[str, Optional[float]]],
        best_metric_name: Optional[str],
        best_metric_value: Optional[float],
        patience_counter: int,
    ) -> None:
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "history": list(history),
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "patience_counter": patience_counter,
            "metric_mode": self.metric_mode,
            "metric_name": self.metric_name,
            "labels": self.label_map.labels,
        }
        epoch_path = self.checkpoints_dir / f"epoch-{epoch + 1:03d}.pth"
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, self.checkpoints_dir / "last.pth")
        self._rotate_checkpoints()

    def _restore_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
    ) -> Tuple[int, List[Dict[str, Optional[float]]], Optional[str], Optional[float], int]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint.get("history", []))
        epoch = int(checkpoint.get("epoch", 0))
        best_metric_name = checkpoint.get("best_metric_name")
        best_metric_value = checkpoint.get("best_metric_value")
        patience_counter = int(checkpoint.get("patience_counter", 0))
        self.metric_mode = checkpoint.get("metric_mode", self.metric_mode)
        self.metric_name = checkpoint.get("metric_name", self.metric_name)
        return epoch, history, best_metric_name, best_metric_value, patience_counter

    def _rotate_checkpoints(self) -> None:
        if self.config.max_checkpoints <= 0:
            return
        all_checkpoints = sorted(self.checkpoints_dir.glob("epoch-*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_ckpt in all_checkpoints[self.config.max_checkpoints :]:
            try:
                old_ckpt.unlink()
            except FileNotFoundError:  # pragma: no cover - carrera improbable
                continue

    def _write_history(self, history: Sequence[Dict[str, Optional[float]]]) -> None:
        with open(self.history_path, "w", encoding="utf-8") as history_file:
            json.dump(list(history), history_file, indent=2)


__all__ = [
    "RobustTrainer",
    "RobustTrainerConfig",
    "TrainingResult",
    "RecoverableTrainingError",
]
