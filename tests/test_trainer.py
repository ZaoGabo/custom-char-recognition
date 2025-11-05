from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from src.training.pipeline import _one_hot, entrenar_modelo, _actualizar_tasa_aprendizaje


def test_one_hot_encoding():
    """Verificar que la codificacion one-hot sea correcta."""
    labels = np.array([0, 2, 1])
    num_clases = 3
    one_hot_labels = _one_hot(labels, num_clases)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.array_equal(one_hot_labels, expected)


@pytest.mark.parametrize(
    "epoca, lr_esperada",
    [
        (0, 0.001),
        (49, 0.001),
        (50, 0.0001),
        (99, 0.0001),
        (100, 0.00001),
    ],
)
def test_actualizar_tasa_aprendizaje_step_decay(epoca, lr_esperada):
    """Verificar que la tasa de aprendizaje se actualice correctamente."""
    scheduler_config = {
        'tipo': 'step_decay',
        'tasa_decaimento': 0.1,
        'epocas_decaimento': 50,
    }

    lr_calculada = _actualizar_tasa_aprendizaje(epoca, base_lr=0.001, scheduler_config=scheduler_config)
    assert lr_calculada == pytest.approx(lr_esperada)


@patch('src.training.pipeline._guardar_modelo')
@patch('src.training.pipeline._train_loop')
@patch('src.training.pipeline._crear_torch_dataloader')
@patch('src.training.pipeline._cargar_dataset')
@patch('src.training.pipeline.DataLoader')
@patch('src.training.pipeline.crear_modelo_cnn_v2')
@patch('src.training.pipeline._build_advanced_augmentations', return_value=None)
def test_entrenar_modelo_flujo_completo(
    mock_build_aug,
    mock_crear_modelo,
    mock_dataloader,
    mock_cargar_dataset,
    mock_crear_dl,
    mock_train_loop,
    mock_guardar_modelo,
    tmp_path,
):
    mock_guardar_modelo.return_value = tmp_path / 'checkpoint.pt'

    model_mock = MagicMock()
    model_mock.to.return_value = model_mock
    mock_crear_modelo.return_value = model_mock

    dataloader_instance = mock_dataloader.return_value
    dataloader_instance.cargar_desde_directorio.return_value = None
    dataloader_instance.dividir_datos.return_value = (
        ['train/img_1.png', 'train/img_2.png'],
        ['val/img_1.png'],
        [0, 1],
        [0],
    )

    mock_cargar_dataset.side_effect = [
        (np.random.rand(2, 1, 28, 28).astype(np.float32), np.array([0, 1], dtype=np.int64)),
        (np.random.rand(1, 1, 28, 28).astype(np.float32), np.array([0], dtype=np.int64)),
    ]

    train_loader_mock = MagicMock()
    val_loader_mock = MagicMock()
    mock_crear_dl.side_effect = [train_loader_mock, val_loader_mock]

    mock_train_loop.return_value = [{'epoch': 1, 'loss_train': 0.5}]

    with patch.dict('src.training.pipeline.PATHS', {'modelos': str(tmp_path)}, clear=False):
        output_dir = entrenar_modelo(
            force=True,
            verbose=False,
            max_epochs=1,
            model_dir_name='modelo_prueba',
            data_dir='ruta/ficticia',
        )

    mock_build_aug.assert_called_once()
    mock_dataloader.assert_called_once()
    dataloader_instance.cargar_desde_directorio.assert_called_once()
    dataloader_instance.dividir_datos.assert_called_once()
    assert mock_cargar_dataset.call_count == 2
    assert mock_crear_dl.call_count == 2
    mock_train_loop.assert_called_once_with(
        model=model_mock,
        train_loader=train_loader_mock,
        val_loader=val_loader_mock,
        device=ANY,
        epocas=1,
        base_lr=ANY,
        scheduler_config=ANY,
        verbose=False,
    )
    mock_guardar_modelo.assert_called_once()
    assert output_dir == tmp_path / 'modelo_prueba'
