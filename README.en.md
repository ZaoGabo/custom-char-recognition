# Custom Character Recognition System

> **Important:** to recognize characters with textured or stylized shapes you must re-train the model with samples that match those styles. Use python scripts/run_pipeline.py --force to refresh the weights and validate them through the Streamlit demo (streamlit run demo/app.py).

## Overview

This repo ships a NumPy-based multilayer perceptron (784 ➜ 512 ➜ 256 ➜ 128 ➜ 52) that classifies 52 characters (A–Z and a–z). Key components:
- src/network.py: MLP with ReLU/softmax, Adam, dropout, and L2 regularization.
- src/data_loader.py: directory-based loader for data/raw/, optional augmentation, manual stratified split when sklearn/cv2 are missing.
- src/trainer.py: training/evaluation logic returning train/val/test metrics.
- scripts/run_pipeline.py: unified CLI (--force, --skip-train, --confusion-report, --save-metrics).
- demo/app.py: Streamlit UI compatible with legacy and new pickled models.

## Requirements

- Python 3.8+
- Packages listed in 
equirements.txt (NumPy required; OpenCV / scikit-learn optional).

## Quick Start

```
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train & Evaluate

1. Place samples under data/raw/<letter>_<upper|lower>/. Any PNG/JPG/JPEG/BMP works; pictures are resized to 28×28.
2. Run the pipeline:
   `
   python scripts/run_pipeline.py --force --confusion-report --limit 5
   `
   - Generates synthetic data if data/raw is empty.
   - Augments training data (noise, shifts, scaling) when DATA_CONFIG['usar_augmentacion'] is enabled.
   - Prints consolidated metrics and class-wise confusions.
3. The trained model is stored at models/modelo_entrenado.pkl.

### Real data / incremental fine-tuning

Drop your real samples into data/raw/ and re-run the pipeline. You may also use 	rain_con_imagenes_reales.py for quick fine-tuning.

## Streamlit Demo

`
streamlit run demo/app.py
`
The app performs preprocessing, displays top-5 predictions, and reads the pickle stored under models/.

## Utilities

- scripts/probar_modelo.py: batch inference against existing files.
- scripts/verificar_sistema.py: sanity-check folder structure and dataset loading.
- src/predictor.py: CSV-based evaluation helper.

## Configuration

src/config.py centralizes network and data hyperparameters:
- NETWORK_CONFIG['capas'], activations, learning rate, dropout, L2, Adam betas.
- DATA_CONFIG: image size, augmentation, split ratios, random seed.
- CUSTOM_LABELS: character set used throughout the project.

## Baseline Metrics (synthetic dataset)

- Training accuracy ≈ 0.998
- Validation accuracy ≈ 0.63
- Test accuracy ≈ 0.75
Numbers improve when adding realistic samples for the weaker classes.

## Suggested Next Steps

- Collect textured/handwritten variants for the classes flagged in the confusion report.
- Tune hyperparameters (dropout_rate, lambda_l2, hidden sizes) according to your dataset.
- Experiment with CNNs or batch normalization by extending src/network.py.

## License

MIT © ZaoGabo