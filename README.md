# Plant Species Classifier

This project is a modular, hardware-optimized deep learning training and evaluation pipeline built in pure PyTorch to classify plant species. It generates high-efficiency classification weights utilized within the offline [FloraLens](https://github.com/abderrahmenex86/FloraLens) ecosystem.

This pipeline strictly implements a flat directory architecture, dynamic hyperparameter configuration, layer-stratified optimization, and defensive execution checks.

______________________________________________________________________

## Directory Architecture

- `dataset.py`: Handles raw image parsing, `torchvision.transforms.v2` preprocessing pipelines, and hardware-specific `DataLoader` configuration (pin memory, prefetching, persistent workers).
- `models.py`: Houses the network architecture (MobileNetV3-Large with modified dropout and classification head) and target losses, including cross-entropy and focal loss formulations.
- `factory.py`: The assembly layer responsible for constructing model instances, stratified optimizers, and learning rate schedulers.
- `trainer.py`: Holds the main execution loop for training and evaluation epochs, along with tracking metrics, saving checkpoints, and loading past runs.
- `tester.py`: Orchestrates a quick sanity check with sliced inputs to verify tensor bounds, gradients, and parameters before launching scaled training.
- `optimize.py`: Manages automatic hyperparameter tuning using Optuna studies via a silenced, global progress-tracked interface.
- `infer.py`: Runs smart target-based predictions by loading historic configuration files directly from saved run directories.
- `main.py`: The single dispatcher entry point for core ML lifecycle operations (`test`, `train`, `optimize`, `infer`).
- `tools.py`: Helper scripts for utilities like data layout verification and training metric plots.

______________________________________________________________________

## Execution Sequence

To prepare, validate, optimize, and execute models on your dataset, follow this standardized execution path.

### 1. Verify Dataset Alignment

Ensure your dataset structures (by default, `dataset/plantnet_300K/images/train` and `dataset/plantnet_300K/images/val`) are correctly formatted, categories match, and image files are uncorrupted:

```bash
python tools.py --mode verify
```

### 2. Run Sanity Check

Test basic network operations on a slice of 5 images. This step validates tensor dimensions, asserts that the forward and backward passes execute correctly, checks that gradients are calculated for active weights, and tests parameter optimizer grouping:

```bash
python main.py --mode test
```

### 3. Hyperparameter Optimization

Execute an Optuna study on a 512-image training slice across 10 epochs. When finished, it prints a terminal-ready string containing the best learning rate and weight decay parameters:

```bash
python main.py --mode optimize
```

### 4. Execute Full Training

Train the network using custom configurations or the exact arguments outputted by the optimization run. Additional parameters can be passed as dynamic flags to override default configurations on the fly:

```bash
python main.py --mode train --learning_rate 0.001 --weight_decay 0.01 --loss_type focal --num_workers_train 8
```

### 5. Resuming from Interrupted States

If a run was stopped, restore training from the last checkpoint folder. The script loads weights, optimizers, learning rate histories, scheduler states, and epoch counts to resume without disruption:

```bash
python main.py --mode train --resume --resume_directory artifacts/YYYYMMDD_HHMMSS_mobilenetv3_large
```

### 6. Analyze History

Generate validation loss, top-1 accuracy, and macro F1 metric trend graphs from a selected training epoch:

```bash
python tools.py --mode plot --run_directory artifacts/YYYYMMDD_HHMMSS_mobilenetv3_large
```

### 7. Run Smart Inference

Pass a specific historic output folder and a sample image path. The script dynamically reads `hyperparameters.json` and `class_names.json`, designs the model, loads `best_model.pth`, and outputs the prediction:

```bash
python main.py --mode infer --run_directory artifacts/YYYYMMDD_HHMMSS_mobilenetv3_large --image_path path/to/test_image.jpg
```

______________________________________________________________________

## System Requirements

- Python 3.10+
- CUDA-capable GPU (highly recommended)
- Dependencies listed in `requirements.txt` (including `torch`, `torchvision`, `torchmetrics`, `torchinfo`, `optuna`, `matplotlib`, and `tqdm`)

## Related Projects

- [FloraLens](https://github.com/abderrahmenex86/FloraLens)
- [Pesti](https://github.com/abderrahmenex86/pesti)
- [Plant-Disease-Segmentation](https://github.com/abderrahmenex86/Plant-Disease-Segmentation)
