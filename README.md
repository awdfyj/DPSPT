This repository contains the source code for our paper "Downstream Task Skewed Pre-training Strategy for
GNNs".The two folders correspond to the experiments in the two fields.
1. Pre-training
```
python pretrain.py --output_model_file OUTPUT_MODEL_PATH
```
This will perform self-supervised pre-training with information control and save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

2. Fine-tuning
```
python finetune.py --model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`