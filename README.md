# Tacotron2
Use [NVIDIA's tacotron2 code](https://github.com/NVIDIA/tacotron2), and modify training mode and data loader.

## Usage
1. Download and unzip LJSpeech dataset in `data`
2. Run `python3 preprocess.py`
3. Run `python3 train.py`
4. Download waveglow pretrained model in `waveglow/pretrained_model`
5. Run `python3 eval.py --step (checkpoint step)`
6. Samples [here]()