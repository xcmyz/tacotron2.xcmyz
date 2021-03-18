import torch

import os
import time
import hparams
import numpy as np

from tqdm import tqdm
from model import Tacotron2
from utils import process_text
from text import text_to_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(checkpoint_path="tacotron2_statedict.pt"):
    model = Tacotron2(hparams).to(device)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    return model


def generator(model):
    with torch.no_grad():
        text = process_text(os.path.join("data", "train.txt"))
        start = time.perf_counter()
        for i in tqdm(range(len(text))):
            mel_gt_name = os.path.join(hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
            mel_gt_target = np.load(mel_gt_name)
            character = text[i][0:len(text[i])-1]
            character = np.array(text_to_sequence(character, hparams.text_cleaners))
            character = torch.stack([torch.from_numpy(character)]).long().to(device)
            mel_gt_target = torch.stack([torch.from_numpy(mel_gt_target.T)]).float().to(device)
            mel_gta = model.gta(character, mel_gt_target)
            print(mel_gt_target.size(), mel_gta.size())

        end = time.perf_counter()
        print("cost {:.2f}s to generate gta data.".format(end - start))


if __name__ == "__main__":
    model = get_model()
    generator(model)
