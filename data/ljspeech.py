import numpy as np
import os
import audio

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def build_from_path(in_dir, out_dir):
    index = 1
    futures = list()

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(_process_utterance(out_dir, index, wav_path, text))

            if index % 100 == 0:
                print("%d Done" % index)
            index = index + 1

    return futures


def _process_utterance(out_dir, index, wav_path, text):
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.tools.get_mel(wav_path).numpy().astype(np.float32)

    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    return text
