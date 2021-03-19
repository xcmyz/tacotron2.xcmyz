import torch
import numpy as np

import text
import hparams

from model import Tacotron2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(checkpoint_path="tacotron2_statedict.pt"):
    model = Tacotron2(hparams).to(device)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    return model


if __name__ == "__main__":
    # Test
    model = get_model()

    words = [
        "All work and no play makes Jack a dull boy is a proverb. It means that without time off from work, a person becomes both bored and boring.",
        "Steven Paul Jobs was an American business magnate, industrial designer, investor, and media proprietor.",
        "As Jobs became more successful with his new company, his relationship with Brennan grew more complex."
    ]
    for i, word in enumerate(words):
        sequence = np.array(text.text_to_sequence(word, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence)
        np.save(f"result{i}.npy", mel_outputs_postnet.cpu()[0].numpy())
