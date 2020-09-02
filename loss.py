import torch
import torch.nn as nn


class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.gate_loss = nn.BCEWithLogitsLoss()

    def forward(self, mel_output, mel_output_postnet, gate_output, mel_target, gate_target):
        mel_loss = self.mse_loss(mel_output, mel_target)
        mel_postnet_loss = self.mse_loss(mel_output_postnet, mel_target)
        gate_loss = self.gate_loss(gate_output, gate_target)

        return mel_loss, mel_postnet_loss, gate_loss
