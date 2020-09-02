import torch
import torch.nn as nn


class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_outputs, duration_predicted, mel_target, duration_predictor_target):
        mel_losses = list()
        mel_target.requires_grad = False
        for mel_output in mel_outputs:
            mel_losses.append(self.mse_loss(mel_output, mel_target))

        duration_predictor_target.requires_grad = False
        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_losses, duration_predictor_loss
