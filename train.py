import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hparams as hp

import argparse
import os
import time
import math
import utils

from loss import DNNLoss
from multiprocessing import cpu_count
from model import Tacotron2
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from optimizer import ScheduledOptim


def main(args):
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    print("Use Tacotron2")
    model = nn.DataParallel(Tacotron2(hp)).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)
    # Get buffer
    print("Load data to buffer")
    buffer = get_data_to_buffer()

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 betas=(0.9, 0.98),
                                 eps=1e-9)
    scheduled_optim = ScheduledOptim(optimizer,
                                     hp.decoder_rnn_dim,
                                     hp.n_warm_up_step,
                                     args.restore_step)
    tts_loss = DNNLoss().to(device)
    print("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Get dataset
    dataset = BufferDataset(buffer)

    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i * hp.batch_expand_size + j + args.restore_step + \
                    epoch * len(training_loader) * hp.batch_expand_size + 1

                # Init
                scheduled_optim.zero_grad()

                # Get Data
                character = db["text"].long().to(device)
                mel_target = db["mel_target"].float().to(device)
                mel_pos = db["mel_pos"].long().to(device)
                src_pos = db["src_pos"].long().to(device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_outputs, duration_predictor_output = model(character,
                                                               src_pos,
                                                               mel_view_pos=mel_view_pos,
                                                               mel_pos=mel_pos,
                                                               mel_max_length=max_mel_len,
                                                               length_target=duration)

                # Cal Loss
                mel_losses, duration_loss = tts_loss(mel_outputs,
                                                     duration_predictor_output,
                                                     mel_target,
                                                     duration)

                total_loss = duration_loss
                for m_l in mel_losses:
                    total_loss = total_loss + m_l

                # Logger
                t_l = total_loss.item()
                d_l = duration_loss.item()

                with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")

                with open(os.path.join("logger", "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp.grad_clip_thresh)

                # Update weights
                if args.frozen_learning_rate:
                    scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    scheduled_optim.step_and_update_lr()

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:"\
                        .format(epoch + 1, hp.epochs, current_step, total_step)
                    str2 = ""
                    for m_l in mel_losses:
                        str2 += " {:.4f}".format(m_l)
                    str2 = "Mel Loss:" + str2 + \
                        ", Duration Loss: {:.4f};".format(d_l)
                    str3 = "Current Learning Rate is {:.6f}."\
                        .format(scheduled_optim.get_learning_rate())
                    str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s."\
                        .format((Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    print(str4)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write("\n")

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
