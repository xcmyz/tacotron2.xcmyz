import torch
import visual
import modules
import utils
import modules


if __name__ == "__main__":
    # --------------------------------- TEST --------------------------------- #

    # visual.plot_data([torch.eye(256, 256) for _ in range(2)], 0)

    # cb = modules.ConvDecoder(256, 3000)
    # print(utils.get_param_num(cb))
    # x = torch.randn(2, 1234, 256)
    # pos = torch.Tensor([[(i+1) for i in range(1234)]for _ in range(2)]).long()
    # print(cb(x, pos, 1234)[0].size())

    # ce = modules.ConvEncoder(256, 300)
    # print(utils.get_param_num(ce))
    # x = torch.Tensor([[i for i in range(123)] for _ in range(2)]).long()
    # pos = torch.Tensor([[(i+1) for i in range(123)] for _ in range(2)]).long()
    # print(ce(x, pos)[0].size())

    # te = tacotron_encoder.Encoder(256)
    # test_input = torch.randn(2, 123, 256)
    # print(te(test_input).size())

    cd = modules.ConvDecoder(512, 270, 3000, 4, 3, 3, 3, 3, 1, 1)
    test_input = torch.randn(2, 123, 512)
    mel_pos = torch.Tensor(
        [[(i+1) for i in range(123)] for _ in range(2)]).long()
    mel_view_pos = torch.Tensor(
        [[(i+1) for i in range(12)] for _ in range(2)]).long()
    o, _ = cd(test_input, mel_pos, mel_view_pos)
    for _o_ in o:
        print(_o_.size())

    # --------------------------------- TEST --------------------------------- #
