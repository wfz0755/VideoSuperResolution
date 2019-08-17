import torch
import torch.nn as nn
from model import common


# pretrained_check_point = {
#     "boat_r20_f21" : None, #medium
#     "boat_r20_f32":None, # high
#     "boat_r20_f48": None # ultra-high
# }
class MDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MDSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        self.sr_ratio = args.scale
       # self.url = url['r{}f{}'.format(n_resblocks, n_feats)] change to the
    #    self.sub_mean = common.MeanShift(args.rgb_range)
    #    self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.pre_process = nn.Sequential(common.ResBlock(conv, n_feats, 5, act=act),common.ResBlock(conv, n_feats, 5, act=act))

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = common.Upsampler(conv, self.sr_ratio, n_feats, act=False)

        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
      #  x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process(x)

        res = self.body(x)
        res += x

        x = self.upsample(res)
        x = self.tail(x)
    #    x = self.add_mean(x)

        return x

    def set_scale(self, sr_ratio):
        self.sr_ratio = sr_ratio

