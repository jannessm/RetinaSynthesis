
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

def spatial_dropout_inference(input, factor, spatial):
    return input

def concatRelu(input):
    a = F.relu(input)
    b = F.relu(-input)
    
    return torch.cat((a, b), 1)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.np_parameters = np.load(os.path.dirname(os.path.abspath(__file__))+"/params.npz")

        self.C__in = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__in.weight.data = torch.from_numpy(self.np_parameters['C__in_filters'])
        self.C__in.bias.data = torch.from_numpy(self.np_parameters['C__in_biases'])
        self.C_B_0_DS_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_0_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_0_DS_p_filters'])
        self.C_B_0_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_0_DS_p_biases'])
        self.C__D_B_0_0_compress_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_compress_p_filters'])
        self.C__D_B_0_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_compress_p_biases'])
        self.C__D_B_0_0_0_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_0_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_0_p_filters'])
        self.C__D_B_0_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_0_p_biases'])
        self.C__D_B_0_0_decompress_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_decompress_p_filters'])
        self.C__D_B_0_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_decompress_p_biases'])
        self.C__D_B_0_1_compress_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_compress_p_filters'])
        self.C__D_B_0_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_compress_p_biases'])
        self.C__D_B_0_1_0_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_0_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_0_p_filters'])
        self.C__D_B_0_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_0_p_biases'])
        self.C__D_B_0_1_decompress_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_decompress_p_filters'])
        self.C__D_B_0_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_decompress_p_biases'])
        self.C_B_1_DS_p = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_1_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_1_DS_p_filters'])
        self.C_B_1_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_1_DS_p_biases'])
        self.C__D_B_1_0_compress_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_compress_p_filters'])
        self.C__D_B_1_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_compress_p_biases'])
        self.C__D_B_1_0_0_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_1_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_0_p_filters'])
        self.C__D_B_1_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_0_p_biases'])
        self.C__D_B_1_0_decompress_p = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_decompress_p_filters'])
        self.C__D_B_1_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_decompress_p_biases'])
        self.C__D_B_1_1_compress_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_compress_p_filters'])
        self.C__D_B_1_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_compress_p_biases'])
        self.C__D_B_1_1_0_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_1_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_0_p_filters'])
        self.C__D_B_1_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_0_p_biases'])
        self.C__D_B_1_1_decompress_p = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_decompress_p_filters'])
        self.C__D_B_1_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_decompress_p_biases'])
        self.C_B_2_DS_p = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_2_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_2_DS_p_filters'])
        self.C_B_2_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_2_DS_p_biases'])
        self.C__D_B_2_0_compress_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_compress_p_filters'])
        self.C__D_B_2_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_compress_p_biases'])
        self.C__D_B_2_0_0_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_2_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_0_p_filters'])
        self.C__D_B_2_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_0_p_biases'])
        self.C__D_B_2_0_decompress_p = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_decompress_p_filters'])
        self.C__D_B_2_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_decompress_p_biases'])
        self.C__D_B_2_1_compress_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_compress_p_filters'])
        self.C__D_B_2_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_compress_p_biases'])
        self.C__D_B_2_1_0_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_2_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_0_p_filters'])
        self.C__D_B_2_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_0_p_biases'])
        self.C__D_B_2_1_decompress_p = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_decompress_p_filters'])
        self.C__D_B_2_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_decompress_p_biases'])
        self.C_B_3_DS_p = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_3_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_3_DS_p_filters'])
        self.C_B_3_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_3_DS_p_biases'])
        self.C__D_B_3_0_compress_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_3_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_3_0_compress_p_filters'])
        self.C__D_B_3_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_3_0_compress_p_biases'])
        self.C__D_B_3_0_0_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_3_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_3_0_0_p_filters'])
        self.C__D_B_3_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_3_0_0_p_biases'])
        self.C__D_B_3_0_decompress_p = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_3_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_3_0_decompress_p_filters'])
        self.C__D_B_3_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_3_0_decompress_p_biases'])
        self.C__D_B_3_1_compress_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_3_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_3_1_compress_p_filters'])
        self.C__D_B_3_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_3_1_compress_p_biases'])
        self.C__D_B_3_1_0_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_3_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_3_1_0_p_filters'])
        self.C__D_B_3_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_3_1_0_p_biases'])
        self.C__D_B_3_1_decompress_p = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_3_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_3_1_decompress_p_filters'])
        self.C__D_B_3_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_3_1_decompress_p_biases'])
        self.C_B_4_DS_p = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_4_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_4_DS_p_filters'])
        self.C_B_4_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_4_DS_p_biases'])
        self.C__D_B_4_0_compress_p = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_4_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_4_0_compress_p_filters'])
        self.C__D_B_4_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_4_0_compress_p_biases'])
        self.C__D_B_4_0_0_p = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_4_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_4_0_0_p_filters'])
        self.C__D_B_4_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_4_0_0_p_biases'])
        self.C__D_B_4_0_decompress_p = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_4_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_4_0_decompress_p_filters'])
        self.C__D_B_4_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_4_0_decompress_p_biases'])
        self.C__D_B_4_1_compress_p = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_4_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_4_1_compress_p_filters'])
        self.C__D_B_4_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_4_1_compress_p_biases'])
        self.C__D_B_4_1_0_p = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_4_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_4_1_0_p_filters'])
        self.C__D_B_4_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_4_1_0_p_biases'])
        self.C__D_B_4_1_decompress_p = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_4_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_4_1_decompress_p_filters'])
        self.C__D_B_4_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_4_1_decompress_p_biases'])
        self.C_B_4_US_p = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_4_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_4_US_p_filters'])
        self.C_B_4_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_4_US_p_biases'])
        self.C__U_B_4_0_compress_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_4_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_4_0_compress_p_filters'])
        self.C__U_B_4_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_4_0_compress_p_biases'])
        self.C__U_B_4_0_0_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_4_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_4_0_0_p_filters'])
        self.C__U_B_4_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_4_0_0_p_biases'])
        self.C__U_B_4_0_decompress_p = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_4_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_4_0_decompress_p_filters'])
        self.C__U_B_4_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_4_0_decompress_p_biases'])
        self.C__U_B_4_1_compress_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_4_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_4_1_compress_p_filters'])
        self.C__U_B_4_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_4_1_compress_p_biases'])
        self.C__U_B_4_1_0_p = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_4_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_4_1_0_p_filters'])
        self.C__U_B_4_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_4_1_0_p_biases'])
        self.C__U_B_4_1_decompress_p = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_4_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_4_1_decompress_p_filters'])
        self.C__U_B_4_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_4_1_decompress_p_biases'])
        self.C_B_3_US_p = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_3_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_3_US_p_filters'])
        self.C_B_3_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_3_US_p_biases'])
        self.C__U_B_3_0_compress_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_3_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_3_0_compress_p_filters'])
        self.C__U_B_3_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_3_0_compress_p_biases'])
        self.C__U_B_3_0_0_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_3_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_3_0_0_p_filters'])
        self.C__U_B_3_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_3_0_0_p_biases'])
        self.C__U_B_3_0_decompress_p = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_3_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_3_0_decompress_p_filters'])
        self.C__U_B_3_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_3_0_decompress_p_biases'])
        self.C__U_B_3_1_compress_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_3_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_3_1_compress_p_filters'])
        self.C__U_B_3_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_3_1_compress_p_biases'])
        self.C__U_B_3_1_0_p = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_3_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_3_1_0_p_filters'])
        self.C__U_B_3_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_3_1_0_p_biases'])
        self.C__U_B_3_1_decompress_p = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_3_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_3_1_decompress_p_filters'])
        self.C__U_B_3_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_3_1_decompress_p_biases'])
        self.C_B_2_US_p = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_2_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_2_US_p_filters'])
        self.C_B_2_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_2_US_p_biases'])
        self.C__U_B_2_0_compress_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_compress_p_filters'])
        self.C__U_B_2_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_compress_p_biases'])
        self.C__U_B_2_0_0_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_2_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_0_p_filters'])
        self.C__U_B_2_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_0_p_biases'])
        self.C__U_B_2_0_decompress_p = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_decompress_p_filters'])
        self.C__U_B_2_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_decompress_p_biases'])
        self.C__U_B_2_1_compress_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_compress_p_filters'])
        self.C__U_B_2_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_compress_p_biases'])
        self.C__U_B_2_1_0_p = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_2_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_0_p_filters'])
        self.C__U_B_2_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_0_p_biases'])
        self.C__U_B_2_1_decompress_p = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_decompress_p_filters'])
        self.C__U_B_2_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_decompress_p_biases'])
        self.C_B_1_US_p = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_1_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_1_US_p_filters'])
        self.C_B_1_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_1_US_p_biases'])
        self.C__U_B_1_0_compress_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_compress_p_filters'])
        self.C__U_B_1_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_compress_p_biases'])
        self.C__U_B_1_0_0_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_1_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_0_p_filters'])
        self.C__U_B_1_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_0_p_biases'])
        self.C__U_B_1_0_decompress_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_decompress_p_filters'])
        self.C__U_B_1_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_decompress_p_biases'])
        self.C__U_B_1_1_compress_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_compress_p_filters'])
        self.C__U_B_1_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_compress_p_biases'])
        self.C__U_B_1_1_0_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_1_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_0_p_filters'])
        self.C__U_B_1_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_0_p_biases'])
        self.C__U_B_1_1_decompress_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_decompress_p_filters'])
        self.C__U_B_1_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_decompress_p_biases'])
        self.C_B_0_US_p = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_0_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_0_US_p_filters'])
        self.C_B_0_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_0_US_p_biases'])
        self.C__U_B_0_0_compress_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_compress_p_filters'])
        self.C__U_B_0_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_compress_p_biases'])
        self.C__U_B_0_0_0_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_0_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_0_p_filters'])
        self.C__U_B_0_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_0_p_biases'])
        self.C__U_B_0_0_decompress_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_decompress_p_filters'])
        self.C__U_B_0_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_decompress_p_biases'])
        self.C__U_B_0_1_compress_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_compress_p_filters'])
        self.C__U_B_0_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_compress_p_biases'])
        self.C__U_B_0_1_0_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_0_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_0_p_filters'])
        self.C__U_B_0_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_0_p_biases'])
        self.C__U_B_0_1_decompress_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_decompress_p_filters'])
        self.C__U_B_0_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_decompress_p_biases'])
        self.C__out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__out.weight.data = torch.from_numpy(self.np_parameters['C__out_filters'])
        self.C__out.bias.data = torch.from_numpy(self.np_parameters['C__out_biases'])

    def forward(self, input):
        C__unet_in = self.C__in(input)
        C__unet__D_Output_0_shortcut_0 = self.C_B_0_DS_p(C__unet_in)
        output_0003 = self.C__D_B_0_0_compress_p(C__unet__D_Output_0_shortcut_0)
        output_0004 = spatial_dropout_inference(output_0003, 0.3, True)
        output_0005 = concatRelu(output_0004)
        output_0006 = self.C__D_B_0_0_0_p(output_0005)
        C__unet__D_Output_0_comp_0 = self.C__D_B_0_0_decompress_p(output_0006)
        C__unet__D_Output_0_shortcut_1 = C__unet__D_Output_0_shortcut_0 + C__unet__D_Output_0_comp_0
        output_0009 = self.C__D_B_0_1_compress_p(C__unet__D_Output_0_shortcut_1)
        output_0010 = spatial_dropout_inference(output_0009, 0.3, True)
        output_0011 = concatRelu(output_0010)
        output_0012 = self.C__D_B_0_1_0_p(output_0011)
        C__unet__D_Output_0_comp_1 = self.C__D_B_0_1_decompress_p(output_0012)
        C__unet__D_Output_0 = C__unet__D_Output_0_shortcut_1 + C__unet__D_Output_0_comp_1
        C__unet__D_Output_1_shortcut_0 = self.C_B_1_DS_p(C__unet__D_Output_0)
        output_0016 = self.C__D_B_1_0_compress_p(C__unet__D_Output_1_shortcut_0)
        output_0017 = spatial_dropout_inference(output_0016, 0.3, True)
        output_0018 = concatRelu(output_0017)
        output_0019 = self.C__D_B_1_0_0_p(output_0018)
        C__unet__D_Output_1_comp_0 = self.C__D_B_1_0_decompress_p(output_0019)
        C__unet__D_Output_1_shortcut_1 = C__unet__D_Output_1_shortcut_0 + C__unet__D_Output_1_comp_0
        output_0022 = self.C__D_B_1_1_compress_p(C__unet__D_Output_1_shortcut_1)
        output_0023 = spatial_dropout_inference(output_0022, 0.3, True)
        output_0024 = concatRelu(output_0023)
        output_0025 = self.C__D_B_1_1_0_p(output_0024)
        C__unet__D_Output_1_comp_1 = self.C__D_B_1_1_decompress_p(output_0025)
        C__unet__D_Output_1 = C__unet__D_Output_1_shortcut_1 + C__unet__D_Output_1_comp_1
        C__unet__D_Output_2_shortcut_0 = self.C_B_2_DS_p(C__unet__D_Output_1)
        output_0029 = self.C__D_B_2_0_compress_p(C__unet__D_Output_2_shortcut_0)
        output_0030 = spatial_dropout_inference(output_0029, 0.3, True)
        output_0031 = concatRelu(output_0030)
        output_0032 = self.C__D_B_2_0_0_p(output_0031)
        C__unet__D_Output_2_comp_0 = self.C__D_B_2_0_decompress_p(output_0032)
        C__unet__D_Output_2_shortcut_1 = C__unet__D_Output_2_shortcut_0 + C__unet__D_Output_2_comp_0
        output_0035 = self.C__D_B_2_1_compress_p(C__unet__D_Output_2_shortcut_1)
        output_0036 = spatial_dropout_inference(output_0035, 0.3, True)
        output_0037 = concatRelu(output_0036)
        output_0038 = self.C__D_B_2_1_0_p(output_0037)
        C__unet__D_Output_2_comp_1 = self.C__D_B_2_1_decompress_p(output_0038)
        C__unet__D_Output_2 = C__unet__D_Output_2_shortcut_1 + C__unet__D_Output_2_comp_1
        C__unet__D_Output_3_shortcut_0 = self.C_B_3_DS_p(C__unet__D_Output_2)
        output_0042 = self.C__D_B_3_0_compress_p(C__unet__D_Output_3_shortcut_0)
        output_0043 = spatial_dropout_inference(output_0042, 0.3, True)
        output_0044 = concatRelu(output_0043)
        output_0045 = self.C__D_B_3_0_0_p(output_0044)
        C__unet__D_Output_3_comp_0 = self.C__D_B_3_0_decompress_p(output_0045)
        C__unet__D_Output_3_shortcut_1 = C__unet__D_Output_3_shortcut_0 + C__unet__D_Output_3_comp_0
        output_0048 = self.C__D_B_3_1_compress_p(C__unet__D_Output_3_shortcut_1)
        output_0049 = spatial_dropout_inference(output_0048, 0.3, True)
        output_0050 = concatRelu(output_0049)
        output_0051 = self.C__D_B_3_1_0_p(output_0050)
        C__unet__D_Output_3_comp_1 = self.C__D_B_3_1_decompress_p(output_0051)
        C__unet__D_Output_3 = C__unet__D_Output_3_shortcut_1 + C__unet__D_Output_3_comp_1
        C__unet__D_Output_4_shortcut_0 = self.C_B_4_DS_p(C__unet__D_Output_3)
        output_0055 = self.C__D_B_4_0_compress_p(C__unet__D_Output_4_shortcut_0)
        output_0056 = spatial_dropout_inference(output_0055, 0.3, True)
        output_0057 = concatRelu(output_0056)
        output_0058 = self.C__D_B_4_0_0_p(output_0057)
        C__unet__D_Output_4_comp_0 = self.C__D_B_4_0_decompress_p(output_0058)
        C__unet__D_Output_4_shortcut_1 = C__unet__D_Output_4_shortcut_0 + C__unet__D_Output_4_comp_0
        output_0061 = self.C__D_B_4_1_compress_p(C__unet__D_Output_4_shortcut_1)
        output_0062 = spatial_dropout_inference(output_0061, 0.3, True)
        output_0063 = concatRelu(output_0062)
        output_0064 = self.C__D_B_4_1_0_p(output_0063)
        C__unet__D_Output_4_comp_1 = self.C__D_B_4_1_decompress_p(output_0064)
        C__unet__D_Output_4 = C__unet__D_Output_4_shortcut_1 + C__unet__D_Output_4_comp_1
        C__unet__U_Output_4_upsampled = self.C_B_4_US_p(C__unet__D_Output_4)
        C__unet__U_Output_4_upsampled_summed = C__unet__U_Output_4_upsampled + C__unet__D_Output_3
        output_0069 = self.C__U_B_4_0_compress_p(C__unet__U_Output_4_upsampled_summed)
        output_0070 = spatial_dropout_inference(output_0069, 0.3, True)
        output_0071 = concatRelu(output_0070)
        output_0072 = self.C__U_B_4_0_0_p(output_0071)
        C__unet__U_Output_4_comp_0 = self.C__U_B_4_0_decompress_p(output_0072)
        C__unet__U_Output_4_shortcut_1 = C__unet__U_Output_4_upsampled_summed + C__unet__U_Output_4_comp_0
        output_0075 = self.C__U_B_4_1_compress_p(C__unet__U_Output_4_shortcut_1)
        output_0076 = spatial_dropout_inference(output_0075, 0.3, True)
        output_0077 = concatRelu(output_0076)
        output_0078 = self.C__U_B_4_1_0_p(output_0077)
        C__unet__U_Output_4_comp_1 = self.C__U_B_4_1_decompress_p(output_0078)
        C__unet__U_Output_4 = C__unet__U_Output_4_shortcut_1 + C__unet__U_Output_4_comp_1
        C__unet__U_Output_3_upsampled = self.C_B_3_US_p(C__unet__U_Output_4)
        C__unet__U_Output_3_upsampled_summed = C__unet__U_Output_3_upsampled + C__unet__D_Output_2
        output_0083 = self.C__U_B_3_0_compress_p(C__unet__U_Output_3_upsampled_summed)
        output_0084 = spatial_dropout_inference(output_0083, 0.3, True)
        output_0085 = concatRelu(output_0084)
        output_0086 = self.C__U_B_3_0_0_p(output_0085)
        C__unet__U_Output_3_comp_0 = self.C__U_B_3_0_decompress_p(output_0086)
        C__unet__U_Output_3_shortcut_1 = C__unet__U_Output_3_upsampled_summed + C__unet__U_Output_3_comp_0
        output_0089 = self.C__U_B_3_1_compress_p(C__unet__U_Output_3_shortcut_1)
        output_0090 = spatial_dropout_inference(output_0089, 0.3, True)
        output_0091 = concatRelu(output_0090)
        output_0092 = self.C__U_B_3_1_0_p(output_0091)
        C__unet__U_Output_3_comp_1 = self.C__U_B_3_1_decompress_p(output_0092)
        C__unet__U_Output_3 = C__unet__U_Output_3_shortcut_1 + C__unet__U_Output_3_comp_1
        C__unet__U_Output_2_upsampled = self.C_B_2_US_p(C__unet__U_Output_3)
        C__unet__U_Output_2_upsampled_summed = C__unet__U_Output_2_upsampled + C__unet__D_Output_1
        output_0097 = self.C__U_B_2_0_compress_p(C__unet__U_Output_2_upsampled_summed)
        output_0098 = spatial_dropout_inference(output_0097, 0.3, True)
        output_0099 = concatRelu(output_0098)
        output_0100 = self.C__U_B_2_0_0_p(output_0099)
        C__unet__U_Output_2_comp_0 = self.C__U_B_2_0_decompress_p(output_0100)
        C__unet__U_Output_2_shortcut_1 = C__unet__U_Output_2_upsampled_summed + C__unet__U_Output_2_comp_0
        output_0103 = self.C__U_B_2_1_compress_p(C__unet__U_Output_2_shortcut_1)
        output_0104 = spatial_dropout_inference(output_0103, 0.3, True)
        output_0105 = concatRelu(output_0104)
        output_0106 = self.C__U_B_2_1_0_p(output_0105)
        C__unet__U_Output_2_comp_1 = self.C__U_B_2_1_decompress_p(output_0106)
        C__unet__U_Output_2 = C__unet__U_Output_2_shortcut_1 + C__unet__U_Output_2_comp_1
        C__unet__U_Output_1_upsampled = self.C_B_1_US_p(C__unet__U_Output_2)
        C__unet__U_Output_1_upsampled_summed = C__unet__U_Output_1_upsampled + C__unet__D_Output_0
        output_0111 = self.C__U_B_1_0_compress_p(C__unet__U_Output_1_upsampled_summed)
        output_0112 = spatial_dropout_inference(output_0111, 0.3, True)
        output_0113 = concatRelu(output_0112)
        output_0114 = self.C__U_B_1_0_0_p(output_0113)
        C__unet__U_Output_1_comp_0 = self.C__U_B_1_0_decompress_p(output_0114)
        C__unet__U_Output_1_shortcut_1 = C__unet__U_Output_1_upsampled_summed + C__unet__U_Output_1_comp_0
        output_0117 = self.C__U_B_1_1_compress_p(C__unet__U_Output_1_shortcut_1)
        output_0118 = spatial_dropout_inference(output_0117, 0.3, True)
        output_0119 = concatRelu(output_0118)
        output_0120 = self.C__U_B_1_1_0_p(output_0119)
        C__unet__U_Output_1_comp_1 = self.C__U_B_1_1_decompress_p(output_0120)
        C__unet__U_Output_1 = C__unet__U_Output_1_shortcut_1 + C__unet__U_Output_1_comp_1
        C__unet_out_upsampled = self.C_B_0_US_p(C__unet__U_Output_1)
        C__unet_out_upsampled_summed = C__unet_out_upsampled + C__unet_in
        output_0125 = self.C__U_B_0_0_compress_p(C__unet_out_upsampled_summed)
        output_0126 = spatial_dropout_inference(output_0125, 0.3, True)
        output_0127 = concatRelu(output_0126)
        output_0128 = self.C__U_B_0_0_0_p(output_0127)
        C__unet_out_comp_0 = self.C__U_B_0_0_decompress_p(output_0128)
        C__unet_out_shortcut_1 = C__unet_out_upsampled_summed + C__unet_out_comp_0
        output_0131 = self.C__U_B_0_1_compress_p(C__unet_out_shortcut_1)
        output_0132 = spatial_dropout_inference(output_0131, 0.3, True)
        output_0133 = concatRelu(output_0132)
        output_0134 = self.C__U_B_0_1_0_p(output_0133)
        C__unet_out_comp_1 = self.C__U_B_0_1_decompress_p(output_0134)
        C__unet_out = C__unet_out_shortcut_1 + C__unet_out_comp_1
        estimate = self.C__out(C__unet_out)
        return estimate
