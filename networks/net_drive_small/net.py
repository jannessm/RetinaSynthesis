
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

        self.C__in = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__in.weight.data = torch.from_numpy(self.np_parameters['C__in_filters'])
        self.C__in.bias.data = torch.from_numpy(self.np_parameters['C__in_biases'])
        self.C_B_0_DS_p = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_0_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_0_DS_p_filters'])
        self.C_B_0_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_0_DS_p_biases'])
        self.C__D_B_0_0_compress_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_compress_p_filters'])
        self.C__D_B_0_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_compress_p_biases'])
        self.C__D_B_0_0_0_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_0_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_0_p_filters'])
        self.C__D_B_0_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_0_p_biases'])
        self.C__D_B_0_0_decompress_p = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_decompress_p_filters'])
        self.C__D_B_0_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_0_decompress_p_biases'])
        self.C__D_B_0_1_compress_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_compress_p_filters'])
        self.C__D_B_0_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_compress_p_biases'])
        self.C__D_B_0_1_0_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_0_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_0_p_filters'])
        self.C__D_B_0_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_0_p_biases'])
        self.C__D_B_0_1_decompress_p = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_0_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_decompress_p_filters'])
        self.C__D_B_0_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_0_1_decompress_p_biases'])
        self.C_B_1_DS_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_1_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_1_DS_p_filters'])
        self.C_B_1_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_1_DS_p_biases'])
        self.C__D_B_1_0_compress_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_compress_p_filters'])
        self.C__D_B_1_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_compress_p_biases'])
        self.C__D_B_1_0_0_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_1_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_0_p_filters'])
        self.C__D_B_1_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_0_p_biases'])
        self.C__D_B_1_0_decompress_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_decompress_p_filters'])
        self.C__D_B_1_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_0_decompress_p_biases'])
        self.C__D_B_1_1_compress_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_compress_p_filters'])
        self.C__D_B_1_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_compress_p_biases'])
        self.C__D_B_1_1_0_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_1_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_0_p_filters'])
        self.C__D_B_1_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_0_p_biases'])
        self.C__D_B_1_1_decompress_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_1_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_decompress_p_filters'])
        self.C__D_B_1_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_1_1_decompress_p_biases'])
        self.C_B_2_DS_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_2_DS_p.weight.data = torch.from_numpy(self.np_parameters['C_B_2_DS_p_filters'])
        self.C_B_2_DS_p.bias.data = torch.from_numpy(self.np_parameters['C_B_2_DS_p_biases'])
        self.C__D_B_2_0_compress_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_compress_p_filters'])
        self.C__D_B_2_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_compress_p_biases'])
        self.C__D_B_2_0_0_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_2_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_0_p_filters'])
        self.C__D_B_2_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_0_p_biases'])
        self.C__D_B_2_0_decompress_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_decompress_p_filters'])
        self.C__D_B_2_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_0_decompress_p_biases'])
        self.C__D_B_2_1_compress_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_compress_p_filters'])
        self.C__D_B_2_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_compress_p_biases'])
        self.C__D_B_2_1_0_p = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__D_B_2_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_0_p_filters'])
        self.C__D_B_2_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_0_p_biases'])
        self.C__D_B_2_1_decompress_p = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__D_B_2_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_decompress_p_filters'])
        self.C__D_B_2_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__D_B_2_1_decompress_p_biases'])
        self.C_B_2_US_p = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_2_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_2_US_p_filters'])
        self.C_B_2_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_2_US_p_biases'])
        self.C__U_B_2_0_compress_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_compress_p_filters'])
        self.C__U_B_2_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_compress_p_biases'])
        self.C__U_B_2_0_0_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_2_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_0_p_filters'])
        self.C__U_B_2_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_0_p_biases'])
        self.C__U_B_2_0_decompress_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_decompress_p_filters'])
        self.C__U_B_2_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_0_decompress_p_biases'])
        self.C__U_B_2_1_compress_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_compress_p_filters'])
        self.C__U_B_2_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_compress_p_biases'])
        self.C__U_B_2_1_0_p = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_2_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_0_p_filters'])
        self.C__U_B_2_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_0_p_biases'])
        self.C__U_B_2_1_decompress_p = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_2_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_decompress_p_filters'])
        self.C__U_B_2_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_2_1_decompress_p_biases'])
        self.C_B_1_US_p = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_1_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_1_US_p_filters'])
        self.C_B_1_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_1_US_p_biases'])
        self.C__U_B_1_0_compress_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_compress_p_filters'])
        self.C__U_B_1_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_compress_p_biases'])
        self.C__U_B_1_0_0_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_1_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_0_p_filters'])
        self.C__U_B_1_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_0_p_biases'])
        self.C__U_B_1_0_decompress_p = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_decompress_p_filters'])
        self.C__U_B_1_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_0_decompress_p_biases'])
        self.C__U_B_1_1_compress_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_compress_p_filters'])
        self.C__U_B_1_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_compress_p_biases'])
        self.C__U_B_1_1_0_p = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_1_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_0_p_filters'])
        self.C__U_B_1_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_0_p_biases'])
        self.C__U_B_1_1_decompress_p = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_1_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_decompress_p_filters'])
        self.C__U_B_1_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_1_1_decompress_p_biases'])
        self.C_B_0_US_p = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.C_B_0_US_p.weight.data = torch.from_numpy(self.np_parameters['C_B_0_US_p_filters'])
        self.C_B_0_US_p.bias.data = torch.from_numpy(self.np_parameters['C_B_0_US_p_biases'])
        self.C__U_B_0_0_compress_p = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_0_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_compress_p_filters'])
        self.C__U_B_0_0_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_compress_p_biases'])
        self.C__U_B_0_0_0_p = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_0_0_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_0_p_filters'])
        self.C__U_B_0_0_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_0_p_biases'])
        self.C__U_B_0_0_decompress_p = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_0_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_decompress_p_filters'])
        self.C__U_B_0_0_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_0_decompress_p_biases'])
        self.C__U_B_0_1_compress_p = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_1_compress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_compress_p_filters'])
        self.C__U_B_0_1_compress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_compress_p_biases'])
        self.C__U_B_0_1_0_p = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__U_B_0_1_0_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_0_p_filters'])
        self.C__U_B_0_1_0_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_0_p_biases'])
        self.C__U_B_0_1_decompress_p = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.C__U_B_0_1_decompress_p.weight.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_decompress_p_filters'])
        self.C__U_B_0_1_decompress_p.bias.data = torch.from_numpy(self.np_parameters['C__U_B_0_1_decompress_p_biases'])
        self.C__out = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.C__out.weight.data = torch.from_numpy(self.np_parameters['C__out_filters'])
        self.C__out.bias.data = torch.from_numpy(self.np_parameters['C__out_biases'])

    def forward(self, input):
        C_unet_in = self.C__in(input)
        C_unet__D_Output_0_shortcut_0 = self.C_B_0_DS_p(C_unet_in)
        output_0003 = self.C__D_B_0_0_compress_p(C_unet__D_Output_0_shortcut_0)
        output_0004 = spatial_dropout_inference(output_0003, 0.5, True)
        output_0005 = concatRelu(output_0004)
        output_0006 = self.C__D_B_0_0_0_p(output_0005)
        C_unet__D_Output_0_comp_0 = self.C__D_B_0_0_decompress_p(output_0006)
        C_unet__D_Output_0_shortcut_1 = C_unet__D_Output_0_shortcut_0 + C_unet__D_Output_0_comp_0
        output_0009 = self.C__D_B_0_1_compress_p(C_unet__D_Output_0_shortcut_1)
        output_0010 = spatial_dropout_inference(output_0009, 0.5, True)
        output_0011 = concatRelu(output_0010)
        output_0012 = self.C__D_B_0_1_0_p(output_0011)
        C_unet__D_Output_0_comp_1 = self.C__D_B_0_1_decompress_p(output_0012)
        C_unet__D_Output_0 = C_unet__D_Output_0_shortcut_1 + C_unet__D_Output_0_comp_1
        C_unet__D_Output_1_shortcut_0 = self.C_B_1_DS_p(C_unet__D_Output_0)
        output_0016 = self.C__D_B_1_0_compress_p(C_unet__D_Output_1_shortcut_0)
        output_0017 = spatial_dropout_inference(output_0016, 0.5, True)
        output_0018 = concatRelu(output_0017)
        output_0019 = self.C__D_B_1_0_0_p(output_0018)
        C_unet__D_Output_1_comp_0 = self.C__D_B_1_0_decompress_p(output_0019)
        C_unet__D_Output_1_shortcut_1 = C_unet__D_Output_1_shortcut_0 + C_unet__D_Output_1_comp_0
        output_0022 = self.C__D_B_1_1_compress_p(C_unet__D_Output_1_shortcut_1)
        output_0023 = spatial_dropout_inference(output_0022, 0.5, True)
        output_0024 = concatRelu(output_0023)
        output_0025 = self.C__D_B_1_1_0_p(output_0024)
        C_unet__D_Output_1_comp_1 = self.C__D_B_1_1_decompress_p(output_0025)
        C_unet__D_Output_1 = C_unet__D_Output_1_shortcut_1 + C_unet__D_Output_1_comp_1
        C_unet__D_Output_2_shortcut_0 = self.C_B_2_DS_p(C_unet__D_Output_1)
        output_0029 = self.C__D_B_2_0_compress_p(C_unet__D_Output_2_shortcut_0)
        output_0030 = spatial_dropout_inference(output_0029, 0.5, True)
        output_0031 = concatRelu(output_0030)
        output_0032 = self.C__D_B_2_0_0_p(output_0031)
        C_unet__D_Output_2_comp_0 = self.C__D_B_2_0_decompress_p(output_0032)
        C_unet__D_Output_2_shortcut_1 = C_unet__D_Output_2_shortcut_0 + C_unet__D_Output_2_comp_0
        output_0035 = self.C__D_B_2_1_compress_p(C_unet__D_Output_2_shortcut_1)
        output_0036 = spatial_dropout_inference(output_0035, 0.5, True)
        output_0037 = concatRelu(output_0036)
        output_0038 = self.C__D_B_2_1_0_p(output_0037)
        C_unet__D_Output_2_comp_1 = self.C__D_B_2_1_decompress_p(output_0038)
        C_unet__D_Output_2 = C_unet__D_Output_2_shortcut_1 + C_unet__D_Output_2_comp_1
        C_unet__U_Output_2_upsampled = self.C_B_2_US_p(C_unet__D_Output_2)
        C_unet__U_Output_2_upsampled_summed = C_unet__U_Output_2_upsampled + C_unet__D_Output_1
        output_0043 = self.C__U_B_2_0_compress_p(C_unet__U_Output_2_upsampled_summed)
        output_0044 = spatial_dropout_inference(output_0043, 0.5, True)
        output_0045 = concatRelu(output_0044)
        output_0046 = self.C__U_B_2_0_0_p(output_0045)
        C_unet__U_Output_2_comp_0 = self.C__U_B_2_0_decompress_p(output_0046)
        C_unet__U_Output_2_shortcut_1 = C_unet__U_Output_2_upsampled_summed + C_unet__U_Output_2_comp_0
        output_0049 = self.C__U_B_2_1_compress_p(C_unet__U_Output_2_shortcut_1)
        output_0050 = spatial_dropout_inference(output_0049, 0.5, True)
        output_0051 = concatRelu(output_0050)
        output_0052 = self.C__U_B_2_1_0_p(output_0051)
        C_unet__U_Output_2_comp_1 = self.C__U_B_2_1_decompress_p(output_0052)
        C_unet__U_Output_2 = C_unet__U_Output_2_shortcut_1 + C_unet__U_Output_2_comp_1
        C_unet__U_Output_1_upsampled = self.C_B_1_US_p(C_unet__U_Output_2)
        C_unet__U_Output_1_upsampled_summed = C_unet__U_Output_1_upsampled + C_unet__D_Output_0
        output_0057 = self.C__U_B_1_0_compress_p(C_unet__U_Output_1_upsampled_summed)
        output_0058 = spatial_dropout_inference(output_0057, 0.5, True)
        output_0059 = concatRelu(output_0058)
        output_0060 = self.C__U_B_1_0_0_p(output_0059)
        C_unet__U_Output_1_comp_0 = self.C__U_B_1_0_decompress_p(output_0060)
        C_unet__U_Output_1_shortcut_1 = C_unet__U_Output_1_upsampled_summed + C_unet__U_Output_1_comp_0
        output_0063 = self.C__U_B_1_1_compress_p(C_unet__U_Output_1_shortcut_1)
        output_0064 = spatial_dropout_inference(output_0063, 0.5, True)
        output_0065 = concatRelu(output_0064)
        output_0066 = self.C__U_B_1_1_0_p(output_0065)
        C_unet__U_Output_1_comp_1 = self.C__U_B_1_1_decompress_p(output_0066)
        C_unet__U_Output_1 = C_unet__U_Output_1_shortcut_1 + C_unet__U_Output_1_comp_1
        C_unet_out_upsampled = self.C_B_0_US_p(C_unet__U_Output_1)
        C_unet_out_upsampled_summed = C_unet_out_upsampled + C_unet_in
        output_0071 = self.C__U_B_0_0_compress_p(C_unet_out_upsampled_summed)
        output_0072 = spatial_dropout_inference(output_0071, 0.5, True)
        output_0073 = concatRelu(output_0072)
        output_0074 = self.C__U_B_0_0_0_p(output_0073)
        C_unet_out_comp_0 = self.C__U_B_0_0_decompress_p(output_0074)
        C_unet_out_shortcut_1 = C_unet_out_upsampled_summed + C_unet_out_comp_0
        output_0077 = self.C__U_B_0_1_compress_p(C_unet_out_shortcut_1)
        output_0078 = spatial_dropout_inference(output_0077, 0.5, True)
        output_0079 = concatRelu(output_0078)
        output_0080 = self.C__U_B_0_1_0_p(output_0079)
        C_unet_out_comp_1 = self.C__U_B_0_1_decompress_p(output_0080)
        C_unet_out = C_unet_out_shortcut_1 + C_unet_out_comp_1
        estimate = self.C__out(C_unet_out)
        return estimate
