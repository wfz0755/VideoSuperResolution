import argparse
from scipy import signal
from scipy import misc
import numpy as np
import torch
import cv2
import re
import os
import math
from utilities.video_op import *
from skimage.measure import compare_ssim as calc_ssim
from skimage.measure import compare_psnr as calc_psnr

def get_parser():
    parser = argparse.ArgumentParser(description='FASTER')

    parser.add_argument('--gpu', type=int, default=0,
                        help='the number of gpu that will be used, by default we use cpu')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--dir_data', type=str, default='../training_data',
                        help='dataset directory')
    parser.add_argument('--train_generic_model', action='store_true',
                        help='train a generic model based on div2k dataset, without this argument, the model will be trained on a video by default')
    parser.add_argument('--dir_test', type=str, default=None,
                        help='low resolution video for testing')
    parser.add_argument('--scale', type=int, default=4,
                        help='super resolution scale(x4,x8)')
    parser.add_argument('--width', type=int, default=480,
                        help='Input LR frame width')
    parser.add_argument('--height', type=int, default=264,
                        help='Input LR frame height')

    parser.add_argument('--patch_size', type=int, default=192,
                        help='patch size for trainning')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--video_name', type=str, default=None,
                        help='Video name for training or testing, remember to remove extension')
    parser.add_argument('--color_mode',type=str,default='RGB',
                        help='Use what color space (RGB/YUV)')

    # Model specifications
    parser.add_argument('--model', default='MDSR',
                        help='model name(MDSR,RCAN)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train_dir', type=str, default='../checkpoints',
                        help='pre-trained model directory, we can use the model for testing and further fine tuning')
    parser.add_argument('--n_resblocks', type=int, default=20,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--load_model_type',type=str,default='generic',
                        help='choose the kind of checkpoint that you what: generic/train_on_video/train_from_generic')

    # Option for Residual channel attention network (RCAN)
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    # Training specifications

    parser.add_argument('--mode', type=str, default='train',
                        help='choose to train/test/fast_related')
    parser.add_argument('--reset', action='store_true',
                        help='reset the training/use checkpoint trained from scratch')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do experiment per every N batches')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='use self-ensemble method for experiment')
    parser.add_argument('--test_only', action='store_true',
                        help='set this option to experiment the model')
    parser.add_argument('--test_speed',action='store_true',
                        help='set this option to test inference speed of model')
    parser.add_argument('--test_all_speed',action='store_true',
                        help='set this option to test inference speed of all models')

    parser.add_argument('--gan_k', type=int, default=1,
                        help='k value for adversarial loss')
    parser.add_argument('--debug',  action='store_true',
                        help='print loss when enabled')
    # Log specifications
    parser.add_argument('--load', type=str, default='',
                        help='checkpoint name to load')
    parser.add_argument('--save_models', action='store_true',
                        help='save the trained model')
    parser.add_argument('--save_results', action='store_true',
                        help='save output results(figure and psnr/ssim/fps)')
    parser.add_argument('--save_interpolated', action='store_true',
                        help='save interpolated video with sr video')

    #FAST options
    parser.add_argument('--inference_mode', type=str, default='normal',
                        help='Choose inference model between normal and fast, normal means to do inference frame by frame, fast make use of pixel transfering, take an YUV file as input.')
    parser.add_argument('--transfer_threshold',type=float, default=10,
                        help='Choose the corresponding transfer_threshold for blocks of residual')
    parser.add_argument('--deblocking',action='store_true',
                        help='Enabling the deblocking CNN or not')
    parser.add_argument('--re_encode',action='store_true',
                        help='Must encode the video using HEVC encoder at the first time we apply FAST to it')
    
    return parser


def np2Tensor(np_arr): 
    if(type(np_arr) is list ):
        tensor = []
        for i in range(len(np_arr)):
            tensor.append(torch.from_numpy(np_arr[i].transpose((2,0,1))).float())
    else:
        tensor= torch.from_numpy(np_arr.transpose((0,3,1,2))).float()
    return tensor

def get_psnr(gt,scale_result,color_mode):
    if(color_mode=='RGB'):
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2YUV)
        scale_result = cv2.cvtColor(scale_result, cv2.COLOR_RGB2YUV)
    return calc_psnr(gt[:,:,0],scale_result[:,:,0],255)
   
def get_ssim(gt,scale_result,color_mode):
    if(color_mode=='RGB'):
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2YUV)
        scale_result = cv2.cvtColor(scale_result, cv2.COLOR_RGB2YUV)
    return calc_ssim(gt[:,:,0],scale_result[:,:,0])


class res_block_info():
    __slots__ = ('x', 'y','w','residual')
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w = None
        self.residual = None
class pu_info():
    __slots__ = ('x', 'y','w','h','mv_x','mv_y','t_r')
    def __init__(self,pu_mv):
        self.x = pu_mv['x']
        self.y = pu_mv['y']
        self.w = pu_mv['w']
        self.h= pu_mv['h']
        self.mv_x = pu_mv['mv_x']
        self.mv_y = pu_mv['mv_y']
        self.t_r = pu_mv['t_r']
        
def parse_dump_file(dump_str): # we only need to get motion vectors and residuals
    pu_all = []
    res_all = []
    res_frame = [] #Y only
    pu_frame = []
    tid= -1
    dump_str = dump_str.split('\n')
    for line in dump_str:
        segments =  line.split(":",maxsplit=1)
        if(segments[0] == 'Coeff'): # coefficients
            #print("coeff")
            if(tid != 0):
                continue
            vals = np.fromstring(segments[1],dtype=np.int16,sep=' ')
            w = vals[2]
            n = w*w
            residuals = vals[(4+3*n):(4+4*n)]
            res_frame[-1].w = w
            res_frame[-1].residual = residuals

        elif(segments[0] == 'TU'): # tranformation unit
            vals = np.fromstring(segments[1],dtype=np.int16,sep=' ')
            x = vals[0]
            y = vals[1]
            text = vals[2]-1 #indicate Y/Cb/Cr
            tid= max(0,text)
            if(tid == 0):
                res_struct = res_block_info(x,y)
                res_frame.append(res_struct)
        elif(segments[0] == 'MV'): # motion vector
            params = segments[1].split()
            pu_mv = {}
            for i in range(0,len(params)):
                p = params[i].split("=")
                pu_mv[p[0]] = int(p[1])
            if(pu_mv['w']!=0):
                pu_frame.append(pu_info(pu_mv))
        elif(segments[0][0:3] == 'POC'): # time to start a new frame
            #print("POC, start a new frame")
            res_all.append(res_frame)
            pu_all.append(pu_frame)
            res_frame = []
            pu_frame = []
    return (pu_all,res_all)
def reconstruct_residual(res_all,size):# height,width
    print("Reconstruct residual")
    total_frames = len(res_all)
    recon_res = np.zeros((total_frames, size[0],size[1]),dtype=np.int16) #since the implementation of HM seems to be buggy(dumped info of chroma is not reasonable),we use residual of Y only 
   # hitmap = np.zeros((total_frames, size[0],size[1]),dtype=np.int8)
    #channel = 0
    #upscale_mat = np.array([[1,1],[1,1]])
    for frame_id in range(total_frames):
        for block in range(len(res_all[frame_id]) ):
            if(res_all[frame_id][block].w is None):
                continue
            x0 = res_all[frame_id][block].x # col
            y0 = res_all[frame_id][block].y # row
            w = res_all[frame_id][block].w
            tmp = np.reshape(res_all[frame_id][block].residual,(w,w))
            # if(channel != 0): # for U and V component
            #     w = 2*w
            #     tmp = np.kron(tmp,upscale_mat)
            recon_res[frame_id, y0:(y0+w),x0:(x0+w) ]= tmp
           # hitmap[frame_id, y0:(y0+w),x0:(x0+w)] += 1
        
   # assert( np.all(hitmap <= 1  ) == True ) #check if the residual blocks cover
    return recon_res
def quantize(sr):
    #pixel_range = 255 / rgb_range
    return sr.clamp(0, 255).round()

def get_ref_patch_sr(sr_frame,r,c,w,h):
    r0 = r -3
    r1 = r + h + 3+1
    c0 = c -3
    c1 = c + w + 3 +1
    h_max,w_max = sr_frame.shape
    tmp_c = np.arange(c0,c1)
    tmp_r = np.arange(r0,r1)
    tmp_c = np.clip(tmp_c,0,w_max-1)
    tmp_r = np.clip(tmp_r,0,h_max-1)
    selected_rows = sr_frame[tmp_r,:]
    ref_patch = selected_rows[:,tmp_c]
    #ref_patch = np.zeros((h,w),dtype=sr_frame.dtype)
    #for ind_i,i in enumerate(tmp_r):
    #    for ind_j,j in enumerate(tmp_c):
    #        ref_patch[ind_i,ind_j] = sr_frame[i,j,0]
    return ref_patch

def fractional_interpolate_sr(ref_patch,mv_x_ind,mv_y_ind):
    #print(ref_patch.dtype)
    filter_coef = np.array([[0, 0, 0, 64, 0, 0, 0, 0],
                            [-1, 4, -10, 58, 17, -5, 1, 0],
                            [-1, 4, -11, 40, 40, -11, 4, -1],
                            [0, 1, -5, 17, 58, -10, 4, -1]])
    patch1 = signal.fftconvolve(ref_patch,np.rot90(filter_coef[mv_x_ind,:].reshape(1,-1),2),'valid')
    patch2 = signal.fftconvolve(patch1,np.transpose(np.rot90(filter_coef[mv_y_ind,:].reshape(1,-1),2)),'valid')
    #patch1 = signal.convolve2d(ref_patch,np.rot90(filter_coef[mv_x_ind,:].reshape(1,-1),2),'valid')
    #patch2 = signal.convolve2d(patch1,np.transpose(np.rot90(filter_coef[mv_y_ind,:].reshape(1,-1),2)),'valid')
    interp_patch = np.floor(patch2/64 + 0.01)
    interp_patch = np.round(interp_patch / 64)
    interp_patch = np.clip(interp_patch,0,255)
   # print(interp_patch.shape)
    return interp_patch


def sr_interpolate(sr_frame,x_h,y_h,w_h,h_h,mv_x,mv_y):
    xp = x_h + math.floor(mv_x+0.01)
    yp = y_h + math.floor(mv_y+0.01)

    ref_patch = get_ref_patch_sr(sr_frame,yp,xp,w_h,h_h)
    mv_x_ind = round(mv_x*4)%4
    mv_y_ind = round(mv_y*4)%4
    ref_patch = fractional_interpolate_sr(ref_patch,mv_x_ind,mv_y_ind)
    return ref_patch
