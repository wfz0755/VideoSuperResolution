import matplotlib
import argparse
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import  feature_extraction
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tqdm import tqdm
from torch.autograd import Variable
#from queue import Queue as thread_queue
# from threading import Thread

from multiprocessing import Process, Queue, Lock

from model import *
from utilities import *
from data import *





def load_checkpoint(model, cur_args):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    check_point_name = None
    if(cur_args.load_model_type=="generic"):
        if(cur_args.model == "MDSR"):
            save_model_name = "{}_r{}_f{}_x{}_{}.pt".format(cur_args.model, cur_args.n_resblocks, cur_args.n_feats,cur_args.scale,cur_args.color_mode)
        elif(cur_args.model=="RCAN"):
            save_model_name=  "{}_rg{}_r{}_f{}_x{}_{}.pt".format(cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,cur_args.scale,cur_args.color_mode)
    elif(cur_args.load_model_type=="train_on_video"):  
        if(cur_args.model == "MDSR"):
            save_model_name = "{}_{}_r{}_f{}_x{}_{}_fromscratch.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resblocks, cur_args.n_feats,cur_args.scale,cur_args.color_mode)
        elif(cur_args.model=="RCAN"):
            save_model_name=  "{}_{}_rg{}_r{}_f{}_x{}_{}_fromscratch.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,cur_args.scale,cur_args.color_mode)
    elif(cur_args.load_model_type=="train_from_generic"):
        if(cur_args.model == "MDSR"):
            save_model_name = "{}_{}_r{}_f{}_x{}_{}.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resblocks, cur_args.n_feats,cur_args.scale,cur_args.color_mode)
        elif(cur_args.model=="RCAN"):
            save_model_name=  "{}_{}_rg{}_r{}_f{}_x{}_{}.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,cur_args.scale,cur_args.color_mode)
    else:
        print("No such checkpoint: {}".format(cur_args.load_model_type))

    if(cur_args.load != ''):
        save_model_name = cur_args.load
    check_point_name = os.path.join(cur_args.pre_train_dir, save_model_name)        
    if os.path.isfile(check_point_name):
        print("=> loading checkpoint '{}'".format(check_point_name))
        checkpoint = torch.load(check_point_name)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> model loaded")
    else:
        print("=> no checkpoint found at '{}'".format(check_point_name))
    return model

def load_certain_checkpoint(model,ckpt_name,cur_args):
    check_point_name = os.path.join(cur_args.pre_train_dir, ckpt_name)
    if os.path.isfile(check_point_name):
        print("=> loading checkpoint '{}'".format(check_point_name))
        checkpoint = torch.load(check_point_name)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> model loaded")
    else:
        print("=> no checkpoint found at '{}'".format(check_point_name))
    return model

def get_avg_psnr_ssim(model,images,device,cur_args):
    avg_psnr = 0
    avg_ssim = 0
    bic_avg_psnr = 0
    bic_avg_ssim = 0
    model.eval()
    torch.no_grad()
    total_frames = len(images)
    #images = np.stack(images,axis=0)
    lr_images = img_downsample(images,cur_args.scale)
    #print(images[0].shape)
    #print(lr_images.shape)
    print("total_frames:{}".format(total_frames))
    bicubic = []
    #sr_result = []
    for i in range(total_frames):
        h,w,_ = images[i].shape
        tmp = cv2.resize(lr_images[i],(w,h),interpolation=cv2.INTER_CUBIC)
        bicubic.append(tmp) 
    lr_images = np2Tensor(lr_images)# now to list of (c,h,w)
    for i in range(total_frames):
        lr = lr_images[i].unsqueeze(0).to(device)
        sr = model(lr)
        sr  = quantize(sr)
        tmp = sr.data.cpu()
        tmp = tmp.numpy().squeeze(0).astype(np.uint8).transpose(1,2,0)
        psnr = get_psnr(images[i],tmp,cur_args.color_mode)
        ssim = get_ssim(images[i],tmp,cur_args.color_mode)
        avg_psnr += psnr
        avg_ssim += ssim
        psnr = get_psnr(images[i],bicubic[i],cur_args.color_mode)
        ssim = get_ssim(images[i],bicubic[i],cur_args.color_mode)
        bic_avg_psnr += psnr
        bic_avg_ssim += ssim
    return (avg_psnr/total_frames,avg_ssim/total_frames,bic_avg_psnr/total_frames,bic_avg_ssim/total_frames)

def get_avg_psnr_ssim_video(model,lr_frames,hr_frames,device,cur_args):
    avg_psnr = 0
    avg_ssim = 0
    bic_avg_psnr = 0
    bic_avg_ssim = 0
    model.eval()
    torch.no_grad()
    total_frames = len(lr_frames)
    
    print("total_frames:{}".format(total_frames))
    bicubic = []
    #sr_result = []
    h_h,w_h,_ = hr_frames[0].shape
    for i in range(total_frames):
        tmp = cv2.resize(lr_frames[i],(w_h,h_h),interpolation=cv2.INTER_CUBIC)
        bicubic.append(tmp) 
    lr_frames = np2Tensor(lr_frames)# now to list of (c,h,w)
    for i in range(total_frames):
        lr = lr_frames[i].unsqueeze(0).to(device)
        sr = model(lr)
        sr  = quantize(sr)
        tmp = sr.data.cpu()
        tmp = tmp.numpy().squeeze(0).astype(np.uint8).transpose(1,2,0)
        psnr = get_psnr(hr_frames[i],tmp,cur_args.color_mode)
        ssim = get_ssim(hr_frames[i],tmp,cur_args.color_mode)
        avg_psnr += psnr
        avg_ssim += ssim
        psnr = get_psnr(hr_frames[i],bicubic[i],cur_args.color_mode)
        ssim = get_ssim(hr_frames[i],bicubic[i],cur_args.color_mode)
        bic_avg_psnr += psnr
        bic_avg_ssim += ssim
    return (avg_psnr/total_frames,avg_ssim/total_frames,bic_avg_psnr/total_frames,bic_avg_ssim/total_frames)

def train_model(cur_args):
    if (args.train_generic_model == True):  # train a generic model using DIV2K
        print("Train generic model")
        patch_size = cur_args.patch_size
        scale = cur_args.scale
        if(cur_args.model == "MDSR"):
            save_model_name = "{}_r{}_f{}_x{}_{}.pt".format(cur_args.model, cur_args.n_resblocks, cur_args.n_feats,scale,cur_args.color_mode)
        elif(cur_args.model=="RCAN"):
            save_model_name=  "{}_rg{}_r{}_f{}_x{}_{}.pt".format(cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,scale,cur_args.color_mode)

        #load training data, in RGB/YUV accordingly
        images = load_images_from_folder(cur_args) #list containing images
        device = None

        #create model
        if(cur_args.model=="MDSR"):
            model = MDSR(cur_args)
        else:
            model = RCAN(cur_args)
        
        if(cur_args.gpu != 0):
            device = torch.device('cuda')
        model.to(device)    
        loss = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay=1e-3)
        avg_losses = []
        avg_loss = 0.0
        patches_per_frame = 16
        start_time = time.time()
        for epo in range(cur_args.epochs):
            running_loss = 0.0
            for i in range(len(images)):
                hr_patches = feature_extraction.image.extract_patches_2d(images[i], (patch_size*scale,patch_size*scale), patches_per_frame)
                lr_patches = np2Tensor(img_downsample(hr_patches, scale))
                hr_patches = np2Tensor(hr_patches)
                if(cur_args.gpu != 0):
                    lr_patches=lr_patches.to(device)
                    hr_patches=hr_patches.to(device)
                optimizer.zero_grad()
                outputs = model(lr_patches)
                cur_loss = loss(outputs, hr_patches)
                cur_loss.backward()
                optimizer.step()
                running_loss += cur_loss.item()
                avg_loss += cur_loss.item()
                if (i % 200 == 199):
                    if(cur_args.debug==True):
                        print('[%d, %5d] loss: %.3f' %(epo + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
                # print('[%d, %5d] loss: %.3f' % (epo + 1, i + 1, running_loss ))
                # running_loss =0.0
            avg_loss /= len(images)
            avg_losses.append(avg_loss)
            avg_loss = 0.0
        elapsed_time = time.time() - start_time
        print("Finished Training using {}".format( str(datetime.timedelta(seconds=elapsed_time))))
        if(cur_args.save_models == True):
            print("Save model as {}".format(save_model_name))
            state = { 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()} # can choose to use the optimizer or not
            torch.save(state,os.path.join(cur_args.pre_train_dir,save_model_name))

        #test performance on images (PSNR and SSIM)

        avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim = get_avg_psnr_ssim(model,images,device,cur_args)

        plt.plot(range(1, cur_args.epochs + 1), avg_losses, "r-")
        plt.axhline(y=min(avg_losses), color='b', linestyle='-')
        plt.title("Min loss:{:.3f}\nPSNR/SSIM {}:({:.2f}/{:.4f})\nBicubic:({:.2f}/{:.4f})".format(min(avg_losses),cur_args.model,avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim))
        plt.xlabel('Epoch')
        plt.ylabel('Average L1 Loss')
        plt.savefig('{}_{}epoch.png'.format("../result/"+save_model_name,cur_args.epochs))

    else:

        if (args.reset==True):

            print("train from the beginning using certain video")
            lr_width,lr_height = encode_video(cur_args)
            hr_height = lr_height*cur_args.scale
            hr_width = lr_width*cur_args.scale
            yuv_path,_ = decode_video(cur_args)
            cap = VideoCaptureYUV(yuv_path, (lr_height,lr_width))
            lr_frames = []
            while(1):
                ret,frame=cap.read_yuv()
                if ret:
                    lr_frames.append(frame)
                else:
                    break
            total_frames = len(lr_frames)

            hr_yuv_path = "../tmp_data/{}_hr.yuv".format(cur_args.video_name)
            cap_hr = VideoCaptureYUV(hr_yuv_path, (hr_height,hr_width))
            while(1):
                ret,frame=cap_hr.read_yuv()
                if ret:
                    hr_frames.append(frame)
                else:
                    break        

            patch_size = cur_args.patch_size
            scale = cur_args.scale
            # vid = cur_args.video_name+".mp4"
            # vid_path = os.path.abspath(os.path.join(cur_args.dir_data,vid))
            # video_name = os.path.splitext(vid)[0]
            # print("Path:"+vid_path)

            #frames = get_frames(vid_path,cur_args)

            if(cur_args.model == "MDSR"):
                save_model_name = "{}_{}_r{}_f{}_x{}_{}_fromscratch.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resblocks, cur_args.n_feats,scale,cur_args.color_mode)
            elif(cur_args.model=="RCAN"):
                save_model_name=  "{}_{}_rg{}_r{}_f{}_x{}_{}_fromscratch.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,scale,cur_args.color_mode)

            device = None
            # create model
            if (cur_args.model == "MDSR"):
                model = MDSR(cur_args)
            else:
                model = RCAN(cur_args)
            #model = load_checkpoint(model,cur_args)
            loss = nn.L1Loss()
            if (cur_args.gpu != 0):
                device = torch.device('cuda')
                model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

            avg_losses = []
            avg_loss = 0.0
            patches_per_frame = 16
            start_time = time.time()
            for epo in range(cur_args.epochs):
                running_loss = 0.0
                for i in range(len(lr_frames)):
                    lr_patches,hr_patches = my_patch_extraction(lr_frames[i],hr_frames[i],patch_size,scale, patches_per_frame)
                    lr_patches = np2Tensor(lr_patches)
                    hr_patches = np2Tensor(hr_patches)
                    if (cur_args.gpu != 0):
                        lr_patches = lr_patches.to(device)
                        hr_patches = hr_patches.to(device)
                    optimizer.zero_grad()
                    outputs = model(lr_patches)
                    cur_loss = loss(outputs, hr_patches)
                    cur_loss.backward()
                    optimizer.step()
                    running_loss += cur_loss.item()
                    avg_loss += cur_loss.item()
                    if (i % 200 == 199):
                        if(cur_args.debug==True):
                            print('[%d, %5d] loss: %.3f' % (epo + 1, i + 1, running_loss / 200))
                        running_loss = 0.0
                    # print('[%d, %5d] loss: %.3f' % (epo + 1, i + 1, running_loss ))
                    # running_loss =0.0
                avg_loss /= len(lr_frames)
                avg_losses.append(avg_loss)
                avg_loss = 0.0
            elapsed_time = time.time() - start_time
            print("Finished Training using {}".format(str(datetime.timedelta(seconds=elapsed_time))))

            if(cur_args.save_models == True):
                print("Save model as {}".format(save_model_name))
                state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}  # can choose to use the optimizer or not
                torch.save(state, os.path.join(cur_args.pre_train_dir,save_model_name))
            

            avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim = get_avg_psnr_ssim_video(model,lr_frames,hr_frames,device,cur_args)

            plt.plot(range(1, cur_args.epochs + 1), avg_losses, "r-")
            plt.axhline(y=min(avg_losses), color='b', linestyle='-')
            plt.title("Min loss:{:.3f}\nPSNR/SSIM {}:({:.2f}/{:.4f})\nBicubic:({:.2f}/{:.4f})".format(min(avg_losses),cur_args.model,avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim))
            plt.xlabel('Epoch')
            plt.ylabel('Average L1 Loss')
            plt.savefig('{}_{}epoch_fromscratch.png'.format("../result/" + save_model_name,cur_args.epochs))



        else:
            print("train from the check point using certain video")
            if(cur_args.re_encode == True):
                lr_width,lr_height = encode_video(cur_args)
            else:
                lr_height = 264
                lr_width = 480

            
            hr_height = lr_height*cur_args.scale
            hr_width = lr_width*cur_args.scale
            yuv_path,_ = decode_video(cur_args)
            cap = VideoCaptureYUV(yuv_path, (lr_height,lr_width))
            lr_frames = []
            hr_frames = []
            while(1):
                ret,frame=cap.read_yuv()
                if ret:
                    lr_frames.append(frame)
                else:
                    break
            total_frames = len(lr_frames)

            hr_yuv_path = "../tmp_data/{}_hr.yuv".format(cur_args.video_name)
            cap_hr = VideoCaptureYUV(hr_yuv_path, (hr_height,hr_width))
            while(1):
                ret,frame=cap_hr.read_yuv()
                if ret:
                    hr_frames.append(frame)
                else:
                    break        


            patch_size = cur_args.patch_size
            scale = cur_args.scale
            
            
            if(cur_args.model == "MDSR"):
                save_model_name = "{}_{}_r{}_f{}_x{}_{}.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resblocks, cur_args.n_feats,scale,cur_args.color_mode)
            elif(cur_args.model=="RCAN"):
                save_model_name=  "{}_{}_rg{}_r{}_f{}_x{}_{}.pt".format(cur_args.video_name,cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,scale,cur_args.color_mode)
            device = None
            # create model
            if (cur_args.model == "MDSR"):
                model = MDSR(cur_args)
            else:
                model = RCAN(cur_args)
            model = load_checkpoint(model,cur_args)
            loss = nn.L1Loss()
            if (cur_args.gpu != 0):
                device = torch.device('cuda')
                model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

            avg_losses = []
            avg_loss = 0.0
            patches_per_frame = 16
            start_time = time.time()
            for epo in range(cur_args.epochs):
                running_loss = 0.0
                for i in range(len(lr_frames)):
                    lr_patches,hr_patches = my_patch_extraction(lr_frames[i],hr_frames[i],patch_size,scale, patches_per_frame)
                    lr_patches = np2Tensor(lr_patches)
                    hr_patches = np2Tensor(hr_patches)
                    if (cur_args.gpu != 0):
                        lr_patches = lr_patches.to(device)
                        hr_patches = hr_patches.to(device)
                    optimizer.zero_grad()
                    outputs = model(lr_patches)
                    cur_loss = loss(outputs, hr_patches)
                    cur_loss.backward()
                    optimizer.step()
                    running_loss += cur_loss.item()
                    avg_loss += cur_loss.item()
                    if (i % 200 == 199):
                        if(cur_args.debug==True):
                            print('[%d, %5d] loss: %.3f' % (epo + 1, i + 1, running_loss / 200))
                        running_loss = 0.0
                    # print('[%d, %5d] loss: %.3f' % (epo + 1, i + 1, running_loss ))
                    # running_loss =0.0
                avg_loss /= len(lr_frames)
                avg_losses.append(avg_loss)
                avg_loss = 0.0
            elapsed_time = time.time() - start_time
            print("Finished Training using {}".format(str(datetime.timedelta(seconds=elapsed_time))))
            if(cur_args.save_models == True):
                print("Save model as {}".format(save_model_name))
                state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}  # can choose to use the optimizer or not
                torch.save(state, os.path.join(cur_args.pre_train_dir,save_model_name))

            avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim = get_avg_psnr_ssim_video(model,lr_frames,hr_frames,device,cur_args)

            plt.plot(range(1, cur_args.epochs + 1), avg_losses, "r-")
            plt.axhline(y=min(avg_losses),color='b', linestyle='-',)
            plt.title("Min loss:{:.3f}\nPSNR/SSIM {}:({:.2f}/{:.4f})\nBicubic:({:.2f}/{:.4f})".format(min(avg_losses),cur_args.model,avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim))
            plt.xlabel('Epoch')
            plt.ylabel('Average L1 Loss')
            plt.savefig('../result/{}_{}epoch.png'.format(save_model_name,cur_args.epochs))

def test_model(cur_args):
    if(cur_args.inference_mode == "normal"):
        print("Test model on video:{}".format(cur_args.video_name))
        if(cur_args.re_encode == True):
            lr_width,lr_height = encode_video(cur_args) # mp4 hr video => cropped hr.yuv + lr.yuv + lr.str 
        else:
            lr_width = cur_args.width
            lr_height = cur_args.height
        hr_height = lr_height*cur_args.scale
        hr_width = lr_width*cur_args.scale
        print("{},{}".format(lr_width,lr_height))

        yuv_path,dumped_str = decode_video(cur_args) # lr.str => dumped_file + lr_recon.yuv
        cap = VideoCaptureYUV(yuv_path, (lr_height,lr_width)) # read lr_recon.yuv
        lr_frames = []
        while(1):
            ret,frame=cap.read_yuv()
            if ret:
                lr_frames.append(frame)
            else:
                break
        total_frames = len(lr_frames)
        print("Total frames: {}".format(total_frames))

        hr_frames = [] # yuv HR frames, finally we can evaluate the performance on Y channel only
        
        hr_yuv_path = "../tmp_data/{}_hr.yuv".format(cur_args.video_name)
        cap_hr = VideoCaptureYUV(hr_yuv_path, (hr_height,hr_width))
        while(1):
            ret,frame=cap_hr.read_yuv()
            if ret:
                hr_frames.append(frame)
            else:
                break
        #get bicubic interpolated frames
        bicubic = []
        for i in range(total_frames):
                bi = cv2.resize(lr_frames[i],(hr_width,hr_height),interpolation=cv2.INTER_CUBIC)
                bicubic.append(bi)
        if (cur_args.model == "MDSR"):
            model = MDSR(cur_args)
        else:
            model = RCAN(cur_args)
    
        model = load_checkpoint(model,cur_args)
        device = None
        if(cur_args.gpu != 0):
            device = torch.device('cuda')
        model.to(device)
        model.eval()
        torch.no_grad()

        start_frame = 0
        sr_min_psnr = 100
        sr_max_psnr = 0
        sr_avg_psnr = 0
        sr_min_ssim = 100
        sr_max_ssim = 0
        sr_avg_ssim = 0
        bi_min_psnr = 100
        bi_max_psnr = 0
        bi_avg_psnr = 0
        bi_min_ssim = 100
        bi_max_ssim = 0
        bi_avg_ssim = 0
        testtime=  0.0

        sr_result = np.zeros( (total_frames,hr_height,hr_width,3),dtype=np.uint8)
        start_time = time.time()
        for i in range(total_frames):
            lr = np.expand_dims(lr_frames[i], axis=0)
            lr = torch.from_numpy( lr.transpose(0,3,1,2) ).float()

            lr = lr.to(device)
            sr_result_tmp = model(lr)
            sr_result_tmp = quantize(sr_result_tmp)
            sr_result_tmp = sr_result_tmp.data.cpu().numpy().squeeze(0).astype(np.uint8).transpose(1,2,0)
            sr_result[i] = sr_result_tmp
        elapsed_time = time.time() - start_time
        print("Inference done")
        for i in range(total_frames):
            psnr = get_psnr(hr_frames[i],sr_result[i],cur_args.color_mode)
            if(sr_min_psnr > psnr):
                sr_min_psnr = psnr
            if(sr_max_psnr < psnr):
                sr_max_psnr = psnr
            sr_avg_psnr += psnr
           
            ssim = get_ssim(hr_frames[i],sr_result[i],cur_args.color_mode)
            if(sr_min_ssim > ssim):
                sr_min_ssim = ssim
            if(sr_max_ssim < ssim):
                sr_max_ssim = ssim
            sr_avg_ssim += ssim
        for i in range(total_frames):
            psnr = get_psnr(hr_frames[i],bicubic[i],cur_args.color_mode)
            if(bi_min_psnr > psnr):
                bi_min_psnr = psnr
            if(bi_max_psnr < psnr):
                bi_max_psnr = psnr
            bi_avg_psnr += psnr
            ssim = get_ssim(hr_frames[i],bicubic[i],cur_args.color_mode)
            if(bi_min_ssim > ssim):
                bi_min_ssim = ssim
            if(bi_max_ssim < ssim):
                bi_max_ssim = ssim
            bi_avg_ssim += ssim


        sr_avg_psnr /= total_frames
        sr_avg_ssim /= total_frames
        bi_avg_psnr /= total_frames
        bi_avg_ssim /= total_frames   
        print("For SR:\nMin PSNR:{}, Max PSNR:{}, Avg PSNR:{}".format(sr_min_psnr,sr_max_psnr,sr_avg_psnr))
        print("Min SSIM:{}, Max SSIM:{}, Avg SSIM:{}".format(sr_min_ssim,sr_max_ssim,sr_avg_ssim))
        print("For Bicubic:\nMin PSNR:{}, Max PSNR:{}, Avg PSNR:{}".format(bi_min_psnr,bi_max_psnr,bi_avg_psnr))
        print("Min SSIM:{}, Max SSIM:{}, Avg SSIM:{}".format(bi_min_ssim,bi_max_ssim,bi_avg_ssim))     
        print("Took {} seconds to finish {} frames (Avg: {} fps) ".format(elapsed_time,total_frames,total_frames/elapsed_time))
      
    else:
        print("FAST Mode")
        if(cur_args.re_encode == True):
            lr_width,lr_height = encode_video(cur_args) # mp4 hr video => cropped hr.yuv + lr.yuv + lr.str 
        else:
            lr_width = cur_args.width
            lr_height = cur_args.height
        hr_height = lr_height*cur_args.scale
        hr_width = lr_width*cur_args.scale
        print("{},{}".format(lr_width,lr_height))


        yuv_path,dumped_str = decode_video(cur_args) # lr.str => dumped_file + lr_recon.yuv
        cap = VideoCaptureYUV(yuv_path, (lr_height,lr_width)) # read lr_recon.yuv
        lr_frames = []
        while(1):
            ret,frame=cap.read_yuv()
            if ret:
                lr_frames.append(frame)
            else:
                break
        total_frames = len(lr_frames)
        print("Total frames: {}".format(total_frames))

        pu_all,res_all = parse_dump_file(dumped_str) # parse syntax from dumped text


        residual_recon = reconstruct_residual(res_all,(lr_height,lr_width))# reconstruct residual for three channels
        #residual_recon shape= (n,h,w)
        residual_recon_hr = np.zeros((total_frames,hr_height,hr_width),dtype=np.int16)
        for i in range(total_frames):
            residual_recon_hr[i] = cv2.resize(residual_recon[i],(hr_width,hr_height),interpolation=cv2.INTER_CUBIC)
        
        #load HR video frames
        hr_frames = [] # yuv HR frames, finally we can evaluate the performance on Y channel only
        
        hr_yuv_path = "../tmp_data/{}_hr.yuv".format(cur_args.video_name)
        cap_hr = VideoCaptureYUV(hr_yuv_path, (hr_height,hr_width))
        while(1):
            ret,frame=cap_hr.read_yuv()
            if ret:
                hr_frames.append(frame)
            else:
                break
        #get bicubic interpolated frames
        bicubic = []
        for i in range(total_frames):
                bi = cv2.resize(lr_frames[i],(hr_width,hr_height),interpolation=cv2.INTER_CUBIC)
                bicubic.append(bi)

        #load SR model
        # currently load hardcoded model first
        #cur_check_pt_name = "MDSR_r20_f21_x4_yuv.pt"

        if (cur_args.model == "MDSR"):
            model = MDSR(cur_args)
        else:
            model = RCAN(cur_args)


        model = load_checkpoint(model,cur_args)
        device = None
        if(cur_args.gpu != 0):
            device = torch.device('cuda')
        model.to(device)
        model.eval()
        torch.no_grad()

        #do super-resolution
        sr_result = np.copy(bicubic)
        sr_interval = 4 # currently hardcoded
        transfer_threshold = cur_args.transfer_threshold



        start_frame = 0
        sr_min_psnr = 100
        sr_max_psnr = 0
        sr_avg_psnr = 0
        sr_min_ssim = 100
        sr_max_ssim = 0
        sr_avg_ssim = 0
        bi_min_psnr = 100
        bi_max_psnr = 0
        bi_avg_psnr = 0
        bi_min_ssim = 100
        bi_max_ssim = 0
        bi_avg_ssim = 0
        testtime=  0.0





        start_time = time.time()

        for i in range(total_frames):
            if(i % sr_interval == 0 or len(pu_all[i])==0):
                #lr = cv2.cvtColor( lr_frames[i].unsqueeze(0),cv2.COLOR_YUV2RGB)
                lr = np.expand_dims(lr_frames[i], axis=0)
                lr = torch.from_numpy( lr.transpose(0,3,1,2) ).float()

                lr = lr.to(device)
                sr_result_tmp = model(lr)
                sr_result_tmp = quantize(sr_result_tmp)
                sr_result_tmp = sr_result_tmp.data.cpu().numpy().squeeze(0).astype(np.uint8).transpose(1,2,0)
                sr_result[i] = sr_result_tmp
            else:
                cnt_transfer = 0
                for cur_pu in pu_all[i]:
                   # cur_pu = pu_all[i][pu_idx]
                    if(cur_pu.w == 0):
                        continue
                    x_l = cur_pu.x
                    y_l = cur_pu.y
                    x_h = cur_args.scale*x_l
                    y_h = cur_args.scale*y_l
                    w_l = cur_pu.w
                    h_l = cur_pu.h
                    w_h = cur_args.scale*w_l
                    h_h = cur_args.scale*h_l
                    res_patch_l = residual_recon[i,y_l:(y_l+h_l),x_l:(x_l+w_l)]
                    mean_abs_val = np.sum(np.absolute(res_patch_l))/res_patch_l.size
                    
                    if(mean_abs_val < transfer_threshold):
                        if(w_h == 0 or h_h == 0):
                            print("w or h can be 0")
                        
                        mv_x_h = round(args.scale*cur_pu.mv_x/4)
                        mv_y_h = round(args.scale*cur_pu.mv_y/4)
                        xp = x_h + mv_x_h
                        yp = y_h + mv_y_h
                        ref_frame_idx = cur_pu.t_r
                        #cnt_transfer += w_h*h_h
                        mv_r1 = max(0, min(yp, hr_height-1))
                        mv_c1 = max(0, min(xp, hr_width-1))
                        mv_r2 = max(0, min(yp+h_h, hr_height))
                        mv_c2 = max(0, min(xp+w_h, hr_width))
                        
                        real_h = mv_r2 - mv_r1
                        real_w = mv_c2 - mv_c1
                        to_r = y_h + mv_r1 - yp
                        to_c = x_h + mv_c1 - xp
                        ref_patch = sr_result[ref_frame_idx][mv_r1:(mv_r1+real_h),mv_c1:(mv_c1+real_w),0]
                        res_patch = residual_recon_hr[i,to_r:(to_r+real_h),to_c:(to_c+real_w)]
                        transfer_patch = ref_patch + res_patch
                        np.clip(transfer_patch,0,255,out=transfer_patch)
                        transfer_patch = transfer_patch.astype(np.uint8)
                        sr_result[i][to_r:(to_r+real_h),to_c:(to_c+real_w),0] = transfer_patch
                #apply deblocking filter on this frame
        print("Inference done")
        elapsed_time = time.time() - start_time
        for i in range(total_frames):
            psnr = get_psnr(hr_frames[i],sr_result[i],cur_args.color_mode)
            if(sr_min_psnr > psnr):
                sr_min_psnr = psnr
            if(sr_max_psnr < psnr):
                sr_max_psnr = psnr
            sr_avg_psnr += psnr
           
            ssim = get_ssim(hr_frames[i],sr_result[i],cur_args.color_mode)
            if(sr_min_ssim > ssim):
                sr_min_ssim = ssim
            if(sr_max_ssim < ssim):
                sr_max_ssim = ssim
            sr_avg_ssim += ssim
        for i in range(total_frames):
            psnr = get_psnr(hr_frames[i],bicubic[i],cur_args.color_mode)
            if(bi_min_psnr > psnr):
                bi_min_psnr = psnr
            if(bi_max_psnr < psnr):
                bi_max_psnr = psnr
            bi_avg_psnr += psnr
            ssim = get_ssim(hr_frames[i],bicubic[i],cur_args.color_mode)
            if(bi_min_ssim > ssim):
                bi_min_ssim = ssim
            if(bi_max_ssim < ssim):
                bi_max_ssim = ssim
            bi_avg_ssim += ssim

        sr_avg_psnr /= total_frames
        sr_avg_ssim /= total_frames
        bi_avg_psnr /= total_frames
        bi_avg_ssim /= total_frames   
        print("For SR:\nMin PSNR:{}, Max PSNR:{}, Avg PSNR:{}".format(sr_min_psnr,sr_max_psnr,sr_avg_psnr))
        print("Min SSIM:{}, Max SSIM:{}, Avg SSIM:{}".format(sr_min_ssim,sr_max_ssim,sr_avg_ssim))
        print("For Bicubic:\nMin PSNR:{}, Max PSNR:{}, Avg PSNR:{}".format(bi_min_psnr,bi_max_psnr,bi_avg_psnr))
        print("Min SSIM:{}, Max SSIM:{}, Avg SSIM:{}".format(bi_min_ssim,bi_max_ssim,bi_avg_ssim))     
        print("Took {} seconds to finish {} frames (Avg: {} fps) ".format(elapsed_time,total_frames,total_frames/elapsed_time))

def playback_process(Q,lock):
    frame=None
    while(True):
        if(Q.qsize() > 0):
            frame = Q.get()
            if(frame=="Done"):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # display the size of the queue on the frame
        if(frame is not None):
            cv2.putText(frame, "Buffer Size: {} frames".format(Q.qsize()),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)    
        # show the frame and update the FPS counter
            cv2.imshow("Frame", frame)
            cv2.waitKey(int(1000/30) ) #frame rate of 30fps
        else:
            time.sleep(0.1)
            # with lock:
            #     print("Waiting for frames")
    cv2.destroyAllWindows()
    with lock:
        print("Finished play back")

class SrPlayer:
    def __init__(self,cur_args):
        self.stopped  = False
        self.Q = Queue()
        self.width = cur_args.width
        self.height = cur_args.height
        self.cur_args = cur_args

    def start(self):
        lock = Lock()
        p = Process(target =playback_process, args=(self.Q,lock))
        p.start()

        if (self.cur_args.model == "MDSR"):
            model = MDSR(self.cur_args)
        else:
            model = RCAN(self.cur_args)
        model = load_checkpoint(model,self.cur_args)
        device = None
        if(self.cur_args.gpu != 0):
            device = torch.device('cuda')
        model.to(device)
        model.eval()
        torch.no_grad()

        if(self.cur_args.inference_mode == "normal"):
            #decode the str
            i = 0
            elapsed_time = 0
            lr_height = self.cur_args.height
            lr_width = self.cur_args.width
            hr_height = lr_height*self.cur_args.scale
            hr_width = lr_width*self.cur_args.scale
            yuv_path,_ = decode_video(self.cur_args) # lr.str => dumped_file + lr_recon.yuv
            fvs =  FileVideoStream(yuv_path,(lr_height,lr_width)).start()
            time.sleep(0.5)
            start_time = time.time()
            while(fvs.more()):
                frame = fvs.read()
                lr = np.expand_dims(frame, axis=0)
                lr = torch.from_numpy( lr.transpose(0,3,1,2) ).float()
                lr = lr.to(device)
                sr_result_tmp = model(lr)
                sr_result_tmp = quantize(sr_result_tmp)
                sr_result_tmp = sr_result_tmp.data.cpu().numpy().squeeze(0).astype(np.uint8).transpose(1,2,0)
                self.Q.put(sr_result_tmp)
                i+=1
            elapsed_time = time.time() - start_time
        else:
            
            lr_height = self.cur_args.height
            lr_width = self.cur_args.width
            hr_height = lr_height*self.cur_args.scale
            hr_width = lr_width*self.cur_args.scale
            yuv_path,dump_str = decode_video(self.cur_args)
            #cap = VideoCaptureYUV(yuv_path, (lr_height,lr_width))
            fvs =  FileVideoStream(yuv_path,(lr_height,lr_width)).start()
            pu_all,res_all = parse_dump_file(dump_str) # parse syntax from dumped text
            residual_recon = reconstruct_residual(res_all,(lr_height,lr_width))# reconstruct residual for three channels
            
            #load deblocker
            #deblocker = FASTARCNN(self.cur_args)
            #checkpoint = torch.load("../checkpoint/FASTARCNN.pt")
            # deblocker = DnCNN(channels=1, num_of_layers=17)
            # deblocker = nn.DataParallel(deblocker, device_ids=[0]).cuda()
            # checkpoint = torch.load("../checkpoints/net.pth")
            # deblocker.load_state_dict(checkpoint)
            # deblocker.eval()

            pre_sr_res = []
            sr_interval = 4
            #transfer_ratio = 0
            i = 0
            start_time = time.time()
            elapsed_time = 0
            while(fvs.more()):
                frame = fvs.read()
                if(i % sr_interval == 0 or len(pu_all[i])==0):
                    #lr = cv2.cvtColor( lr_frames[i].unsqueeze(0),cv2.COLOR_YUV2RGB)
                    lr = np.expand_dims(frame, axis=0)
                    lr = torch.from_numpy( lr.transpose(0,3,1,2) ).float()
                    lr = lr.to(device)
                    sr_result_tmp = model(lr)
                    sr_result_tmp = quantize(sr_result_tmp)
                    sr_result_tmp = sr_result_tmp.data.cpu().numpy().squeeze(0).astype(np.uint8).transpose(1,2,0)
                    self.Q.put(sr_result_tmp)
                    pre_sr_res.append(sr_result_tmp)
                    #pre_sr_res = sr_result_tmp
                else:
                    residual_recon_hr = cv2.resize(residual_recon[i],(hr_width,hr_height),interpolation=cv2.INTER_CUBIC)
                    sr_result_tmp = cv2.resize(frame,(hr_width,hr_height),interpolation=cv2.INTER_CUBIC)
                    #cnt_transfer = 0
                    for cur_pu in pu_all[i]:
                        x_l = cur_pu.x
                        y_l = cur_pu.y
                        x_h = self.cur_args.scale*x_l
                        y_h = self.cur_args.scale*y_l
                        w_l = cur_pu.w
                        h_l = cur_pu.h
                        w_h = self.cur_args.scale*w_l
                        h_h = self.cur_args.scale*h_l
                        ref_index = cur_pu.t_r
                        res_patch_l = residual_recon[i,y_l:(y_l+h_l),x_l:(x_l+w_l)]
                        mean_abs_val = np.sum(np.absolute(res_patch_l))/res_patch_l.size

                        if(mean_abs_val < self.cur_args.transfer_threshold):
                            mv_x_h = round(args.scale*cur_pu.mv_x/4)
                            mv_y_h = round(args.scale*cur_pu.mv_y/4)
                            xp = x_h + mv_x_h
                            yp = y_h + mv_y_h
                            #cnt_transfer += w_h*h_h
                            mv_r1 = max(0, min(yp, hr_height-1))
                            mv_c1 = max(0, min(xp, hr_width-1))
                            mv_r2 = max(0, min(yp+h_h, hr_height))
                            mv_c2 = max(0, min(xp+w_h, hr_width))
                            
                            real_h = mv_r2 - mv_r1
                            real_w = mv_c2 - mv_c1
                            to_r = y_h + mv_r1 - yp
                            to_c = x_h + mv_c1 - xp
                            ref_patch = pre_sr_res[ref_index][mv_r1:(mv_r1+real_h),mv_c1:(mv_c1+real_w),0]
                            res_patch = residual_recon_hr[to_r:(to_r+real_h),to_c:(to_c+real_w)]
                            transfer_patch = ref_patch + res_patch
                            np.clip(transfer_patch,0,255,out=transfer_patch)
                            transfer_patch = transfer_patch.astype(np.uint8)
                            sr_result_tmp[to_r:(to_r+real_h),to_c:(to_c+real_w),0] = transfer_patch 
                            #cnt_transfer += w_h*h_h

                    # in_sr = np.float32(sr_result_tmp[:,:,0])/255
                    # in_sr = np.expand_dims(in_sr,0)
                    # in_sr = np.expand_dims(in_sr,1)
                    # Isource = torch.Tensor(in_sr)
                    # Isource = Variable(Isource.to(device))
                    # with torch.no_grad():
                    #     out = torch.clamp(Isource - deblocker(Isource),0.,1.)*255
                    # out = out.data.cpu().numpy().astype(np.uint8)
                    # sr_result_tmp[:,:,0] = out
                    
                    # kernel = np.ones((5,5),np.float32)/25
                    # dst = cv2.filter2D(sr_result_tmp[:,:,0],-1,kernel)
                    # sr_result_tmp[:,:,0] = dst

                    #transfer_ratio += cnt_transfer/(hr_width*hr_height)
                    self.Q.put(sr_result_tmp)
                    pre_sr_res.append(sr_result_tmp)
                i+=1
                
            elapsed_time += time.time() - start_time
            #print("Transfer ratio: {:.2f}",transfer_ratio/i)
        print("Finished sr({})\nInference frame rate:{} fps".format(i,i/elapsed_time))
        self.Q.put("Done")
        p.join()

def video_sr(cur_args): # may consider start another process to collect super-resolution output frames
    player = SrPlayer(cur_args)
    player.start()

def test_model_speed(cur_args):

    def test_speed():
        if(cur_args.model=="MDSR"):
            model = MDSR(cur_args)
        elif(cur_args.model=="FASTARCNN"):
            model = FASTARCNN(cur_args)
        elif(cur_args.model=="DnCNN"):
            model= DnCNN()
        else:
            model = RCAN(cur_args)
        device = None
        if(cur_args.gpu != 0):
            device = torch.device('cuda')
        model.to(device)
        model.eval()
        torch.no_grad()
        #batch_candi = [1,2,4]
        #for batch_size in batch_candi:
        #    print("For {} batch".format(batch_size))
        if(cur_args.model=="DnCNN"):
            random_fig = torch.randint(0,255,(1,1,int(1080/cur_args.scale),int(1920/cur_args.scale)),dtype=torch.float32,device=device)
        else:
            random_fig = torch.randint(0,255,(1,3,int(1080/cur_args.scale),int(1920/cur_args.scale)),dtype=torch.float32,device=device)
        start_time = time.time()
        for i in range(60):
            tmp = model(random_fig)
        elapsed_time = time.time()-start_time
        print(tmp.shape)
        if(cur_args.model=="MDSR"):
            print("Average Inference speed for MDSR {} resblocks, {} feats:{} fps".format(cur_args.n_resblocks,cur_args.n_feats,60/elapsed_time))
        elif(cur_args.model=="RCAN"):
            print("Average Inference speed for RCAN {} resgroups, {} resblocks, {} feats:{} fps".format(cur_args.n_resgroups,cur_args.n_resblocks,cur_args.n_feats,60/elapsed_time))
        elif(cur_args.model=='FASTARCNN'):
            print("Average Inference speed for FASTARCNN 48 feats:{} fps".format(60/elapsed_time))
        else:
            print("Average Inference speed for DnCNN :{} fps".format(60/elapsed_time))
    if(cur_args.test_all_speed):
        print("Test all speed:")
        model_candi = ["MDSR","RCAN"]
        params = {
        "MDSR": { "n_resblocks": [20,25,30],"n_feats":[21,32,48]},
        "RCAN": {"n_resgroups":[2,3,4],"n_resblocks":[4,6,8], "n_feats":[21,32,48] }
        }
        for m in model_candi:
            if(m == "MDSR"):
                for n_resblocks in params[m]["n_resblocks"]:
                    for n_feats in params[m]["n_feats"]:
                        cur_args.model = m
                        cur_args.n_resblocks = n_resblocks
                        cur_args.n_feats = n_feats
                        test_speed()
            else:
                for n_resgroups in params[m]["n_resgroups"]:
                    for n_resblocks in params[m]["n_resblocks"]:
                        for n_feats in params[m]["n_feats"]:
                            cur_args.model = m
                            cur_args.n_resgroups = n_resgroups
                            cur_args.n_resblocks = n_resblocks
                            cur_args.n_feats = n_feats
                            test_speed()
    else:
        test_speed()


def compare_yuv_rgb_performance(cur_args):
    patch_size = cur_args.patch_size
    scale = cur_args.scale
    if(cur_args.model == "MDSR"):
        save_model_name_yuv = "{}_r{}_f{}_x{}_yuv.pt".format(cur_args.model, cur_args.n_resblocks, cur_args.n_feats,scale)
    elif(cur_args.model=="RCAN"):
        save_model_name_yuv=  "{}_rg{}_r{}_f{}_x{}_yuv.pt".format(cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,scale)
    if(cur_args.model == "MDSR"):
        save_model_name_rgb = "{}_r{}_f{}_x{}_rgb.pt".format(cur_args.model, cur_args.n_resblocks, cur_args.n_feats,scale)
    elif(cur_args.model=="RCAN"):
        save_model_name_rgb=  "{}_rg{}_r{}_f{}_x{}_rgb.pt".format(cur_args.model, cur_args.n_resgroups,cur_args.n_resblocks, cur_args.n_feats,scale)

    #load training data, in RGB already
    images_rgb = load_images_from_folder(cur_args.dir_data)
    # images_yuv = []
    # for i in range(len(images_rgb)):
    #     images_yuv.append(cv2.cvtColor(images_rgb[i],cv2.COLOR_RGB2YUV))
    # #print(images[0].shape)
    device = None

    #create model
    if(cur_args.model=="MDSR"):
        model_yuv = MDSR(cur_args)
        model_rgb = MDSR(cur_args)
    else:
        model_yuv = RCAN(cur_args)
        model_rgb = RCAN(cur_args)
    
    if(cur_args.gpu != 0):
        device = torch.device('cuda')
    model_yuv.to(device)    
    model_rgb.to(device)    
    loss_yuv = nn.L1Loss()
    loss_rgb = nn.L1Loss()

    optimizer_yuv = optim.Adam(model_yuv.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay=1e-3)
    optimizer_rgb = optim.Adam(model_rgb.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay=1e-3)
    avg_losses_yuv = []
    avg_loss_yuv = 0.0

    avg_losses_rgb = []
    avg_loss_rgb = 0.0
    patches_per_frame = 16
    start_time = time.time()
    for epo in range(cur_args.epochs):
        running_loss_yuv = 0.0
        running_loss_rgb = 0.0
        
        for i in range(len(images_rgb)):
            hr_patches_rgb = feature_extraction.image.extract_patches_2d(images_rgb[i], (patch_size*scale,patch_size*scale), patches_per_frame)
            hr_patches_yuv = np.zeros(hr_patches_rgb.shape) # actually not all images have the same size
            for j in range(hr_patches_rgb.shape[0]):
                hr_patches_yuv[j] = cv2.cvtColor(hr_patches_rgb[j],cv2.COLOR_RGB2YUV)
            lr_patches_rgb = np2Tensor(img_downsample(hr_patches_rgb, scale))
            hr_patches_rgb = np2Tensor(hr_patches_rgb)

            lr_patches_yuv = np2Tensor(img_downsample(hr_patches_yuv, scale))
            hr_patches_yuv = np2Tensor(hr_patches_yuv)
            if(cur_args.gpu != 0):
                lr_patches_rgb=lr_patches_rgb.to(device)
                hr_patches_rgb=hr_patches_rgb.to(device)
                lr_patches_yuv=lr_patches_yuv.to(device)
                hr_patches_yuv=hr_patches_yuv.to(device)


            optimizer_yuv.zero_grad()
            optimizer_rgb.zero_grad()

            outputs_yuv = model_yuv(lr_patches_yuv)
            outputs_rgb = model_rgb(lr_patches_rgb)
            cur_loss_yuv = loss_yuv(outputs_yuv, hr_patches_yuv)
            cur_loss_rgb = loss_rgb(outputs_rgb, hr_patches_rgb)
            
            cur_loss_yuv.backward()
            cur_loss_rgb.backward()

            optimizer_yuv.step()
            optimizer_rgb.step()
            running_loss_yuv += cur_loss_yuv.item()
            running_loss_rgb += cur_loss_rgb.item()
            avg_loss_yuv += cur_loss_yuv.item()
            avg_loss_rgb += cur_loss_rgb.item()

            if (i % 200 == 199):
                if(cur_args.debug==True):
                    print('YUV [%d, %5d] loss: %.3f' %(epo + 1, i + 1, running_loss_yuv / 200))
                    print('RGB [%d, %5d] loss: %.3f' %(epo + 1, i + 1, running_loss_rgb / 200))
                running_loss_yuv = 0.0
                running_loss_rgb = 0.0
            # print('[%d, %5d] loss: %.3f' % (epo + 1, i + 1, running_loss ))
            # running_loss =0.0
        avg_loss_yuv /= len(images_rgb)
        avg_loss_rgb /= len(images_rgb)

        avg_losses_yuv.append(avg_loss_yuv)
        avg_losses_rgb.append(avg_loss_rgb)
        avg_loss_yuv = 0.0
        avg_loss_rgb = 0.0
    elapsed_time = time.time() - start_time
    print("Finished Training using {}".format( str(datetime.timedelta(seconds=elapsed_time))))

    print("Save model as {}".format(save_model_name_yuv))
    state = { 'state_dict': model_yuv.state_dict(),
             'optimizer': optimizer_yuv.state_dict()} # can choose to use the optimizer or not
    torch.save(state,os.path.join(cur_args.pre_train_dir,save_model_name_yuv))

    print("Save model as {}".format(save_model_name_rgb))
    state = { 'state_dict': model_rgb.state_dict(),
             'optimizer': optimizer_rgb.state_dict()} # can choose to use the optimizer or not
    torch.save(state,os.path.join(cur_args.pre_train_dir,save_model_name_rgb))

    plt.plot(range(1, cur_args.epochs + 1), avg_losses_yuv, "r--")
    plt.axhline(y=min(avg_losses_yuv), color='r', linestyle='-')
    plt.plot(range(1, cur_args.epochs + 1), avg_losses_rgb, "b--")
    plt.axhline(y=min(avg_losses_rgb), color='b', linestyle='-')

    plt.title("Min loss: YUV({0:.3f})/RGB({1:.3f}) ".format(min(avg_losses_yuv),min(avg_losses_rgb)))
    plt.xlabel('Epoch')
    plt.ylabel('Average L1 Loss')
    plt.savefig('{}{}_compare_yuv_rgb_{}epoch.png'.format("../result/",cur_args.model, cur_args.epochs))

def train_deblocking_network(cur_args):
    print("Now train deblocking network")
    


def check_psnr_ssim_diff(cur_args):
    scale = cur_args.scale
    if (cur_args.model == "MDSR"):
        model = MDSR(cur_args)
    else:
        model = RCAN(cur_args)
    ckpt_name = cur_args.load
    if(ckpt_name==''):
        print("Didn't select checkpoint")
        return
    model = load_certain_checkpoint(model,ckpt_name,cur_args)
    device = None
    if(cur_args.gpu != 0):
        device = torch.device('cuda')
    model.to(device)
    model.eval()
    torch.no_grad()
    images = load_images_from_folder(cur_args) #list containing images

    avg_psnr,avg_ssim,bic_avg_psnr,bic_avg_ssim = get_avg_psnr_ssim(model,images,device,cur_args)
    print("Color mode: {}".format(cur_args.color_mode))
    print("Average PSNR/SSIM for {}: {:.2f}/{:.4f}".format(cur_args.model,avg_psnr,avg_ssim))
    print("Average PSNR/SSIM for Bicubic: {:.2f}/{:.4f}".format(bic_avg_psnr,bic_avg_ssim))

parser = get_parser()
args = parser.parse_args()
if not os.path.exists("../result"):
    os.mkdir("../result")
if not os.path.exists("../tmp_data"):
    os.mkdir("../tmp_data")
if not os.path.exists(args.pre_train_dir):
    os.mkdir(args.pre_train_dir)


if(args.mode=="train"): # train a model
    train_model(args)
elif(args.mode=="test"):
    if(args.test_only==True):
        test_model(args)
    elif(args.test_speed==True):
        test_model_speed(args)
    else:
        video_sr(args)
elif(args.mode=="compare"):
    print("Compare yuv and rgb")
    #compare_yuv_rgb_performance(args)
    check_psnr_ssim_diff(args)
