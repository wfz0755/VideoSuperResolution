import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import platform
import random

from queue import Queue as thread_queue
from threading import Thread




# def crop_hr( filename):
#     vidcap = cv2.VideoCapture(filename)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print("Frames: {}".format(total_frames))
#     ori_w = vidcap.get(3)
#     ori_h = vidcap.get(4)
#     expected_w_for_hr = ori_w - ori_w % (16 * xxx)
#     expected_h_for_hr = enc_param["h"] - enc_param["h"] % (enc_param["clip_dim"] * enc_param["sr_ratio"])
#     rgb_arr = np.zeros((total_frames, expected_h_for_hr, expected_w_for_hr, 3),
#                        dtype=np.uint8)  # setting dtype is quite important
#     print("Original size: {}x{}, now: {}x{}".format(enc_param["w"], enc_param["h"], expected_w_for_hr,
#                                                     expected_h_for_hr))
#     for i in range(total_frames):
#         success, hr = vidcap.read() # currently hr is BGR
#         hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # plt. cv2 reads images as BGR by default, where as plt uses RGB format. So your Blue and Red color will get flipped.
#         rgb_arr[i] = hr[0:expected_h_for_hr, 0:expected_w_for_hr]
#         assert np.array_equal(rgb_arr[i], hr[0:expected_h_for_hr, 0:expected_w_for_hr]), "copy failed"
#     return rgb_arr

def crop_hr(hr_frames,scale):
    min_cu = 8
    _,h,w,_ = hr_frames.shape
    new_h = h - h%(scale*min_cu)
    new_w = w - w%(scale*min_cu)
    hr_frames = hr_frames[:,0:new_h,0:new_w,:]
    return hr_frames

def cur_get_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        suc, hr = vidcap.read()
        if not suc:
            break
        frames.append(hr) # simply get BGR frames

    return frames

def img_downsample( hr_images, ratio):
    if(type(hr_images) is list):
        n_frames = len(hr_images)
        lr_frames = []
        for i in range(n_frames):
            h,w,c = hr_images[i].shape
            assert (int(h / ratio) * ratio == h), "Frame size not appropriate"
            assert (int(w / ratio) * ratio == w), "Frame size not appropriate"
            lr_frames.append(cv2.resize(hr_images[i], dsize=(int(w / ratio), int(h / ratio)),
                                  interpolation=cv2.INTER_LINEAR))
    else:
        (n_frames, h, w, c) = hr_images.shape
        assert (int(h / ratio) * ratio == h), "Frame size not appropriate"
        assert (int(w / ratio) * ratio == w), "Frame size not appropriate"
        lr_h = int(h / ratio)
        lr_w = int(w / ratio)
        lr_frames = np.zeros( (n_frames,lr_h,lr_w,c),dtype=np.uint8 )
        for i in range(n_frames):
            lr_frames[i] = cv2.resize(hr_images[i], dsize=(lr_w, lr_h),
                                  interpolation=cv2.INTER_LINEAR)
     # considering images may not have the same shape(if not for video)
    
    return lr_frames



class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.f = open(filename, 'rb')
        self.shape = (self.height,self.width)
        self.upscale_mat = np.array([[1,1],[1,1]])
    def read_yuv(self):
        try:
            yuv = np.zeros((self.height,self.width,3),dtype=np.uint8)
            buf = self.f.read(self.width * self.height)
            yuv[:,:,0] = np.reshape(np.frombuffer(buf, dtype=np.uint8),self.shape)
            buf = self.f.read( int(self.width/2 * self.height/2) )
            yuv[:,:,1] = np.kron(np.reshape(np.frombuffer(buf, dtype=np.uint8),(int(self.height/2),int(self.width/2) )),self.upscale_mat)
            buf = self.f.read( int(self.width/2 * self.height/2) )
            yuv[:,:,2] = np.kron(np.reshape(np.frombuffer(buf, dtype=np.uint8),(int(self.height/2),int(self.width/2) )),self.upscale_mat)
        except Exception as e:
            #print( str(e))
            self.f.close()
            return False, None
        return True, yuv

    def read_rgb(self):
        ret, yuv = self.read_yuv()
        if not ret:
            self.f.close()
            return ret, yuv
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return ret, rgb
    def close(self):
        self.f.close()

class FileVideoStream:
    def __init__(self, path, size,queueSize=600):
        self.stream = VideoCaptureYUV(path,size)
        self.stopped = False
        self.Q = thread_queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read_yuv()
                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)

    def read(self):
        return self.Q.get()
    def more(self):
        return self.Q.qsize() > 0
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def encode_video(cur_args):
    print("Based on hr video, generate lr video binary and per-sequence configuration file")
    vid_path = os.path.abspath(os.path.join(cur_args.dir_data,"{}.mp4".format(cur_args.video_name) ))
    maincfg_path = "../utils/hm/cfg/encoder_lowdelay_P_main.cfg"
    per_seq_cfg_path = "../utils/hm/cfg/per-sequence/{}.cfg".format(cur_args.video_name)
    hr_frames = cur_get_frames(vid_path)
    hr_frames = np.stack( hr_frames, axis=0 )
    hr_frames = crop_hr(hr_frames,cur_args.scale)
    lr_frames = img_downsample(hr_frames,cur_args.scale)
    total_frames = hr_frames.shape[0]
    size = (lr_frames.shape[2],lr_frames.shape[1])# width, height
    yuv_path = "../tmp_data/{}_{}_{}.yuv".format(cur_args.video_name,lr_frames.shape[2],lr_frames.shape[1]) # width,height
    video_write = cv2.VideoWriter(yuv_path,cv2.VideoWriter_fourcc(*'I420'), 30, size)

    for i in range(total_frames):
        video_write.write(lr_frames[i])
    video_write.release()

    hr_yuv_path = "../tmp_data/{}_hr.yuv".format(cur_args.video_name)
    hr_size = (hr_frames.shape[2],hr_frames.shape[1])# width, height
    hr_video_write = cv2.VideoWriter(hr_yuv_path,cv2.VideoWriter_fourcc(*'I420'), 30, hr_size)

    for i in range(total_frames):
        hr_video_write.write(hr_frames[i])
    hr_video_write.release()

    #get YUV file

    cfgfile = open(per_seq_cfg_path,"w")
    cfgfile.write("InputBitDepth                 : 8\nFrameRate                     : 30\nFrameSkip                     : 0\n")
    cfgfile.write("SourceWidth                   : {}\nSourceHeight                  : {}\nFramesToBeEncoded              : {}".format(size[0],size[1],total_frames))
    cfgfile.close()
    other_opts = "--QP={}".format(27)
    binary_name = "../tmp_data/{}.str".format(cur_args.video_name)
    yuv_recon_name = "../tmp_data/{}_recon.yuv".format(cur_args.video_name)
    if(platform.system() == 'Linux' or platform.system() == 'Linux2'):
        encoder_path = "../utils/hm/bin/TAppEncoderStatic"
    elif(platform.system()=='Darwin'):
        encoder_path = "../utils/hm/bin/TAppEncoder"
    elif(platform.system()=='Win32'):
        print("Not supported on windows now")
        return None
    else:
        print("Unknown platform")
        return None
    encode_yuv_cmd = "{} -c {} -c {} -i {} -o {} -b {} {}".format(encoder_path, maincfg_path, per_seq_cfg_path, yuv_path,yuv_recon_name,
                                                                  binary_name,
                                                                  other_opts)
    os.system(encode_yuv_cmd)
    print("Finished encoding.")
    return lr_frames.shape[2],lr_frames.shape[1]

def decode_video(cur_args):
    print("Decoding start")
    #env_cmd = 'env PRINT_COEFF=1 PRINT_INTRA=0 PRINT_MV=1 SAVE_PREFILT=0' 
    if(platform.system() == 'Linux' or platform.system() == 'Linux2'):
        decoder_path = "../utils/hm/bin/TAppDecoderStatic"
    elif(platform.system()=='Darwin'):
        decoder_path = "../utils/hm/bin/TAppDecoder"
    elif(platform.system()=='Win32'):
        print("Not supported on windows now")
        return None
    else:
        print("Unknown platform")
        return None
    binary_name = "../tmp_data/{}.str".format(cur_args.video_name)

    yuv_recon_name = "../tmp_data/{}_recon.yuv".format(cur_args.video_name)
   # dump_txt_name = "../tmp_data/{}_dumped_syntax.txt".format(cur_args.video_name)
    #command = "{} {} -b {} -o {} ".format(env_cmd, decoder_path,binary_name,yuv_recon_name)
  #  print(command)
    my_env = os.environ.copy()
    my_env['PRINT_COEFF'] = '1'
    my_env['PRINT_INTRA'] = '0'
    my_env['PRINT_MV'] = '1'
    my_env['SAVE_PREFILT'] = '0'

    out = subprocess.Popen([decoder_path,'-b',binary_name,'-o',yuv_recon_name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env = my_env)
    stdout,stderr = out.communicate()
    if(stderr is not None):
        print("Something wrong happend during decoding")
        print(stderr)
    else:
    #os.system(command)
        print("Finished decoding.")  
        return (yuv_recon_name,stdout.decode())

def my_patch_extraction(lr_frame,hr_frame,patch_size,scale, patches_per_frame):
    h_l,w_l,_ = lr_frame.shape
    r_candi = None
    c_candi = None
    lr_patches = []
    hr_patches= []
    max_r = h_l - patch_size
    max_c = w_l - patch_size
    for i in range(patches_per_frame):
        r_candi = random.randint(0,max_r)
        c_candi = random.randint(0,max_c)
        lr_patches.append(lr_frame[r_candi:(r_candi+patch_size),c_candi:(c_candi+patch_size)])
        hr_patches.append(hr_frame[r_candi*scale:(r_candi+patch_size)*scale,c_candi*scale:(c_candi+patch_size)*scale])
    lr_patches = np.stack(lr_patches,axis = 0)
    hr_patches = np.stack(hr_patches,axis = 0)
    return lr_patches,hr_patches
