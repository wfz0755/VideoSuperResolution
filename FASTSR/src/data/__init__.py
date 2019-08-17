import cv2
import os


def load_images_from_folder(cur_args):
    folder =cur_args.dir_data
    images = []
    for filename in os.listdir(folder):
        if(os.path.splitext(filename)[1]!='.png'):
            continue
        img = cv2.imread(os.path.join(folder,filename))
        if(img is not None):
            if( cur_args.color_mode == "RGB" ):
                images.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            else:
                images.append(cv2.cvtColor(img,cv2.COLOR_BGR2YUV))
    return images


def load_images_for_deblocking_network(cur_args):
    #TODO
    folder =cur_args.dir_data
    images = []
    for filename in os.listdir(folder):
        if(os.path.splitext(filename)[1]!='.png'):
            continue
        img = cv2.imread(os.path.join(folder,filename))
        if(img is not None):
            if(cur_args.color_mode== "YUV"):
                images.append(cv2.cvtColor(img,cv2.COLOR_BGR2YUV))
            else:
                images.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return images


def get_vid(folder):
    vids = []
    for filename in os.listdir(folder):
        if(os.path.splitext(filename)[1]=='.mp4'):
            vids.append(filename)
    return vids


def get_frames(video_path,cur_args):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        suc, hr = vidcap.read()
        if not suc:
            break
        if(cur_args.color_mode=='RGB'):
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        else:
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YUV)
        frames.append(hr)

    return frames
