#python main.py --epochs 250 --dir_data /home/ubuntu/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21
#python main.py --epochs 250 --dir_data /home/ubuntu/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32
#python main.py --epochs 250 --dir_data /home/ubuntu/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 21
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 32
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 48
#python main.py --mode test --test_speed --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 

#python main.py --epochs 2 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21
#python main.py --epochs 2 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 21




#test the psnr and ssim 
#python main.py --epochs 300 --load RCAN_rg3_r4_f21_x4_rgb.pt  --mode compare --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 21 --color_mode RGB
#python main.py --epochs 300 --load RCAN_rg3_r4_f21_x4_yuv.pt --dir_data /home/fangzhou/DIV2K_train_HR --mode compare --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 21 --color_mode YUV
#python main.py --epochs 300 --load MDSR_r20_f21_x4_rgb.pt --dir_data /home/fangzhou/DIV2K_train_HR --mode compare --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode RGB
#python main.py --epochs 300 --load MDSR_r20_f21_x4_yuv.pt --dir_data /home/fangzhou/DIV2K_train_HR --mode compare --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV


# python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode RGB
# python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode RGB
# python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode RGB
# python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV


#python main.py --epochs 1 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 21 --color_mode RGB
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 32 --color_mode RGB
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 48 --color_mode RGB

#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 16 --color_mode YUV
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 6 --n_feats 16 --color_mode YUV
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 8 --n_feats 16 --color_mode YUV

#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 30 --n_feats 21 --color_mode YUV
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 40 --n_feats 21 --color_mode YUV
#python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV


# python main.py --epochs 300 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 16 --color_mode YUV
# python main.py --epochs 300 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 6 --n_feats 16 --color_mode YUV
# python main.py --epochs 300 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 30 --n_feats 21 --color_mode YUV
# python main.py --epochs 300 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 40 --n_feats 21 --color_mode YUV
# python main.py --epochs 300 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --epochs 300 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV

#python main.py --epochs 500 --mode train  --re_encode --video_name yacht --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 16 --color_mode YUV
#python main.py --epochs 500 --mode train  --video_name yacht --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 6 --n_feats 16 --color_mode YUV
#python main.py --epochs 500 --mode train  --video_name yacht --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
#python main.py --epochs 500 --mode train  --video_name yacht --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
#python main.py --epochs 500 --mode train  --video_name yacht --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV

#python main.py --epochs 500 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
#python main.py --epochs 500 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
#python main.py --epochs 500 --mode train  --video_name jocky --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
python main.py --epochs 500 --mode train --re_encode --video_name bee --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --epochs 500 --mode train  --video_name bee --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --epochs 500 --mode train  --video_name bee --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV

