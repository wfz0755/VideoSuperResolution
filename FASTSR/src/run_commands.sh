#compare the performance of RGB and YUV
python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode compare --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21

python main.py --epochs 300 --dir_data /home/fangzhou/DIV2K_train_HR --mode compare --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 21

#Train generic models
python main.py --epochs 300 --dir_data [the folder storing images] --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
python main.py --epochs 300 --dir_data [the folder storing images] --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode RGB

python main.py --epochs 300 --dir_data [the folder storing images] --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 32 --color_mode YUV
python main.py --epochs 300 --dir_data [the folder storing images] --mode train --reset --train_generic_model --save_models --gpu 1 --scale 4 --patch_size 48 --model RCAN --n_resgroups 3 --n_resblocks 4 --n_feats 32 --color_mode RGB

#Continue training on video
python main.py --mode train --save_models --video_name [video_name without suffix] --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode train --save_models --video_name [video_name without suffix] --gpu 1 --scale 4 --patch_size 48 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV

#Test the speed of different models
python main.py --mode test --test_speed --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48
python main.py --mode test --test_speed --test_all_speed --gpu 1  --scale 4

#Test the performance on video sequence(normal(default) or fast mode)
python main.py --mode test --test_only --inference_mode normal --video_name [video_name without suffix] --re_encode --load [checkpoint name] --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only --inference_mode fast --video_name [video_name without suffix] --re_encode --load [checkpoint name] --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
#if the checkpoint name is not provided, the system will try to load the checkpoint according to the video name, model specification and load model type.
python main.py --mode test --test_only  --video_name [video_name without suffix] --re_encode --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only  --video_name [video_name without suffix] --re_encode --load_model_type train_on_video --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only  --video_name [video_name without suffix] --re_encode --load_model_type generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

#SR playback
python main.py --mode test --inference_mode normal --video_name [video_name without suffix] --load [checkpoint name] --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --inference_mode fast --video_name [video_name without suffix] --load [checkpoint name] --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV --transfer_threshold 10
