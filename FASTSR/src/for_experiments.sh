
# #Test normal
# python main.py --mode test --test_only --inference_mode normal --video_name bee4s  --load bee_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name bee --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only --inference_mode normal  --video_name yacht4s  --load yacht_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name yacht --load_model_type train_from_generic  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only --inference_mode normal  --video_name jocky4s  --load jocky_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name jocky --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# #Test fast
# python main.py --mode test --test_only  --inference_mode fast --video_name bee4s  --load bee_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only  --inference_mode fast --video_name bee --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only  --inference_mode fast --video_name yacht4s  --load yacht_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only  --inference_mode fast --video_name yacht --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only  --inference_mode fast  --video_name jocky4s  --load jocky_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
python main.py --mode test --test_only  --inference_mode fast  --video_name jocky --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV


# python main.py --mode test --test_only --inference_mode normal --video_name bee4s  --load bee_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name bee --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only --inference_mode normal  --video_name yacht4s  --load yacht_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name yacht --load_model_type train_from_generic  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only --inference_mode normal  --video_name jocky4s  --load jocky_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name jocky --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# #Test fast
# python main.py --mode test --test_only  --inference_mode fast --video_name bee4s  --load bee_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --mode test --test_only  --inference_mode fast --video_name bee --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only  --inference_mode fast --video_name yacht4s  --load yacht_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --mode test --test_only  --inference_mode fast --video_name yacht --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only  --inference_mode fast  --video_name jocky4s  --load jocky_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
python main.py --mode test --test_only  --inference_mode fast  --video_name jocky --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV


# python main.py --mode test --test_only --inference_mode normal --video_name bee4s  --load bee_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name bee --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only --inference_mode normal  --video_name yacht4s  --load yacht_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name yacht --load_model_type train_from_generic  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only --inference_mode normal  --video_name jocky4s  --load jocky_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
python main.py --mode test --test_only --inference_mode normal  --video_name jocky --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# #Test fast
# python main.py --mode test --test_only  --inference_mode fast --video_name bee4s  --load bee_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
# #python main.py --mode test --test_only  --inference_mode fast --video_name bee --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only  --inference_mode fast --video_name yacht4s  --load yacht_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
# #python main.py --mode test --test_only  --inference_mode fast --video_name yacht --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --test_only  --inference_mode fast  --video_name jocky4s  --load jocky_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV
# #python main.py --mode test --test_only  --inference_mode fast  --video_name jocky --load_model_type train_from_generic --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV



#SR
#test actual frame rate when playback is added


# python main.py --mode test --inference_mode normal   --video_name yacht  --load yacht_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --inference_mode normal   --video_name yacht  --load yacht_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --mode test --inference_mode normal   --video_name yacht  --load yacht_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV

# python main.py --mode test --inference_mode fast   --video_name yacht  --load yacht_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --inference_mode fast   --video_name yacht  --load yacht_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --mode test --inference_mode fast   --video_name yacht  --load yacht_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV

# python main.py --mode test --inference_mode normal   --video_name bee  --load bee_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --inference_mode normal   --video_name bee  --load bee_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --mode test --inference_mode normal   --video_name bee  --load bee_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV

# python main.py --mode test --inference_mode fast   --video_name bee  --load bee_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --inference_mode fast   --video_name bee  --load bee_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --mode test --inference_mode fast   --video_name bee  --load bee_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV


# python main.py --mode test --inference_mode normal   --video_name jocky  --load jocky_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --inference_mode normal   --video_name jocky  --load jocky_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --mode test --inference_mode normal   --video_name jocky  --load jocky_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV

# python main.py --mode test --inference_mode fast   --video_name jocky  --load jocky_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test --inference_mode fast   --video_name jocky  --load jocky_MDSR_r20_f32_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 32 --color_mode YUV
# python main.py --mode test --inference_mode fast   --video_name jocky  --load jocky_MDSR_r20_f48_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 48 --color_mode YUV






# python main.py --mode test --inference_mode normal   --video_name yacht4s  --load yacht_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --mode test  --inference_mode normal  --video_name yacht  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test  --inference_mode normal  --video_name jocky4s  --load jocky_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --mode test  --inference_mode normal  --video_name jocky  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

#test actual frame rate when 

# python main.py --mode test   --inference_mode fast  --video_name bee4s  --load bee_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --mode test   --inference_mode fast  --video_name bee  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test   --inference_mode fast  --video_name yacht4s  --load yacht_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --mode test   --inference_mode fast  --video_name yacht  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV

# python main.py --mode test  --inference_mode fast   --video_name jocky4s  --load jocky_MDSR_r20_f21_x4_YUV.pt --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
# python main.py --mode test   --inference_mode fast  --video_name jocky  --gpu 1  --scale 4 --model MDSR --n_resblocks 20 --n_feats 21 --color_mode YUV
