# Video_SR
A video super-resolution system that can train DNN using video and make inference in real-time with normal GPUs.


#FAST mode instruction
Steps in test_model:FAST:  
1. Read .mp4 HR video as input and generated cropped frames and .cfg file.  
2. Those frames are encoded in to HR .yuv file and downsampled to generate LR .yuv file. After getting .yuv file, we can use HEVC encoder to encode YUV file into binary .str file.  
3. Use HEVC dumper to decode LR .yuv file and generate dumped syntax file plus reconstructed LR .yuv file.  
4. Read LR frames from .yuv file using self-defined VideoCapture.  
5. Parse dumped syntax file and reconstruct Residual for all frames.  Apply bicubic interpolation on those residual for upsampling.  (In the mean while, we read HR frames from HR.yuv file for performance evaluation.)  
6. Load pretrained DNN model.  
7. Now we start pixel transfering + super-resolution