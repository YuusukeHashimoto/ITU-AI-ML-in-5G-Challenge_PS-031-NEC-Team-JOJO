# TU AI/ML in 5G Challenge  (PS-031-NEC, Japan)
Theme : Network State Estimation by Analyz-ing Raw Video Data (NEC, Japan)

# Environment
- Python 3.7.2
- Keras 2.3.1
- Tensorflow 2.0.0

# Problem statement
You can check the details of the problem statement via [this link](https://www.ieice.org/~rising/AI-5G/#theme1).

# Date Set
The data set of Theme 2 can be downloaded [here](https://www.ieice.org/~rising/AI-5G/dataset/theme2-NEC/dataset_and_issue.tar.gz).


# Brief usage

## (Step1) Decomposition of raw video data into frames

## (Step2) Calculating Time Series Data for Peak Signal to Noise Ratio (PSNR)

## (Step3) Extraction of information about the frames that make up a video
We use [FFmpeg](https://ffmpeg.org/) to extract information from the frames that make up a video.

## (Step4) Model training and network state estimation 
Training and Testing the model using the PSNR time series data calculated in (Step 2) and the frame size information of the frames that make up the original video extracted in (Step 3).

Example of run: 
```
python3 Solusion.py [path1] [path2] [path3] [path4]
```

Description of the arguments given to the program : 

- [*path1*] : Path to the directory where the JSON data is stored (for training)  
- [*path2*] : Path to the directory where the PSNR time series data is stored (for training)  
- [*path3*] : Path to the directory where the JSON data is stored (for test)  
- [*path4*] : Path to the directory where the PSNR time series data is stored (for test)  

# Performance Evaluation
Please refer to [report](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031.1_NEC_JOJO/blob/main/ITU_Challenge_FinalConference_JOJO%20.pdf)
