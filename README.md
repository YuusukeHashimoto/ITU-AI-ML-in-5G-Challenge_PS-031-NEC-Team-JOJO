# TU AI/ML in 5G Challenge  (PS-031-NEC, Japan)
Theme : Network State Estimation by Analyz-ing Raw Video Data (NEC, Japan)

# Environment
- Operating System : macOS Big Sur 11.0.1
- Python 3.7.2
- Keras 2.3.1
- Tensorflow 2.0.0
- OpenCV 3.4.2
- Ffmpeg 4.3.1

# Problem statement
You can check the details of the problem statement via [this link](https://www.ieice.org/~rising/AI-5G/#theme1)!!

# Date Set
The data set of Theme 2 can be downloaded [here](https://www.ieice.org/~rising/AI-5G/dataset/theme2-NEC/dataset_and_issue.tar.gz) !!


# Brief usage
Our solution consists of four major steps. If you want to skip (Step 1) to (Step 3), you can use the [data](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/tree/main/Data) uploaded to this repository.

## (Step1) Decomposition of raw video data into frames
In this step, the video data is decomposed into frames. [OpenCV](https://opencv.org/) is used for the decomposition process.

example of run: 
```
python3 DecomposingVideo_into_Frames.py [path1] [path2]
```

Description of the arguments given to the program : 

- [*path1*] : Path to the directory containing the video data to be decomposed into frames. If there is more than one video data in the directory, process them all.
- [*path2*] : The path to the directory that will output the frames decomposed by video. If the directory does not exist, a new one will be created.

See the image below for an overview of the paths specified by the arguments.
<img src="image/step1.png" width="700px">

## (Step2) Calculating Time Series Data for Peak Signal to Noise Ratio (PSNR)
In this step, we use the frame group generated in (Step 1). Comparing the frames of the original video and the frames of the received video, the time series data of PSNR is calculated. The PSNR calculation process uses [OpenCV](https://opencv.org/).

example of run: 
```
python3 CalcPSNR.py [path1] [path2]
```

Description of the arguments given to the program : 

- [*path1*] : Path to the directory where the frames of the original video data reside
- [*path2*] : Path to the Directory containing multiple directories in which frames of the received video data exist.

See the image below for an overview of the paths specified by the arguments.
<img src="image/Step2.png" width="700px">

[Here](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/blob/main/example/0GHpTnbnTZs_1100kbps_01.csv) is an example of the output data.

## (Step3) Extraction of information about the frames that make up the original video data
In this step, the video data is analyzed to obtain frame-by-frame information.

We use [FFmpeg](https://ffmpeg.org/) in the program to extract information from the frames that make up a video.  
When this program is executed, the information of the processed original video data is output in JSON format.  The output file contains frame-by-frame information. In the following steps, we will use the information about frame size from this information.

Example of run: 
```
sh Extract_FrameInformation.sh [path] 
```
Description of the arguments given to the program : 

- [*path*] : Path to the directory containing the video files to be processed

See the image below for an overview of the paths specified by the arguments.
<img src="image/Step3.png" width="700px">

[Here](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/blob/main/example/0GHpTnbnTZs.mp4.json) is an example of the output data.

## (Step4) Training the model and estimating the network state
Training and Testing the model using the PSNR time series data calculated in (Step 2) and the frame size information of the frames that make up the original video extracted in (Step 3).

Example of run: 
```
python3 Solusion.py [path1] [path2] [path3] [path4]
```

Description of the arguments given to the program : 

- [*path1*] : Path to the directory where the JSON file generated from the original video exists (for training data)   
→ You can use "[Data/JSON_Data/JSON_forTraining](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/tree/main/Data/JSON_Data/JSON_forTraining)" in this repository.

- [*path2*] : Path to a directory where there are multiple directories containing PSNR time series data files (for training data)   
→ You can use "[Data/PSNR_Data/PSNR_forTraining](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/tree/main/Data/PSNR_Data/PSNR_forTraining)" in this repository.

- [*path3*] : Path to the directory where the JSON file generated from the original video exists (for test data)    
→ You can use "[Data/JSON_Data/JSON_forTest](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/tree/main/Data/JSON_Data/JSON_forTest)" in this repository.

- [*path4*] : Path to a directory where there are multiple directories containing PSNR time series data files (for test data)    
→ You can use "[Data/PSNR_Data/PSNR_forTest](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031-NEC-Team-JOJO/tree/main/Data/PSNR_Data/PSNR_forTest)" in this repository.

See the image below for an overview of the paths specified by the arguments.
<img src="image/Step4.png" width="700px">

# Performance Evaluation
Please refer to our [report](https://github.com/ITU-AI-ML-in-5G-Challenge/PS-031.1_NEC_JOJO/blob/main/ITU_Challenge_FinalConference_JOJO%20.pdf) !!
