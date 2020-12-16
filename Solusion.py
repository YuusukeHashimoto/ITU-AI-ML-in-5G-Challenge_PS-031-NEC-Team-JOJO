from keras.models import Input, Model
from keras.layers import LSTM, Dense, TimeDistributed, concatenate, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import to_categorical
from keras.layers import CuDNNLSTM
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping
from keras import metrics
import numpy as np
import os, random, itertools, csv, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from keras.utils import plot_model
import random
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sys

# Global Variables
SegmentSize = 30 #Segment size (number of frames)
nExtractedSegments = 100 # Number of segments to be extracted
nTotalExtractedPSNRData = SegmentSize*nExtractedSegments # Number of total PSNR data to be extracted
PSNR_Threshold = 50 # The upper limits of PSNR
batchsize = 16
epoch = 100
k = 10 # Number of divisions of cross-validation

###########################################################
# Input : 
# Path to the directory where the JSON files are stored
# Return : 
# Index of segments with high data rate (top 10)
# Index of segments with low data rate (top 10)
###########################################################
def CalculateIndex_ExtractedSegment(path):
    
    # Get a list of JSON file names
    JSON_FILES = os.listdir(path)
    JSON_FILES.sort()
    nSample = len(JSON_FILES)

    highFrameSizeSection_Index = [] # Array to store the index of the segment with the highest data rate
    lowFrameSizeSection_Index = []  # Array to store the index of the segment with the lowest data rate

    # Read all JSON files sequentially
    for file in JSON_FILES:

        # Initialization Division
        framecount = 0      
        sum_frame_size = 0  
        FrameSection = []   
        
        json_open = open(path + "/" + file, 'r') # Open JSON file
        json_load = json.load(json_open)         # Load opened JSON file

        for frame in json_load['frames']:   # Extract the frames section in the JSON file
            framesize = (int(frame['pkt_size']) * 8)/1000   # Calculate the frame size [kbit]
            sum_frame_size += framesize
            framecount+=1


            if (framecount%SegmentSize == 0):   # Loaded frames for a segment size
                FrameSection.append(sum_frame_size)                
                # Initialize the variables
                sum_frame_size = 0
                framecount = 0


        # Convert to a numpy array for sorting
        FrameSection = np.array(FrameSection)
        
        # Sort by ascending order
        FrameSection_SortedIndex = np.argsort(FrameSection) 
       
        n = nExtractedSegments + 1  # Extract (nExtractedSegments + 1) segments to avoid errors (The last one is a spare) [2020/12/8]

        # Extract the top (nExtractedSegments + 1) segments of the data rate
        highFrameSizeSection_Index = np.append(highFrameSizeSection_Index, FrameSection_SortedIndex[:n])
        # Extract the Worst (nExtractedSegments + 1) segments of the data rate
        lowFrameSizeSection_Index = np.append(lowFrameSizeSection_Index, FrameSection_SortedIndex[-n:])
       

    return highFrameSizeSection_Index.reshape(nSample, n), lowFrameSizeSection_Index.reshape(nSample, n)   

            

def SegmentExtraction(Top10, Worst10, path_PSNR):

    PSNR_SequenceData_Top10Section = []     # 対象となる区間の全フレームを格納していく1次元配列(最後に(480, 500, 1)となるようreshapeする)
    PSNR_SequenceData_Worst10Section = []   # 対象となる区間の全フレームを格納していく1次元配列(最後に(480, 500, 1)となるようreshapeする)
    FileNameList = np.array([])             #csvファイルを取り出した順にファイル名を格納する（のちに教師データの生成に使用）
   
    Each_PSNR_directory = os.listdir(path_PSNR)
    Each_PSNR_directory.sort()

    VideoContentCount = 0
    nDataFile = 0

    for Target_PSNR_directory in Each_PSNR_directory:   # This loop will be repeated 8 times (the number of original video types)
        TargetTop10 = Top10[VideoContentCount]
        TargetWorst10 = Worst10[VideoContentCount]


        PSNR_CSVfiles = os.listdir(path_PSNR + "/" + Target_PSNR_directory)
        PSNR_CSVfiles.sort()
        

        # Load a csv file containing PSNR time series data one by one
        for tagetfile in PSNR_CSVfiles:                 # This loop is repeated 60 times (for received video data)
            df = pd.read_csv(path_PSNR + '/' + Target_PSNR_directory + '/' + tagetfile, dtype=float, names=['temp']) # Read csv file and convert to DataFrame
            df = df.values.tolist()                         
            df = list(itertools.chain.from_iterable(df))   
            DataLength = len(df)  #PSNR Time Series Data Length

            # Truncate unusually high values of PSNR
            for psnr_index in range(len(df)):
                if(df[psnr_index] >= PSNR_Threshold):
                    df[psnr_index] = PSNR_Threshold

            # Normalizing PSNR time series data
            MS_Scalor =MinMaxScaler(feature_range=(0, 1), copy=True)  # MinMaxScaler
            df = np.array([df]).reshape(-1, 1)
            df = MS_Scalor.fit_transform(df)

            #--- Extraction of PSNR time series data for the corresponding segment ---#
            # Extract the segments with high data rates
            Errorflag = False   # ErrorFlag
            for i in range(nExtractedSegments): # Extract the Top nExtractedSegments segment index of the data rate (df[i,n] to get the elements i ~ (n-1))
                
                Segment_StartPosition = int(TargetTop10[i])*SegmentSize
                Segment_EndPosition = (int(TargetTop10[i]) + 1)*SegmentSize

                if (DataLength < Segment_EndPosition):  # Skip the processing if the extracted frame segments exceed the PSNR data length
                    Errorflag = True
                else:
                    PSNR_SequenceData_Top10Section.append(df[Segment_StartPosition : Segment_EndPosition])
            
            if(Errorflag == True):   # Add a spare 1 segment if the error flag is True
                Segment_StartPosition = int(TargetTop10[nExtractedSegments])*SegmentSize
                Segment_EndPosition = (int(TargetTop10[nExtractedSegments]) + 1)*SegmentSize
                PSNR_SequenceData_Top10Section.append(df[Segment_StartPosition : Segment_EndPosition])

            # Extract the segments with low data rates
            Errorflag = False   # ErrorFlag
            for i in range(nExtractedSegments): # Extract the Worst n segment index of the data rate (df[i,n] to get the elements i ~ (n-1))
                Segment_StartPosition = int(TargetWorst10[i])*SegmentSize
                Segment_EndPosition = (int(TargetWorst10[i]) + 1)*SegmentSize
                
                if (DataLength < Segment_EndPosition):  # Skip the processing if the extracted frame segments exceed the PSNR data length
                    Errorflag = True
                else:
                    PSNR_SequenceData_Worst10Section.append(df[Segment_StartPosition : Segment_EndPosition])

            if(Errorflag == True):   # Add a spare 1 segment if the error flag is True
                Segment_StartPosition = int(TargetTop10[nExtractedSegments])*SegmentSize
                Segment_EndPosition = (int(TargetTop10[nExtractedSegments]) + 1)*SegmentSize
                PSNR_SequenceData_Top10Section.append(df[Segment_StartPosition : Segment_EndPosition])


            # Storing the processed file names
            FileNameList = np.append(FileNameList, tagetfile)
            # Count the number of data read
            nDataFile += 1
            
        # Move on to the next video file.
        VideoContentCount += 1
      
    

    PSNR_SequenceData_Top10Section = np.array(PSNR_SequenceData_Top10Section)
    PSNR_SequenceData_Worst10Section = np.array(PSNR_SequenceData_Worst10Section)
    

    PSNR_SequenceData_Top10Section = np.ravel(PSNR_SequenceData_Top10Section)
    PSNR_SequenceData_Worst10Section = np.ravel(PSNR_SequenceData_Worst10Section)

    return PSNR_SequenceData_Top10Section.reshape(nDataFile, nTotalExtractedPSNRData, 1), PSNR_SequenceData_Worst10Section.reshape(nDataFile, nTotalExtractedPSNRData, 1), FileNameList
    


####################################################
### Extract the correct label from the file name ###
####################################################
def create_teacherdata(CSVFILES):

    nSample = len(CSVFILES)

    y_train_Throughput = []
    y_train_PLR  = []
    for i in CSVFILES:
        tmp = i.replace('.csv','').split('_')
        # Extracting "bandwidth" label information from file names
        tmpA = tmp[1]
        Throughput = tmpA.replace('kbps',".")
        # Extracting "Packet Loss Rate" label information from file names
        tmpA = list(tmp[2])
        tmpA.insert(1,".")

        try:
            PLR = float("".join(tmpA))
        except ValueError as e:
            tmpA = tmp[2]
            Throughput = tmpA.replace('kbps',".")

            tmpA = list(tmp[3])
            tmpA.insert(1,".")
            PLR = float("".join(tmpA))

        y_train_Throughput.append(float(Throughput))
        y_train_PLR.append(float(PLR))

    y_train_Throughput = np.array(y_train_Throughput)
    y_train_PLR = np.array(y_train_PLR)


    y_train_Throughput = y_train_Throughput.reshape(-1, 1)
    y_train_PLR = y_train_PLR.reshape(-1, 1)  
    
    return y_train_Throughput.reshape(nSample, 1), y_train_PLR.reshape(nSample, 1) 


args = sys.argv
path_JSON_training = args[1]  # Path to the directory where the JSON data is stored (for training)
path_PSNR_training = args[2]  # Path to the directory where the PSNR time series data is stored (for training)
path_JSON_test = args[3]  # Path to the directory where the JSON data is stored (for test)
path_PSNR_test = args[4]  # Path to the directory where the PSNR time series data is stored (for test)


#################################
###  Generating training data ###
#################################
#path_JSON = '/Users/yusukehashimoto/MySpace/ITU_Challenge/FarameData_JSON/original/' 
Top10, Worst10 = CalculateIndex_ExtractedSegment(path_JSON_training)  

#path_PSNRData = '/Users/yusukehashimoto/MySpace/ITU_Challenge/PSNR'
Top10Section_PSNRSequenceData, Worst10Section_PSNRSequenceData, FileNameList = SegmentExtraction(Top10, Worst10, path_PSNR_training) 
print(Top10Section_PSNRSequenceData.shape)
print(Worst10Section_PSNRSequenceData.shape)

y_Throughput, y_PacketLossRate = create_teacherdata(FileNameList)

# Shuffle the dataset (to eliminate bias due to it being initially sorted by name)
p = np.random.permutation(len(Top10Section_PSNRSequenceData))
Top10Section_PSNRSequenceData = Top10Section_PSNRSequenceData[p]
Worst10Section_PSNRSequenceData = Worst10Section_PSNRSequenceData[p]
y_Throughput = y_Throughput[p]
y_PacketLossRate = y_PacketLossRate[p]
FileNameList = FileNameList[p]


MAE_Bandwidth = []
MAE_LossRate = []
# k-Cross Validation
kf = KFold(n_splits=k, shuffle=True)
for train_index, val_index in kf.split(Top10Section_PSNRSequenceData, y_Throughput):
    # Data for training 
    x_train_w = Worst10Section_PSNRSequenceData[train_index]
    x_train_t = Top10Section_PSNRSequenceData[train_index]
    y_train_p = y_PacketLossRate[train_index]
    y_train_b = y_Throughput[train_index]
    
    # Data for validation     
    x_train_w_val = Worst10Section_PSNRSequenceData[val_index]
    x_train_t_val = Top10Section_PSNRSequenceData[val_index]
    y_train_p_val = y_PacketLossRate[val_index]
    y_train_b_val = y_Throughput[val_index]
    x_train_val = [x_train_w_val, x_train_t_val]
    y_train_val = [y_train_p_val, y_train_b_val]

        
    ####################
    # Generation model #
    ####################
    input1 = Input(shape=(nTotalExtractedPSNRData, 1))
    x1 = Conv1D(32, SegmentSize, strides = SegmentSize, activation='linear')(input1)
    x1 = GlobalAveragePooling1D()(x1)
    x1 = Dense(16, activation='linear')(x1)
    PLR_OUT = Dense(1, activation='linear', name='Output_PacketLossRate1')(x1)   

    input2 = Input(shape=(nTotalExtractedPSNRData, 1))
    x2 = Conv1D(32, SegmentSize, strides = SegmentSize, activation='linear')(input2) 
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dense(16, activation='linear')(x2)
    THROUGHTPUT_OUT = Dense(1, activation='linear', name='Output_Bandwidth')(x2)     

    model = Model(input=[input1, input2], outputs=[PLR_OUT, THROUGHTPUT_OUT])

    # Output the model summary
    print(model.summary())
    # Model overview figure file output
    plot_model(model, to_file='MyModel.png', show_shapes=True)

    # Compilation of the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])

    ############
    # Training #
    ############
    hist = model.fit([x_train_w, x_train_t], [y_train_p, y_train_b], epochs=epoch, batch_size=batchsize, verbose=1)

    ############
    # Evaluate # 
    ############
    score = model.evaluate(x_train_val, y_train_val, verbose=0)
    print(str(score[3]) + " " +str(score[3]))
    MAE_LossRate.append(score[3])
    MAE_Bandwidth.append(score[4])
    

tmp = 0.0
with open("MAE_Bandwidth(Evaluate).txt", 'w') as h:
    for i in MAE_Bandwidth:
        h.write(str(i) + "\n")
        tmp += i
    h.write("--average--\n")
    h.write(str(tmp/k) + "\n")

tmp = 0.0
with open("MAE_LossRate(Evaluate).txt", 'w') as h:
    for i in MAE_LossRate:
        h.write(str(i) + "\n")
        tmp += i
    h.write("--average--\n")
    h.write(str(tmp/k) + "\n")

###########
# Predict #
###########
#path_JSON_test= '/Users/yusukehashimoto/MySpace/ITU_Challenge/FarameData_JSON/issue/original'
Top10, Worst10 = CalculateIndex_ExtractedSegment(path_JSON_test) 
#path_PSNR_test = '/Users/yusukehashimoto/MySpace/ITU_Challenge/PSNR_issue'
x_train_t_test, x_train_w_test, Test_FileNameList = SegmentExtraction(Top10, Worst10, path_PSNR_test) 

nTestSample = 10 
pre_throughputs_array = []        # An array for storing the estimated bandwidth (later used for calculating the mean square error)
pre_PacketLossRates_array = []    # An array for storing the estimated packet loss rate (later used for calculating the mean square error)

# Thes label for tasting data
y_train_b_test = np.array([1200, 1800, 1400, 1600, 1400, 1100, 1300, 1700, 1900, 1400])
y_train_p_test = np.array([0.01, 0.001, 0.25, 0.025, 0.25, 0.001, 0.001, 0.025, 0.025, 0.25])

for index in range(nTestSample):
    # Predicted results
    predictions = model.predict([x_train_w_test[index].reshape(1, nTotalExtractedPSNRData ,1), x_train_t_test[index].reshape(1, nTotalExtractedPSNRData ,1)], verbose=0)

    pre_PacketLossRate = predictions[0]
    pre_throughput = predictions[1]
    
    print("Bandwidth : " + str(pre_throughput) + "   Loss Rate : " + str(pre_PacketLossRate))

    # Store the estimation results in an array
    pre_throughputs_array.append(pre_throughput[0])
    pre_PacketLossRates_array.append(pre_PacketLossRate[0])
    
# Compute MAE with estimated results and correct labels
MSE_Throughput = mean_absolute_error(y_train_b_test, pre_throughputs_array)
MSE_PacketLossRate = mean_absolute_error(y_train_p_test, pre_PacketLossRates_array)
print("[MSE]  Bandwidth :" + str(MSE_Throughput) + " Loss Rate :" + str(MSE_PacketLossRate))

