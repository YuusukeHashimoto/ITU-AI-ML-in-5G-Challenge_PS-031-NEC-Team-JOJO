
import cv2
import os
import sys
import csv

# originalのフレーム群
original_dir = sys.argv[1] 

# receivedの動画（ディレクトリ）
received_dir = sys.argv[2]

path_original = original_dir
path_received = received_dir

files_original = os.listdir(path_original)
files_received = os.listdir(path_received) 
files_original.sort()
files_received.sort()

for target in files_received:

    # List of frames for received videos
    received_frames = os.listdir(path_received + "/" + target)
    received_frames.sort()

    # Number of frames that make up a Received video
    count = len(received_frames)

    # Consider the difference in the number of frames from the original video
    # !! Received video is missing frames due to degradation, and has fewer frames than the original video !!
    Sabun = len(files_original) - len(received_frames)

    # Create a csv file for writing
    filename = target.replace('.mp4', '')  + '.csv'  

    # Ignore the file if it already exists ( For parallel processing ) 
    if(os.path.exists(filename)==False):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for i in range(count):  # Perform the following for all frames that make up a video
                
                img_received = cv2.imread(path_received + "/" + target + "/" + received_frames[i])
                
                MAX_PSNR = 0.0      # A temporary variable that stores the maximum PSNR in a window.
                frame_index = -1    # Index of the frame with the maximum PSNR in the window
                for r in range(i , i + Sabun + 1):       # Refers to the original frame ahead of the current position by the number of difference frames
           
                    img_original = cv2.imread(path_original + "/" + files_original[r])
        
                    # Calculate PSNR by comparing two frames
                    psnr = cv2.PSNR(img_original, img_received)

                    if(MAX_PSNR < psnr): # Save the idex and PSNR value of the frame with the highest PSNR within the window size range.
                        MAX_PSNR = psnr
                        frame_index = r

                # Write the maximum psnr value to the csv filess
                writer.writerow([MAX_PSNR])
    else:
        print("File Exsist!!!!!")



