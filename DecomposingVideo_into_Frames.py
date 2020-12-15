import cv2
import os
import sys

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return


# The path to the directory where the video data to be processed exists is specified as an argument
videofile_path = sys.argv[1]
# The path to the directory where the data will be stored after the video is broken down into frames.
output_dir = sys.argv[2]
video_files = os.listdir(videofile_path)
video_files.sort()

for i in video_files:
    
    print('Processing..... ' + i )

    target_videofile = videofile_path + "/" + i
    output_directory = output_dir + "/" + i

    #もしすでにデイレクトリが生成済みの場合
    if(os.path.exists(output_directory)):
        print(">>Directry is Exist!!")
        continue

    save_all_frames(target_videofile, output_directory, 'frame')
