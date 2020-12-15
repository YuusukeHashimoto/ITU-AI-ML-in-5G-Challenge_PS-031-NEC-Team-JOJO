#!/bin/sh

# As the first argument, specify the path to the directory containing the video files to be processed.
if [ $# != 1 ]; then
    echo "usage: $0 strings" 1>&2
    exit 0
fi

ARGV1=$1

for file in $ARGV1/*.mp4; do
    echo $file  
    if [ -e $file.json ]; then
        echo "--------File exists.-----------"
    fi

    if [ ! -e $file.json ];then
        echo "---------File not exists.------------"
        ffprobe -show_frames -select_streams v -print_format json $file > $file.json
    fi  
done 