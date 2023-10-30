#!/bin/bash
cmd="ffmpeg -f v4l2 -framerate 30 -video_size 640x480 -i /dev/cams/camEntrance -fflags nobuffer -r 30 -tune zerolatency -probesize 32 -preset ultrafast -tune zerolatency -b:v 2000k -f rtsp -rtsp_transport udp rtsp://localhost:8554/entrance"

until $cmd > /dev/null 2>&1 < /dev/null; do
	echo "restarting ffmpeg command..."
	sleep 2
done
