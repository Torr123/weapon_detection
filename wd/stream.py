import threading

import cv2
from imutils.video import VideoStream

from wd.settings import local_mode, remote_ip

vs = None
cap = cv2.VideoCapture()
outputFrame = None
lock = threading.Lock()
frame_idx = 0

static_back = None
motion = 0
motion_list = [None, None]

if local_mode:
    vs = VideoStream(src=0).start()
else:
    cap.open(remote_ip)

