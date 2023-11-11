from pathlib import Path

import cv2
import imutils

from utils.torch_utils import select_device

from wd.settings import local_mode, weights_path, data, conf_thres
from wd.run_detection import Detector
from wd.stream import vs, cap, outputFrame, lock, frame_idx
FILE = Path(__file__).resolve()

device_gl = select_device('0')
print(f'Gun Detector conf thres = {conf_thres}')

detector = Detector(
    weights=weights_path,  # model path or triton URL
    data=data,  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=conf_thres,  # confidence threshold,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device=device_gl,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    # view_img=False,  # show results
    # save_txt=False,  # save results to *.txt
    # save_csv=False,  # save results in CSV format
    # save_conf=False,  # save confidences in --save-txt labels
    # save_crop=False,  # save cropped prediction boxes
    # nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    # augment=False,  # augmented inference
    # visualize=False,  # visualize features
    # update=False,  # update all models
    # name='exp',  # save results to project/name
    # exist_ok=False,  # existing project/name ok, do not increment
    # line_thickness=3,  # bounding box thickness (pixels)
    # hide_labels=False,  # hide labels
    # hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    # vid_stride=1,  # video frame-rate stride
    bs=None
)
