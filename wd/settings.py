import os
remote_ip = os.environ.get('REMOTE_IP', "0.0.0.0")
local_mode = (remote_ip == "0.0.0.0")
weights_path = 'weights/last.pt'  # model path or triton URL
data = 'data/coco128.yaml',  # dataset.yaml path
conf_thres = 0.6
