import cv2
import numpy as np
import torch
from torchvision import transforms

from human_det.utils.datasets import letterbox
from human_det.utils.general import non_max_suppression_kpt
from human_det.utils.plots import output_to_keypoint


def draw(image, bboxes):
    h, w, _ = image.shape
    color = (255, 0, 0)
    thickness = 2
    print(f"h = {h} w = {w}")
    for bbox in bboxes:
        start_point = (int(bbox[0] - bbox[2] / 2), int(bbox[1]  - bbox[3] / 2))
        end_point = (int((bbox[0] + bbox[2] / 2) * 1), int((bbox[1] + bbox[3] / 2) * 1))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


class HumanDetector:
    def __init__(self, weights, device='cpu'):
        self.cuda_avail = False
        if device:
            self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'RUN DETECTOR ON {self.device}')

        self.model = torch.load(weights, map_location='cpu')['model']
        self.model = self.model.float().eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.cuda_avail = True
            self.model.half().to(self.device)

    def preprocess(self, image):
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if self.cuda_avail:
            image = image.half().to(self.device)

        return image

    def detect(self, image):
        im = self.preprocess(image)
        self.size = im.shape[-2:]
        # print('QQQQQQQQQ', self.size)
        output, _ = self.model(im)
        # input(output)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'],
                                         kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        
        if len(output) == 0:
            return output
        
        output = output[:, 2:6]
        f_output = output.copy()
        output[:, [0, 1]] = f_output[:, [0, 1]] - f_output[:, [2, 3]] / 2
        output[:, [2, 3]] = f_output[:, [0, 1]] + f_output[:, [2, 3]] / 2
        h, w, _ = image.shape
        output[:, [0, 2]] = output[:, [0, 2]] / self.size[1] * w
        output[:, [1, 3]] = output[:, [1, 3]] / self.size[0] * h
        
        return output

    def get_img_with_preds(self, image, bboxes):
        h, w, _ = image.shape
        print(f'draw : {h}, {w}')
        color = (255, 0, 0)
        thickness = 2
        print(f"h = {h} w = {w}")
        for bbox in bboxes:
            # start_point = (int((bbox[0] - bbox[2] / 2) / 640 * w), int((bbox[1]  - bbox[3] / 2) / 480 * h))
            # end_point = (int((bbox[0] + bbox[2] / 2) / 640 * w), int((bbox[1] + bbox[3] / 2) / 480 * h))
            # image = cv2.rectangle(image, start_point, end_point, color, thickness)

            color = (0, 255, 0) if bbox[-1] == 0 else (0, 0, 255)
            print(f'With Gun : {bbox[-1]}')
            
            # start_point = (
            # int((bbox[0] - bbox[2] / 2) / self.size[1] * w), int((bbox[1] - bbox[3] / 2) / self.size[0] * h))
            # end_point = (
            # int((bbox[0] + bbox[2] / 2) / self.size[1] * w), int((bbox[1] + bbox[3] / 2) / self.size[0] * h))
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

        return image
