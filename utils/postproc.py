import torch
import numpy as np

def intersec(box1, box2, eps=1e-7):

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    print(box1, box2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((b2 - b1).prod(2) + eps)

def with_gun(humans, guns, inter_thres=0.2):
    humans = torch.tensor(humans) if not isinstance(humans, torch.Tensor) else humans
    guns = torch.tensor(guns) if not isinstance(guns, torch.Tensor) else guns

    humans = humans.cpu()
    guns = guns.cpu()

    if len(humans) == 0 or len(guns) == 0:
        humans = torch.cat([humans, torch.zeros((humans.shape[0], 1), dtype=humans.dtype)], dim=1)
        return humans

    humans = torch.cat([humans, torch.zeros((humans.shape[0], 1), dtype=humans.dtype)], dim=1)

    print(f'humans : {humans}')
    print(f'guns : {guns}')

    inters = intersec(humans[:, :4], guns[:, :4])

    print(f'inters : {inters}, thresh : {inter_thres}')

    # correct = np.zeros(humans.shape[0]).astype(bool)

    # x = torch.where(inters >= inter_thres)  # IoU > threshold and classes match

    # if x[0].shape[0]:
    #     matches = torch.cat((torch.stack(x, 1), inters[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
    #     if x[0].shape[0] > 1:
    #         matches = matches[matches[:, 2].argsort()[::-1]]
    #         matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    #         # matches = matches[matches[:, 2].argsort()[::-1]]
    #         matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    #     correct[matches[:, 1].astype(int)] = True

    correct = (inters >= inter_thres).any(dim=1)

    humans[correct, -1] = 1
    return humans