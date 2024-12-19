# -*- coding: utf-8 -*-
import torch
import time
import numpy
import torchvision
import cv2
import yaml
import logging
import os
import logging.config
import platform

class Colors:
    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "2C99A8", "00C2FF", "344593", "FFB21D",
            "CFD231", "48F90A", "92CC17", "6473FF", "0018EC", "8438FF", "520085",
            "CB38FF", "FF95C8", "FF37C7", "FFDDFF", "00E436", "FFD23F", "0089BA",
            "D100F1", "F1C100", "F1D300", "BEDE0D", "FCC200", "F1A100", "F1CA7D",
            "8CE600", "00D8FF", "D1F1F1", "F1E2E2", "F1DADA", "F1B2B2", "F1A1A1",
            "F1D4D4", "F1E8E8", "F1F4F4", "F1BBBB", "F1BFBF", "F1D6D6", "F1E2E2",
            "F1EEEE", "F1F9F9", "F1C4C4", "F1D1D1", "F1E6E6", "F1F3F3", "F1ECEC",
            "F1F7F7", "F1D9D9", "F1E5E5", "F1F2F2", "F1DEDE", "F1EFEF", "F1F8F8",
            "F1E1E1", "F1F6F6", "F1D8D8", "F1E4E4", "F1F1F1", "F1DDDD", "F1E9E9",
            "F1F5F5", "F1C8C8", "F1D3D3", "F1E7E7", "F1F4F4", "F1C6C6", "F1D2D2",
            "F1E6E6", "F1F3F3", "F1C5C5", "F1D1D1", "F1E5E5", "F1F2F2", "F1CDCD",
            "F1D7D7", "F1E3E3", "F1F0F0", "F1CACACA", "F1D4D4", "F1E8E8", "F1F5F5"
        )

        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[i + 1 : 1 + i + 2], 16) for i in (0, 2, 4))

# def xywh2xyxy(x):
#     y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
#     y[..., 0] = x[..., 0] - x[..., 2] / 2
#     y[..., 1] = x[..., 1] - x[..., 3] / 2
#     y[..., 2] = x[..., 0] - x[..., 2] / 2
#     y[..., 3] = x[..., 1] - x[..., 3] / 2
#     return y
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_min
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_min
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_max
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_max
    return y


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1).clamp(0).prod(2))
    return inter / ((a1 - a2).prod(2) + (b2 - b1).prod(2) - inter + eps)

def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def box_label(im, box, label='', color=(128,128,128), txt_color=(255,255,255)):
    lw = 2
    color = (255, 0, 0)
    w = 0
    h = 0
    # if label:
    #     tf = max(lw - 1, 1)
    #     w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, 2, lineType=cv2.LINE_AA)
    # cv2.imshow('img', im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    if label:
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 2
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
        # cv2.imshow('img', im)
        # cv2.waitKey()
        # cv2.destroyAllWin
        #
        # dows()
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        # cv2.imshow('img', im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    return im

def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

def post_process_yolov5(det, im, label_path="coco_label.yaml"):
    if len(det):
        det[:, :4] = scale_boxes(im.shape[:2], det[:, :4], im.shape).round()
        names = yaml_load(label_path)['names']
        colors = Colors()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = names[c]
            box_label(im, xyxy, label, color=colors(c, True))

            # cv2.imshow('img', im)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    return im

LOGGING_NAME = "yolov5"

def set_logging(name=LOGGING_NAME, verbose=True):
    """Configures logging with specified verbosity; `name` sets the logger's name, `verbose` controls logging level."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )

set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
# if platform.system() == "Windows":
#     for fn in LOGGER.info, LOGGER.warning:
#         setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output



