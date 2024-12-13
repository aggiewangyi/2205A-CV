# -*- coding: utf-8 -*-
import torch
import time
import numpy as np
import torchvision
import cv2
import yaml
import logging.config
import os

class Colors:
    def __init__(self) -> None:
        hexs = ("FF3838","FF9D97","FF701F","FFB21D","CFD231","48F90A","92CC17",
                "3DDB86","1A9334","00D4BB","2C99A8","00C2FF","344593","6473FF",
            "0018EC","8438FF","520085","CB38FF","FF95C8","FF37C7")
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i)  % self.n]
        return (c[2],c[1],c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2],16) for i in (0,2,4))

def xywh2xyxy(x):
    y = x.clone() if isinstance(x,torch.Tensor) else np.copy(x)
    y[...,0] = x[...,0] - x[...,2]/2
    y[...,1] = x[...,1] - x[...,3]/2
    y[...,2] = x[...,0] + x[...,2]/2
    y[...,3] = x[...,1] + x[...,3]/2
    return y

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

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
    # print(boxes)

def scale_boxes(img0_shape, boxes, img1_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        print(gain)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain

    clip_boxes(boxes, img1_shape)
    return boxes

def box_label(im, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = 2
    p1,p2 = (int(box[0]),int(box[1])),(int(box[2]),int(box[3]))
    cv2.rectangle(im,p1,p2,color,thickness=lw,lineType=cv2.LINE_AA)
    if label:
        tf = max(lw-1,1)
        w,h = cv2.getTextSize(label,0,fontScale=lw/3,thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] +  w,p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im,p1,p2,color,-1,cv2.LINE_AA)
        cv2.putText(im,label,(p1[0],p1[1]-2 if outside else p1[1] + h + 2),0,lw/3,txt_color,
                    thickness=tf,lineType=cv2.LINE_AA)
    return im

def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

# def post_process_yolov5(det,im,label_path = 'coco_label.yaml'):
def post_process_yolov5(det,org_data,names,new_img):
    # post_process_yolov5(pred[0], org_data, label_name, im)
    if len(det):
        det[:,:4] = scale_boxes(org_data.shape[:2],det[:,:4],new_img.shape[2:]).round()
        # def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):

        # names = yaml_load(label_path)['names']
        colors= Colors()
        for *xyxy,conf,cls in reversed(det):
            c = int(cls)
            label = names[c]
            box_label(org_data,xyxy,label,color=colors(c,True))
    return org_data


def non_max_suppression(
        prediction,
        conf_thres = 0.25,
        iou_thres = 0.45,
        classes = None,
        agnostic = False,
        multi_label = False,
        labels = (),
        max_det = 300,
        nm = 0,
):
    assert 0 <= conf_thres <= 1,f'Invalid Confidence threshold {conf_thres}'
    assert 0 <= iou_thres <= 1,f'Invalid Iou {iou_thres}'
    if isinstance(prediction,(list,tuple)):
        prediction = prediction[0]

    device = prediction.device
    mps = 'mps' in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = prediction.shape[2]
    xc = prediction[...,4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 +0.05  * bs
    redundant = True
    multi_label &= nc>1
    merge = False

    t =time.time()
    mi = 5 + nc
    output = [torch.zeros((0,6+nm),device = prediction.device)] * bs
    for xi,x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb),nc + nm + 5),device=x.device)
            v[:,:4] = lb[:,1:5]
            v[:,4] = 1.0
            v[range(len(lb)),lb[:,0].long() + 5] = 1.0
            x = torch.cat((x,v),0)
        if not x.shape[0]:
            continue
        x[:,5:] *= x[:,4:5]

        box = xywh2xyxy(x[:,:4])
        mask = x[:,mi:]

        if multi_label:
            i,j = (x[:,5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i],x[i,5+j,None],j[:,None].float(),mask[i]),1)
        else:
            conf,j = x[:,5:mi].max(1,keepdim=True)
            x = torch.cat((box,conf,j.float(),mask),1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:,5:6] ==torch.tensor(classes,device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:,4].argsort(descending=True)[:max_nms]]

        c = x[:,5:6]*(0 if agnostic else max_wh)
        boxes,scores =x[:,:4] + c,x[:,4]
        i = torchvision.ops.nms(boxes,scores,iou_thres)
        i = i[:max_det]
        if merge and (1<n<3E3):
            iou = box_iou(boxes[i],boxes) > iou_thres
            weights = iou * scores[None]
            x[i,:4] = torch.mm(weights,x[:,:4]).float() /weights.sum(1,keepdim=1)
            if  redundant:
                i =  i[iou.sum(1)>1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time()-t)>time_limit:
            break
        if (time.time()-t)>time_limit:
            break
    return output

def resize_image_cv2(image,size):
    ih,iw,ic = image.shape
    w,h = size

    scale = min(w/iw,h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image,(nw,nh))
    new_image = np.ones((size[0],size[1],3),dtype='uint8')*128
    start_h = (h-nh)/2
    start_w = (w-nw)/2

    end_h = size[1] - start_h
    end_w = size[0] - start_w

    new_image[int(start_h):int(end_h),int(start_w):int(end_w)] = image

    return new_image,nw,nh

def data_process_cv2(frame,input_shape):
    image_data,nw,nh = resize_image_cv2(frame,(input_shape[1],input_shape[0]))
    np_data = np.array(image_data,np.float32)
    np_data = np_data/255
    image_data = np.expand_dims(np.transpose(np_data,(2,0,1)),0)
    image_data = np.ascontiguousarray(image_data)

    return image_data,frame






        # if mps:
        #     output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
        #     break  # time limit exceeded
# LOGGING_NAME = "yolov5"
#
# def set_logging(name=LOGGING_NAME, verbose=True):
#     """Configures logging with specified verbosity; `name` sets the logger's name, `verbose` controls logging level."""
#     rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
#     level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
#     logging.config.dictConfig(
#         {
#             "version": 1,
#             "disable_existing_loggers": False,
#             "formatters": {name: {"format": "%(message)s"}},
#             "handlers": {
#                 name: {
#                     "class": "logging.StreamHandler",
#                     "formatter": name,
#                     "level": level,
#                 }
#             },
#             "loggers": {
#                 name: {
#                     "level": level,
#                     "handlers": [name],
#                     "propagate": False,
#                 }
#             },
#         }
#     )
#
# set_logging(LOGGING_NAME)  # run before defining LOGGER
# LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
# # if platform.system() == "Windows":
# #     for fn in LOGGER.info, LOGGER.warning:
# #         setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging
#

