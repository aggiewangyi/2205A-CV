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

# 定义了一个名为 Colors 的类。  这个类可以被用来获取一个颜色列表中的颜色，并且可以方便地将颜色转换为RGB或BGR格式。
class Colors:
    # 定义了类的构造函数 __init__，它会在创建类的实例时自动调用。
    def __init__(self):
        # 在构造函数中定义了一个字符串元组 hexs，包含了20个十六进制颜色代码。
        hexs = ('FE3838','FF9D97','FE701F','FFB21D','CED231','48F90A','92CC17','3DDB86','1A9334','00D4BB',
                '2C99A8','00C2FE','344593','6473FE','0018EC','8438EE','520085','CB38FE','EE95C8','FE37C7')
        # 使用列表推导式和 hex2rgb 方法将十六进制颜色代码转换为RGB元组，并将结果列表赋值给实例变量 self.palette。
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        # 计算颜色列表的长度，并将其存储在实例变量 self.n 中，用于后续的颜色索引循环。
        self.n = len(self.palette)

    # 定义了一个特殊方法 __call__，使得类的实例可以被当作函数来调用。它接受一个索引 i 和一个布尔值 bgr。
    def __call__(self, i, bgr=False):
        # 通过对索引 i 取模 self.n 来确保索引在颜色列表的范围内，并从 self.palette 中获取相应的颜色。
        c = self.palette[int(i) % self.n]
        # 如果 bgr 参数为 True，则返回一个元组，其顺序为蓝、绿、红（BGR），这是某些图像处理库（如OpenCV）使用的颜色顺序。如果 bgr 为 False，则按原样返回RGB颜色。
        return (c[2], c[1], c[0]) if bgr else c

    # 定义了一个静态方法 hex2rgb，它不需要访问任何类或实例的状态。这个方法将一个十六进制颜色字符串转换为RGB元组。
    @staticmethod
    def hex2rgb(h):
        # 使用列表推导式将十六进制字符串的每两位转换为一个整数，表示RGB颜色值中的一个分量。
        # 字符串切片 h[i + 1 : 1 + i + 2] 用于提取两位十六进制数，int(..., 16) 将其转换为十进制整数。
        return tuple(int(h[i + 1 : 1 + i + 2], 16) for i in (0, 2, 4))

# def xywh2xyxy(x):
#     y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
#     y[..., 0] = x[..., 0] - x[..., 2] / 2
#     y[..., 1] = x[..., 1] - x[..., 3] / 2
#     y[..., 2] = x[..., 0] - x[..., 2] / 2
#     y[..., 3] = x[..., 1] - x[..., 3] / 2
#     return y


# 定义了一个函数 xywh2xyxy，它用于将边界框的表示从 (x, y, width, height) 转换为 (x_min, y_min, x_max, y_max)。
# 定义了一个名为 xywh2xyxy 的函数，它接受一个参数 x，这个参数可以是一个包含边界框信息的张量（使用PyTorch）或者一个NumPy数组。
def xywh2xyxy(x):
    # 检查输入 x 是否是 torch.Tensor 类型。如果是，使用 .clone() 方法创建 x 的一个副本。如果不是，使用 numpy.copy() 创建一个副本。副本 y 将用于存储转换后的边界框信息。
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    # 计算边界框的左上角 x 坐标（x_min）。这通过从中心 x 坐标（x[..., 0]）减去边界框宽度的一半（x[..., 2] / 2）来完成。
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_min
    # 计算边界框的左上角 y 坐标（y_min）。这通过从中心 y 坐标（x[..., 1]）减去边界框高度的一半（x[..., 3] / 2）来完成。
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_min
    # 计算边界框的右下角 x 坐标（x_max）。这通过将中心 x 坐标（x[..., 0]）加上边界框宽度的一半（x[..., 2] / 2）来完成。
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_max
    # 计算边界框的右下角 y 坐标（y_max）。这通过将中心 y 坐标（x[..., 1]）加上边界框高度的一半（x[..., 3] / 2）来完成。
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_max
    # 返回转换后的边界框表示 y。
    return y

# 定义了一个名为 box_iou 的函数，它接受两个参数 box1 和 box2，这两个参数都是边界框的张量。eps 是一个很小的正数，用于在计算中防止除以零的错误。
def box_iou(box1, box2, eps=1e-7):
    # box1.unsqueeze(1) 在 box1 的第1维（索引从0开始）增加一个维度，使得 box1 的形状变为 [N, 1, 4]，其中 N 是边界框的数量。
    # .chunk(2, 2) 在第2维（索引从0开始）上将张量分割为两个部分，每个部分包含两个元素。对于 box1 和 box2，这将产生四个张量，分别表示边界框的左上角和右下角坐标。
    # (a1, a2) 和 (b1, b2) 分别代表 box1 和 box2 中每个边界框的左上角和右下角坐标。
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)\
    # torch.min(a2, b2) 计算两个边界框右下角坐标的最小值。
    # torch.max(a1, b1) 计算两个边界框左上角坐标的最大值。
    # .clamp(0) 将坐标限制为非负值，确保计算的交集区域是有效的。
    # inter 计算了两个边界框的交集区域的宽度和高度，然后通过 .prod(2) 计算交集区域的面积。
    inter = (torch.min(a2, b2) - torch.max(a1, b1).clamp(0).prod(2))
    # (a1 - a2).prod(2) 计算每个 box1 边界框的面积。
    # (b2 - b1).prod(2) 计算每个 box2 边界框的面积。
    # inter / (...) 计算交并比（IoU），即交集面积除以两个边界框的并集面积。并集面积是两个边界框的面积之和减去交集面积。
    # + eps 在分母中添加一个小的正数，以避免除以零的情况。
    # 函数返回的是 box1 和 box2 中每一对边界框的 IoU 值，结果是一个形状为 [N, M] 的张量，其中 N 是 box1 的数量，M 是 box2 的数量。
    return inter / ((a1 - a2).prod(2) + (b2 - b1).prod(2) - inter + eps)

def clip_boxes(boxes, shape):
    """
    将边界框坐标（xyxy格式）裁剪到指定的图像形状（高度，宽度）内。

    :param boxes: 一个包含边界框坐标的张量或numpy数组，每个边界框由[x1, y1, x2, y2]表示。
    :param shape: 一个元组（高度，宽度），表示图像的尺寸。
    """
    # 检查输入的boxes是否为PyTorch张量
    if isinstance(boxes, torch.Tensor):
        # 对每个边界框的x1坐标进行裁剪，使其在[0, shape[1]]范围内
        boxes[..., 0].clamp_(0, shape[1])  # x1
        # 对每个边界框的y1坐标进行裁剪，使其在[0, shape[0]]范围内
        boxes[..., 1].clamp_(0, shape[0])  # y1
        # 对每个边界框的x2坐标进行裁剪，使其在[0, shape[1]]范围内
        boxes[..., 2].clamp_(0, shape[1])  # x2
        # 对每个边界框的y2坐标进行裁剪，使其在[0, shape[0]]范围内
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:
        # 如果boxes是numpy数组，则以分组方式执行裁剪操作
        # 对每个边界框的x1和x2坐标进行裁剪，使其在[0, shape[1]]范围内
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        # 对每个边界框的y1和y2坐标进行裁剪，使其在[0, shape[0]]范围内
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2



def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    缩放 (xyxy) 边界框从 img1_shape 到 img0_shape，可选地使用提供的 `ratio_pad`。

    :param img1_shape: 包含原始图像尺寸的元组，格式为 (height, width)。
    :param boxes: 包含边界框坐标的张量或数组，每个边界框由 [x1, y1, x2, y2] 表示。
    :param img0_shape: 目标图像尺寸的元组，格式为 (height, width)。
    :param ratio_pad: 一个可选的元组，包含缩放比例和填充大小。
    """
    if ratio_pad is None:  # 如果 ratio_pad 没有提供，则从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 计算填充大小
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # 调整边界框的 x 坐标，使其适应目标图像的尺寸
    boxes[..., [1, 3]] -= pad[1]  # 调整边界框的 y 坐标，使其适应目标图像的尺寸
    boxes[..., :4] /= gain  # 调整边界框的宽度和高度，使其适应目标图像的尺寸
    clip_boxes(boxes, img0_shape)  # 裁剪边界框，确保它们在目标图像的范围内
    return boxes  # 返回调整后的边界框

def box_label(im, box, label='', color=(128,128,128), txt_color=(255,255,255)):
    lw = 2  # 线宽设置为2
    # color = (255, 0, 0)  # 这行代码被注释掉了，如果启用，将设置边界框的颜色为红色
    w = 0  # 初始化文本宽度为0
    h = 0  # 初始化文本高度为0

    # 获取边界框的左上角和右下角坐标，并将它们转换为整数
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

    # 在图像上绘制边界框，使用cv2.rectangle函数
    cv2.rectangle(im, p1, p2, color, lw, lineType=cv2.LINE_AA)
    # cv2.imshow('org', im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # 如果提供了标签字符串，则在边界框旁边绘制标签
    if label:
        tf = max(lw - 1, 1)  # 设置文本框的线宽，至少为1
        # 使用cv2.getTextSize获取标签文本的尺寸
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        # 判断是否在边界框外部绘制文本
        outside = p1[1] - h >= 3
        # 计算文本框的右下角坐标
        p2 = (p1[0] + w, p1[1] - h - 3) if outside else (p1[1] + h + 2)
        # 在图像上绘制文本框背景
        try:
            cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
        except:
            pass
        # 在图像上绘制文本
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    # 返回绘制了边界框和标签的图像
    return im


# 定义了一个名为 yaml_load 的函数，它接受一个参数 file，该参数默认值为 "data.yaml"。
# 这意味着如果不提供 file 参数，函数将尝试加载名为 "data.yaml" 的文件。
def yaml_load(file="data.yaml"):
    # 这是一个多行字符串，用作函数的文档字符串（docstring），它描述了函数的作用。这里说明函数安全地加载并返回由 file 参数指定的 YAML 文件的内容。
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    # 使用 with 语句打开由 file 参数指定的文件。open 函数用于打开文件，并返回一个文件对象。
    # errors="ignore" 参数告诉 Python 在读取文件时忽略任何编码错误。
    # as f 将打开的文件对象赋值给变量 f。使用 with 语句的好处是，无论在文件操作过程中是否发生异常，文件都会在代码块执行完毕后自动关闭。
    with open(file, errors="ignore") as f:
        # yaml.safe_load(f) 调用 yaml 库的 safe_load 函数，该函数安全地加载由文件对象 f 指向的 YAML 文件内容。
        # safe_load 函数不会执行任何潜在的危险代码，只会解析 YAML 文件中的数据结构。
        # return 语句将加载的数据返回给函数的调用者。
        return yaml.safe_load(f)

# def post_process_yolov5(det, im, label_path="coco_label.yaml"):

# 定义了一个名为 post_process_yolov5 的函数，它接受三个参数：
# det：检测到的对象列表，每个对象是一个包含位置、置信度和类别信息的数组。
# im：原始图像，通常是一个numpy数组。
# names：一个列表，包含与类别索引相对应的类别名称。
def post_process_yolov5(det, im, names, org_img):
    # 检查 det 列表是否为空，即是否有检测到的对象。
    if len(det):
        # 如果检测列表不为空，对检测到的每个对象的边界框坐标进行缩放，
        # 以适应原始图像的大小，并将结果四舍五入到整数。
        # im.shape[:2] 获取图像的高度和宽度。
        # det[:, :4] 获取每个检测到的对象的边界框坐标。
        # scale_boxes 是一个未在代码中定义的函数，它用于根据图像的原始尺寸调整边界框坐标。
        # .round() 方法将坐标值四舍五入到最近的整数。
        det[:, :4] = scale_boxes(org_img.shape[2:], det[:, :4], im.shape).round()
        # cv2.imshow('org', im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # names = yaml_load(label_path)['names']
        colors = Colors()
        # 检
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

def resize_image_cv2(image, size):
    ih, iw, ic = image.shape
    w, h = size

    scale = min(w/w, h/h)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw, nh))
    new_image = numpy.ones((size[0], size[1], 3), dtype='uint8') * 128
    start_h = (h - nh) / 2
    start_w = (w - nw) / 2

    end_h = size[1] - start_h
    end_w = size[0] - start_w

    new_image[int(start_h):int(end_h), int(start_w):int(end_w)] = image
    return new_image, nw, nh


def data_process_cv2(frame, input_shape):
    image_data, nw, nh = resize_image_cv2(frame, (input_shape[1], input_shape[0]))
    org_data = image_data.copy()
    np_data = numpy.array(image_data, numpy.float32)
    np_data = np_data / 255
    image_data = numpy.expand_dims(numpy.transpose(np_data, (2, 0, 1)), 0)
    image_data = numpy.ascontiguousarray(image_data)
    return image_data, org_data
