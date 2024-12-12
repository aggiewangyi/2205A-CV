import numpy as np
import torch
from PIL import Image
import copy
import cv2
import math
import yaml
import torchvision
import time

def box_iou(box1, box2, eps=1e-7):  # 计算IOU
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(1).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # .clamp(0) 方法将张量中的所有元素限制在 [0, +inf] 范围内。这意味着任何负值都会替换为 0。因为在计算交集面积时，我们不希望宽度或高度为负值。
    # .prod(2) 方法在指定的维度（这里是维度 2，即最后一个维度，对应于宽度和高度）上计算乘积。这给出了交集区域的面积。
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,         # 模型输出的检测结果，通常是一个包含边界框坐标、置信度和类别预测的张量。
        conf_thres=0.25,    # 置信度阈值，低于此阈值的检测框将被忽略。
        iou_thres=0.45,     # 交并比（IoU）阈值，用于确定哪些检测框相互重叠并应被抑制。
        classes=None,       # 可选参数，指定要保留的类别索引列表。
        agnostic=False,     # 如果为True，则执行类别不可知的NMS（即不考虑类别，仅根据IoU抑制重叠框）。
        multi_label=False,  # 如果为True，则允许每个检测框有多个标签（即，每个框可以有多个类别的高置信度预测）。
        labels=(),          # 可选参数，用于自动标注的先验标签列表。
        max_det=300,        # 每张图像保留的最大检测框数量。
        nm=0,               # 锚框的数量
):
    # 第一行断言确保0<=conf_thres（置信度阈值）<=1。否则抛出一个AssertionError异常，并显示提供的错误消息，指出无效的置信度阈值。
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    # 第二行断言确保iou_thres（交并比阈值，IoU）的值也在0到1之间。否则将抛出异常并显示错误消息。
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    """
    检查prediction变量是否是列表（list）或元组（tuple）类型。模型的输出可能是一个包含多个元素的元组，其中第一个元素是推理输出（inference_out），
    第二个元素是损失输出（loss_out）。如果prediction是列表或元组，代码将选择第一个元素作为prediction的新值，即只保留推理输出。
    这是因为在某些情况下，你可能只对模型的推理结果感兴趣，而不关心损失值。
    """
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device # prediction.device 获取了prediction张量所在的设备（如CPU、CUDA设备等）。
    # 然后，通过检查设备类型字符串中是否包含"mps"来判断是否在使用Apple的MPS。MPS是Apple为Metal图形API提供的高性能计算加速库。
    mps = "mps" in device.type  # Apple MPS

    if mps:  # 如果检测到在使用MPS设备
        # 将prediction张量从MPS设备转移到CPU。因为某些操作（如非极大值抑制，NMS）在MPS设备上可能不受支持或性能不佳。
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # 批处理大小batch size
    # 预测张量的第三个维度（通常包含类别概率、边界框坐标等信息）减去某个值（nm加上5）。这里的nm表示锚框的数量，5代表边界框的四个坐标加上一个置信度值。
    nc = prediction.shape[2] - nm - 5  # 类别数number of classes
    # 比较预测张量中每个元素的置信度（假设位于每个元素的第五个位置，即..., 4）与置信度阈值conf_thres来筛选候选目标。
    xc = prediction[..., 4] > conf_thres  # 结果xc是一个布尔张量，其中True表示对应元素的置信度超过了阈值，因此被视为候选目标。

    # Settings
    # min_wh = 2 # 过滤掉宽度和高度小于指定像素数的边界框。这有助于去除太小的、可能是噪声的边界框。
    max_wh = 7680 # 最大边界框宽度和高度（以像素为单位）。这用于过滤掉过大的边界框，这些边界框可能超出了图像的边界或表示不合理的对象。
    max_nms = 30000  # 在调用torchvision.ops.nms()（非极大值抑制，NMS）之前允许的最大边界框数量。防止NMS操作因处理过多边界框而变得过于缓慢。
    # time_limit后处理步骤允许的最大时间（秒）。在实时应用中控制处理延迟。
    time_limit = 0.5 + 0.05 * bs # 时间限制包括所有后处理步骤，如NMS、边界框过滤等。bs（批次大小）用于动态调整时间限制，以适应不同数量的图像。
    redundant = True # 冗余检测。如果设为True，对每个对象检测多个边界框(即使非常相似)。保留更多的边界框，但也可能增加后处理的复杂性和计算成本。
    multi_label &= nc > 1 # 布尔值，用于指示是否允许多标签（即每个边界框可以属于多个类别）。
    # 这个设置是通过检查类别数nc是否大于1来动态确定的。如果nc > 1，则multi_label被设置为True，表示每个边界框可以包含多个类别的标签。
    merge = False  # 是否使用合并NMS（merge-NMS）。如果设置为False，则不使用合并NMS技术。
    # 合并NMS是一种特殊的NMS变体，旨在进一步减少边界框的数量，同时保持较高的检测准确性。然而，它可能会增加一些额外的计算成本。

    t = time.time() # 记录起始时间
    mi = 5 + nc  # 定义了一个掩码起始索引mi，它基于类别数nc来计算。定义类别索引
    # 初始化空列表output，包含bs（批次大小）个形状为(0, 6 + nm)的全零张量。6+nm表示边界框坐标、置信度和类别标签（加上一些额外信息如锚框数量nm）。
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # 遍历了prediction张量中的每个图像预测结果。xi是图像的索引，x是对应图像的预测。
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 注释掉的代码用于根据边界框的宽度和高度来过滤预测结果。边界框的宽度或高度小于min_wh或大于max_wh，则置信度设为0，后续会忽略这些边界框。
        x = x[xc[xi]]  # 使用布尔张量xc（基于置信度阈值筛选的候选）来筛选当前图像的预测结果。只有置信度高于阈值的预测才会被保留下来。

        if labels and len(labels[xi]): # 如果提供了标签（labels）并且当前图像有标签，则执行自动标注的逻辑。
            # 这部分代码会创建一个新的张量v，其中包含标签信息（边界框、置信度和类别标签）。
            lb = labels[xi]  # 类别标签
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0) # 与当前的预测结果x合并。

        # 如果经过筛选后，当前图像没有任何预测结果（即x的形状的第一个维度为0），则跳过当前循环迭代，直接处理下一个图像。
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # 目标存在置信度（obj_conf）与类别置信度（cls_conf）相乘，得到每个预测框的最终置信度（conf）。

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # 这行代码将边界框从中心点+宽高的格式（xywh）转换为左上角和右下角的坐标格式（xyxy）。
        mask = x[:, mi:] # 类别

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label: # 如果模型是多标签配置（即一个对象可以同时属于多个类别）
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T # 这里会找出所有类别置信度高于阈值conf_thres的预测
            # 构建一个新的检测矩阵，包含边界框、置信度最高的类别及其置信度、掩码信息（如果有）。
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # 如果模型是单标签配置（即一个对象只能属于一个类别）
            conf, j = x[:, 5:mi].max(1, keepdim=True) # 这里会找出置信度最高的类别及其置信度
            # 构建一个新的检测矩阵，包含边界框、最高置信度及其对应的类别、掩码信息（如果有）。然后，找出所有类别置信度高于阈值conf_thres的预测
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 注释代码用于检查x（模型输出）是否包含非有限数（如NaN或Inf），并过滤掉包含这些值的行。
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 这里检查x的第一维大小（即边界框的数量），如果没有边界框，则跳过当前循环迭代（这通常是在一个更大的循环中，比如遍历图像的批次）。
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 根据置信度（x[:, 4]）对边界框进行降序排序，并保留前max_nms个框。

        # 这里首先计算一个类偏移量c（在类别不可知（agnostic）的情况下为0，否则为max_wh，但这里的max_wh似乎没有用到，可能是一个遗留的变量或错误）。
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 然后，将边界框和置信度分离出来，并应用NMS来去除重叠度高于iou_thres的框。最后，限制检测数量不超过max_det。
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # 如果启用了合并（merge）且框的数量在合理范围内
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            # 则使用加权平均值来合并重叠框。这包括计算IOU矩阵、权重、更新合并后的框坐标，以及在需要时保留冗余框（redundant）。
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i] # 将处理后的边界框和相关信息存储到output字典中，xi是当前的索引或键。
        if mps: # 如果使用MPS设备。
            output[xi] = output[xi].to(device) # 则将输出转移到指定设备（如GPU）。
        if (time.time() - t) > time_limit: # 如果当前处理时间超过了预设的时间限制，则跳出循环。这通常用于实时应用或批量处理中的性能优化。
            break
    return output

def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def resize_image_cv2(image, size):  # 图像（image）和一个目标尺寸（size）
    ih, iw, ic = image.shape  # 这里 ih 是图像的高度，iw 是图像的宽度，ic 是图像的通道数（对于彩色图像通常是3）。
    w, h = size  # size 是一个包含目标宽度和高度的元组。
    scale = min(w / iw, h / ih)  # scale 是目标尺寸和原始图像尺寸的缩放比例，以确保调整大小后的图像能够完全适应目标尺寸，同时保持图像的宽高比。
    nw = int(iw * scale)  # nw是调整大小后的图像宽度。
    nh = int(ih * scale)  # nh是调整大小后的图像高度。
    image = cv2.resize(image, (nw, nh))  # 将图像调整为新的尺寸。
    # 创建了一个与目标尺寸相同的新图像，所有像素值初始化为128（对于灰度值来说是一个中间值，对于彩色图像来说，每个通道都是128，结果是一个灰色图像）。
    new_image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 128
    # 这些值用于计算在新图像中放置调整大小后的图像的位置，以确保它在目标尺寸内居中。
    start_h = (h - nh) / 2
    start_w = (w - nw) / 2
    end_h = size[1] - start_h
    end_w = size[0] - start_w
    new_image[int(start_h):int(end_h), int(start_w):int(end_w)] = image
    return new_image, nw, nh


def preprocess_input(image):
    image /= 255.0
    return image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def box_label(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):  # 在图像上绘制矩形框，并添加文本标签。
    lw = 2  # 定义矩形框的线宽。
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  # (x1,y1),(x2,y2)
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)  # 绘制矩形框
    if label:  # 如果有标签

        tf = max(lw - 1, 1)  # 计算文本标签的字体粗细 tf，确保它至少为 1。
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # 计算文本标签的宽度和高度。
        outside = p1[1] - h >= 3  # 判断标签是否应该绘制在矩形框的上方（outside 为 True）还是下方（outside 为 False）。
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3  # 计算文本标签的右下角坐标 p2。
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # 绘制一个背景矩形（用于突出显示文本标签），thickness = -1来填充矩形。
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)  # 绘制文本标签
        return im


def yaml_load(file="coco128.yaml"):
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)
    # return yaml.safe_load(f)这行代码使用yaml模块的safe_load函数来读取并解析文件对象f中的YAML内容。safe_load函数用于安全地加载YAML文件
    # 它只允许加载Python的基本数据类型，从而避免了执行YAML文件中的任意函数或加载不受信任的数据结构可能带来的安全风险。最后，函数返回解析后的内容


def clip_boxes(boxes, shape):  # 将一组边界框（boxes）的坐标限制在给定图像尺寸（shape）的范围内。
    if isinstance(boxes, torch.Tensor):  # 如果 boxes 是一个 PyTorch 张量（torch.Tensor）
        boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # # 如果 boxes 不是一个 PyTorch 张量（torch.Tensor）
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])  # x1,x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])  # y1,y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # img1_shape（目标图像的尺寸，通常是元组(height, width)），
    # boxes（边界框的坐标，通常是形状为[N, 4]的数组，其中N是边界框的数量，每个边界框由[x1, y1, x2, y2]表示），
    # img0_shape（源图像的尺寸，同样是(height, width)的元组），
    # 以及可选的ratio_pad（一个包含缩放比例和填充量的元组）。
    if ratio_pad is None:  # 如果 ratio_pad 为 None
        # 根据 img0_shape 和 img1_shape 计算缩放比例 gain（目标尺寸相对于源尺寸的比例，取宽度和高度的最小比例以保证图像完全适应目标尺寸），
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # 并计算填充量 pad（在宽度和高度上分别需要添加的填充量，以保持图像的中心位置不变）。
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:  # 如果提供了 ratio_pad，直接使用它提供的缩放比例和填充量。
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x 方向padding
    boxes[..., [1, 3]] -= pad[1]  # y 方向padding
    boxes[..., :4] /= gain  # 将边界框的尺寸除以 gain 以适应目标图像的尺寸。
    clip_boxes(boxes, img0_shape)  # 将边界框的坐标裁剪到 [0, img0_shape[1]]（y）和 [0, img0_shape[0]]（x）的范围内。
    return boxes


class Colors:  # 定义了一个名为Colors的类。
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):  # 这是类的初始化方法（构造器），在创建类的实例时会自动调用。
        # 在初始化方法中，定义了一个名为hexs的元组，其中包含了20个十六进制颜色代码字符串。
        hexs = ("FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7")
        # 通过列表推导式，将hexs元组中的每个十六进制颜色代码字符串转换为RGB元组，并将这些RGB元组存储在self.palette列表中。
        # self.hex2rgb方法用于执行此转换，而f'#{c}'则是为了确保颜色代码以  # 开头，符合十六进制颜色代码的格式。
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)  # 计算并存储调色板中颜色的总数。

    # 特殊方法__call__，使Colors的实例能像函数一样调用。两个参数：i（颜色索引）和bgr（布尔值，指定返回的颜色是否BGR格式，默认False，即RGB）。
    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod  # 这是一个装饰器，用于声明hex2rgb方法是一个静态方法。静态方法不依赖于类的实例，因此不需要self或cls参数。
    def hex2rgb(h):  # 定义了hex2rgb静态方法，它接受一个十六进制颜色代码字符串h作为参数。
        # 通过生成器表达式，将十六进制颜色代码的每个通道（红、绿、蓝）转换为整数，并将这些整数作为一个元组返回。
        # 这里，h[1 + i:1 + i + 2]用于提取每个两位数的十六进制通道值，int(..., 16)则将这些值从十六进制转换为十进制整数。
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class DataLoader():
    def __init__(self, image_size=[473, 473]):
        self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                       (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                       (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0),
                       (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        self.input_shape = image_size

    def data_process(self, image):
        image = cvtColor(image)
        # -------------------------------------------------------#
        #   对输入对象进行一个备份，后面用于绘图
        # -------------------------------------------------------#
        self.orininal_h = np.array(image).shape[0]
        self.orininal_w = np.array(image).shape[1]
        # -------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # -------------------------------------------------------#
        image_data, self.nw, self.nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # -------------------------------------------------------#
        #   添加上batch_size维度
        # -------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        image_data = np.ascontiguousarray(image_data)
        return image_data

    def data_process_cv2(self, frame):
        self.orininal_h = frame.shape[0]
        self.orininal_w = frame.shape[1]
        # 将输入的frame调整大小到(input_shape[1],input_shape[0])，即模型的宽度和高度。返回调整大小后的图像image_data，新的宽度nw和高度nh
        image_data, nw, nh = resize_image_cv2(frame, (self.input_shape[1], self.input_shape[0]))
        org_data = image_data.copy()  # 复制原始图像
        # 通过np.transpose(np_data, (2, 0, 1))将图像的通道顺序从(高, 宽, 通道)转换为(通道, 高, 宽)，
        # 然后np.expand_dims在第一个维度（通道维度之前）增加一个维度，使得最终的形状变为(1, 通道, 高, 宽)，这通常表示一个批次的单个图像。
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,np.float32)), (2, 0, 1)), 0)  # (1, 通道, 高, 宽)
        image_data = np.ascontiguousarray(image_data)  # 使image_data数组在内存中是连续存储
        return image_data, org_data

    def post_process(self, pr):
        nh = self.nh
        nw = self.nw
        input_shape = self.input_shape
        orininal_h = self.orininal_h
        orininal_w = self.orininal_w

        # -------------------------------------------------------#
        #   将灰条部分截掉
        # -------------------------------------------------------#
        pr = pr[int((input_shape[0] - nh) // 2):int((input_shape[0] - nh) // 2 + nh),
             int((input_shape[0] - nw) // 2):int((input_shape[0] - nw) // 2 + nw)]
        # -------------------------------------------------------#
        #   进行图片的resize
        # -------------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # -------------------------------------------------------#
        #   取出每一个像素点的种类
        # -------------------------------------------------------#
        pr = pr.argmax(axis=1)

        pr_1d = pr.reshape(-1)
        counts = np.bincount(pr_1d)

        label_index = np.argsort(-counts)

        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

        return seg_img, label_index

    def post_process_yolov5(self, det, im, label_path="coco128.yaml"):
        # det：一个包含检测结果的二维数组，其中每行代表一个检测到的目标，包含边界框坐标、置信度和类别索引。
        # im：原始图像，通常是一个三维数组（高度、宽度、通道数）。
        # label_path：包含类别名称的YAML文件的路径，默认值为"coco_label.yaml"。
        if len(det):  # 如果有检测结果
            # 自定义scale_boxes函数将边界框坐标从缩放后的图像大小img_size缩放到原始图像大小im.shape[:2](高度和宽度),.round()方法四舍五入成整数。
            det[:, :4] = scale_boxes(im.shape[:2], det[:, :4], im.shape).round()
            names = yaml_load(label_path)['names']  # 使用自定义yaml_load函数来加载YAML文件中定义的类别名称。
            colors = Colors()  # 实例化一个颜色类
            for *xyxy, conf, cls in reversed(det):  # # 使用解包和reversed函数来遍历检测结果数组。
                # *xyxy 捕获边界框的四个坐标（x_min, y_min, x_max, y_max）,conf捕获置信度,cls捕获类别索引。
                # reversed()函数接受一个可迭代对象（如列表、元组、字符串等）作为参数，并返回一个迭代器。使用for循环遍历这个迭代器，能以逆序访问元素。
                c = int(cls)  # 类别索引
                label = names[c]  # 从字典取出对应类别名称
                box_label(im, xyxy, label, color=colors(c, True))  # 为每个检测到的目标绘制边界框和标签。
        return im

    def save_data(self, pred):
        pred_np = pred.cpu().numpy()
        with open("./img_out/trt_output.txt", 'w') as outfile:
            np.savetxt(outfile, pred_np[0][8], fmt='%f', delimiter='')
