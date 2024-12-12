# -*- coding: utf-8 -*-
import os, random, onnxruntime
import albumentations
import ast
import albumentations.pytorch
import albumentations.core.composition
import torchvision
import cv2
import torch.cuda
from demo01 import *
import time
random.seed(0)

# def data_process_cv2(img, img_size):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     compose = albumentations.Compose([
#         albumentations.Resize(img_size[0], img_size[1]),
#         albumentations.Normalize(),
#         albumentations.pytorch.ToTensorV2()])
#     im = compose(image=img)
#     org_data = {'org_w': img.shape[1], 'org_h': img.shape[0]}
#     return im, org_data

# def data_process_cv2(img, imgsz):
#     # 将图像从BGR转换为RGB
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # 将图像缩放到imgsz x imgsz
#     img = cv2.resize(img, (imgsz))
#
#     # 归一化图像
#     img = img / 255.0
#
#     input_array = numpy.expand_dims(img, axis=0)
#     input_array = numpy.transpose(input_array, (0, 3, 1, 2))
#     print(input_array.shape)
#
#     return input_array, img

def onnx_load(w):
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    # 这一行使用 onnxruntime 库创建一个 InferenceSession 对象，该对象用于执行 ONNX 模型的推理。w 参数是 ONNX 模型文件的路径，providers 参数指定了推理的执行提供者。
    # session 是 onnxruntime 库中 InferenceSession 类的一个实例。
    session = onnxruntime.InferenceSession(w, providers=providers)
    output_name = [x.name for x in session.get_outputs()]
    return session, output_name

if __name__ == '__main__':
    w = "yolov5s.onnx"
    img_dir = r"J:\RGZN\CV3\yolov5-master\yolov5-master\coco128\coco128\images\train2017"
    imgsz = [640, 640]
    session, output_name = onnx_load(w)
    label_name = session.get_modelmeta().custom_metadata_map['names']
    label_name = ast.literal_eval(label_name)
    device = torch.device("cuda:0")
    img_list  = os.listdir(img_dir)
    random.shuffle(img_list)
    for img_name in img_list:
        start_time = time.time()
        path = os.path.join(img_dir, img_name)
        # try:
        img = cv2.imread(path)
        # cv2.imshow('re', img)
        # cv2.waitKey(1)
        #     img = img.astype(numpy.float32)
        # except:
        #     continue

        if img is None:
            continue
        im, org_data = data_process_cv2(img, imgsz)
        y = session.run(output_name, {session.get_inputs()[0].name:im})
        pred = torch.from_numpy(y[0]).to(device)
        pred = non_max_suppression(pred)
        print("spend time: {0} ms".format((time.time() - start_time) * 1000))
        # print(pred)
        # print(pred[0])
        res_img = post_process_yolov5(pred[0], img, label_name, im)
        cv2.imshow('res', res_img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()



