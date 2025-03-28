# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('../')
import common
import time
import tensorrt as trt
import torch
import pycuda.autoinit
from dataloader import DataLoader
import cv2
from general import non_max_suppression

TRT_LOGGER = trt.Logger()


class trt_engine:
    def __init__(self, engine_path='yolov5s.engine'):
        f = open(engine_path, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        print(engine.get_binding_shape(0))
        print(engine.get_binding_shape(1))
        context = engine.create_execution_context()
        # context.active_optimization_profile =0
        # cuda_ctx=pycuda.autoinit.context
        # cuda_ctx.push()
        origin_inputshape = context.get_binding_shape(0)
        # origin_inputshape[0],origin_inputshape[l], origin_inputshape[2], origin_inputshape[3]=img_in.shape
        # 若每个输入的size不一样，可根据inputs的size更改对应的context中的size
        context.set_binding_shape(0, (origin_inputshape))
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, context)
        self.__dict__.update(locals())  # assign all variables to self

    def trt_inference(self, input_image):
        self.cuda_ctx = pycuda.autoinit.context
        self.cuda_ctx.push()
        self.inputs[0].host = input_image
        start = time.time()
        trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs,
                                          outputs=self.outputs, stream=self.stream, batch_size=1)
        end = time.time()
        # print('inference time = ',(end-start)*1000)
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        prediction = trt_outputs[0].reshape(1, 25200, 85)
        prediction = torch.tensor(prediction)
        result = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                     multi_label=False, labels=(), max_det=300, nm=0)
        return result


if __name__ == "__main__":
    trt = trt_engine()
    dataloader = DataLoader(image_size=[640, 640])
    # cap = cv2.VideoCapture(0)
    # while True:
        # 从摄像头中读取一帧图像
        # ret,frame=cap.read()
    image_dir = r"C:\Users\quant\Desktop\cs\coco128\images\train2017"
    image_list = os.listdir(image_dir)
    for image_item in image_list:
        start_time = time.time()
        image_path = os.path.join(image_dir,image_item)
        src = cv2.imread(image_path)
        print(src)
        input_image,org_image = dataloader.data_process_cv2(src)
        # print("input_image shape:",input_image.shape)
        predict = trt.trt_inference(input_image)
        print("spend time:{0}".format((time.time()-start_time)*1000))
        res_img = dataloader.post_process_yolov5(predict[0],org_image)
        cv2.imshow('res',res_img)
        cv2.waitKey(0)