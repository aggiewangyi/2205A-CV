# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('../../')
import common
import time
import tensorrt as trt
import torch
import pycuda.autoinit
from dataloader import DataLoader
import cv2
from general import non_max_suppression

import os
import random
import onnxruntime
import ast
from flask import Flask, request, render_template
import os
import json
import cv2 as cv
import base64
app = Flask(__name__, template_folder='.')
import numpy as np

#将图片转编码成二进制流
def cv2_to_base64(image):
    # image = cv2.imencode('.jpg',image)[1]
    image = jpeg.encode(image, quality=90)
    return base64.b64encode(image).decode('utf-8')


#将二进制流解码成图片
def base64_to_cv2(base64_string):

    image_data = base64.b64decode(base64_string)
    data = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return image








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
        print('inference time = ',(end-start)*1000)
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        prediction = trt_outputs[0].reshape(1, 25200, 85)
        prediction = torch.tensor(prediction)
        result = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                     multi_label=False, labels=(), max_det=300, nm=0)
        return result

trt = trt_engine()
dataloader = DataLoader(image_size=[640, 640])
url = 'http://127.0.0.1:6008/post'

from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

@app.route('/post', methods=['POST', 'GET'])
def pre():

    f_obj = request.files.get('x')
    data = f_obj.read()
    data = np.frombuffer(data, np.uint8)
    im0 = jpeg.decode(data)
    # im0 = cv2.imdecode(data, cv2.IMREAD_COLOR)


    # data = json.loads(request.data)
    # data = data['feed'][0]['x'].encode('utf8')
    # # start_time = time.time()
    # im0 = base64_to_cv2(data)
    # # cv2.imshow('res', im0)
    # # cv2.waitKey(0)

    input_image,org_image = dataloader.data_process_cv2(im0)
    # print("input_image shape:",input_image.shape)
    predict = trt.trt_inference(input_image)
    res_img = dataloader.post_process_yolov5(predict[0],org_image)
    # cv2.imshow('res',res_img)
    # cv2.waitKey(1)
    res_img = cv2_to_base64(res_img)

    res_data = {"feed": [{"img": res_img}, {"label": 'name_list'}], "fetch": ["res"]}
    return res_data

app.run(host='0.0.0.0', port=6009, debug=True)