import sys
import os
import common
import time
import torch
import cv2
import pycuda
import tensorrt as trt
from tool import *
trt_logger = trt.Logger()

class trt_engine:
    def __init__(self,engine_path='yolov5s.engine'):
        f = open(engine_path,'rb')
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(f.read())
        print(engine.get_binding_shape(0))
        print(engine.get_binding_shape(1))

        context = engine.create_execution_context()

        origin_input_shape = context.get_binding_shape(0)
        context.set_binding_shape(0,(origin_input_shape))
        inputs,outpus,bindings,stream = common.allocate_buffers(engine,context)
        self.__dict__.update(locals())
    def trt_inference(self,input_image):
        self.cuda_ct = pycuda.autoinit.context
        self.cuda_ct.push()
        self.inputs[0].host = input_image
        start_time = time.time()
        trt_outputs = common.do_inference(self.context,bindings=self.bindings,
                                          inputs=self.inputs,outputs=self.outpus,
                                          stream=self.stream,batch_size=1)
        print('tensorrt inference time = ',1000 * (time.time()-start_time))

        if self.cuda_ct:
            self.cuda_ct.pop()
        pred = trt_outputs[0].reshape(1,25200,85)
        pred = torch.tensor(pred)
        result = non_max_suppression(pred)
        return result

if __name__ == '__main__':
    trt = trt_engine()
    img_dir = 'images'
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        start_time = time.time()
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        input_img,ori_img = DataLoder.data_process(img,[640,640])
        print('input img shape: ',input_img.shape)
        prediction = trt.trt_inference(input_img)
        res_img = post_process_yolov5_engine(prediction[0],img,input_img)
        cv2.imshow('res',res_img)
        cv2.waitKey(0)