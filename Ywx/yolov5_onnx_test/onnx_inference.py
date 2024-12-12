# import onnxruntime
# import numpy as np
# import onnxruntime as ort
# #import ultralytics
# model_path = r'D:\ZG6\class\yolov5\best.onnx'
#
# providers = ['CPUExecutionProvider']
# ort_session = ort.InferenceSession(model_path, providers=providers)
# # ort_session = onnxruntime.InferenceSession('test-model.onnx', providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
# batch_size = 1
# input_size = 10  # set input size to 10
#
# input_data = np.random.rand(1, 3,640,640).astype(np.float32)
# #input_data = input_data.reshape(batch_size, input_size)  # reshape input_data to (batch_size, 10)
# # batch_size = 1
# # input_size = 784  # assuming you have a model that takes 784 inputs
# #
# # input_data = np.random.rand(batch_size, input_size).astype(np.float32)
#
# output = ort_session.run(None, {'images': input_data})
#
# print(output[0].shape)
#


import os
import random
import onnxruntime
from tool import *


def onnx_load(w):
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    return session, output_names


if __name__ == '__main__':
    w = "best.onnx"
    # image dir = "images"
    image_dir = r'/class/coco/images'
    imgsz = [640, 640]
    session, output_names = onnx_load(w)
    label_names = session.get_modelmeta().custom_metadata_map['names']
    device = torch.device("cpu")
    image_list = os.listdir(image_dir)
    random.shuffle(image_list)
    for image_item in image_list:
        # while True:
        start_time = time.time()
        path = os.path.join(image_dir, image_item)
        im0 = cv2.imread(path)  # BGR
        im = data_process_cv2(im0, imgsz)

        y = session.run(output_names, {session.get_inputs()[0].name: im})
        pred = torch.from_numpy(y[0]).to(device)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
        print("spend time: {0} ms".format((time.time() - start_time) * 1000))
        # label_name = session.get_modelmeta().custom_metadata_map['names']
        label_names = eval(label_names)
        res_img = post_process_yolov5(pred[0], im0, im.shape[2:], label_names)
        cv2.imshow("res", res_img)
        cv2.waitKey(0)
