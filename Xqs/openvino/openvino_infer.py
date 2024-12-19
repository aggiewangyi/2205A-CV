import cv2
import os
import time
import yaml
import re
import numpy as np
from openvino.runtime import Core,Model,CompiledModel

class openvino:
    def __init__(self,model_path,conf_thres=0.25,iou_thres=0.5):
        core = Core()
        model_xml = os.path.join(model_path,"yolov5.xml")
        model_bin = os.path.join(model_path,"yolov5.bin")
        model = core.read_model(model=model_xml,weights=model_bin)
        device = "CPU"
        compiled_model = core.compile_model(model,device,)
        self.infer_request = compiled_model.create_infer_request()
        self.input = compiled_model.input(0)
        self.output = compiled_model.output(0)
        self.input_size = 640
        self.classes = self.yaml_load('coco128.yaml')['names']
        self.color_palette = np.random.uniform(0,255,(len(self.classes),3))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self,img):
        self.infer_request.infer({self.input:img})
        output_data = self.infer_request.get_output_tensor()
        numpy_output = np.copy(output_data.data)
        return numpy_output

    def preprocess(self,input_img):
        self.img = input_img
        self.img_height,self.img_width = self.img.shape[:2]
        self.img = cv2.resize(self.img,(self.input_size,self.input_size),interpolation=cv2.INTER_NEAREST)
        input_img = np.stack([self.img])
        image_data = input_img[...,::-1].transpose((0,3,1,2))
        new_image_data = cv2.normalize(image_data,None,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        return new_image_data
    def yaml_load(self,file='data.yaml'):
        # Single-line safe yaml loading
        with open(file, errors='ignore') as f:
            return yaml.safe_load(f)

    def draw_detection(self,img,box,score,classid):
        x,y,w,h = box
        color = self.color_palette[classid]
        cv2.rectangle(img,pt1=(int(x),int(y)),pt2=(int(x+w),int(y+h)),color=color,thickness=2)
        label = f'{self.classes[classid]} : {score:.2f}'
        (label_width,label_height),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        label_x = x
        label_y = y-10 if y-10 > label_height else y+10
        cv2.putText(img,label,(label_x,label_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,0,0],1)
    def enhance_postprocess(self,output):
        outputs = np.squeeze(output[0])
        score_mat = outputs[:,4:5] * outputs[:,5:]
        boxes = outputs[:,:4]
        row_max = np.amax(score_mat,axis=1)
        max_ind = np.argmax(score_mat,axis=1)
        indices = np.where(row_max >= self.conf_thres)
        class_ids = max_ind[indices]
        obj_box = boxes[indices,:][0]
        obj_score = row_max[indices]
        obj_box[:,0] = (obj_box[:,0] - obj_box[:,2]/2)
        obj_box[:,1] = (obj_box[:,1] - obj_box[:,3]/2)
        obj_box = obj_box.astype(int)

        res_ind = cv2.dnn.NMSBoxes(obj_box.tolist(),obj_score.tolist(),self.conf_thres,self.iou_thres)
        res_boxes = obj_box[res_ind].tolist()
        res_score = obj_score[res_ind].tolist()
        res_class = class_ids[res_ind].tolist()

        for index in range(len(res_boxes)):
            self.draw_detection(self.img,res_boxes[index],res_score[index],res_class[index])
        cv2.imshow('result',self.img)
        cv2.waitKey(0)
        return

if __name__ == "__main__":
    model_path = r'C:\Users\26746\Desktop\c5\openvino_infer_simple'
    ov = openvino(model_path,0.5,0.7)
    image_path = os.path.join(model_path,'images')
    image_list = os.listdir(image_path)
    for image_name in image_list:
        img_path = os.path.join(image_path,image_name)
        src = cv2.imread(img_path)
        input_data = ov.preprocess(src)

        start_time = time.time()
        numpy_output = ov.run(input_data)
        spend_time = (time.time() - start_time) * 1000
        print('openvino inference time cost: ',spend_time)
        ov.enhance_postprocess(numpy_output)
