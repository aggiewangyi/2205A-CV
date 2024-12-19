import cv2
import os
import time
import yaml
import re
import numpy as np
from openvino.runtime import Core, Model, CompiledModel


class OpenVINO:
    def __init__(self, model_path, confidence_thres=0.25, iou_thres=0.5):
        core = Core()
        model_xml = os.path.join(model_path, r"D:\ZG6\class\yolov5\yolov5n_openvino_model\yolov5n.xml")
        model_bin = os.path.join(model_path, r"D:\ZG6\class\yolov5\yolov5n_openvino_model\yolov5n.bin")
        model = core.read_model(model=model_xml, weights=model_bin)
        device = "CPU"
        compiled_model = core.compile_model(model, device)
        self.infer_request_handle = compiled_model.create_infer_request()
        self.input_tensor = compiled_model.input(0)
        self.output_tensor = compiled_model.output(0)
        self.input_width = 640
        self.input_height = 640
        data = self.yaml_load(r'D:\ZG6\class\yolov5\yolov5n_openvino_model\yolov5n.yaml')
        self.classes = data["names"]
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

    def run(self, input_image):
        self.infer_request_handle.infer({self.input_tensor: input_image})
        # 获取输出数据
        output_data = self.infer_request_handle.get_tensor(self.output_tensor)
        numpy_output = np.copy(output_data.data)
        return numpy_output

    def preprocess(self, input_image):
        self.img = input_image
        self.img_height, self.img_width = self.img.shape[:2]
        # Resize the image to match the input shape
        self.img = cv2.resize(self.img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        input_image = np.stack([self.img])
        image_data = input_image[..., ::- 1].transpose((0, 3, 1, 2))
        image_data = cv2.normalize(image_data, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return image_data

    def yaml_load(self, file=r"D:\ZG6\class\yolov5\yolov5n_openvino_model\yolov5n.yaml", append_filename=False):
        # assert Path (file) .suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load ()"
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()  # string
            # if not s.isprintable():
            #     s = re.sub(r"[\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load () may return None for empty files)
            if append_filename:
                data["yaml_file"] = str(file)
            return data

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score :.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def enhance_postporcess(self, output):
        outputs = np.squeeze(output[0])
        score_matrix = outputs[:, 5:] * outputs[:, 4:5]
        boxes = outputs[:, :4]
        row_max_values = np.amax(score_matrix, axis=1)
        max_values_index = np.argmax(score_matrix, axis=1)
        indices = np.where(row_max_values > self.confidence_thres)
        class_ids = max_values_index[indices]
        obj_boxes = boxes[indices, :][0]
        obj_scores = row_max_values[indices]
        obj_boxes[:, 0] = (obj_boxes[:, 0] - obj_boxes[:, 2] / 2)
        obj_boxes[:, 1] = (obj_boxes[:, 1] - obj_boxes[:, 3] / 2)
        obj_boxes = obj_boxes.astype(int)
        res_indices = cv2.dnn.NMSBoxes(obj_boxes.tolist(), obj_scores.tolist(), self.confidence_thres, self.iou_thres)
        res_boxes = obj_boxes[res_indices].tolist()
        res_scores = obj_scores[res_indices].tolist()
        res_class = class_ids[res_indices].tolist()
        for index in range(len(res_boxes)):
            self.draw_detections(self.img, res_boxes[index], res_scores[index], res_class[index])
        cv2.imshow("src", self.img)
        cv2.waitKey(0)
        return


if __name__ == '__main__':

    model_path = r"D:\ZG6\class\yolov5\yolov5n_openvino_model"
    openvino = OpenVINO(model_path)
    image_dir = r"D:\ZG6\class\coco\images\val"
    images = os.listdir(image_dir)
    for image_item in images:
        image_path = os.path.join(image_dir, image_item)
        src = cv2.imread(image_path)
        input_data = openvino.preprocess(src)
        start_time = time.time()
        numpy_output = openvino.run(input_data)
        start_time = time.time()
        numpy_output = openvino.run(input_data)
        spend_time = (time.time() - start_time) * 1000
        print("inference spend time:", spend_time)
        openvino.enhance_postporcess(numpy_output)
