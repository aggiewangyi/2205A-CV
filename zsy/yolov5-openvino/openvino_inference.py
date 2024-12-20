import cv2
import os
import time
import yaml
import re
import numpy as np
from openvino.runtime import Core, Model, CompiledModel


class OpenVINO:
    def __init__(self, model_path, confidence_thres=0.25, iou_thres=0.5):
        # 初始化OpenVINO核心对象
        core = Core()
        # 模型文件路径
        model_xml = os.path.join(model_path, "yolov5l6.xml")
        model_bin = os.path.join(model_path, "yolov5l6.bin")
        # 加载模型
        model = core.read_model(model=model_xml, weights=model_bin)
        device = "CPU"
        # 编译模型用于特定设备
        compiled_model = core.compile_model(model, device)
        # 执行推理
        self.infer_request_handle = compiled_model.create_infer_request()
        self.input_tensor = compiled_model.input(0)
        self.output_tensor = compiled_model.output(0)

        self.input_width = 640
        self.input_height = 640

        self.classes = self.yaml_load("coco128.yaml")["names"]
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
        image_data = input_image[..., ::-1].transpose((0, 3, 1, 2))

        # start_time = time.time()
        image_data = cv2.normalize(image_data, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # spend_time = (time.time() - start_time) * 1000
        # print("norm time: ", spend_time)

        # image_data = np.ascontiguousarray(image_data)

        return image_data


    def yaml_load(self, file="data.yaml", append_filename=False):
        # assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()  # string
            # Remove special characters
            if not s.isprintable():
                s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            # Add YAML filename to dict and return
            data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
            if append_filename:
                data["yaml_file"] = str(file)
            return data


    def draw_detections(self, img, box, score, class_id):

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)



    def enhance_postporcess(self, output):
        # start_time = time.time()
        outputs = np.squeeze(output[0])
        score_matrix = outputs[:, 5:]*outputs[:, 4:5]
        boxes = outputs[:, :4]
        #获取每行最大分值
        row_max_values = np.amax(score_matrix, axis=1)
        max_values_index = np.argmax(score_matrix, axis=1)
        indices = np.where(row_max_values > self.confidence_thres)
        class_ids = max_values_index[indices]
        obj_boxes = boxes[indices, :][0]
        obj_scores = row_max_values[indices]
        obj_boxes[:, 0] = (obj_boxes[:, 0] - obj_boxes[:, 2]/2)
        obj_boxes[:, 1] = (obj_boxes[:, 1] - obj_boxes[:, 3]/2)
        obj_boxes = obj_boxes.astype(int)

        res_indices = cv2.dnn.NMSBoxes(obj_boxes.tolist(), obj_scores.tolist(), self.confidence_thres, self.iou_thres)
        res_boxes = obj_boxes[res_indices].tolist()
        res_scores = obj_scores[res_indices].tolist()
        res_class = class_ids[res_indices].tolist()

        # spend_time = (time.time() - start_time) * 1000
        # print("post spend time:", spend_time)

        for index in range(len(res_boxes)):
            self.draw_detections(self.img, res_boxes[index], res_scores[index], res_class[index])
        cv2.imshow("src", self.img)
        cv2.waitKey(0)
        return


if __name__ == "__main__":

    model_path = "yolov5_openvino_model"

    # model_path = "fish_best_int8_openvino_model"
    openvino = OpenVINO(model_path)

    image_dir = r"E:\coco\images\val2017"

    # image_dir = r"E:\fish\finish\reapair\huizhong\picture"
    images = os.listdir(image_dir)

    for image_item in images:
        # print("image_item:", image_item)
        image_path = os.path.join(image_dir, image_item)
        src = cv2.imread(image_path)

        # start_time = time.time()
        input_data = openvino.preprocess(src)
        # spend_time = (time.time() - start_time)*1000
        # print("preprocess time: ", spend_time)

        start_time = time.time()
        numpy_output = openvino.run(input_data)
        spend_time = (time.time() - start_time) * 1000
        print("inference spend time:", spend_time)

        # start_time = time.time()
        openvino.enhance_postporcess(numpy_output)
        # spend_time = (time.time() - start_time)
        # print("post spend time:", spend_time)
        # print(f"time took {spend_time * 1000:.4f} ms.")

        # fps = 1 / spend_time
        # print("inference fps:", int(fps))