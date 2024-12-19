import os
import torch
from datetime import datetime
import numpy as np
from utils.general import (check_requirements, LOGGER, yaml_save, colorstr)
from pathlib import Path
from utils.dataloaders import LoadImagesAndLabels
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
import openvino.runtime as ov  # noqa
from openvino.tools import mo  # noqa
import nncf


def transform_fn(data_item):
    """Quantization transform function."""
    assert (
            data_item[0].dtype == torch.uint8
    ), "Input image must be uint8 for the quantization preprocessing"
    im = data_item[0].numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
    return np.expand_dims(im, 0) if im.ndim == 3 else im


class OpenvinoExport():
    def __init__(self, weight_path, data=r"E:\PycharmProject\yolov5-master\data\coco128.yaml"):
        check_requirements("openvino-dev>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        check_requirements("nncf>=2.5.0")
        self.prefix = colorstr("OpenVINO:")
        LOGGER.info(f"\n{self.prefix} starting export with openvino {ov.__version__}...")
        file_name = os.path.basename(weight_path)
        self.pretty_name = os.path.splitext(file_name)[0]
        file = Path(file_name)
        f = str(file).replace(file.suffix, f"_openvino_model{os.sep}")
        self.fq = str(file).replace(file.suffix, f"_int8_openvino_model{os.sep}")
        self.f_onnx = file.with_suffix(".onnx")
        self.f_ov = str(Path(f) / file.with_suffix(".xml").name)
        self.fq_ov = str(Path(self.fq) / file.with_suffix(".xml").name)
        device = select_device("cpu")
        self.model = DetectMultiBackend(weight_path, device=device, data=data)


    def serialize(self, ov_model, file):
        """Set RT info, serialize and save metadata YAML."""
        # ov_model.set_rt_info("YOLOv8", ["model_info", "model_type"])
        ov_model.set_rt_info("YOLOv5", ["model_info", "model_type"])
        ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
        ov_model.set_rt_info(114, ["model_info", "pad_value"])
        ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
        iou_threshold = 0.7
        ov_model.set_rt_info(iou_threshold, ["model_info", "iou_threshold"])
        ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
        ov.serialize(ov_model, file)  # save



    def quantize(self, image_path=r"E:\PycharmProject\yolov5-master\datasets\coco128"):
        ov_model = mo.convert_model(
            self.f_onnx, model_name=self.pretty_name, framework="onnx", compress_to_fp16=False
        )  # export
        dataset = LoadImagesAndLabels(image_path)
        n = len(dataset)
        if n < 300:
            LOGGER.warning(f"{self.prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images.")
        quantization_dataset = nncf.Dataset(dataset, transform_fn)
        # ignored_scope = nncf.IgnoredScope(types=["Multiply", "Subtract", "Sigmoid"])  # ignore operation
        ignored_scope = nncf.IgnoredScope(types=["Multiply", "Sigmoid"])  # ignore operation
        quantized_ov_model = nncf.quantize(
            ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED, ignored_scope=ignored_scope
           )
        self.serialize(quantized_ov_model, self.fq_ov)
        return self.fq



if __name__ == "__main__":
    weight_path = "yolov5l6.pt"
    data = r"E:\PycharmProject\yolov5-master\data\coco128.yaml"
    image_path = r"E:\PycharmProject\yolov5-master\datasets\coco128"
    openvino_export = OpenvinoExport(weight_path, data)
    openvino_export.quantize(image_path)

