# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
from utils.general import colorstr, check_requirements, LOGGER
from utils.general import yaml_save, check_yaml, check_dataset
from pathlib import Path
import os
from datetime import datetime
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
import numpy as np
from openvino.tools import mo
from utils.dataloaders import create_dataloader


class YOLOOpenVINOExporter:
    def __init__(self, onnx_model_path, data_yaml_path, prefix=colorstr("OpenVINO:")):
        self.onnx_model_path = onnx_model_path
        self.data_yaml_path = data_yaml_path
        self.prefix = prefix

    def export_openvino(self):
        """YOLO OpenVINO export."""
        check_requirements(f'openvino{"<=2024.0.0" if False else ">=2024.0.0"}')  # fix OpenVINO issue on ARM64
        import openvino as ov

        LOGGER.info(f"\n{self.prefix} starting export with openvino {ov.__version__}...")

        device = select_device('cpu')
        weight = self.onnx_model_path
        model = DetectMultiBackend(weights=weight, device=device)

        fq = str(self.onnx_model_path).replace(self.onnx_model_path.suffix, f"_int8_openvino_model{os.sep}")
        fq_ov = str(Path(fq) / self.onnx_model_path.with_suffix(".xml").name)
        f_onnx = self.onnx_model_path.with_suffix(".onnx")

        metadata = {
            "description": 'Ultralytics YOLO5l6 model trained on /usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml',
            "author": "zkl",
            "date": datetime.now().isoformat(),
            "version": '8.31.4',
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": 32,
            "task": "detect",
            "batch": 1,
            "imgsz": [640, 640],
            "names": model.names,
        }
        ov_model = mo.convert_model(f_onnx, model_name=self.onnx_model_path.stem, framework="onnx", compress_to_fp16=False)

        def serialize(ov_model,file):
            """Set RT info, serialize and save metadata YAML."""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            iou = 0.7
            ov_model.set_rt_info(iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in model.names.values()], ["f_ovmodel_info", "labels"])

            ov.runtime.save_model(ov_model, file, compress_to_fp16=False)
            yaml_save(Path(file).parent / "metadata.yaml", metadata)  # add metadata.yaml

        if True:
            check_requirements("nncf>=2.8.0")
            import nncf

            def transform_fn(data_item) -> np.ndarray:
                """Quantization transform function."""
                data_item: torch.Tensor = data_item[0] if isinstance(data_item, dict) else data_item
                im = data_item[0].numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
                return np.expand_dims(im, 0) if im.ndim == 3 else im

            def gen_dataloader(yaml_path, task="train", imgsz=640, workers=4):
                """Generates a DataLoader for model training or validation based on the given YAML dataset configuration."""
                data_yaml = check_yaml(yaml_path)
                data = check_dataset(data_yaml)
                dataloader = create_dataloader(
                    data[task], imgsz=imgsz, batch_size=1, stride=32, pad=0.5, single_cls=False, rect=False,
                    workers=workers
                )[0]
                return dataloader

            dataset = gen_dataloader(self.data_yaml_path)

            n = len(dataset)
            if n < 300:
                LOGGER.warning(f"{self.prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images.")

            quantization_dataset = nncf.Dataset(dataset, transform_fn)
            quantized_ov_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)
            serialize(quantized_ov_model, fq_ov)
            return fq


if __name__ == '__main__':
    model = YOLOOpenVINOExporter('D:\onnx_to_oepnvino\yolov5-master\yolov5l6.onnx','D:\onnx_to_oepnvino\coco128.yaml')
    model.export_openvino()

