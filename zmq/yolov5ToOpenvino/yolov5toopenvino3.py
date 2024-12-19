# -*- coding: utf-8 -*-
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    print_args,
    strip_optimizer,
    yaml_save,
    check_yaml,
    check_dataset,
)
import torch
from pathlib import Path
import os
import numpy as np
from models.yolo import Detect
from models.common import DetectMultiBackend
from datetime import datetime
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_dataloader
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader

class YoloToOpenvino():
    def __init__(self, model_path_pt='yolov5s.pt', model_path_onnx='yolov5s.onnx', device='cpu', format='openvino',
                 yaml_path='coco128.yaml'):
        weights = Path(model_path_pt)
        self.weights = weights
        device = select_device(device)
        self.device = device
        model = DetectMultiBackend(weights, device=device)
        self.model = model
        self.model_path_onnx = model_path_onnx
        metadata = {
            # "description": description,
            # "author": "Ultralytics",
            # "date": datetime.now().isoformat(),
            # "version": torch.__version__,
            # "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            # "docs": "https://docs.ultralytics.com",
            # "stride": int(max(model.stride)),
            "stride": 32,
            # "task": model.task,
            # "batch": 1,
            # "imgsz": [640, 640],
            "names": model.names,
        }  # model metadata
        self.metadata = metadata
        m_batch = 1
        self.m_batch = m_batch
        self.format = format
        self.yaml_path = yaml_path

    def run(self):
        self.export_openvino(prefix=colorstr("OpenVINO:"), int8=True)

    def gen_dataloader(self,yaml_path, task="train", imgsz=640, workers=4):
        """Generates a DataLoader for model training or validation based on the given YAML dataset configuration."""
        data_yaml = check_yaml(yaml_path)
        data = check_dataset(data_yaml)
        dataloader = create_dataloader(
            data[task], imgsz=imgsz, batch_size=1, stride=32, pad=0.5, single_cls=False, rect=False, workers=workers
        )[0]
        return dataloader

    def export_openvino(self,prefix=colorstr("OpenVINO:"), int8=True):
        """YOLO OpenVINO export."""
        # check_requirements(f'openvino{"<=2024.0.0" if ARM64 else ">=2024.0.0"}')  # fix OpenVINO issue on ARM64
        import openvino as ov
        from openvino.tools import mo  # noqa

        TORCH_1_13 = True

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
        assert TORCH_1_13, f"OpenVINO export requires torch>=1.13.0 but torch=={torch.__version__} is installed"

        file = Path(self.yaml_path)

        # f_onnx = file.with_suffix(".onnx")
        f_onnx = self.model_path_onnx
        half = False
        ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework="onnx", compress_to_fp16=half)  # export

        # im = torch.zeros(m_batch, 3, *[640,640]).to(device)
        # im = im.half()
        # ov_model = ov.convert_model(
        #     model,
        #     input= [im.shape],
        #     example_input= im,
        # )

        def serialize(ov_model, file):
            """Set RT info, serialize and save metadata YAML."""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            # ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info(0.7, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            # if model.task != "classify":
            #     ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            # ov.runtime.save_model(ov_model, file, compress_to_fp16=self.args.half)
            # print(file)
            ov.runtime.save_model(ov_model, file, compress_to_fp16=False)
            # print(metadata)
            yaml_save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

        if int8:
            save_name = ''
            file = Path(self.model_path_onnx)
            fq = str(file).replace(file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / file.with_suffix(".xml").name)
            print(fq_ov)
            check_requirements("nncf>=2.8.0")
            import nncf

            def transform_fn(data_item) -> np.ndarray:
                """Quantization transform function."""
                # data_item: torch.Tensor = data_item[0] if isinstance(data_item, dict) else data_item
                # assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
                assert data_item[0].dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
                im = data_item[0].numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
                return np.expand_dims(im, 0) if im.ndim == 3 else im

            # Generate calibration data for integer quantization
            ignored_scope = None
            # model = DetectMultiBackend(weights, device=device)
            # if isinstance(model.model[-1], Detect):
            if True:
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # ignore operations
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.dfl.*",
                    ],
                    types=["Sigmoid"],
                )

            # quantized_ov_model = nncf.quantize(
            #     model=ov_model,
            #     calibration_dataset=nncf.Dataset(get_int8_calibration_dataloader(prefix), transform_fn),
            #     preset=nncf.QuantizationPreset.MIXED,
            #     ignored_scope=ignored_scope,
            # )

            ds = self.gen_dataloader(Path(self.yaml_path))
            # ds = get_int8_calibration_dataloader(prefix)
            quantization_dataset = nncf.Dataset(ds, transform_fn)
            quantized_ov_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)

            serialize(quantized_ov_model, fq_ov)
            return fq, None

        # f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        # f_ov = str(Path(f) / self.file.with_suffix(".xml").name)
        #
        # serialize(ov_model, f_ov)
        # return f, None


if __name__ == '__main__':
    to_openvino = YoloToOpenvino(model_path_pt='yolov5l6.pt', model_path_onnx='yolov5l6.onnx', device='cpu', format='openvino',yaml_path='coco128.yaml')
    to_openvino.run()
    # export_openvino(prefix=colorstr("OpenVINO:"), int8=True)
