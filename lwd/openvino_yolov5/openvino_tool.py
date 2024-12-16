from openvino.runtime import Core
from openvino.runtime import serialize

ie = Core()
onnx_model_path = r"yolov5s.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
serialize(model=model_onnx, xml_path="yolov5s.xml", bin_path="yolov5s.bin", version="UNSPECIFIED")