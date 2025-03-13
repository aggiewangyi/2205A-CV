from ultralytics import YOLO
# 从头开始构建新模型
model = YOLO("yolov8n.yaml")
# 加载预训练模型（建议用于训练）
model = YOLO("yolov8n.pt")
## 训练模型
model.train(data="coco128.yaml", epochs=3)
# 在验证集上评估模型性能
metrics = model.val()
# 将模型导出为 ONNX 格式
success = model.export(format="onnx")
