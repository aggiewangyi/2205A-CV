import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import configparser

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 获取配置信息
model_folder = config.get('Paths', 'model_folder', fallback=r"E:\clip-vit-large-patch14")
image_db_path = config.get('Paths', 'image_db_path', fallback="image_db")
faiss_index_path = config.get('Paths', 'faiss_index_path', fallback="faiss_index.bin")
image_list_path = config.get('Paths', 'image_list_path', fallback="image_files.txt")
device = config.get('Settings', 'device', fallback="cpu")
num_workers = int(config.get('Settings', 'num_workers', fallback=4))

# 加载 CLIP 模型
model = CLIPModel.from_pretrained(model_folder).to(device)
processor = CLIPProcessor.from_pretrained(model_folder)


# 获取所有图像文件
image_files = [os.path.join(image_db_path, f) for f in os.listdir(image_db_path)
               if f.endswith(('.png', '.jpg', '.jpeg'))]


# 定义提取单张图像特征的函数
def extract_image_features(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy().flatten()

# 并行提取图像特征向量
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    image_features = list(executor.map(extract_image_features, image_files))

# 过滤掉处理出错的特征
image_features = [feat for feat in image_features if feat is not None]

# 转换为 NumPy 数组
image_features = np.array(image_features).astype('float32')

# 构建 Faiss 索引
dimension = image_features.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(image_features)

# 保存索引到文件
faiss.write_index(index, faiss_index_path)


# 保存文件路径

with open(image_list_path, "w") as f:
    for file in image_files:
        f.write(file + "\n")
print(f"保存至 `{image_list_path}`")

