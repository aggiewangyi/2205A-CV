import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu" # 检查是否有可用的 GPU。如果有则使用 CUDA 加速，否则使用 CPU
model, preprocess = clip.load('ViT-B/32', device) # 加载 CLIP 模型的 Vision Transformer (ViT-B/32) 版本。

# 获取 CIFAR-100 数据集
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

image, class_id = cifar100[0] # 从测试集中选取索引为 3637 的图片及其对应的真实类别 ID。

import numpy as np
import cv2
# 将PIL图像转换为NumPy数组
np_img = np.array(image)
# 如果图像是RGB格式，转换为BGR格式（OpenCV默认使用BGR）
if np_img.ndim == 3:  # 检查是否为彩色图像
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

# 创建一个可以调整大小的窗口
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# 设置窗口大小
window_width = 800
window_height = 600
cv2.resizeWindow('image', window_width, window_height)
# 使用OpenCV显示图像-
cv2.imshow("image", np_img)
# 等待按键按下，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

image_input = preprocess(image).unsqueeze(0).to(device)
# preprocess(image)：对图片进行预处理（如缩放、归一化等），以符合 CLIP 模型的输入要求。
# .unsqueeze(0)：增加批次维度，变成 [1, 3, H, W] 的形状。

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
# clip.tokenize(f"a photo of a {c}")：将每个类别名（如“apple”、“bicycle”）生成对应的文本 token，用于与图像特征进行对比。
# torch.cat([...])：将所有类别的 token 拼接成一个大的张量。

with torch.no_grad():
    image_features = model.encode_image(image_input)  # 提取图像特征。
    text_features = model.encode_text(text_inputs)  # 提取文本特征。

image_features /= image_features.norm(dim=-1, keepdim=True)
# 这行代码计算图像特征向量的L2范数（即向量的长度或大小），然后将图像特征向量的每个元素除以这个L2范数。
# dim=-1表示沿着最后一个维度（即特征向量的维度）计算范数。
# keepdim=True确保计算范数后结果的维度与原始特征向量的维度一致，以便可以直接进行除法操作。
# 这种归一化处理确保了特征向量的长度为1，仅保留向量的方向信息，这对于后续的相似性计算非常重要。
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# image_features @ text_features.T：计算图像特征与每个文本特征的点积，得到相似度分数。
# .softmax(dim=-1)：对相似度分数进行 softmax 归一化，得到概率分布。
# 100.0 *：将相似度值缩放到 0~100 的范围。

values, indices = similarity[0].topk(5)
# similarity[0]：取出批次中第一张图片（也是唯一一张）的相似度分布。
# .topk(5)：选出相似度最高的 5 个类别及其对应的分数。


# 打印结果
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")