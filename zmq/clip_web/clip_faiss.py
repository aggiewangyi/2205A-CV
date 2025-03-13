# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
from PIL import Image
import os
import faiss
import numpy as np
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root=r'C:\Users\quant\Desktop\cs\datapath\pretraind_weights')
model.eval()
# image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
# image = preprocess(Image.open("examples/OIP-C.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘",'飞机','奥特曼','奶龙']).to(device)
image_dir = 'G:/coco/images/val'
with torch.no_grad():
    file_list = os.listdir(image_dir)
    img_n = len(file_list)
    # 4. 使用faiss构建索引
    d = 512  # 向量维度
    index = faiss.IndexFlatIP(d)  # 使用内积作为相似度度量
    for i,imgname in enumerate(file_list):
        print(f"进度：{i}/{img_n}")
        image = preprocess(Image.open(image_dir+"/"+imgname)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
        index.add(image_features)
    # 保存索引到文件
    index_file = "image_index.faiss"
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")

print(image_features)












    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # logits_per_image, logits_per_text = model.get_similarity(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]