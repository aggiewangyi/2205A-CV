# -*- coding: utf-8 -*-
# 人工智能 CV-AIGC 项目-文搜图 任务
import cv2
import faiss
import numpy as np
import os
import pickle
import gradio as gr
import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import mysql.connector

# 禁用 TensorFlow 的 OpenMP,多个OpenMP库加载问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    print("Available models:", available_models())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device,
                                       download_root=r'C:\Users\quant\Desktop\cs\datapath\pretraind_weights')
    model.eval()
    # 连接到 MySQL 服务器
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="img_db"
    )
    # 创建游标对象
    mycursor = mydb.cursor()
    # 选择数据库
    mydb.database = "img_db"

    # 读取保存的索引
    index_file = "image_index2.faiss"
    loaded_index = faiss.read_index(index_file)
    print(f"Index loaded from {index_file}")


    def clip_find(input_text):
        ####第一步生成了特征向量数据库,这时候每当输入新的文本时,就不需要再重新计算图片特征向量
        text = clip.tokenize(input_text).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
            text_features /= text_features.norm(dim=-1, keepdim=True)
        # 定义要搜索的最近邻数量
        k = 5
        # 执行搜索操作
        text_features = text_features.cpu().numpy()
        distances, indices = loaded_index.search(text_features, k)
        output_image_list = []
        for i in indices[0]:
            # 编写 SQL 查询语句
            sql = "SELECT img_name FROM products WHERE img_id = %s"
            val = (i.item(),)
            # 执行 SQL 查询
            mycursor.execute(sql, val)
            # 获取查询结果
            result = mycursor.fetchone()
            # http://localhost/000000000009.jpg
            output_image_list.append("http://localhost/" + result[0])

        # 打印查询结果
        for row in output_image_list:
            print(row)

        return output_image_list, distances

    with gr.Blocks() as demo:
        gr.Markdown("# 文搜图 CLIP 网站")
        gr.Markdown("输入文本描述，搜索与之最相似的图像。")
        input_text = gr.Textbox(label="输入文本描述", value="踢足球的人")
        greet_button = gr.Button("生成")
        with gr.Column():
            image_gallery = gr.Gallery(label="前5张相似图片组合", show_label=True, columns=5)
            output_scores_text = gr.Textbox(label="相似度分数")
        # 按钮点击事件
        greet_button.click(fn=clip_find, inputs=input_text, outputs=[image_gallery, output_scores_text])

    demo.launch()
    # 关闭游标和数据库连接
    mycursor.close()
    mydb.close()
