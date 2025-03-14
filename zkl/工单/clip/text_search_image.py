import mysql.connector
import json
import chromadb
import mysql.connector

from image_show import show_image
from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip import tokenize
import torch
import gradio as gr
import atexit


#加载jsonl函数
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

#读取jsonl文件
image_vectors_data = read_jsonl(r'C:\Users\26296\Desktop\Chinese-CLIP-master\DATAPATH\datasets\test_imgs.img_feat.jsonl')
text_vectors_data = read_jsonl(r'C:\Users\26296\Desktop\Chinese-CLIP-master\DATAPATH\datasets\test_texts.txt_feat.jsonl')


# 连接 MySQL 数据库
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="image_search_db"
)
mycursor = mydb.cursor()


# 连接 Chroma 客户端
client = chromadb.Client()

# 创建文本和图像集合
text_collection = client.create_collection(name="text_vectors")
image_collection = client.create_collection(name="image_vectors")

# 插入图像向量数据到 Chroma
image_ids = []
image_embeddings = []
for item in image_vectors_data:
    image_ids.append(str(item["image_id"]))
    image_embeddings.append(item["feature"])

image_collection.add(
    ids=image_ids,
    embeddings=image_embeddings
)

# 插入文本向量数据到 Chroma
text_ids = []
text_embeddings = []
for item in text_vectors_data:
    text_ids.append(str(item["text_id"]))
    text_embeddings.append(item["feature"])

text_collection.add(
    ids=text_ids,
    embeddings=text_embeddings
)

# 获取文本和图像集合
text_collection = client.get_collection(name="text_vectors")
image_collection = client.get_collection(name="image_vectors")


# 加载用于文本特征提取的模型
with open(r"C:\Users\26296\Desktop\Chinese-CLIP-master\cn_clip\clip\model_configs\ViT-B-16.json", 'r') as fv, open(r"C:\Users\26296\Desktop\Chinese-CLIP-master\cn_clip\clip\model_configs\RoBERTa-wwm-ext-base-chinese.json", 'r') as ft:
    model_info = json.load(fv)
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])
    for k, v in json.load(ft).items():
        model_info[k] = v

model = CLIP(**model_info)
convert_weights(model)
convert_models_to_fp32(model)
'加载模型权重'
state_dict = torch.load(r"C:/Users/26296\Desktop\Chinese-CLIP-master\DATAPATH\pretrained_weights\clip_cn_vit-b-16.pt")
sd = state_dict["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
model.load_state_dict(sd)


def text_to_image_search(input_text):
    # 输入文本
    #提取文本特征
    text = tokenize([input_text], context_length=52)
    text_features = model(None, text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.tolist()[0]

    # 使用 Chroma 检索与输入文本向量最相似的文本向量
    results = text_collection.query(
        query_embeddings=[text_features],
        n_results=1  # 检索最相似的 1 个文本向量
    )

    # 获取最相似的文本向量的 ID
    similar_text_ids = results['ids'][0]

    # 根据文本向量的 ID 在 MySQL 数据库中查询与之关联的图像 ID
    image_ids = []
    for text_id in similar_text_ids:
        select_query = "SELECT image_id FROM image_text_relation WHERE text_id = %s"
        mycursor.execute(select_query, (text_id,))
        result = mycursor.fetchall()
        for row in result:
            image_ids.append(row[0])

    # 根据ID去匹配对应的图像
    image_paths = show_image(image_ids)
    return image_paths


# 定义一个函数用于关闭数据库连接
def close_db_connection():
    mycursor.close()
    mydb.close()
    print("数据库连接已关闭")
# 注册关闭数据库连接的函数，在程序退出时执行
atexit.register(close_db_connection)



# 创建 Gradio 界面
def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 文搜图系统")
        input_text = gr.Textbox(label="输入文本", placeholder="请输入要搜索的文本")
        search_button = gr.Button("搜索")
        output_images = gr.Gallery(label="搜索结果图片", columns=3)

        # 绑定按钮点击事件到文搜图函数
        search_button.click(
            fn=text_to_image_search,
            inputs=input_text,
            outputs=output_images
        )

    # 启动 Gradio 应用
    demo.launch()

if __name__ == "__main__":
    main()
