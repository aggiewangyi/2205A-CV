from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json

from deepseek_api import call_ollama_api

with open('train.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

documents = []
for item in json_data:
    # 将instruction和output合并为完整上下文
    content = f"Instruction: {item['instruction']} Response: {item['output']}"
    documents.append(Document(page_content=content))
# print(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name=r"E:\rag\bge-small-zh-v1.5")

# 创建 Chroma 向量存储库
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="medical_dialogue_collection"
)

def augment_prompt(query):
    #获取文本片段
    result = vectorstore.similarity_search(query, k=2)
    source_knowledge = "\n".join([x.page_content for x in result])
    #构建prompt
    augment_prompt = f'使用以下内容去回答下面的问题，内容：{source_knowledge}。问题：{query}'
    return augment_prompt

if __name__ == '__main__':
    query = "自去年底开始感觉吞咽功能减退，今年1月份首发肺部感染抗炎治疗20余天，一周前再发感染。神态尚清，因脑梗引起血管性痴呆5年多，现语言功能退化，大小便表达不清，但尚未卧床。"
    prompt = augment_prompt(query)
    print(prompt)
    result = call_ollama_api(prompt)
    print(result)