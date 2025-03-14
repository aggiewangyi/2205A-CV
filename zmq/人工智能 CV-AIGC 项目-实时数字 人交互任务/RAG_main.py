from langchain.embeddings import HuggingFaceEmbeddings, huggingface  # 用于加载 Hugging Face 的预训练嵌入模型
from langchain.vectorstores import Chroma  # 用于创建和管理向量存储库
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 用于将文档分割成较小的文本块
from langchain.docstore.document import Document  # 用于表示文档对象
import json  # 用于处理 JSON 数据
import requests  # 用于发送 HTTP 请求

# 加载 JSON 数据
json_data = []
with open('dev.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            json_data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
# 创建文档列表
documents = []
for item in json_data:
    # 将instruction和output合并为完整上下文
    content = f"用户输入: {item['content']} 回答: {item['summary']}"
    documents.append(Document(page_content=content))  # 封装为Document对象
# print(documents)
# 页面内容很多的话需要分割，其实还是Document对象
# 文本分割 每个文本块的大小为 500 个字符，重叠部分为 50 个字符
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)

docs = documents
# 初始化嵌入模型
# embeddings = HuggingFaceEmbeddings(model_name=r"bge-small-zh-v1.5")
embeddings = HuggingFaceEmbeddings(model_name=r"./bge-small-zh-v1___5")

# 创建 Chroma 向量存储库
vectorstore = Chroma.from_documents(  # 使用Chroma.from_documents方法将分割后的文本块转换为向量
    documents=docs,  # 文本分割结果，其实还是Document对象
    embedding=embeddings,  # 选择嵌入模型
    collection_name="medical_dialogue_collection"  # 存储在名为medical_dialogue_collection的 Chroma 向量存储库中。
)


# 增强查询提示
def augment_prompt(query):
    # 获取文本片段
    result = vectorstore.similarity_search(query, k=2)
    source_knowledge = "\n".join([x.page_content for x in result])
    # 构建prompt
    augment_prompt = f'你需要写一段服装介绍,使用以下内容去回答下面的问题，内容：{source_knowledge}。问题：{query}'
    print(augment_prompt)
    return augment_prompt


def call_ollama_api(prompt, model_name="deepseek-r1:1.5b", base_url="http://localhost:11434"):
    """
    调用 Ollama 本地服务的 API 进行流式文本生成。

    :param prompt: 输入的提示文本
    :param model_name: 使用的模型名称，默认为 "deepseek-r1:7b"
    :param base_url: Ollama 服务的基础 URL，默认为本地服务地址
    """
    url = f"{base_url}/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt
    }
    try:
        # 发送流式 POST 请求
        response = requests.post(url, json=data, stream=True)
        # 检查请求是否成功
        response.raise_for_status()
        think = 0
        print("\n\ndeepseek回答：", end='', flush=True)
        for line in response.iter_lines():
            if line:
                import json
                line_data = json.loads(line)
                if 'response' in line_data:
                    # print(line_data['response'], end='', flush=True)
                    if think == 1:
                        print(line_data['response'], end='', flush=True)
                    if line_data['response'] == '</think>':
                        think = 1

    except requests.RequestException as e:
        print(f"请求出错: {e}")
    except ValueError as e:
        print(f"JSON 解析出错: {e}")


if __name__ == '__main__':
    query = "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"
    prompt = augment_prompt(query)
    call_ollama_api(prompt)
