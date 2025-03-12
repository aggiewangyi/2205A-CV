# -*- coding: utf-8 -*-
#对比结果
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch
import copy
model_path = r"C:\Users\quant\Desktop\cs\modelscope\chatglm2-6b-int4"
# 加载分词器
tokenizer_model = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 加载基础的模型
base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True).half()
# 加载微调后的模型
lora_model = copy.deepcopy(base_model)
# 加载微调后的权重
model_path2 = "/root/autodl-tmp/chat-pth/ChatGLM-6B-main/results/checkpoint-2000/pytorch_model.bin"
lora_model.load_state_dict(torch.load(model_path2), strict=False)

# 将模型移动到 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = base_model.to(device)
lora_model = lora_model.to(device)
#输入文本
input_text = "连续稀便四天，每天一次，昨天下午躺沙发没盖被子睡了大约四十分钟，起来后出去发了个快递，大约四点回来后感觉有点酸懒，五点钟体温测量37.7度，五点半时候吃了一袋感冒冲剂，大约六点多一点的时候体温恢复正常，晚八点半后又感觉有点冷，测量体温37.8度，随后吃了一粒复方氨酚烷胺胶囊，大约一个小时后一切恢复正常，至今天下午四点之前一直体温正常，今天早起吃了一袋感冒冲剂，中午时候吃了三粒肠炎宁片，和六克补脾益肠丸，调理肠胃，下午四点以后感觉有点发冷，六点时候测量体温37.8度，刚刚把感冒冲剂吃了。"

base_model = base_model.eval()
response, history = base_model.chat(tokenizer_model, input_text, history=[])
print('基础模型回答:',response)

lora_model = lora_model.eval()
lora_response, lora_history = lora_model.chat(tokenizer_model, input_text, history=[])
print('lora模型回答',lora_response)
