# -*- coding: utf-8 -*-
# 导入包
import os
import torch
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    AutoModel

# 加载数据集
data = load_dataset('json', data_files=r"C:\Users\quant\Desktop\cs\covid-数据集\train.json")
print(data)
model_path = r"C:\Users\quant\Desktop\cs\modelscope\chatglm2-6b-int4"
# 加载分词模型 加载该模型的分词
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# 定义预处理函数
def preprocess_function(examples):
    # 构建输入文本，添加提示词
    inputs = [f"输入: {inst}" for inst in examples["instruction"]]
    # 构建目标文本，添加提示词
    targets = [f"输出: {out}" for out in examples["output"]]
    # 对输入文本进行分词处理，设置填充和截断
    # max_length 参数指定了分词后的词元序列的最大长度。
    # truncation 参数用于控制是否对超过 max_length 的词元序列进行截断处理。
    # padding 参数用于控制是否对词元序列进行填充操作，以确保所有输入的词元序列长度一致。max_length=512
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    # 对目标文本进行分词处理，设置填充和截断
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    # 将目标文本的分词结果中的 input_ids 添加到 model_inputs 中，作为 labels 字段
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# map 方法是 datasets.Dataset 和 datasets.DatasetDict 对象的一个方法，
# 其作用是对数据集中的每个样本或者样本批次应用指定的函数。
tokenized_data = data.map(preprocess_function, batched=True)
# 当 batched=True 时，preprocess_function 会一次性接收一批样本进行处理。
# print(tokenized_data.data)
# print(tokenized_data)

# 加载因果语言模型
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
# LoRA微调，生成config 配置表
config = LoraConfig(
    r=8,  # LoRA模块的秩（rank），即低秩矩阵的维度
    lora_alpha=8,  # LoRA模块的缩放因子
    target_modules=["query_key_value"],  # 要应用 LoRA 的目标模块，指定了模型中哪些模块将应用LoRA技术
    lora_dropout=0.1,  # 丢弃率,LoRA模块的dropout概率
    bias="none",  # 不调整偏置项
    task_type="CAUSAL_LM",  # 任务类型为因果语言模型
    inference_mode=False,  # 推理模式，这里设置为False，表示进行训练
)
model = get_peft_model(model, config)  # 这一行代码将LoRA配置应用于加载的模型，返回一个经过LoRA微调的模型
model.config.torch_dtype = torch.float32  # 确保模型在训练或推理过程中使用特定的数据类型
model.cuda()

#
training_args = TrainingArguments(
    output_dir='./results',  # 训练结果保存的目录
    num_train_epochs=10,  # 训练轮数
    per_device_train_batch_size=1,  # 每个设备的训练批次大小
    per_device_eval_batch_size=1,  # 每个设备的评估批次大小
    warmup_steps=500,  # 热身步数
    weight_decay=0.01,  # 权重衰减率
    logging_dir='./logs',  # 日志保存的目录
    logging_steps=10,  # 每多少步记录一次日志
    save_steps=1000,  # 每多少步保存一次模型
    # evaluation_strategy="steps",  # 评估策略，按步数进行评估
    # eval_steps=500,  # 每多少步进行一次评估
    evaluation_strategy="no",  # 禁用评估操作
    fp16=True  # 是否使用混合精度训练
)
# 创建一个训练器对象 trainer
trainer = Trainer(
    model=model,  # 指定要训练的模型
    args=training_args,  # 训练过程中的各种参数设置
    train_dataset=tokenized_data["train"],  # 指定训练数据集
    eval_dataset=tokenized_data["validation"] if "validation" in tokenized_data else None  # 指定评估数据集
)
# 开始训练
trainer.train()

# 保存 LoRA 模型
model.save_pretrained("./medical_lora_chatglm2")

#
