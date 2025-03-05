# -*- coding: utf-8 -*-
# 工单编号：人工智能CV-AIGC-项目-语音克隆任务

import os
import torch
import librosa
import soundfile as sf
import numpy as np
from gpt_sovits import GPTSoVITS  # 需从官方仓库获取模型代码

# 配置参数
config = {
    "data_dir": "./user_voice",  # 用户语音数据目录
    "output_dir": "./results",  # 输出目录
    "model_path": "gpt-sovits-base",  # 预训练模型路径
    "sr": 22050,  # 采样率
    "n_mfcc": 13,  # MFCC特征维度
    "epochs": 50,  # 训练轮次
}


def preprocess_data():
    """数据预处理：提取语音特征并保存"""
    os.makedirs(config["output_dir"], exist_ok=True)
    wav_files = [f for f in os.listdir(config["data_dir"]) if f.endswith(".wav")]

    features = []
    for file in wav_files:
        path = os.path.join(config["data_dir"], file)
        y, _ = librosa.load(path, sr=config["sr"])
        mfcc = librosa.feature.mfcc(y=y, sr=config["sr"], n_mfcc=config["n_mfcc"])
        features.append(mfcc.T)

    # 保存预处理数据
    np.save(os.path.join(config["output_dir"], "user_mfcc.npy"), features)
    print("数据预处理完成！")


def train_model():
    """微调语音合成模型"""
    # 加载模型
    model = GPTSoVITS.load_from_checkpoint(config["model_path"])

    # 加载数据
    mfcc_data = np.load(os.path.join(config["output_dir"], "user_mfcc.npy"))
    dataset = torch.utils.data.TensorDataset(torch.tensor(mfcc_data, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(config["epochs"]):
        for batch in loader:
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss.item():.4f}")

    # 保存微调后的模型
    torch.save(model.state_dict(), os.path.join(config["output_dir"], "fine_tuned_model.pt"))
    print("模型微调完成！")


def synthesize(text):
    """语音合成推理"""
    # 加载微调后的模型
    model = GPTSoVITS.load_from_checkpoint(config["model_path"])
    model.load_state_dict(torch.load(os.path.join(config["output_dir"], "fine_tuned_model.pt")))

    # 生成语音
    waveform = model.generate(text)
    output_path = os.path.join(config["output_dir"], "output.wav")
    sf.write(output_path, waveform, config["sr"])
    print(f"语音已生成：{output_path}")


if __name__ == "__main__":
    # 执行流程
    preprocess_data()  # 步骤1：数据预处理
    train_model()  # 步骤2：模型微调
    synthesize("欢迎使用八维信息集团的语音克隆服务")  # 步骤3：语音合成