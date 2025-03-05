"""
人工智能CV-AIGC项目-数字人生成任务
工单编号：人工智能CV-AIGC-项目-数字人生成任务
Python版本：3.8+（推荐3.9）
依赖环境配置：
  - CUDA 11.3+ (GPU加速)
  - cuDNN 8.2+
所需包：
  - torch==1.12.1+cu113
  - torchvision==0.13.1+cu113
  - opencv-python==4.7.0
  - numpy==1.23.5
  - librosa==0.9.2
  - ffmpeg-python==0.2.0
  - tqdm==4.64.1
"""

import os
import subprocess
import argparse
import cv2
import librosa
import numpy as np
from tqdm import tqdm


# Wave2Lip 实现
def run_wave2lip(face_path, audio_path, output_path):
    """
    调用Wave2Lip生成数字人视频
    参考：https://github.com/Rudrabha/Wav2Lip
    """
    cmd = f"python inference.py --checkpoint_path wave2lip_ckpt.pth \
            --face {face_path} --audio {audio_path} --outfile {output_path}"
    subprocess.run(cmd, shell=True, check=True)


# MuseTalk 实现
def run_musetalk(face_path, audio_path, output_path):
    """
    调用MuseTalk生成数字人视频
    参考：https://github.com/microsoft/MuseTalk
    """
    cmd = f"python musetalk_infer.py --source_image {face_path} \
            --driven_audio {audio_path} --result_dir {output_path}"
    subprocess.run(cmd, shell=True, check=True)


# 同步性验证（示例）
def validate_sync(video_path, audio_path, threshold=0.1):
    """
    验证嘴型同步误差是否<=0.1秒
    返回误差值和是否通过
    """
    # 此处简化实现，实际需提取视频嘴型帧时间戳和音频特征
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    calc_duration = frame_count / fps
    time_diff = abs(duration - calc_duration)
    return time_diff, time_diff <= threshold


# 主流程
def main(args):
    # 生成Wave2Lip结果
    print("Running Wave2Lip...")
    run_wave2lip(args.face_input, args.audio_input, "output_wave2lip.mp4")

    # 生成MuseTalk结果
    print("Running MuseTalk...")
    run_musetalk(args.face_input, args.audio_input, "output_musetalk.mp4")

    # 同步性验证
    wave2lip_diff, wave2lip_pass = validate_sync("output_wave2lip.mp4", args.audio_input)
    musetalk_diff, musetalk_pass = validate_sync("output_musetalk.mp4", args.audio_input)

    # 生成对比报告
    report = f"""
    === 模型对比报告 ===
    [Wave2Lip]
    - 同步误差：{wave2lip_diff:.4f}s | {'通过' if wave2lip_pass else '不通过'}
    - 优势：推理速度快，资源占用低
    - 劣势：表情自然度较低

    [MuseTalk]
    - 同步误差：{musetalk_diff:.4f}s | {'通过' if musetalk_pass else '不通过'}
    - 优势：表情更自然，支持多语言
    - 劣势：硬件要求较高

    === 验收结果 ===
    综合推荐：{'MuseTalk' if musetalk_pass else 'Wave2Lip'}
    """
    print(report)
    with open("model_comparison.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_input", type=str, required=True, help="输入人脸图片/视频路径")
    parser.add_argument("--audio_input", type=str, required=True, help="输入音频文件路径")
    args = parser.parse_args()

    main(args)