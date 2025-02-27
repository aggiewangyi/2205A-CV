# -*- coding: utf-8 -*-
import json
import nltk
import requests
from Wav2Lip_master import wav2lip_api
import re
# Ollama API 的基本 URL，默认情况下 Ollama 运行在本地的 11434 端口
base_url = "http://localhost:11434/api/generate"
# 要使用的模型名称
model = "deepseek-r1:1.5b"
try:
    # 下载 averaged_perceptron_tagger_eng 资源
    nltk.download('averaged_perceptron_tagger_eng')
except:
    print('')

import sys

# sys.path.append('../../')
from VITS.GPT_SoVITS import GPT_SoVITS
import gradio as gr
from FunASR import FunASR


def recognize_speech(filepath):
    if filepath:
        try:
            print(filepath)
            text = model.transcribe(filepath)
            # print(text)
            return text
        except:
            return '错误0'


def ollama_api(prompt):
    model = "deepseek-r1:1.5b"
    data = {
        "model": model,
        "prompt": prompt
    }
    # 发送 POST 请求
    response = requests.post(base_url, json=data)
    result_s = ''
    think = 0
    # 检查响应状态码
    if response.status_code == 200:
        # 逐行处理响应内容
        for line in response.text.splitlines():
            if line:
                try:
                    result = json.loads(line)
                    if 'response' in result:
                        print(result['response'], end='', flush=True)
                        if think == 1:
                            result_s = result_s + str(result['response'])
                        if result['response'] == '</think>':
                            think = 1
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON: {line}")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
    return result_s


def au_predict(text):
    print(text)

    result_s = ollama_api(str(text))
    # 使用正则表达式匹配所有换行符并替换为空字符串
    result_s = re.sub(r'[\r\n]+', '', result_s)
    GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, result_s, text_language, how_to_cut,
                                 save_audio_file)
    print(save_audio_file)
    wav2lip_api.main(r'C:\Users\quant\Desktop\cs\Linly-Talker\result.wav')
    return r'.\results\result_voice.mp4'


if __name__ == '__main__':
    GPT_SoVITS_inference = GPT_SoVITS()
    gpt_path = "D:/数字人/GPT-SoVITS/GPT-SoVITS-v3-202502123fix2/GPT_weights_v2/pm-v2-e15.ckpt"
    sovits_path = "D:/数字人/GPT-SoVITS/GPT-SoVITS-v3-202502123fix2/SoVITS_weights_v2/pm-v2_e8_s1296.pth"
    GPT_SoVITS_inference.load_model(gpt_path, sovits_path)
    ref_wav_path = "C:/Users/quant/Desktop/cs/Linly-Talker/GPT_SoVITS/cs1/vo_AHLQ001_1_paimon_03.wav"
    # 参考音频的文本
    prompt_text = "啊，看来艾尔海森要当代理贤者，这件事传的很广呢，不过居然也有人不认识他，他似乎不是那么有名。"
    prompt_language = "中文"
    text = "大家好"
    text_language = "中英混合"
    # text_language = "中文"
    how_to_cut = "不切"  # ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"]
    print("参考音频文本：", prompt_text)
    print("目标文本：", text)
    save_audio_file = "./result.wav"
    # GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut,
    #                              save_audio_file)

    # 语音识别
    model = FunASR()
    # iface = gr.Interface(fn=recognize_speech, inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
    #                      # 返回文件路径
    #                      outputs="text", live=True)  # live=True自动更新，不用点按钮
    #
    # iface.launch()
    with gr.Blocks() as demo:
        input_au = gr.Audio(sources=['microphone', 'upload'], type='filepath')

        output_text = gr.Textbox(lines=2, label='识别结果：')
        # 当音频组件的值发生改变时（即录完音或上传完文件），触发语音识别函数
        input_au.change(fn=recognize_speech, inputs=input_au, outputs=output_text)
        greet_button = gr.Button("生成")
        # au_path = gr.Audio(type="filepath")
        # voice_path = gr.Interface(type="filepath")

        # 用于播放视频的组件
        voice_path = gr.Video(label="播放生成的视频")

        # 按钮点击事件
        greet_button.click(fn=au_predict, inputs=output_text, outputs=voice_path)

    demo.launch()
