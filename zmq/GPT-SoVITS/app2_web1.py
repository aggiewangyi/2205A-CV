# -*- coding: utf-8 -*-
import nltk

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


def au_predict(text):
    print(text)
    GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut,
                                 save_audio_file)
    return save_audio_file


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
    GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut,
                                 save_audio_file)

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
        greet_button = gr.Button("Greet")
        au_path = gr.Audio(type="filepath")
        # 按钮点击事件
        greet_button.click(fn=au_predict, inputs=output_text, outputs=au_path)

    demo.launch()
