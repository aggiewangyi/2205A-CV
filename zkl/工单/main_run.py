from cProfile import label

import gradio as gr
import time
import ffmpeg
import os


from GPTSoVITS import api接口 as api
from GPTSoVITS.FunASR import FunASR
from deepseekapi.main import call_ollama_api
from Wav2Lip_master.api import main
from wav2lip_arg import parse_wav2lip_args

system_path = os.environ.get('PATH', '')
new_path = f"{system_path};C:/python/Anaconda/an/envs/ASR/Library/bin"
os.environ['PATH'] = new_path


wavlip_pth = r'C:\Users\26296\Desktop\Digital human\wav2lip_pth\wav2lip_gan.pth'
face_path = r"C:\Users\26296\Desktop\full_body_2.png"

# 模拟语音转文本的 API 接口
def speech_to_text(audio_file):
    asr = FunASR()
    prompt_text = asr.transcribe(audio_file)
    return prompt_text

# 模拟 DeepSeek 的 API 接口
def deepseek_api(text):
    s = call_ollama_api(text)
    return s

def generate_char_by_char(text):
    for char in text:
        yield char
        time.sleep(0.1)  # 控制每个字输出的间隔时间

#文本转语音
def text_to_speech(text):
    GPT_SoVITS_inference = api.GPT_SoVITS()
    gpt_path = r"C:\Users\26296\Desktop\Digital human\GPTSoVITS\GPT_weights_v2\yuanshen-e5.ckpt"
    sovits_path = r"C:\Users\26296\Desktop\Digital human\GPTSoVITS\SoVITS_weights_v2\yuanshen_e8_s1288.pth"
    GPT_SoVITS_inference.load_model(gpt_path, sovits_path)
    #参考音频
    ref_wav_path = r"C:\Users\26296\Desktop\Digital human\GPTSoVITS\output\slicer_opt\vo_ABLQ002_2_ambor_02.wav"
    asr = FunASR()
    #参考音频转文本
    prompt_text = asr.transcribe(ref_wav_path)
    prompt_language = "中文"
    text_language = "中英混合"
    how_to_cut = "不切"
    save_audio_file = "./result.wav"
    GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut,
                                 save_audio_file)
    return save_audio_file


#wav2lip视频合成
def wav2lip_process(wav_pth,face_path,audio_path):
    args = parse_wav2lip_args(wav_pth,face_path,audio_path)
    wav2lip_path = main(args)
    return wav2lip_path

# 组合 API 的函数
def process_speech(audio):
    text = speech_to_text(audio)
    deepseek_text = deepseek_api(text)
    result = ""
    for char in generate_char_by_char(deepseek_text):
        result += char
        yield result,None,None

    # 语音合成
    audio_path = text_to_speech(deepseek_text)
    video_path = wav2lip_process(wavlip_pth,face_path,audio_path)
    yield result,audio_path,video_path


# 创建 Gradio 界面
iface = gr.Interface(
    fn=process_speech,
    # inputs=gr.Audio(sources=["microphone"], type="filepath", label="请录制语音"),
    inputs=gr.Audio(sources=["upload","microphone"], type="filepath", label="请上传音频"),
    outputs=[gr.Textbox(label="处理结果"),gr.Audio(label="语音合成结果"),gr.Video(label="数字人视频")],
    title="语音处理系统",
    description="输入语音，经过语音转文本和 DeepSeek 处理后输出结果"
)

# 启动 Gradio 界面
iface.launch()