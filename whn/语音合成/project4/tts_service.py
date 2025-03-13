# 人工智能CV-AIGC 项目-语音合成任务
from paddlespeech.cli.tts import TTSExecutor

def generate_speech(text: str, output_path: str):
    tts = TTSExecutor()
    tts(
        text=text,
        output=output_path,
        am="fastspeech2_mix",      # 声学模型
        voc="hifigan_csmsc",       # 声码器
        lang="mix",                # 支持中英文混合
        spk_id=174                # 说话人ID（174为女声）
    )