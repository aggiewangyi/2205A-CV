from funasr import AutoModel


class FunASR:
    def __init__(self):
        super().__init__()
        model_path = "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        vad_model_path = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        punc_model_path = "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        self.model = AutoModel(model=model_path, vad_model=vad_model_path, punc_model=punc_model_path)

    def transcribe(self, path):
        res = self.model.generate(input=path, batch_size_s=300)
        print(res)
        return res[0]['text']


if __name__ == '__main__':
    audio_file = "vo_ABDLQ001_1_paimon_02.wav"
    asr = FunASR()
    result = asr.transcribe(audio_file)
    print(result)
