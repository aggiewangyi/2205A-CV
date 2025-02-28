
import os
import warnings
import time
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from funasr import AutoModel
from modelscope.utils.constant import Tasks
from modelscope.hub.snapshot_download import snapshot_download
from zsq.cost_time import calculate_time


class FunASR:
    def __init__(
            self,
            use_denoise: bool = True,
            model_dir: Optional[Path] = None,
            denoise_model_dir: Optional[Path] = None,
            device: Optional[str] = None,
            njobs: int = 4,
            batch_size_s: int = 300,
            download_retry: int = 3
    ) -> None:
        self._init_device(device)
        self.njobs = njobs
        self.batch_size_s = batch_size_s
        self.download_retry = download_retry

        # 初始化模型路径
        self.model_dir = self._init_model_dir(model_dir)
        self.denoise_model_dir = denoise_model_dir or self.model_dir / "denoise_models"

        # 自动下载模型
        self._validate_models([
            ("speech_seaco_paraformer*", self.model_dir),
            ("speech_fsmn_vad*", self.model_dir),
            ("punc_ct-transformer*", self.model_dir),
            ("speech_dfsmn_denoise*", self.denoise_model_dir)
        ])

        # 初始化ASR模型
        self.model = self._init_asr_model(use_denoise)

        # 备用降噪管道
        self.denoise_pipeline = None
        if use_denoise and not self.model.denoise_model:
            self._init_fallback_denoise()

    def _init_device(self, device: Optional[str]) -> None:
        """初始化计算设备"""
        if device and device not in ["cuda", "cpu"]:
            raise ValueError(f"不支持的设备类型: {device}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA不可用，已自动回退到CPU")
            self.device = "cpu"

    def _init_model_dir(self, model_dir: Optional[Path]) -> Path:
        """初始化模型存储目录"""
        default_dir = Path(__file__).parent / "model_weights"
        model_dir = model_dir or default_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _validate_models(self, model_list: List[Tuple[str, Path]]) -> None:
        """验证并自动下载缺失模型"""
        for pattern, base_dir in model_list:
            if not list(base_dir.glob(pattern)):
                print(f"正在下载模型: {pattern.replace('*', '')}")
                self._download_model(pattern.split("*")[0], base_dir)

    def _download_model(self, model_name: str, target_dir: Path) -> None:
        """下载模型并处理异常"""
        from modelscope.hub.api import HubApi
        api = HubApi()

        for attempt in range(self.download_retry):
            try:
                snapshot_download(f"damo/{model_name}",
                                  cache_dir=str(target_dir),
                                  revision='master')
                return
            except Exception as e:
                if attempt == self.download_retry - 1:
                    raise RuntimeError(f"模型 {model_name} 下载失败: {str(e)}")
                print(f"下载失败，正在重试 ({attempt +1}/{self.download_retry})...")
                time.sleep(2)

    def _init_asr_model(self, use_denoise: bool) -> AutoModel:
        """初始化FunASR模型实例"""
        return AutoModel(
            model=self._find_model("speech_seaco_paraformer*", self.model_dir) or "paraformer-zh",
            vad_model=self._find_model("speech_fsmn_vad*", self.model_dir) or "fsmn-vad",
            punc_model=self._find_model("punc_ct-transformer*", self.model_dir) or "ct-punc-c",
            denoise_model=self._find_model("speech_dfsmn_denoise*", self.denoise_model_dir) if use_denoise else None,
            njobs=self.njobs,
            device=self.device
        )

    def _find_model(self, pattern: str, base_dir: Path) -> Optional[str]:
        """查找模型路径"""
        matches = list(base_dir.glob(pattern))
        return str(matches[0]) if matches else None

    def _init_fallback_denoise(self) -> None:
        """初始化备用降噪方案"""
        from modelscope.pipelines import pipeline

        try:
            self.denoise_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model="damo/speech_frcrn_ans_cirm_16k",
                device=self.device
            )
        except Exception as e:
            warnings.warn(f"备用降噪初始化失败: {str(e)}")
            self.denoise_pipeline = None

    def _preprocess_audio(self, audio_path: str) -> str:
        """音频预处理"""
        if not self.denoise_pipeline:
            return audio_path

        try:
            output_path = self._generate_output_path(audio_path)
            self.denoise_pipeline(audio_path, output_path=output_path)
            return output_path
        except Exception as e:
            warnings.warn(f"降噪处理失败: {str(e)}，使用原始音频")
            return audio_path

    def _generate_output_path(self, input_path: str) -> str:
        """生成唯一输出路径"""
        timestamp = int(time.time() * 1000)
        original_path = Path(input_path)
        return str(original_path.with_name(f"{original_path.stem}_denoised_{timestamp}{original_path.suffix}"))

    @calculate_time
    def transcribe(self, audio_file: str) -> str:

        if not Path(audio_file).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")

        processed_path = self._preprocess_audio(audio_file)

        try:
            result = self.model.generate(
                input=processed_path,
                batch_size_s=self.batch_size_s,
                frontend_processing=True
            )
            return result[0]['text']
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("显存不足，请尝试减小batch_size_s")
        except Exception as e:
            raise RuntimeError(f"ASR处理失败: {str(e)}")
        finally:
            self._clean_temp_file(processed_path, audio_file)

    def _clean_temp_file(self, processed_path: str, original_path: str) -> None:
        """清理临时文件"""
        if processed_path != original_path and Path(processed_path).exists():
            try:
                os.remove(processed_path)
            except Exception as e:
                warnings.warn(f"临时文件清理失败: {str(e)}")


if __name__ == "__main__":
    # 示例用法
    try:
        asr = FunASR(
            use_denoise=True,
            model_dir=Path("./custom_models"),  # 自定义模型目录
            batch_size_s=200
        )

        result = asr.transcribe("test_audio.wav")
        print(f"识别成功: {result}")

    except Exception as e:
        print(f"发生错误: {str(e)}")