import abc
from dashscope.audio.asr import (
    # Gummy (Translation) imports
    TranslationRecognizerRealtime,
    TranslationRecognizerCallback,
    TranscriptionResult,
    # Paraformer (Recognition) imports
    Recognition,
    RecognitionCallback,
    RecognitionResult,
)

class ASRModelHandler(abc.ABC):
    """
    一个抽象基类，定义了所有ASR模型处理器必须遵循的接口。
    """
    @abc.abstractmethod
    def create_recognizer(self, callback):
        """根据模型类型，创建并返回一个实时的识别器实例。"""
        pass

    @abc.abstractmethod
    def create_callback(self, on_sentence_callback: callable):
        """
        创建一个与特定模型SDK兼容的回调类实例。
        这个方法将封装处理不同 Result 类型的逻辑。
        """
        pass

class GummyASRHandler(ASRModelHandler):
    """'gummy-realtime-v1' 模型的具体实现。"""
    
    def create_recognizer(self, callback):
        return TranslationRecognizerRealtime(
            model="gummy-realtime-v1",
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            callback=callback
        )

    def create_callback(self, on_sentence_callback: callable):
        class _GummyCallback(TranslationRecognizerCallback):
            def on_event(self, _, result: TranscriptionResult, *args) -> None:
                if result and result.is_sentence_end:
                    sentence = result.text
                    on_sentence_callback(sentence)
        
        return _GummyCallback()


class ParaformerASRHandler(ASRModelHandler):
    """'paraformer-realtime-v2' 模型的具体实现。"""

    def create_recognizer(self, callback):
        # print("[ASR Handler] Creating recognizer for: paraformer-realtime-v2")
        return Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=16000,
            callback=callback
        )

    def create_callback(self, on_sentence_callback: callable):
        # 定义一个只属于 Paraformer 的内部回调类
        class _ParaformerCallback(RecognitionCallback):
            def on_event(self, result: RecognitionResult, *args, **kwargs):
                if result.get_sentence()["sentence_end"]:
                    sentence = result.get_sentence()["text"]
                    on_sentence_callback(sentence)

        return _ParaformerCallback()

_MODEL_HANDLERS = {
    "gummy-realtime-v1": GummyASRHandler,
    "paraformer-realtime-v2": ParaformerASRHandler,
}

def get_asr_handler(model_name: str) -> ASRModelHandler:
    """
    工厂函数：根据模型名称返回一个具体的ASR模型处理器实例。
    """
    handler_class = _MODEL_HANDLERS.get(model_name)
    if not handler_class:
        raise ValueError(f"不支持的ASR模型: {model_name}. 可用模型: {list(_MODEL_HANDLERS.keys())}")
    
    return handler_class()
