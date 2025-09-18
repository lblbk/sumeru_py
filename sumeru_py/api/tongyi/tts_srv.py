import os
import time
import dashscope
import base64
import numpy as np
from collections import deque
from .audio_manager import AudioManager

class TTSService:
    """
    只负责文本到语音。现在通过 AudioManager 与音频硬件解耦。
    """
    def __init__(self, audio_manager: AudioManager, voice: str = "Chelsie"):
        self.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.audio_manager = audio_manager
        self.voice = voice

    def speak(self, text: str):
        if not text or not text.strip():
            print(f"[TTS ERROR] 输入文本有误 {text}")
            return
        stream = None
        try:
            stream = self.audio_manager.open_output_stream()
            responses = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model="qwen-tts", api_key=self.api_key, text=text,
                voice=self.voice, stream=True
            )

            if responses is None:
                print("[LLM ERROR] API 返回为空")
                return            

            PRE_BUFFER_SIZE = 5 
            audio_buffer = deque()
            is_playing = False
            
            for chunk in responses:
                if 'output' in chunk and 'audio' in chunk['output'] and 'data' in chunk['output']['audio']:
                    wav_bytes = base64.b64decode(chunk["output"]["audio"]["data"])
                    audio_data = np.frombuffer(wav_bytes, dtype=np.int16).tobytes()
                    audio_buffer.append(audio_data)

                    # if not is_playing and len(audio_buffer) >= PRE_BUFFER_SIZE:
                    #     is_playing = True
                    
                    # if is_playing:
                    stream.write(audio_buffer.popleft())

                    # stream.write(np.frombuffer(wav_bytes, dtype=np.int16).tobytes())
            while audio_buffer:
                stream.write(audio_buffer.popleft())
            
            time.sleep(0.5)
        except Exception as e:
            print(f"   [TTS] 语音播放时发生错误: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
