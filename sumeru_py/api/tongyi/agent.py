import os
import sys
import time
import threading
import json
import dashscope
from concurrent.futures import ThreadPoolExecutor

from api.tongyi.audio_manager import AudioManager
from api.tongyi.asr_srv import ASRService
from api.tongyi.tts_srv import TTSService
from api.tongyi.llm_srv import LLMService
from api.tongyi.srv_pusher import FastAPIPusher

def set_api(api_key):
    os.environ["DASHSCOPE_API_KEY"] = api_key
    dashscope.api_key = api_key
    if os.environ["DASHSCOPE_API_KEY"] == "":
        print("[ERROR] DASHSCOPE_API_KEY 设置失败")
        sys.exit(0)

class VoiceAssistantApp:
    def __init__(self, llm_model: str = "qwen-plus", tts_voice: str = "Chelsie", 
                 pusher: FastAPIPusher = None):
        self.is_speaking = False
        self._app_lock = threading.Lock()

        # 1. 首先创建硬件管理器
        self.audio_manager = AudioManager()

        self.pusher = pusher

        # 2. 初始化所有服务，并将 audio_manager 实例注入到需要的服务中
        self.llm_service = LLMService(model=llm_model)
        self.tts_service = TTSService(audio_manager=self.audio_manager, voice=tts_voice)
        self.asr_service = ASRService(audio_manager=self.audio_manager, on_sentence_callback=self._on_sentence_recognized)

        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='AppWorker')

    def _on_sentence_recognized(self, text: str):
        print(f"[ASR] {text}")
        with self._app_lock:
            if self.is_speaking:
                return
        # threading.Thread(target=self._process_text_async, args=(text,)).start()        
        self.executor.submit(self._process_text_async, text)

    def _process_text_async(self, text: str):
        """
        在一个独立的线程中执行耗时的LLM和TTS任务。
        """
        future_classification = self.executor.submit(self.llm_service.get_classification, text)
        future_response = self.executor.submit(self.llm_service.get_conversational_response, text)

        classification = future_classification.result()

        # 任务1: 获取JSON分类结果，并放入队列
        # classification = self.llm_service.get_classification(text)
        if not classification:
            print("[APP ERROR] 分类任务失败")
            return
        if classification and self.pusher:
            try:
                json_payload = json.dumps(classification, ensure_ascii=False)
                self.executor.submit(self.pusher.push, json_payload)
            except Exception as e:
                print(f"[APP ERROR] pusher 不存在或者有问题 {e}")

        # 任务2: 获取对话式回复
        # response_text = self.llm_service.get_conversational_response(text)
        response_text = future_response.result()
        print(f"[LLM] {response_text}")
        
        # 任务3: 使用TTS服务播放回复
        if response_text:
            self.asr_service.pause_listening()

            with self._app_lock:
                self.is_speaking = True

            self.tts_service.speak(response_text)

            with self._app_lock:
                self.is_speaking = False

            self.asr_service.resume_listening()

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def run(self):
        try:
            print("语音助手应用已启动。按下 Ctrl+C 退出。")
            self.asr_service.activate()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n检测到退出指令，正在关闭所有服务...")
        finally:
            self.asr_service.close()
            self.audio_manager.close()
            self.shutdown()
            print("应用已安全退出。")

if __name__ == "__main__":
    # from utils import set_api
    API_KEY = "sk-8fcc0ea8b3314ae3bc0693ab3f99e2c8" 
    set_api(API_KEY)

    # 2. 创建应用实例
    pusher = FastAPIPusher(server_url="http://localhost:8000")
    assistant = VoiceAssistantApp(pusher=pusher)

    # 3. 运行应用
    assistant.run()
