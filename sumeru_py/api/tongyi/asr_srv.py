import os
import time
import threading
from queue import Queue, Empty
import dashscope

from .asr_model import get_asr_handler
from .audio_manager import AudioManager



class ASRService:
    """
    只负责实时语音识别。现在通过 AudioManager 与音频硬件解耦。


    """
    def __init__(self, audio_manager: AudioManager, on_sentence_callback: callable, 
                 model_name: str = "paraformer-realtime-v2"):
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.audio_manager = audio_manager
        self.on_sentence_received = on_sentence_callback

        self.model_handler = get_asr_handler(model_name)
        
        self.recognizer  = None
        self.input_stream = None # 用于持有音频流对象
        self.audio_queue = Queue()
        
        self._shutdown_event = threading.Event()
        self._session_should_be_active = threading.Event()
        self._is_sending_audio = False # Dashscope会话是否准备好接收音频
        self._session_lock = threading.Lock()

        threading.Thread(target=self._recognizer_manager_loop, daemon=True).start()
        threading.Thread(target=self._audio_sender_loop, daemon=True).start()

    def _recognizer_manager_loop(self):
        """
        守护线程：管理 ASR 服务和音频输入流的生命周期。
        """
        while not self._shutdown_event.is_set():
            self._session_should_be_active.wait(timeout=1)
            if not self._session_should_be_active.is_set():
                continue
            
            with self._session_lock:
                if self.recognizer is not None:
                    continue
                try:
                    callback = self.model_handler.create_callback(self.on_sentence_received)
                    self.recognizer = self.model_handler.create_recognizer(callback)
                    self.recognizer.start()

                    self.input_stream = self.audio_manager.open_input_stream(
                        callback=self._audio_stream_callback
                    )
                    self.input_stream.start_stream()
                    self._is_sending_audio = True
                    print("[ASR] Dashscope 会话已创建")

                except Exception as e:
                    print(f"[ASR] 启动识别器或音频流时发生错误: {e}")
                    self._cleanup_session()
                    time.sleep(3)
    
    def _audio_sender_loop(self):
        """
        消费者线程：不断从队列中取出音频数据并发送到云端。
        """
        while not self._shutdown_event.is_set():
            if not self._is_sending_audio:
                time.sleep(0.5)
                continue

            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                if self.recognizer:
                    self.recognizer.send_audio_frame(audio_chunk)
            except Empty:
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"[ASR] 发送音频帧时出错: {e}。会话已断开，将触发自动重建。")
                self._cleanup_session()
                time.sleep(1)

    def _audio_stream_callback(self, in_data, frame_count, time_info, status):
        """
        生产者回调：此函数由 PyAudio 高频调用。
        唯一任务：将音频数据放入队列。必须极快！
        """
        self.audio_queue.put(in_data)
        import pyaudio
        return (None, pyaudio.paContinue) # 返回 None 而不是 in_data，因为我们不直接处理它
    
    # def _create_dashscope_callback(self):
    #     parent_instance = self
    #     class _Callback(TranslationRecognizerCallback):
    #         def on_event(self, _, transcription_result: TranscriptionResult, *args) -> None:
    #             if transcription_result and transcription_result.is_sentence_end:
    #                 parent_instance.on_sentence_received(transcription_result.text)
    #     return _Callback()

    def activate(self):
        """
        激活ASR服务，开始建立连接并准备监听。
        这是一个重量级操作，通常只在应用启动时调用一次。
        """
        if not self._session_should_be_active.is_set():
            print("[ASR] 激活ASR服务...")
            self._session_should_be_active.set()

    def pause_listening(self):
        """硬暂停：通过 stop_stream() 直接停止麦克风音频流"""
        with self._session_lock:
            if self.input_stream and self.input_stream.is_active():
                # print("[ASR] 暂停监听...")
                try:
                    self.input_stream.stop_stream()
                    # 清空队列，丢弃在暂停前瞬间进入的任何音频数据
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                except Exception as e:
                    print(f"[ASR ERROR] 停止监听麦克风报错：{e}")

    def resume_listening(self):
        """通过 start_stream() 重新启动麦克风音频流。"""
        with self._session_lock:
            if self.input_stream and not self.input_stream.is_active():
                # print("[ASR] 重启监听")
                try:
                    self.input_stream.start_stream()
                except Exception as e:
                    print(f"[ASR ERROR] 重启麦克风监听报错：{e}")
    
    def close(self):
        """完全关闭服务"""
        self._shutdown_event.set()
        self._session_should_be_active.clear() 
        self._cleanup_session()
        print("[ASR] ASR服务已成功关闭。")

    def _cleanup_session(self):
        """
        公共的清理方法，它会获取锁以确保线程安全。
        """
        with self._session_lock:
            self._cleanup_session_internal()

    def _cleanup_session_internal(self):
        """
        实际执行清理工作的内部方法。必须在已持有锁的情况下调用。
        """
        if not self._is_sending_audio and self.recognizer is None:
            return # 避免重复清理
        
        self._is_sending_audio = False
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception as e:
                print(f"[ASR ERROR] 关闭音频流时出错: {e}")
            finally:
                self.input_stream = None

        if self.recognizer:
            self._is_sending_audio = False
            try:
                self.recognizer.stop()
            except Exception as e:
                print(f"[ASR ERROR] 停止识别器时出错: {e}")
            finally:
                self.recognizer = None

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        print("[ASR] 会话清理完成...")
