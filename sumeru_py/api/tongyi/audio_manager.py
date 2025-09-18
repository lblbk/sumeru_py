import pyaudio

class AudioManager:
    """
    一个专门用于管理 PyAudio 实例和音频流的类。
    它是整个应用中唯一与 PyAudio直接交互的地方。
    """
    def __init__(self):
        self._p = pyaudio.PyAudio()

    def open_input_stream(self, callback: callable):
        """
        打开一个非阻塞的输入流（用于麦克风）。
        :param callback: 当有新的音频数据时，PyAudio会调用这个函数。
        :return: PyAudio stream 对象。
        """
        if not callable(callback):
            raise ValueError("Callback 必须是一个可调用对象")
            
        stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=3200,
            stream_callback=callback
        )
        return stream

    def open_output_stream(self):
        """
        打开一个阻塞的输出流（用于扬声器）。
        :return: PyAudio stream 对象。
        """
        stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )
        return stream

    def close(self):
        """
        终止 PyAudio 会话，释放所有系统资源。
        """
        self._p.terminate()
        print("[AudioManager] PyAudio 已关闭。")
