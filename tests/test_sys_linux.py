from sumeru_py.system.linux.volume_ctl import PulseAudioAppVolumeController, PulseAudioSystemVolumeController

if __name__ == "__main__":
    sys_vol = PulseAudioSystemVolumeController()
    if sys_vol.get_volume() != 150:
        sys_vol.set_volume(150)

    app_vol = PulseAudioAppVolumeController("ALSA plug-in [python3.11]")

    # 检查是否活跃
    if app_vol.is_active():
        print("当前音量:", app_vol.get_volume())
        print("是否静音:", app_vol.is_muted())

        # 设置音量
        app_vol.set_volume(60)
        print("当前音量:", app_vol.get_volume())

        # 静音 2 秒后恢复
        app_vol.mute(True)
        import time; time.sleep(2)
        app_vol.mute(False)
    else:
        print("应用未播放声音")