from sumeru_py.system.linux.volume_ctl import PulseAudioVolumeController

if __name__ == "__main__":
    # 控制系统音量
    sys_vol = PulseAudioVolumeController.for_system()
    if sys_vol.get_volume() != 150:
        sys_vol.set_volume(150)

    # # 控制特定应用音量
    app_vol = PulseAudioVolumeController.for_app("ALSA plug-in [python3.11]")
    if app_vol.is_active():
        print("应用音量:", app_vol.get_volume())
        app_vol.set_volume(40)
