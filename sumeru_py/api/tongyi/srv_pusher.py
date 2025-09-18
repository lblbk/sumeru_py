import requests

class FastAPIPusher:
    def __init__(self, server_url: str):
        if not server_url.endswith('/'):
            server_url += '/'
        self.endpoint = f"{server_url}push/classification"
        print(f"[Pusher] 将向此端点推送数据: {self.endpoint}")

    def push(self, json_data: str):
        """
        通过 HTTP POST 请求将字典数据发送到FastAPI服务器。
        """
        try:
            headers = {
                'Content-Type': 'application/json; charset=utf-8'
            }
            
            response = requests.post(
                self.endpoint,
                data=json_data.encode('utf-8'),
                headers=headers,
                timeout=3 # 设置3秒超时，避免网络问题卡住
            )
            # 检查服务器是否成功处理了请求
            if response.status_code == 200:
                # print(f"[Pusher] 数据推送成功.") # 可以取消注释用于详细调试
                pass
            else:
                print(f"[Pusher] 推送失败，服务器返回状态码: {response.status_code}, 响应: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"[Pusher] 推送时发生网络错误: {e}")
