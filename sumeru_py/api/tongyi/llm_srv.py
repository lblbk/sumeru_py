import os
import time
import json
import re
import dashscope
from dashscope import Generation

def clean_json_with_markdown(raw_str: str):
    """从可能包含Markdown格式的字符串中提取并解析JSON。"""
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, raw_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("提取的内容不是有效的JSON格式")
    else:
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError:
            raise ValueError("输入字符串不是有效的JSON格式")

class LLMService:
    """
    负责所有与大语言模型（LLM）相关的文本处理任务。
    这个类是“无状态”的，不依赖于语音输入或输出。
    """
    def __init__(self, model: str = "qwen-plus"):
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.model = model

        # 任务1：JSON分类的系统提示
        self._classification_prompt = [
            {
                "role": "system",
                "content": """你是一个任务机器人。你的任务是分析用户的话，并严格按照以下JSON格式进行回复：

                规则定义
                - 话语中包含或暗示了任何关于【水果】类型（例如苹果、香蕉、吃水果），你就必须回复数字 '1'。
                - 话语中包含或暗示了任何关于【水】或【口渴】类型，你就必须回复数字 '2'。
                - 话语中包含或暗示了任何关于【工具】类型（例如锤子、扳手、修理），你就必须回复数字 '3'。
                - 话语中内容不确定或者不存在的类型归于【其他】类型，回复数字 '0'。

                输出格式
                你的回复必须是一个严格的JSON字符串，格式如下：
                {"category": "识别出的分类名称", "code": 对应的代码数字}
                """,
            }
        ]

        # 任务2：生活助手的系统提示
        self._conversation_prompt = [
            {
                "role": "system", 
                "content": """你是一个生活小助手，名字是小美，请使用中文回复用户提问的问题，
                回复尽量简洁明了，不要添加任何表情符号或者特殊字符""",
            }
        ]

    def _call_llm(self, messages):
        """通用的LLM调用方法。"""
        response = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
        )
        return response.output.choices[0].message.content

    def get_classification(self, user_input: str) -> dict:
        """
        执行JSON分类任务。
        :param user_input: 用户输入的文本。
        :return: 一个包含分类结果的字典。
        """
        try:
            current_messages = self._classification_prompt + [{"role": "user", "content": user_input}]
            assistant_output = self._call_llm(current_messages)
            ret = clean_json_with_markdown(assistant_output)
            ret['input'] = user_input
            ret['id'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            return ret
        except Exception as e:
            print(f"   [LLM-JSON] 调用文本理解模型时出错: {e}")
            return None

    def get_conversational_response(self, user_input: str) -> str:
        """
        执行对话生成任务。
        :param user_input: 用户输入的文本。
        :return: 生成的对话式回复字符串。
        """
        try:
            # current_messages = self._conversation_prompt + [{"role": "user", "content": user_input}]
            self._conversation_prompt = self._conversation_prompt + [{"role": "user", "content": user_input}]
            assistant_output = self._call_llm(self._conversation_prompt)
            self._conversation_prompt = self._conversation_prompt + [{"role": "system", "content": assistant_output}]
            return assistant_output
        except Exception as e:
            print(f"   [LLM-TTS] 生成对话时出错: {e}")
            return "抱歉，我好像遇到了一点问题。"
