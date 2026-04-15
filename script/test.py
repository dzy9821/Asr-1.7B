import base64
import os
import httpx
from openai import OpenAI

# 临时清除代理环境变量（相当于 unset http_proxy 和 https_proxy）
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# 读取本地音频文件并转换为 Base64
local_audio_path = "/home/d1465299/funasr-nano/data/近远场测试.wav"  # 修改为您的本地文件路径

with open(local_audio_path, "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

# 创建不使用代理的 httpx 客户端
http_client = httpx.Client()

# Initialize client
client = OpenAI(
    base_url="http://148.148.52.127:15002/v1",
    api_key="EMPTY",
    http_client=http_client
)

# Create multimodal chat completion request
response = client.chat.completions.create(
    model="Qwen3-ASR-1.7B",  # 根据 curl 返回的模型 ID
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_base64}"
                    }
                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
