# -*- coding: utf-8 -*-
"""
CardioAI - 模块3: AI 语音问答助手 (Flask)
===========================================
流程: 用户文字问题
  -> DeepSeek(ChatOpenAI, SystemPrompt="专业心血管健康顾问") 生成文字回答
  -> CosyVoice(SpeechSynthesizer.call 同步调用) 合成完整 MP3 音频
  -> base64 编码，返回 JSON {"text_answer": ..., "audio_base64": ...}

API Key 从项目根目录的 .env 读取（python-dotenv）:
  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
  DASHSCOPE_API_KEY

运行: python module3_voice_assistant/voice_assistant_app.py
访问: http://127.0.0.1:5001/
"""

import base64
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# 加载项目根目录的 .env（存放 DeepSeek / DashScope 的 API Key）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = "专业心血管健康顾问"
# CosyVoice 单次合成文本有长度限制，超长回答截断用于合成（响应仍返回完整 text_answer）
TTS_MAX_CHARS = 500

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# 惰性单例：避免缺少 API Key 时 import 即崩溃；LLM 连接可复用
_llm = None


def get_llm() -> ChatOpenAI:
    """初始化 DeepSeek 大模型（通过 langchain-openai 调用，走 OpenAI 兼容协议）。"""
    global _llm
    if _llm is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 中填写")
        _llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.3,
        )
    return _llm


def create_synthesizer() -> SpeechSynthesizer:
    """创建 CosyVoice 语音合成器（cosyvoice-v2 / longxiaochun_v2 / MP3）。

    注意：必须每次请求新建实例。dashscope 的 SpeechSynthesizer 在 call() 结束后
    会关闭 WebSocket 但不重置 self.ws，同一实例第二次 call() 会抛
    "WebSocket connection is not established or has been closed"。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，请在项目根目录 .env 中填写")
    import dashscope

    dashscope.api_key = api_key
    return SpeechSynthesizer(
        model="cosyvoice-v2",
        voice="longxiaochun_v2",
        format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
    )


@app.route("/")
def index():
    return render_template("voice_index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """接收用户问题 -> DeepSeek 回答 -> CosyVoice 合成音频 -> base64 返回。"""
    data = request.get_json(force=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    try:
        # 1) DeepSeek 文字回答
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
        text_answer = str(response.content).strip()
        if not text_answer:
            return jsonify({"error": "DeepSeek 未返回有效回答"}), 502

        # 2) CosyVoice 将完整文字回答同步合成为 MP3 音频（call 返回 bytes）
        synthesizer = create_synthesizer()  # 每次新建，避免实例复用 bug
        tts_text = text_answer[:TTS_MAX_CHARS]  # 超长截断，防 CosyVoice 长度限制
        audio_bytes = synthesizer.call(tts_text)

        # 3) Base64 编码
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return jsonify({"text_answer": text_answer, "audio_base64": audio_base64})

    except Exception as e:  # noqa: BLE001 - 统一转为可读的 500
        return jsonify({"error": f"语音问答服务失败: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
