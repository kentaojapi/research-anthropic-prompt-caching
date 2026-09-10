# /// script
# dependencies = ["anthropic", "python-dotenv"]
# ///
"""
system prompt 内に2つの明示的キャッシュブレークポイント (`cache_control`) を設定し、
実際にAnthropic API (claude-sonnet-4-5) へ同一リクエストを2回連続で送信して、
1回目は cache_creation_input_tokens が発生し、2回目は cache_read_input_tokens が
発生する（＝キャッシュが効いている）ことを確認するスクリプト。

参考:
https://platform.claude.com/docs/ja/build-with-claude/prompt-caching
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が .env から読み込めませんでした")

client = anthropic.Anthropic(api_key=api_key)

MODEL = "claude-sonnet-4-5"

# --- ブレークポイント1: めったに変わらない「役割・指示」セクション -----------------
# Claude Sonnet 4.5 のキャッシュ最小長は1,024トークンなので、しきい値を確実に
# 超えるようそれぞれのブロックを十分な長さにしている。
_INSTRUCTION_PARAGRAPH = (
    "You are a senior support engineer for a fictional SaaS product called "
    "TripleTail Analytics. You must always answer in a professional, concise "
    "tone, cite the relevant internal policy number when applicable, and never "
    "invent numbers that are not present in the provided context. "
)
SYSTEM_BLOCK_INSTRUCTIONS = (_INSTRUCTION_PARAGRAPH * 60).strip()

# --- ブレークポイント2: 毎日更新されうるが数分〜数時間は使い回す「ナレッジベース」 ---
_CONTEXT_PARAGRAPH = (
    "Policy 4.2.1: Refunds are issued within 5 business days once approved. "
    "Policy 4.2.2: Enterprise plan customers receive priority queue support. "
    "Policy 4.2.3: API rate limits reset every 60 seconds per API key. "
    "Policy 4.2.4: Data exports are retained for 30 days after generation. "
)
SYSTEM_BLOCK_KNOWLEDGE = (_CONTEXT_PARAGRAPH * 60).strip()

system = [
    {
        "type": "text",
        "text": SYSTEM_BLOCK_INSTRUCTIONS,
        "cache_control": {"type": "ephemeral"},  # ブレークポイント1
    },
    {
        "type": "text",
        "text": SYSTEM_BLOCK_KNOWLEDGE,
        "cache_control": {"type": "ephemeral"},  # ブレークポイント2
    },
]

messages = [
    {"role": "user", "content": "返金までに何営業日かかりますか？"},
]


def call(label: str):
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=messages,
    )
    usage = response.usage
    print(f"=== {label} ===")
    print("stop_reason:", response.stop_reason)
    print("input_tokens:", usage.input_tokens)
    print("cache_creation_input_tokens:", usage.cache_creation_input_tokens)
    print("cache_read_input_tokens:", usage.cache_read_input_tokens)
    for block in response.content:
        if block.type == "text":
            print("text:", block.text)
    print()


if __name__ == "__main__":
    # 1回目: system の2ブロック分がキャッシュに新規書き込みされる想定
    call("1回目のリクエスト（キャッシュ書き込み想定）")
    # 2回目: 同一プレフィックスのため、キャッシュ読み取りに切り替わる想定
    call("2回目のリクエスト（キャッシュ読み取り想定）")
