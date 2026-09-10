# /// script
# dependencies = ["anthropic", "python-dotenv"]
# ///
"""
先の回答で提示した messages 例が Anthropic API のスキーマバリデーションを
通るかを実際にリクエストして確認するテストスクリプト。
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# .env を読み込む（このファイルから見てリポジトリ直下の .env）
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が .env から読み込めませんでした")

client = anthropic.Anthropic(api_key=api_key)

messages = [
    # 1. ユーザーの最初の入力（テキストのみ）
    {
        "role": "user",
        "content": "東京の現在の天気を教えて。",
    },
    # 2. AIの出力：テキスト + tool_use（関数呼び出し）
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "東京の天気を調べます。"},
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq917835lq9",
                "name": "get_weather",
                "input": {"location": "Tokyo, Japan", "unit": "celsius"},
            },
        ],
    },
    # 3. ユーザー側から返す tool_result
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                "content": [{"type": "text", "text": '{"temp": 29, "condition": "晴れ"}'}],
            }
        ],
    },
    # 4. AIの最終出力
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "東京は現在29℃、晴れです。"}],
    },
    # 5. 追加のユーザー質問
    {
        "role": "user",
        "content": "明日の予定と天気も一緒に教えて。",
    },
    # 6. AIが複数tool_useを並列で呼ぶケース
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "予定と天気を両方調べますね。"},
            {
                "type": "tool_use",
                "id": "toolu_02B",
                "name": "get_calendar",
                "input": {"date": "2026-09-05"},
            },
            {
                "type": "tool_use",
                "id": "toolu_03C",
                "name": "get_weather",
                "input": {"location": "Tokyo, Japan", "unit": "celsius", "date": "2026-09-05"},
            },
        ],
    },
    # 7. 複数tool_resultを1つのuserメッセージにまとめて返す
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_02B",
                "content": [{"type": "text", "text": '{"events": ["10:00 会議", "14:00 歯科"]}'}],
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_03C",
                "content": [{"type": "text", "text": '{"temp": 26, "condition": "曇り"}'}],
            },
        ],
    },
]

try:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=[
            {
                "name": "get_weather",
                "description": "指定地域の天気を取得する",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        "date": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_calendar",
                "description": "指定日の予定を取得する",
                "input_schema": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}},
                    "required": ["date"],
                },
            },
        ],
        messages=messages,
    )
    print("=== SUCCESS ===")
    print("stop_reason:", response.stop_reason)
    for block in response.content:
        print("-", block.type, getattr(block, "text", getattr(block, "input", None)))
except anthropic.APIStatusError as e:
    print("=== API STATUS ERROR ===")
    print("status_code:", e.status_code)
    print("body:", e.body)
except anthropic.APIError as e:
    print("=== API ERROR ===")
    print(e)
