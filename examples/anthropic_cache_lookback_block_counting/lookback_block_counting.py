# /// script
# dependencies = ["anthropic", "python-dotenv"]
# ///
"""
Anthropic Prompt Caching の「ルックバックウィンドウ = 20ブロック」における
「ブロック」の数え方を実際のAPIレスポンス(usage)から検証するスクリプト。

疑問:
  content 配列の中に複数の要素（例: text ブロックを10個並べる）を入れた場合、
  ルックバックウィンドウのカウント上、
    (a) 配列内の要素それぞれが1ブロックとして個別にカウントされるのか
    (b) content 配列全体（＝そのメッセージ）が1ブロックとして丸っとカウントされるのか
  を、実際に確かめる。

検証方法:
  1. Turn1: 大きな system ブロック（cache_control付き）だけでキャッシュを書き込む。
  2. Turn2: system の cache_control を外し（Turn1のエントリへのルックバックのみで
     ヒットするかを見たいため）、間に「assistant の content 配列に25個の独立した
     text ブロックを詰めたメッセージ」を挟んでから、新しいブレークポイントを設定する。
     - もし配列要素が個別にカウントされるなら、ブレークポイントからTurn1のエントリ
       までの距離は 1(breakpoint自身)+25(assistant配列要素)+1(user "OK") = 27 以上
       となり、20ブロックのルックバックウィンドウを超えるため cache miss になるはず。
     - もし content 配列全体が1ブロックとしてカウントされるなら、距離は
       1(breakpoint)+1(assistantメッセージ)+1(user "OK") = 3 程度で済み、
       20以内なので cache hit するはず。
  3. 比較対照として、assistant のメッセージが単一のtextブロック（配列要素1個）の
     ケースも実行し、ベースラインとして必ずヒットすることを確認する。

結果は response.usage の cache_creation_input_tokens / cache_read_input_tokens
で判定する（読み取りが発生していれば hit、発生していなければ miss）。
"""
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
import anthropic

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が .env から読み込めませんでした")

client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-sonnet-4-5"

# 実行のたびにキャッシュ内容を完全に一意にするためのnonce。
# これを system / 最終質問に埋め込むことで、「前回の実行で書き込んだキャッシュに
# 偶然完全一致してヒットしてしまう」ことを防ぎ、毎回まっさらな状態で
# ルックバックの挙動だけを検証できるようにする（TTL=5分は関係なくなる）。
RUN_ID = uuid.uuid4().hex[:12]

# claude-sonnet-4-5 の最小キャッシュ可能長(1,024トークン)を確実に超える静的ブロック
_PARAGRAPH = (
    "You are a senior support engineer for a fictional SaaS product called "
    "TripleTail Analytics. Always answer concisely and cite policy numbers. "
)
BIG_SYSTEM_TEXT = (_PARAGRAPH * 60).strip() + f" [session:{RUN_ID}]"


def show(label: str, response: anthropic.types.Message):
    usage = response.usage
    print(f"--- {label} ---")
    print("cache_creation_input_tokens:", usage.cache_creation_input_tokens)
    print("cache_read_input_tokens:", usage.cache_read_input_tokens)
    print("input_tokens:", usage.input_tokens)
    print()


# ============================================================
# Turn1: システムブロックに cache_control を付けてキャッシュを新規作成する
# ============================================================
turn1 = client.messages.create(
    model=MODEL,
    max_tokens=64,
    system=[
        {
            "type": "text",
            "text": BIG_SYSTEM_TEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": "OK"}],
)
show("Turn1: system blockへの新規キャッシュ書き込み", turn1)


# ============================================================
# Turn2-A（ベースライン）: 間に挟む assistant メッセージの content が
# 単一のtextブロック（配列要素1個）。距離は小さいので必ずヒットするはず。
# ============================================================
turn2_baseline = client.messages.create(
    model=MODEL,
    max_tokens=64,
    system=[{"type": "text", "text": BIG_SYSTEM_TEXT}],  # cache_control なし
    messages=[
        {"role": "user", "content": "OK"},
        {"role": "assistant", "content": "了解しました。"},  # content配列要素1個
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"2+2は？ [{RUN_ID}-A]",
                    "cache_control": {"type": "ephemeral"},  # 新しいブレークポイント
                }
            ],
        },
    ],
)
show("Turn2-A（ベースライン: assistant content要素1個）", turn2_baseline)


# ============================================================
# Turn2-B（検証本体）: 間に挟む assistant メッセージの content 配列に
# 25個の独立したtextブロックを詰める。
# ============================================================
filler_blocks = [{"type": "text", "text": f"補足メモ{i}"} for i in range(25)]

turn2_many_blocks = client.messages.create(
    model=MODEL,
    max_tokens=64,
    system=[{"type": "text", "text": BIG_SYSTEM_TEXT}],  # cache_control なし
    messages=[
        {"role": "user", "content": "OK"},
        {"role": "assistant", "content": filler_blocks},  # content配列要素25個
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"2+2は？ [{RUN_ID}-B]",
                    "cache_control": {"type": "ephemeral"},  # 新しいブレークポイント
                }
            ],
        },
    ],
)
show("Turn2-B（検証: assistant content要素25個）", turn2_many_blocks)


print("=== 判定 ===")
print(
    "Turn2-A cache_read>0 なら基本のルックバックが機能している証拠(ベースラインOK)。"
)
print(
    "Turn2-B で cache_read>0 のままなら「content配列は丸ごと1ブロック」、"
    "cache_read=0 (miss)になれば「配列内の要素は個別に1ブロックとしてカウントされる」。"
)
