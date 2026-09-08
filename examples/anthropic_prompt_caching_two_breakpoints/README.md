# Anthropic Prompt Caching: system prompt に2つのブレークポイント

参考記事: https://platform.claude.com/docs/ja/build-with-claude/prompt-caching

## やっていること

`system` を単一の文字列ではなく **content blockの配列** にし、それぞれのブロック末尾に
`cache_control: {"type": "ephemeral"}` を付けることで、2つの独立したキャッシュ
ブレークポイントを作成する。

1. **ブレークポイント1**: `SYSTEM_BLOCK_INSTRUCTIONS`
   めったに変わらない役割・振る舞いの指示（静的プロンプトの先頭部分）。
2. **ブレークポイント2**: `SYSTEM_BLOCK_KNOWLEDGE`
   数分〜数時間単位で更新されうるナレッジベース／ポリシー文書。
   ブレークポイントを分けることで、指示部分とナレッジ部分を独立してキャッシュでき、
   どちらかだけ更新されても、もう片方のキャッシュ資産（安いキャッシュ読み取り）を
   使い回せる。

`claude-sonnet-4-5` はキャッシュ可能な最小プロンプト長が1,024トークンのため、
各ブロックを十分な長さ（同じ文の繰り返しで約数千トークン相当）にして、
確実にキャッシュ対象になるようにしている。

## 実行方法

```bash
uv run examples/anthropic_prompt_caching_two_breakpoints/prompt_caching_two_breakpoints.py
```

`.env`（リポジトリ直下）に `ANTHROPIC_API_KEY` が必要。

## 実行結果（検証済み）

同一リクエストを2回連続で送信し、`usage` フィールドで挙動を確認:

| 回数 | `cache_creation_input_tokens` | `cache_read_input_tokens` |
|------|-------------------------------|----------------------------|
| 1回目 | 8101（新規キャッシュ書き込み） | 0 |
| 2回目 | 0 | 8101（キャッシュ読み取り＝ヒット） |

2回目のリクエストで `cache_read_input_tokens` が発生しており、
system プロンプトの2ブロック分がキャッシュから読み取られたことが確認できた。
