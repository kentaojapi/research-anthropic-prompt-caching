# Anthropic Messages スキーマ動作確認

`client.messages.create` に渡す `messages` 配列に、text / tool_use / tool_result
など複数のcontentタイプを混在させた例が、実際にAnthropic API
（スキーマバリデーション）を通過するかを確認するためのスクリプト。

## 内容

`messages` には以下の7件を含む:

1. `user`: テキストのみ
2. `assistant`: `text` + `tool_use`（単発ツール呼び出し）
3. `user`: `tool_result`
4. `assistant`: `text`（最終回答）
5. `user`: テキストのみ（追加質問）
6. `assistant`: `text` + `tool_use` を2つ（並列ツール呼び出し）
7. `user`: `tool_result` を2つまとめて返却

画像コンテンツ (`type: "image"`) はトークン消費を避けるため未検証。

## 実行方法

`ANTHROPIC_API_KEY` をリポジトリ直下の `.env` に設定した上で、以下を実行:

```bash
uv run experiments/anthropic_messages_schema_check/test_messages.py
```

スクリプト冒頭のPEP 723インラインメタデータ（`anthropic`, `python-dotenv`）
により、`uv run` が依存関係を自動解決する。リポジトリの `pyproject.toml` の
依存には影響しない。

## 実行結果（検証済み）

`model="claude-sonnet-4-5"` に対してリクエストし、`stop_reason: end_turn` で
正常応答を確認済み。スキーマバリデーションエラーは発生しなかった。
