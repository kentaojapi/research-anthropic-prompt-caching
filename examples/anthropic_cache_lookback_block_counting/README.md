# Anthropic Prompt Caching の20位置lookback

Anthropic Prompt Caching は、過去に書き込んだキャッシュエントリを、後続リクエストのcache pointから最大20位置まで遡って探す。ここでいう位置は、すべてのcontent blockを同じように数えた数ではない。公式ドキュメントは、連続する`tool_use`群を1位置、連続する`tool_result`群も1位置として扱う。

このディレクトリでは、通常のtext blockと、並列ツール呼び出しのblock群を実APIで比較する。すべての実験は、実行ごとに一意な値を静的プレフィックスへ埋め込む。別の実行が残した5分TTLのキャッシュを偶然読むことはない。

参照: <https://platform.claude.com/docs/ja/build-with-claude/prompt-caching>

## 実験一覧

| スクリプト | API経路 | 検証する内容 |
|---|---|---|
| `lookback_block_counting.py` | Anthropic Messages API | 25個の連続text blockがlookbackを使い切るか |
| `contiguous_tool_blocks.py` | Anthropic Messages API | 15個の連続`tool_use`と15個の連続`tool_result`がキャッシュを失わせるか |
| `strands_parallel_tool_lookback.py` | Strands `Agent` + AnthropicModel | Strandsの実ツールループでも同じ性質が保たれるか |

実行にはリポジトリ直下の`.env`に`ANTHROPIC_API_KEY`を設定する。

```bash
uv run examples/anthropic_cache_lookback_block_counting/lookback_block_counting.py
uv run examples/anthropic_cache_lookback_block_counting/contiguous_tool_blocks.py
uv run examples/anthropic_cache_lookback_block_counting/strands_parallel_tool_lookback.py
```

## text blockの実験

最初のリクエストで、1,861トークンのsystem prefixにcache pointを置いてキャッシュを書き込む。次のリクエストではsystem側のcache pointを外し、末尾のユーザー入力に新しいcache pointを置く。これにより、後続リクエストは新しいcache pointから過去の書き込みを探す。

対照実験では、途中のassistantメッセージにtext blockを1個だけ置いた。検証本体では、同じ位置に25個のtext blockを置いた。

| ケース | `cache_creation_input_tokens` | `cache_read_input_tokens` | 結果 |
|---|---:|---:|---|
| 最初の書き込み | 1,861 | 0 | 新規キャッシュ |
| text block 1個 | 22 | 1,861 | ヒット |
| text block 25個 | 2,028 | 0 | ミス |

25個のtext blockを置いたケースでは、前回のキャッシュ書き込みが20位置の外側にあり、読み取りが発生しなかった。少なくとも通常のtext blockは、content配列の要素ごとにlookback位置を消費する。

## 生のAnthropic APIでの並列ツール実験

次の実験では、最初のリクエストが静的system prefixにキャッシュを書き込む。次のリクエストは、同じtools定義とsystem prefixを送ったうえで、15個の連続`tool_use` blockと、それに対応する15個の連続`tool_result` blockを含める。最後のtool resultにcache pointを置く。

| リクエスト | `cache_creation_input_tokens` | `cache_read_input_tokens` | `input_tokens` |
|---|---:|---:|---:|
| 静的prefixの書き込み | 2,329 | 0 | 324 |
| tool use/resultを含む後続リクエスト | 1,398 | 2,329 | 7 |

後続リクエストは2,329トークンをキャッシュから読んだ。15件ずつの連続block群は、前回のエントリを20位置の外へ押し出していない。

## Strandsの実ツールループでの並列ツール実験

Strandsの実験では、Claudeに`lookup_record`を15回並列実行するよう指示する。Strandsが保持する会話履歴を検査し、`toolUse`を15個含むassistantメッセージが1件、`toolResult`を15個含むuserメッセージが1件であることをスクリプトが検証する。

最初の本物のuserターンには`BeforeModelCallEvent`でcache pointを置く。ツール実行後は`AfterToolsEvent`で、連続するtool result群の直後にcache pointを置く。この配置なら、tool resultを受け取る次のモデル呼び出しが、最初の書き込みをlookbackで探す。

| Strandsのモデル応答 | `cache_creation_input_tokens` | `cache_read_input_tokens` | `input_tokens` |
|---|---:|---:|---:|
| ウォームアップ | 2,832 | 0 | 3 |
| 15件のtool useを返す応答 | 0 | 2,832 | 53 |
| 15件のtool resultを処理する応答 | 1,127 | 2,832 | 7 |

最後の応答でも2,832トークンがキャッシュから読まれた。Strandsのメッセージ変換と標準ツールループは、連続した並列tool use/resultを多数の独立位置へ分解していない。

## キャッシュポイント配置への影響

大量の並列ツール呼び出しだけを理由に、会話履歴のcache pointを前方へ移動する必要はない。キャッシュミスを招くのは、並列数ではなく、前回の書き込みから20位置以上離れることである。

直列のツールループでは、assistant側のtool use群とuser側のtool result群がラウンドごとに増える。text blockなどが間に挟まる場合も、連続群として数えられなくなる。長い会話でこの距離が20位置を超えるなら、あらかじめ複数の明示的cache pointを置く。

画像やドキュメントなどの添付ファイルが多数あるケースは、このディレクトリでは未検証である。公式に連続群を1位置とする特例が明記されているのは`tool_use`と`tool_result`である。添付blockを同じようにまとめて数えられるとは仮定せず、実際の添付形式で`cache_read_input_tokens`を計測する。
