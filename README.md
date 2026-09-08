# Bedrock / Strands Agents Prompt Caching Validation

このリポジトリは、Amazon Bedrock上のClaude Sonnet 5とStrands Agentsを使い、
4層のPrompt Caching戦略を実際のAPI呼び出しで検証するための実験コードです。

アプリケーションとして提供することは目的にしていません。キャッシュポイントの配置、
会話履歴の引き継ぎ、Bedrockが返すトークン使用量を確認するための最小構成です。

## 検証するキャッシュ層

Anthropic ClaudeのPrompt Cachingでは、キャッシュ対象が `tools`、`system`、
`messages` の順に連結されます。この実験では、Bedrock Converseを通じて
その順序に4つのキャッシュポイントを置きます。

| 層 | キャッシュする範囲 | ブレークポイントの位置 |
| --- | --- | --- |
| **L1** | Tool定義と全テナント共通のsystem prompt | 共通system promptの直後 |
| **L2** | L1とテナント固有のsystem prompt | テナント固有system promptの直後 |
| **L3** | L2と前回のモデル呼び出し時の末尾までの会話履歴 | 前回末尾のuser messageの直後 |
| **L4** | L3以降から今回のモデル呼び出し時の末尾までの会話履歴 | 今回末尾のuser messageの直後 |

`TurnBoundaryCachePointHook` はモデル呼び出し直前に実行されます。前回末尾の
ブレークポイントを残し、今回の末尾へ新しいブレークポイントを追加します。
古いブレークポイントは削除するため、会話が長くなってもmessages内の
キャッシュポイントは最大2点です。L1とL2を含めても、Bedrockの上限である4点を超えません。

この実験はTool定義をL1に含めますが、Toolは呼び出しません。モデルへの入力には
`Do not call a tool.` を含め、予期しない `toolUse` が返った場合は失敗させます。

## 必要なもの

- Python 3.11以上
- Poetry
- AWS CLIと有効なAWS認証情報
- `us-east-1` で利用可能なClaude Sonnet 5のBedrockモデルアクセス

`agent_factory.py` は `BedrockModel(..., region_name="us-east-1")` を指定します。
AWS CLIのデフォルトリージョンとは関係なく、すべての検証リクエストは`us-east-1`へ送信されます。

実行前にAWS認証を確認します。

```bash
aws sts get-caller-identity
```

必要なら、使用するプロファイルを指定します。

```bash
export AWS_PROFILE=your-profile-name
```

## セットアップ

依存関係をインストールします。

```bash
poetry install
```

リポジトリ直下に`.env`を作成し、Bedrockで利用可能なモデルIDを設定します。

```dotenv
BEDROCK_CLAUDE_SONNET_5_MODEL_ID=us.anthropic.claude-sonnet-5
```

`.env`にはAWSアクセスキーなどの認証情報を書かず、AWS CLIまたは実行環境の
認証情報を利用してください。`.env`はGit管理対象外です。

## 4層キャッシュの実行

次のスクリプトを実行します。

```bash
./run_cache_validation.sh
```

5分TTLを明示する場合は、次のとおりです。

```bash
./run_cache_validation.sh --ttl 5m
```

1時間TTLを使う場合は、次のとおりです。

```bash
./run_cache_validation.sh --ttl 1h
```

スクリプトはリポジトリルートへ移動してから、次のコマンドを実行します。

```bash
poetry run python test_cache_hit_rate.py "$@"
```

## 実験のリクエスト構成

検証では、まず2つのseedリクエストを順番に送り、L1からL4までのキャッシュエントリを作成します。
その後、会話履歴を分岐させた4つのリクエストを送ります。

| ケース | 送信する構成 | 期待する最長キャッシュヒット |
| --- | --- | --- |
| `seed-1` | L1 + L2 + seed message 1 | L1、L2、最初のmessages境界を作成 |
| `seed-2` | L1 + L2 + seed message 1 + assistant message 1 + seed message 2 | L3とL4を作成 |
| `L1` | L1 + 異なるL2 + new message 1 | L1 |
| `L1-L2` | L1 + L2 + new message 2 | L1からL2 |
| `L1-L3` | L1 + L2 + L3までの履歴 + new message 3 | L1からL3 |
| `L1-L4` | L1 + L2 + L4までの履歴 + new message 4 | L1からL4 |

各分岐の`new message`は異なる内容です。そのため、各ケースは前の層までのキャッシュを
読み込み、その後の新しい入力をキャッシュへ書き込みます。

## 成功時の出力

実行すると、Bedrockが返す使用量をケースごとに表示し、同じ内容を
`cache_layer_usage.jsonl`へ保存します。

```text
case       cache read  cache write   uncached  expected hit
seed-1              0         9105          2  create L1, L2, and L4
seed-2           9105         1294          2  create L3 and L4
L1               6429         2616          2  Tool and common system prompt
L1-L2            7750         1295          2  L1 and tenant-specific system prompt
L1-L3            9105         1234          2  L1, L2, and the previous model-call boundary
L1-L4           10399         1234          2  L1, L2, L3, and the previous latest-user boundary
PASS: L1, L1-L2, L1-L3, and L1-L4 cache reuse was observed.
```

トークン数は、モデルのトークナイズ結果や実験用のrun IDによって変動します。成功の判定には、
絶対値ではなく次の関係を使います。

```text
L1.cache_read_input_tokens
  < L1-L2.cache_read_input_tokens
  < L1-L3.cache_read_input_tokens
  < L1-L4.cache_read_input_tokens
```

加えて、`seed-1`はキャッシュ書込みを行い、`seed-2`は既存キャッシュの読込みと
新規書込みの両方を行う必要があります。これらの条件を満たさない場合、
`test_cache_hit_rate.py`は`AssertionError`で終了します。

## ローカルテスト

Bedrockを呼び出さずに、ブレークポイントの配置と実測値の判定ロジックを確認するには、
次を実行します。

```bash
poetry run pytest
```

このテストはAWS認証情報もモデルアクセスも必要としません。

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `agent_factory.py` | L1とL2を含むStrands Agentを作成する |
| `cache_hooks.py` | 前回と今回の会話境界へL3とL4を配置するHook |
| `cache_config.py` | キャッシュTTLの設定 |
| `test_cache_hit_rate.py` | Bedrockへ6リクエストを送り、4層の利用を実測する |
| `run_cache_validation.sh` | `.env`を確認して検証スクリプトを実行する |
| `tests/` | Hook、Agent生成、キャッシュ使用量の判定ロジックのテスト |
