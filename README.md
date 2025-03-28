# LangSmith RAG 評価

このプロジェクトは、LangSmithを使用してRetrieval-Augmented Generation (RAG)アプリケーションを評価するためのサンプルコードです。

## 概要

このアプリケーションは、Lilian Wengのブログ記事に基づく質問応答システムを構築し、以下の評価指標を使ってその性能を評価します：

1. **正確性 (Correctness)**: 生成された回答が参照回答と事実的に一致しているか
2. **関連性 (Relevance)**: 生成された回答が質問に関連しているか
3. **根拠 (Groundedness)**: 生成された回答が取得された文書に根拠があるか
4. **検索関連性 (Retrieval Relevance)**: 取得された文書が質問に関連しているか

## セットアップ

### 前提条件

- Node.js 18以上
- LangSmith APIキー
- OpenAI APIキー

### インストール

```bash
# 依存関係のインストール
npm install
```

### 環境変数の設定

`.env`ファイルを作成し、以下の環境変数を設定してください：

```
LANGSMITH_API_KEY=あなたのLangSmith APIキー
OPENAI_API_KEY=あなたのOpenAI APIキー
LANGSMITH_TRACING=true
```

## 使用方法

プロジェクトのディレクトリで以下のコマンドを実行します：

```bash
# アプリケーションの実行
npm start
```

実行すると、以下のステップが実行されます：

1. ブログ記事の読み込みとインデックス作成
2. RAGシステムの構築
3. テストデータセットの作成（または既存のものを使用）
4. 評価指標の定義
5. 評価の実行

## エラーのトラブルシューティング

TypeScriptの型定義に関するエラーが発生した場合は、依存関係を最新のものに更新してみてください：

```bash
npm install @langchain/core@latest @langchain/openai@latest langsmith@latest langchain@latest @langchain/community@latest
```

また、以下のいずれかの方法で型チェックをスキップして実行することも可能です：

```bash
# ts-nodeでスキップ
npx ts-node --transpile-only src/index.ts

# またはnodeで直接実行（トランスパイルが必要）
npx tsc 
node dist/index.js
```

## 評価結果

評価結果はLangSmithのUIで確認できます。評価が終わると、コンソールに評価結果のサマリーが表示されます。

詳細な分析や各サンプルごとの評価結果を確認するには、LangSmithのWebインターフェイスを使用してください。 