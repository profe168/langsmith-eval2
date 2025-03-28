import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Client, type Dataset } from "langsmith";
import { PromptTemplate } from "@langchain/core/prompts";
import { formatDocumentsAsString } from "langchain/util/document";
import { Document } from "langchain/document";
import * as dotenv from "dotenv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import * as fs from "fs";
import { 
  RunEvalType,
  correctnessInstructions,
  relevanceInstructions,
  groundedInstructions,
  retrievalRelevanceInstructions,
  ragPromptTemplate,
  pdfFiles,
  testData,
  datasetName,
  projectName
} from "./constants.js";
import {
  EvaluationResult,
  Example,
  TestItem,
  RagResult,
  EvaluationSummary
} from "./types.js";
import {
  createEvaluator,
  loadDocumentsFromPDF,
  createVectorStore,
  buildRagChain,
  evaluateWithEvaluator
} from "./functions.js";

// 環境変数の読み込み
dotenv.config();

async function main() {
  console.log("RAG評価を開始します");

  // 1. PDFファイルの読み込みとインデックス作成
  console.log("ドキュメントを読み込んでいます...");
  const docs = await loadDocumentsFromPDF(pdfFiles);
  console.log(`${docs.length}個のドキュメントを読み込みました`);

  // ベクトルストアの作成
  const vectorstore = await createVectorStore(docs);
  console.log(`ドキュメントをベクトル化しました`);

  // リトリーバーの作成
  const retriever = vectorstore.asRetriever(6);

  // 2. RAGチェーンの構築
  const ragBot = buildRagChain(retriever, ragPromptTemplate);

  // 3. テスト用データセットの作成
  console.log("テスト用データセットを作成しています...");

  // 入力と出力の形式を整える
  const inputs = testData.map((item: TestItem) => ({ question: item.question }));
  const outputs = testData.map((item: TestItem) => ({ answer: item.answer }));

  // LangSmithクライアントを初期化
  const client = new Client();
  
  // データセットの作成（または既存のデータセットを使用）
  let dataset: Dataset;
  
  try {
    // 既存のデータセットを検索
    const datasets = await client.listDatasets({
      datasetName
    });
    
    let foundDataset = null;
    for await (const ds of datasets) {
      if (ds.name === datasetName) {
        foundDataset = ds;
        break;
      }
    }
    
    if (foundDataset) {
      dataset = foundDataset;
      console.log(`既存のデータセット "${datasetName}" を使用します (ID: ${dataset.id})`);
    } else {
      // 新しいデータセットを作成
      dataset = await client.createDataset(datasetName);
      console.log(`新しいデータセット "${datasetName}" を作成しました (ID: ${dataset.id})`);
      
      // データセットにサンプルを追加
      for (let i = 0; i < inputs.length; i++) {
        await client.createExample(inputs[i], outputs[i], {
          datasetId: dataset.id
        });
      }
      console.log(`${inputs.length}個のサンプルをデータセットに追加しました`);
    }
  } catch (error) {
    console.error("データセットの作成中にエラーが発生しました:", error);
    return;
  }

  // 4. 評価指標の定義
  console.log("評価指標を定義しています...");

  // 正確性の評価（回答と参照回答の比較）
  const correctnessEvaluator = createEvaluator({
    evaluationName: "correctness",
    evaluationType: RunEvalType.comparison,
    evaluatorLLM: new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0,
    }).bind({
      response_format: { type: "json_object" }
    }),
    prompt: correctnessInstructions,
    outputParser: {
      parseResult: (result: string) => {
        try {
          const parsed = JSON.parse(result);
          return {
            score: parsed.correct === true ? 1 : 0,
            value: parsed.correct,
            reasoning: parsed.explanation,
          };
        } catch (e) {
          return {
            score: 0,
            value: false,
            reasoning: "Failed to parse evaluator output: " + result,
          };
        }
      }
    }
  });

  // 関連性の評価（回答と質問の比較）
  const relevanceEvaluator = createEvaluator({
    evaluationName: "relevance",
    evaluationType: RunEvalType.qa,
    evaluatorLLM: new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0,
    }).bind({
      response_format: { type: "json_object" }
    }),
    prompt: relevanceInstructions,
    outputParser: {
      parseResult: (result: string) => {
        try {
          const parsed = JSON.parse(result);
          return {
            score: parsed.relevant === true ? 1 : 0,
            value: parsed.relevant,
            reasoning: parsed.explanation,
          };
        } catch (e) {
          return {
            score: 0,
            value: false,
            reasoning: "Failed to parse evaluator output: " + result,
          };
        }
      }
    }
  });

  // 根拠の評価（回答と取得文書の比較）
  const groundedEvaluator = createEvaluator({
    evaluationName: "grounded",
    evaluationType: RunEvalType.custom,
    evaluatorLLM: new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0,
    }).bind({
      response_format: { type: "json_object" }
    }),
    prompt: groundedInstructions,
    inputVariables: ["facts", "answer"],
    outputParser: {
      parseResult: (result: string) => {
        try {
          const parsed = JSON.parse(result);
          return {
            score: parsed.grounded === true ? 1 : 0,
            value: parsed.grounded,
            reasoning: parsed.explanation,
          };
        } catch (e) {
          return {
            score: 0,
            value: false,
            reasoning: "Failed to parse evaluator output: " + result,
          };
        }
      }
    }
  });

  // 検索関連性の評価（取得文書と質問の比較）
  const retrievalRelevanceEvaluator = createEvaluator({
    evaluationName: "retrieval_relevance",
    evaluationType: RunEvalType.custom,
    evaluatorLLM: new ChatOpenAI({
      modelName: "gpt-4o-mini",
      temperature: 0,
    }).bind({
      response_format: { type: "json_object" }
    }),
    prompt: retrievalRelevanceInstructions,
    inputVariables: ["facts", "question"],
    outputParser: {
      parseResult: (result: string) => {
        try {
          const parsed = JSON.parse(result);
          return {
            score: parsed.relevant === true ? 1 : 0,
            value: parsed.relevant,
            reasoning: parsed.explanation,
          };
        } catch (e) {
          return {
            score: 0,
            value: false,
            reasoning: "Failed to parse evaluator output: " + result,
          };
        }
      }
    }
  });

  // 5. 評価の実行
  console.log("評価を実行しています...");

  // 評価対象の関数
  const targetFunc = async (input: any) => {
    try {
      const result = await ragBot(input.question);
      return result;
    } catch (error) {
      console.error("RAG処理中にエラーが発生しました:", error);
      return {
        answer: "エラーが発生しました",
        documents: [],
        facts: ""
      };
    }
  };

  try {
    // 評価の実行 - LangSmithのAPIを直接使用する代わりに、サンプルごとに実行して手動で評価を行う
    console.log("データセットの各サンプルに対してRAGを実行します...");
    
    const results = [];
    
    // データセットからサンプルを取得
    const examples = client.listExamples({ datasetId: dataset.id });
    
    for await (const example of examples) {
      const typedExample = example as unknown as Example;
      console.log(`質問を処理中: ${typedExample.inputs.question}`);
      
      // RAGを実行
      const prediction = await targetFunc(typedExample.inputs);
      
      if (!typedExample.outputs) {
        console.log(`サンプル ${typedExample.id} には出力データがありません。スキップします。`);
        continue;
      }
      
      // 各評価器で評価
      const correctnessResult = await evaluateWithEvaluator(
        correctnessEvaluator,
        {
          question: typedExample.inputs.question,
          answer: prediction.answer,
          reference_answer: typedExample.outputs.answer
        }
      );
      
      const relevanceResult = await evaluateWithEvaluator(
        relevanceEvaluator,
        {
          question: typedExample.inputs.question,
          answer: prediction.answer
        }
      );
      
      const groundedResult = await evaluateWithEvaluator(
        groundedEvaluator,
        {
          facts: prediction.facts,
          answer: prediction.answer
        }
      );
      
      const retrievalRelevanceResult = await evaluateWithEvaluator(
        retrievalRelevanceEvaluator,
        {
          facts: prediction.facts,
          question: typedExample.inputs.question
        }
      );
      
      // 結果を保存
      results.push({
        question: typedExample.inputs.question,
        reference_answer: typedExample.outputs.answer,
        generated_answer: prediction.answer,
        correctness: correctnessResult,
        relevance: relevanceResult,
        grounded: groundedResult,
        retrieval_relevance: retrievalRelevanceResult
      });
      
      // トレースをLangSmithに送信
      await client.createRun({
        name: "RAG Evaluation",
        inputs: typedExample.inputs,
        outputs: prediction,
        project_name: projectName,
        run_type: "chain"
      });
    }
    
    console.log("評価が完了しました");
    
    // 集計結果を表示
    const summary: EvaluationSummary = {
      total: results.length,
      correctness: results.filter(r => r.correctness.value).length,
      relevance: results.filter(r => r.relevance.value).length,
      grounded: results.filter(r => r.grounded.value).length,
      retrieval_relevance: results.filter(r => r.retrieval_relevance.value).length
    };
    
    console.log("評価結果サマリー:");
    console.log(`合計サンプル数: ${summary.total}`);
    console.log(`正確性 (Correctness): ${summary.correctness}/${summary.total} (${(summary.correctness/summary.total*100).toFixed(2)}%)`);
    console.log(`関連性 (Relevance): ${summary.relevance}/${summary.total} (${(summary.relevance/summary.total*100).toFixed(2)}%)`);
    console.log(`根拠 (Grounded): ${summary.grounded}/${summary.total} (${(summary.grounded/summary.total*100).toFixed(2)}%)`);
    console.log(`検索関連性 (Retrieval Relevance): ${summary.retrieval_relevance}/${summary.total} (${(summary.retrieval_relevance/summary.total*100).toFixed(2)}%)`);
    
    console.log("詳細な評価結果はLangSmithのUIで確認してください");
  } catch (error) {
    console.error("評価中にエラーが発生しました:", error);
  }
}

// メイン関数の実行
main().catch(console.error); 