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

// 環境変数の読み込み
dotenv.config();

// 評価結果の型定義
interface EvaluationResult {
  score: number;
  value: boolean;
  reasoning: string;
}

// 例の型定義
interface Example {
  id: string;
  inputs: {
    question: string;
  };
  outputs?: {
    answer: string;
  };
}

// 評価タイプの列挙型
enum RunEvalType {
  qa = "qa",
  comparison = "comparison",
  custom = "custom"
}

// カスタム評価器を作成する関数
function createEvaluator(options: {
  evaluationName: string;
  evaluationType: RunEvalType;
  evaluatorLLM: any;
  prompt: string;
  outputParser: {
    parseResult: (result: string) => EvaluationResult;
  };
  inputVariables?: string[];
}) {
  return {
    name: options.evaluationName,
    evaluationType: options.evaluationType,
    llm: options.evaluatorLLM,
    prompt: options.prompt,
    parseResult: options.outputParser.parseResult,
    inputVariables: options.inputVariables
  };
}

async function main() {
  console.log("RAG評価を開始します");

  // 1. PDFファイルの読み込みとインデックス作成
  const pdfFiles = [
    "documents/agent.pdf",
    "documents/prompt-engineering.pdf",
    "documents/adversarial-attack.pdf",
  ];

  console.log("ドキュメントを読み込んでいます...");
  const docs = [];
  
  // PDFファイルを確認
  for (const filePath of pdfFiles) {
    if (!fs.existsSync(filePath)) {
      console.error(`ファイルが見つかりません: ${filePath}`);
      continue;
    }
    
    const loader = new PDFLoader(filePath);
    const loadedDocs = await loader.load();
    docs.push(...loadedDocs);
  }

  console.log(`${docs.length}個のドキュメントを読み込みました`);

  // テキスト分割
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 0
  });

  const docSplits = await textSplitter.splitDocuments(docs);
  console.log(`${docSplits.length}個のチャンクに分割しました`);

  // ベクトルストアの作成
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(
    docSplits,
    embeddings
  );

  // リトリーバーの作成
  const retriever = vectorstore.asRetriever(6);

  // 2. RAGチェーンの構築
  const prompt = PromptTemplate.fromTemplate(`
あなたはLilian Wengのブログに関する質問に答えるアシスタントです。
以下の情報を使って、質問に答えてください。

コンテキスト:
{context}

質問: {question}

回答:`);

  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0
  });

  // RAGチェーンの構築方法を変更
  const retrievalChain = async (input: { question: string }) => {
    const docs = await retriever.getRelevantDocuments(input.question);
    const context = formatDocumentsAsString(docs);
    const formattedPrompt = await prompt.format({
      context: context,
      question: input.question
    });
    const response = await llm.invoke(formattedPrompt);
    return response.content as string;
  };

  // このチェーンをラップして、必要な形式で出力を返す関数
  const ragBot = async (question: string) => {
    const docs = await retriever.getRelevantDocuments(question);
    const answer = await retrievalChain({ question });
    return {
      question,
      documents: docs,
      answer,
      // 評価のために追加
      facts: docs.map((doc: Document) => doc.pageContent).join("\n")
    };
  };

  // 3. テスト用データセットの作成
  console.log("テスト用データセットを作成しています...");
  const testData = [
    {
      question: "LLMエージェントとは何ですか？",
      answer: "LLMエージェントは、大規模言語モデル（LLM）を中心として構築され、環境と相互作用して指示に従い目標を達成するシステムです。エージェントは通常、計画立案、実行、ツール使用、記憶などの能力を持ち、複雑なタスクを実行できます。"
    },
    {
      question: "プロンプトエンジニアリングの主要な技術は何ですか？",
      answer: "プロンプトエンジニアリングの主要な技術には、ゼロショット推論、フューショット学習、チェーン・オブ・ソート、自己一貫性、指示の微調整などがあります。これらの技術は、LLMの出力の質と適切さを向上させるために使用されます。"
    },
    {
      question: "LLMに対する敵対的攻撃の種類を説明してください。",
      answer: "LLMに対する敵対的攻撃には、プロンプトインジェクション、ジェイルブレイク、データ抽出、有害出力の生成などがあります。これらの攻撃は、モデルの安全対策をバイパスし、望ましくない行動を引き起こすことを目的としています。"
    }
  ];

  // 入力と出力の形式を整える
  const inputs = testData.map(item => ({ question: item.question }));
  const outputs = testData.map(item => ({ answer: item.answer }));

  // LangSmithクライアントを初期化
  const client = new Client();
  
  // データセットの作成（または既存のデータセットを使用）
  const datasetName = "Lilian Weng Blogs Q&A";
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
  const correctnessInstructions = `You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please return your response in JSON format with the following structure:
{
  "correct": boolean,
  "explanation": "your reasoning here"
}`;

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
  const relevanceInstructions = `You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please return your response in JSON format with the following structure:
{
  "relevant": boolean,
  "explanation": "your reasoning here"
}`;

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
  const groundedInstructions = `You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please return your response in JSON format with the following structure:
{
  "grounded": boolean,
  "explanation": "your reasoning here"
}`;

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
  const retrievalRelevanceInstructions = `You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please return your response in JSON format with the following structure:
{
  "relevant": boolean,
  "explanation": "your reasoning here"
}`;

  const retrievalRelevanceEvaluator = createEvaluator({
    evaluationName: "retrieval_relevance",
    evaluationType: RunEvalType.custom,
    evaluatorLLM: new ChatOpenAI({
      modelName: "gpt-4o",
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
        project_name: "rag-evaluation-experiment",
        run_type: "chain"
      });
    }
    
    console.log("評価が完了しました");
    
    // 集計結果を表示
    const summary = {
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

// 評価器を使って評価を実行する関数
async function evaluateWithEvaluator(evaluator: any, inputs: any): Promise<EvaluationResult> {
  try {
    // プロンプトの作成
    let prompt = evaluator.prompt;
    
    // 入力変数を置換
    Object.entries(inputs).forEach(([key, value]) => {
      prompt = prompt.replace(`{${key}}`, value as string);
    });
    
    // LLMで評価を実行
    const result = await evaluator.llm.invoke(prompt);
    
    // 結果をパース
    return evaluator.parseResult(result);
  } catch (error) {
    console.error("評価中にエラーが発生しました:", error);
    return {
      score: 0,
      value: false,
      reasoning: "評価中にエラーが発生しました: " + String(error)
    };
  }
}

// メイン関数の実行
main().catch(console.error); 