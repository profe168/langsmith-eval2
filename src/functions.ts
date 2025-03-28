import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";
import { Document } from "langchain/document";
import * as fs from "fs";
import { PromptTemplate } from "@langchain/core/prompts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { EvaluationResult, EvaluatorOptions, Evaluator } from "./types.js";
import { RunEvalType } from "./constants.js";

/**
 * カスタム評価器を作成する関数
 */
export function createEvaluator(options: EvaluatorOptions): Evaluator {
  return {
    name: options.evaluationName,
    evaluationType: options.evaluationType,
    llm: options.evaluatorLLM,
    prompt: options.prompt,
    parseResult: options.outputParser.parseResult,
    inputVariables: options.inputVariables
  };
}

/**
 * PDFファイルからドキュメントを読み込む関数
 */
export async function loadDocumentsFromPDF(pdfFiles: string[]): Promise<Document[]> {
  const docs: Document[] = [];
  
  for (const filePath of pdfFiles) {
    if (!fs.existsSync(filePath)) {
      console.error(`ファイルが見つかりません: ${filePath}`);
      continue;
    }
    
    const loader = new PDFLoader(filePath);
    const loadedDocs = await loader.load();
    docs.push(...loadedDocs);
  }
  
  return docs;
}

/**
 * ドキュメントをチャンクに分割してベクトルストアを作成する関数
 */
export async function createVectorStore(docs: Document[]): Promise<MemoryVectorStore> {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 0
  });

  const docSplits = await textSplitter.splitDocuments(docs);
  const embeddings = new OpenAIEmbeddings();
  
  return await MemoryVectorStore.fromDocuments(
    docSplits,
    embeddings
  );
}

/**
 * RAGチェーンを構築する関数
 */
export function buildRagChain(retriever: any, promptTemplate: string) {
  const prompt = PromptTemplate.fromTemplate(promptTemplate);
  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0
  });

  // リトリーバルチェーン
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

  // ラップ関数
  const ragBot = async (question: string) => {
    const docs = await retriever.getRelevantDocuments(question);
    const answer = await retrievalChain({ question });
    return {
      question,
      documents: docs,
      answer,
      facts: docs.map((doc: Document) => doc.pageContent).join("\n")
    };
  };

  return ragBot;
}

/**
 * 評価器を使って評価を実行する関数
 */
export async function evaluateWithEvaluator(evaluator: Evaluator, inputs: any): Promise<EvaluationResult> {
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