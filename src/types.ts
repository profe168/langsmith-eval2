import { RunEvalType } from "./constants.js";

// 評価結果の型定義
export interface EvaluationResult {
  score: number;
  value: boolean;
  reasoning: string;
}

// LangSmithの例の型定義
export interface Example {
  id: string;
  inputs: {
    question: string;
  };
  outputs?: {
    answer: string;
  };
}

// テストデータの型定義
export interface TestItem {
  question: string;
  answer: string;
}

// 評価器の型定義
export interface Evaluator {
  name: string;
  evaluationType: RunEvalType;
  llm: any;
  prompt: string;
  parseResult: (result: string) => EvaluationResult;
  inputVariables?: string[];
}

// RAG結果の型定義
export interface RagResult {
  question: string;
  documents: any[];
  answer: string;
  facts: string;
}

// 評価結果サマリーの型定義
export interface EvaluationSummary {
  total: number;
  correctness: number;
  relevance: number;
  grounded: number;
  retrieval_relevance: number;
}

// 評価器オプションの型定義
export interface EvaluatorOptions {
  evaluationName: string;
  evaluationType: RunEvalType;
  evaluatorLLM: any;
  prompt: string;
  outputParser: {
    parseResult: (result: string) => EvaluationResult;
  };
  inputVariables?: string[];
} 