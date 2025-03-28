// 評価タイプの列挙型
export enum RunEvalType {
  qa = "qa",
  comparison = "comparison",
  custom = "custom"
}

// 正確性の評価（回答と参照回答の比較）のプロンプト
export const correctnessInstructions = `You are a teacher grading a quiz. 

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

// 関連性の評価（回答と質問の比較）のプロンプト
export const relevanceInstructions = `You are a teacher grading a quiz. 

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

// 根拠の評価（回答と取得文書の比較）のプロンプト
export const groundedInstructions = `You are a teacher grading a quiz. 

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

// 検索関連性の評価（取得文書と質問の比較）のプロンプト
export const retrievalRelevanceInstructions = `You are a teacher grading a quiz. 

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

// RAGのプロンプトテンプレート
export const ragPromptTemplate = `
あなたはLilian Wengのブログに関する質問に答えるアシスタントです。
以下の情報を使って、質問に答えてください。

コンテキスト:
{context}

質問: {question}

回答:`;

// PDFファイルのパス
export const pdfFiles = [
  "documents/# LLMエージェント.pdf",
  "documents/# プロンプトエンジニアリング.pdf",
  "documents/# LLMに対する敵対的攻撃.pdf",
];

// テスト用データ
export const testData = [
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

// LangSmithのデータセット名
export const datasetName = "Lilian Weng Blogs Q&A";

// プロジェクト名
export const projectName = "rag-evaluation-experiment"; 