from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


@dataclass
class Document:
    doc_id: int
    topic: str
    text: str


def parse_documents(raw_text: str) -> List[Document]:
    documents: List[Document] = []
    blocks = [doc.strip() for doc in raw_text.split("\n\n") if doc.strip()]
    for idx, block in enumerate(blocks):
        if block.startswith("Topic:"):
            header, _, body = block.partition("|")
            topic = header.replace("Topic:", "").strip()
            text = body.strip()
        else:
            topic = "General"
            text = block
        documents.append(Document(doc_id=idx, topic=topic, text=text))
    return documents


@dataclass
class Retriever:
    docs: List[Document]
    top_k: int = 2
    min_score: float = 0.1

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform([doc.text for doc in self.docs])

    def retrieve(self, query: str, topics: Optional[Iterable[str]] = None) -> List[Tuple[Document, float]]:
        query_vec = self.vectorizer.transform([query])
        topic_set = set(topics) if topics else None
        candidate_indices = [
            idx
            for idx, doc in enumerate(self.docs)
            if not topic_set or doc.topic in topic_set
        ]
        if not candidate_indices:
            return []

        candidate_vectors = self.doc_vectors[candidate_indices]
        scores = cosine_similarity(query_vec, candidate_vectors).flatten()
        ranked = np.argsort(scores)[::-1][: self.top_k]
        if scores[ranked[0]] < self.min_score:
            return []

        results: List[Tuple[Document, float]] = []
        for rank in ranked:
            doc_index = candidate_indices[rank]
            results.append((self.docs[doc_index], float(scores[rank])))
        return results


@dataclass
class Generator:
    max_tokens: int = 120

    def __post_init__(self) -> None:
        self.pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            framework="pt",
        )

    def generate(self, query: str, context: str, history: str = "") -> str:
        prompt_parts = [
            "Answer the healthcare-related question using ONLY the context below.",
            "If the answer is not in the context, respond with \"Information not available.\"",
            "Answer in 3-5 sentences and include practical details when present in the context.",
            "",
        ]
        if history:
            prompt_parts.extend(["Conversation so far:", history, ""])
        prompt_parts.extend(["Context:", context, "", f"Question: {query}", "Answer:"])
        prompt = "\n".join(prompt_parts)
        output = self.pipe(prompt, max_new_tokens=self.max_tokens)
        return output[0]["generated_text"].strip()


class HealthcareChatbot:
    def __init__(
        self,
        top_k: int = 2,
        max_tokens: int = 120,
        min_score: float = 0.1,
        max_history_turns: int = 3,
        data_path: str | Path | None = None,
    ) -> None:
        resolved_path = Path(data_path) if data_path else Path(__file__).with_name("healthcare_data.txt")
        with resolved_path.open("r", encoding="utf-8") as file:
            raw_text = file.read()
        documents = parse_documents(raw_text)

        self.retriever = Retriever(documents, top_k=top_k, min_score=min_score)
        self.generator = Generator(max_tokens=max_tokens)
        self.max_history_turns = max_history_turns

    def _format_history(self, history: Optional[Sequence[Tuple[str, str] | dict]]) -> str:
        if not history:
            return ""
        trimmed = list(history)[-self.max_history_turns * 2 :]
        lines = []
        for item in trimmed:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
            else:
                role, content = item
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def chat(
        self,
        query: str,
        topics: Optional[Iterable[str]] = None,
        history: Optional[Sequence[Tuple[str, str] | dict]] = None,
    ) -> dict:
        retrieved = self.retriever.retrieve(query, topics=topics)
        context = "\n".join(doc.text for doc, _ in retrieved)
        history_text = self._format_history(history)
        answer = self.generator.generate(query, context, history=history_text)
        sources = [
            {
                "doc_id": doc.doc_id,
                "topic": doc.topic,
                "score": score,
                "text": doc.text,
            }
            for doc, score in retrieved
        ]
        return {"answer": answer, "sources": sources}


def run_experiments() -> None:
    experiment_chatbot = HealthcareChatbot(top_k=2, max_tokens=80)

    top_k_values = [2, 3, 5]
    max_token_values = [50, 80, 150]

    rows = []

    for top_k in top_k_values:
        experiment_chatbot.retriever.top_k = top_k
        result = experiment_chatbot.chat("What are diabetes symptoms?")
        rows.append((top_k, experiment_chatbot.generator.max_tokens, "What are diabetes symptoms?", result["answer"]))

    for max_tokens in max_token_values:
        experiment_chatbot.generator.max_tokens = max_tokens
        result = experiment_chatbot.chat("How to manage blood pressure?")
        rows.append((experiment_chatbot.retriever.top_k, max_tokens, "How to manage blood pressure?", result["answer"]))

    header = "| top_k | max_tokens | query | response |"
    separator = "|---|---|---|---|"
    print(header)
    print(separator)
    for top_k, max_tokens, query, response in rows:
        print(f"| {top_k} | {max_tokens} | {query} | {response} |")

    # Observation: higher top_k -> more context but risk of irrelevant info
    # Observation: higher max_tokens -> longer but more complete answers
    # Observation: bigram ngram_range=(1,2) -> better matching for multi-word queries


if __name__ == "__main__":
    TOP_K = 2
    MAX_TOKENS = 120

    chatbot = HealthcareChatbot(top_k=TOP_K, max_tokens=MAX_TOKENS)

    print("Healthcare FAQ Chatbot")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Session ended.")
            break
        if not user_input:
            continue
        result = chatbot.chat(user_input)
        print(f"Bot: {result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for source in result["sources"]:
                print(
                    f"- Doc {source['doc_id']} | {source['topic']} | score={source['score']:.3f}"
                )
            print("")
