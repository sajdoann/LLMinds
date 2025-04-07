import spacy
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

        dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)
        self.document_store = {}
        self.nlp = spacy.load("en_core_web_sm")

    def load_documents(self):
        with open('documents.txt', 'r', encoding='utf-8') as f:
            raw_text = f.read()

        chunks = self.split_into_chunks(raw_text, max_words=100)
        print(f"[INFO] Split into {len(chunks)} chunks.")

        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.document_store[i] = chunk

        return self.document_store

    def split_into_chunks(self, text, max_words=100, overlap=20):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks = []
        chunk = []
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > max_words:
                chunks.append(" ".join(chunk))
                chunk = chunk[-overlap:] if overlap > 0 else []
                word_count = len(chunk)
            chunk.extend(words)
            word_count += len(words)

        if chunk:
            chunks.append(" ".join(chunk))

        return chunks

    def retrieve(self, query, tokenizer=None, max_tokens=7000, top_k=20):
        user_embedding = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(user_embedding, k=top_k)

        retrieved_docs = []
        seen = set()
        token_count = 0

        for idx in I[0]:
            print (f"[INFO] Retrieved index: {idx}")
            if idx == -1 or idx not in self.document_store:
                continue

            doc = self.document_store[idx].strip()
            if doc in seen:
                continue

            seen.add(doc)
            retrieved_docs.append(doc)

            if tokenizer:
                tokens = tokenizer(doc, return_tensors="pt")["input_ids"].shape[1]
                token_count += tokens
                if token_count >= max_tokens:
                    break

        return "\n".join(retrieved_docs)
