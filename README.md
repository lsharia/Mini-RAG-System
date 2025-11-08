# Mini-RAG-System

### 🎯 Goal

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system completely from scratch by combining the three key components of modern LLM pipelines:

1. **Embeddings (Retriever)** — Find the most relevant knowledge.
2. **Retrieval** — Match user queries to the right document.
3. **Generation (Answering)** — Extract the final answer using a question-answering model.

---

## 🧩 Project Overview

The Mini RAG System is a simplified implementation of how systems like ChatGPT or Gemini use external knowledge to answer questions.

It performs three major steps:

1. **Retrieve:** Find the document in the knowledge base most similar to the query.
2. **Generate:** Use a pretrained transformer model to extract the specific answer.
3. **Assess:** Automatically evaluate retrieval and generation accuracy.

---

## ⚙️ Installation

Run this command to install all required dependencies:

```bash
!pip install sentence-transformers transformers torch
```

### Dependencies

* `sentence-transformers` — for text embeddings (retriever)
* `transformers` — for the QA pipeline (generator)
* `torch` — for tensor operations

---

## 🧠 System Components

### **1. Retriever**

Model: `all-MiniLM-L6-v2`

* Converts text into embeddings.
* Computes cosine similarity to find the most relevant document.

**Code Snippet:**

```python
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_embeddings = retriever_model.encode(knowledge_base, convert_to_tensor=True)
```

---

### **2. Generator**

Model: `distilbert-base-cased-distilled-squad`

* Performs question answering on the retrieved context.

**Code Snippet:**

```python
from transformers import pipeline
generator = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
```

---

### **3. RAG Pipeline Evaluation**

Each test includes:

* **Retrieval Accuracy:** Did the retriever find the correct context?
* **Generation Accuracy:** Did the generator extract the correct answer?

Example test question:

```python
{
  "question": "Which city is home to the Louvre museum?",
  "expected_keyword": "France",
  "expected_answer": "Paris"
}
```

**Function:**

```python
run_rag_assessment()
```

✅ Output shows retrieved context, generated answer, and a total score.

---

## 🧪 Tasks

### **Task 1 — Run & Understand**

* Execute the notebook and observe the results.
* Check if your final score is **6 / 6**.

### **Task 2 — Add a New Question**

* Extend the assessment with a new question about existing knowledge.
* Expected total score: **8 / 8**.

Example addition:

```python
{
  "question": "What is the largest tropical rainforest in the world?",
  "expected_keyword": "Amazon",
  "expected_answer": "Amazon rainforest"
}
```

### **Task 3 — Add New Knowledge**

* Add a new fact to the knowledge base.
* Re-encode embeddings.
* Test with a matching question.

Example:

```python
knowledge_base_task_3 = [
    "The speed of light is approximately 186,282 miles per second in a vacuum."
]
```

---

## 🧾 Sample Output

```
--- 🚀 Starting RAG System Assessment ---
✅ Retrieval Correct!
✅ Generation Correct!
🎯 Final Score: 6 / 6
🎉 Perfect! Your RAG system is working as expected!
```

---

## 🧠 Concepts Learned

* **Sentence Embeddings:** Representing text as numeric vectors.
* **Cosine Similarity:** Measuring how close two embeddings are.
* **RAG Architecture:** How retrieval improves generation.
* **Evaluation Metrics:** Scoring retrieval and generation quality.

---

## 🏆 Results

✅ Perfect RAG performance across all tests:

* Task 1: **6 / 6**
* Task 2: **8 / 8**
* Task 3: **2 / 2**




