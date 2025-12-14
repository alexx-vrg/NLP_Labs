# NLP Labs

A collection of Natural Language Processing labs covering transformer models, text classification, and LLM fine-tuning.

## Repository Structure

```
NLP_Labs/
├── Lab2_Intro_to_maths_optimisation/
├── Lab3_Transformer_Introduction/
├── Lab4_Transformer_Classification/
├── Lab5_Text_Classification/
├── Lab6_Finetune_llama/
├── Lab7_Prompting_LangChain_Ollama/
└── README.md
```

## Lab Descriptions

### Lab 2 - Introduction to Maths Optimisation

Foundations of mathematical optimization for machine learning, covering gradient descent and optimization techniques used in neural network training.

### Lab 3 - Transformer Introduction

Introduction to Hugging Face Transformers library through practical NLP tasks:

- **Text Classification** - Sentiment analysis with DistilBERT
- **Named Entity Recognition** - Identifying entities (persons, organizations, locations)
- **Question Answering** - Extractive QA with SQuAD-trained models
- **Summarization** - Abstractive summarization with DistilBART
- **Translation** - English to German with MarianMT
- **Text Generation** - Autoregressive generation with GPT-2

### Lab 4 - Training Transformer Models for Text Classification

End-to-end pipeline for training transformers on emotion classification:

- Dataset loading and exploration with Hugging Face Datasets
- Tokenization with DistilBERT tokenizer
- Feature extraction using pre-trained transformer hidden states
- Training a classifier on extracted embeddings
- Fine-tuning with Hugging Face Trainer API

### Lab 5 - Text Classification with Generative Models

Comparing different approaches to text classification:

- **Task-specific models** - Pre-trained RoBERTa for sentiment
- **Supervised embeddings** - Sentence-transformers + Logistic Regression
- **Zero-shot classification** - Label embeddings with cosine similarity
- **Generative models** - FLAN-T5 for prompt-based classification

### Lab 6 - Fine-tuning Llama 3.2

Fine-tuning a large language model on a medical Q&A dataset:

- Loading Llama 3.2 with Hugging Face
- LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning
- Training on medical dataset
- Model evaluation and inference

### Lab 7 - Prompting with LangChain & Ollama

_Empty for now_

## Setup

Each lab folder contains:

- `lab.ipynb` - Jupyter notebook with code and answers
- `requirements.txt` - Python dependencies

To run a lab:

```bash
cd Lab<N>_<name>/
pip install -r requirements.txt
jupyter notebook lab.ipynb
```
