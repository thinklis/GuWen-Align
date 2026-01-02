# A Weakly Supervised Preference Alignment Framework for Robust Ancient Chinese Translation

[![Paper Status](https://img.shields.io/badge/Status-Under_Revision-orange.svg)](#)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Robust--Erya-yellow)](https://huggingface.co/datasets/thinklis/Robust-Erya)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official repository for the paper: **"A Weakly Supervised Preference Alignment Framework for Robust Ancient Chinese Translation"**, currently under revision at *Pattern Recognition*.

This project focuses on enhancing the translation of Ancient Chinese into Modern Chinese, specifically addressing the challenge of linguistic variations and model robustness through a weakly supervised preference alignment framework.

---

## 📅 News
- **[2025-12]**: Released the **Robust-Erya** benchmark on Hugging Face Hub!
- **[Coming Soon]**: The core implementation of **Stage 2 (Preference Alignment)** will be fully open-sourced upon official paper acceptance.

---

## 📊 Datasets

We utilize two main datasets for our research. Detailed descriptions, noise taxonomy, and data samples can be found on our Hugging Face page.

### 1. Erya Benchmark
The foundation of our training and standard evaluation. 
- **Source:** [Erya Official Repository](https://github.com/RUCAIBox/Erya)

### 2. Robust-Erya Benchmark
An extension of the Erya test suite designed for robustness evaluation. It covers 5 domains with 3 categories of noise at 5 intensity levels.

👉 **Access and download the dataset here:** [Hugging Face: thinklis/Robust-Erya](https://huggingface.co/datasets/thinklis/Robust-Erya)

---

## ⚙️ Training Framework

Our framework follows a two-stage training paradigm:

- **Stage 1 (Supervised Fine-Tuning):** Initial SFT using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework on base models (e.g., LLaMA-3, Qwen2.5, InternLM3.
- **Stage 2 (Preference Alignment):** Our proposed framework aligns model outputs with human-preferred stylistic and linguistic norms using weak supervision. 
  *(Code for Stage 2 is currently withheld for the review process)*.

---

## 🚀 Usage

### 1. Inference
Inference and generation are performed via **LLaMA-Factory**. Please download the **Robust-Erya** test files from Hugging Face and follow the LLaMA-Factory [Prediction Guide](https://github.com/hiyouga/LLaMA-Factory) to generate translations and obtain initial **BLEU-4** scores.

### 2. Evaluation
For advanced semantic evaluation, we employ an **LLM-as-a-Judge** approach:

* **`evaluation/llm_judge_vllm.py`**: A Python script that utilizes LLMs to evaluate the semantic alignment and translation fluency between the model outputs and ground-truth references.

---
## 📜 Citation

If you find this project or the **Robust-Erya** dataset helpful, please cite this repository:

```bibtex
@misc{GuWenAlign2025,
  author = {Thinklis},
  title = {A Weakly Supervised Preference Alignment Framework for Robust Ancient Chinese Translation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/thinklis/GuWen-Align](https://github.com/thinklis/GuWen-Align)}}
}