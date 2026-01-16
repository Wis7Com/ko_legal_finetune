# Korean Legal LLM Fine-tuning with QLoRA

Fine-tuning **Kanana Nano 2.1B Instruct** model on Korean legal terminology using **QLoRA** (Quantized Low-Rank Adaptation) for efficient training on consumer GPUs.

## 🎯 Overview

This repository contains a complete fine-tuning pipeline for adapting the Kakao's Kanana Nano 2.1B Instruct model to Korean legal domain using parameter-efficient fine-tuning techniques.

### Key Features

- ✅ **4-bit Quantization**: Efficient memory usage with NF4 quantization
- ✅ **QLoRA**: Low-rank adaptation for parameter-efficient fine-tuning
- ✅ **Optimized for Colab**: Runs on free Google Colab T4 GPU
- ✅ **Production-ready**: Includes evaluation and model saving

## 📊 Model & Dataset

### Base Model
- **Model**: [kakaocorp/kanana-nano-2.1b-instruct](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)
- **Size**: 2.1B parameters
- **Architecture**: Transformer-based causal language model
- **Language**: Korean

### Dataset
- **Dataset**: [flyingcarycoder/korean-legal-terminology](https://huggingface.co/datasets/flyingcarycoder/korean-legal-terminology)
- **Samples**: 17,484 legal term definitions
- **Format**: Instruction-following (input/output pairs)
- **Domain**: Korean legal terminology and concepts

## 🔧 Fine-tuning Configuration

### QLoRA Settings

```python
# 4-bit Quantization
- Quantization Type: NF4 (Normal Float 4-bit)
- Compute dtype: bfloat16
- Double Quantization: Enabled

# LoRA Configuration
- LoRA Rank (r): 16
- LoRA Alpha: 32
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- LoRA Dropout: 0.05
- Trainable Parameters: ~23M (1.95% of total)
```

### Training Hyperparameters

```python
# Training Setup
- Epochs: 3
- Batch Size: 4 (per device)
- Gradient Accumulation Steps: 4
- Effective Batch Size: 16

# Optimization
- Optimizer: Paged AdamW 8-bit
- Learning Rate: 2e-4
- LR Scheduler: Cosine
- Warmup Ratio: 0.03
- Weight Decay: 0.01
- Max Gradient Norm: 0.3

# Precision
- Mixed Precision: bfloat16
- Max Sequence Length: 2048
```

## 🚀 Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wis7com/ko_legal_finetune/blob/main/notebooks/finetune_kanana_legal.ipynb)

### 2. Run the Notebook

The notebook includes:
1. **Environment Setup**: Install dependencies
2. **Data Loading**: Load dataset from Hugging Face
3. **Model Loading**: Load base model with 4-bit quantization
4. **Training**: Fine-tune with QLoRA
5. **Evaluation**: Validate on test set
6. **Saving**: Save to Google Drive

### 3. Local Setup (Optional)

```bash
# Clone repository
git clone https://github.com/Wis7com/ko_legal_finetune.git
cd ko_legal_finetune

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/finetune_kanana_legal.ipynb
```

## 💻 Hardware Requirements

### Minimum (Google Colab Free)
- **GPU**: T4 (16GB VRAM)
- **RAM**: 12GB
- **Training Time**: ~3-4 hours

### Recommended
- **GPU**: A100 (40GB/80GB VRAM)
- **RAM**: 32GB+
- **Training Time**: ~1-2 hours

### Memory Optimization

The QLoRA approach reduces memory requirements:
- **Full Fine-tuning**: ~40GB VRAM required
- **QLoRA (4-bit)**: ~8-10GB VRAM required
- **Memory Savings**: ~75% reduction

## 📈 Training Results

After fine-tuning, the model achieves:
- Better understanding of Korean legal terminology
- Improved accuracy on legal concept explanations
- Maintained general Korean language capabilities

## 🔍 Model Architecture

```
Base Model: Kanana Nano 2.1B Instruct
├── Quantization: 4-bit NF4
├── LoRA Adapters (trainable)
│   ├── Attention Layers: q_proj, k_proj, v_proj, o_proj
│   └── MLP Layers: gate_proj, up_proj, down_proj
└── Base Model (frozen)
```

## 📝 Prompt Format

The model is trained on the following instruction format:

```
### 질문:
다음 법률 용어(한자: 吸收合倂)를 설명해줘: 흡수합병

### 답변:
법률이 정하는 절차에 의하여 2 이상의 법인 전부 또는 그중 1개의 법인이외의 법인이 해산하여...
```

## 🛠️ Technical Details

### Why QLoRA?

1. **Memory Efficient**: 4-bit quantization reduces memory by 75%
2. **Performance**: Minimal accuracy loss compared to full fine-tuning
3. **Accessible**: Enables training on consumer GPUs
4. **Fast**: Reduced computation requirements

### Optimization Techniques

- **Gradient Checkpointing**: Reduces memory during backpropagation
- **Paged Optimizers**: Efficient memory management for optimizer states
- **Mixed Precision**: bfloat16 for faster computation
- **Gradient Accumulation**: Simulates larger batch sizes

## 📚 Citation

If you use this fine-tuning pipeline, please cite:

```bibtex
@misc{ko_legal_finetune_2026,
  title={Korean Legal LLM Fine-tuning with QLoRA},
  author={flyingcarycoder},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Wis7com/ko_legal_finetune}}
}
```

### Dataset Citation

```bibtex
@dataset{korean_legal_terminology_2026,
  title={Korean Legal Terminology Dataset},
  author={flyingcarycoder},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/flyingcarycoder/korean-legal-terminology}}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- **Base Model**: [Kakao Corp](https://huggingface.co/kakaocorp) for Kanana Nano 2.1B Instruct
- **Dataset**: Korean Legal Terminology dataset contributors
- **QLoRA**: [Tim Dettmers et al.](https://arxiv.org/abs/2305.14314) for the QLoRA method
- **Libraries**: HuggingFace Transformers, PEFT, TRL, bitsandbytes
