# Natural Language Inference with Neural Architectures

A comparative study of three neural architectures for Natural Language Inference (NLI), implementing Bidirectional GRU, Variational Siamese Autoencoder, and Self-Attention Transformer models to classify premise-hypothesis pairs as entails or neutral.

**Course:** CITS4012 Natural Language Processing, Semester 2, 2025  
**Team:** Mahit Gupta (23690265), James Wigfield (23334375), Mitchell Otley (23475725)

---

## 📊 Results Summary

| Model | Test Accuracy | F1-Score | Key Strength |
|-------|---------------|----------|--------------|
| **BiGRU** | 70.0% | 0.650 | Highest accuracy |
| **VSAE** | 68.1% | **0.678** | Best class balance |
| **Transformer (Self-Attention)** | **69.1%** | 0.654 | Balanced performance |
| Transformer (Cross-Attention) | 62.8% | 0.622 | Ablation baseline |

**Key Finding:** The VSAE model achieved the highest F1-score (0.678), demonstrating superior precision-recall balance despite lower raw accuracy—making it most suitable for imbalanced datasets.

---

## 🔬 Project Overview

Natural Language Inference (NLI) tasks models to determine if a hypothesis can be logically inferred from a premise. This project implements and compares three distinct neural architectures:

1. **Bidirectional GRU** - Recurrent architecture with layer normalization and max pooling
2. **Variational Siamese Autoencoder (VSAE)** - Combines VAE generative regularization with Siamese pairwise reasoning
3. **Self-Attention Transformer** - Parallel processing with multi-head attention mechanisms

### Dataset
- **Training:** 23,088 samples
- **Validation:** 1,304 samples  
- **Test:** 2,126 samples
- **Classes:** Entails (minority) and Neutral (majority)
- **Source:** Science exam questions converted to premise-hypothesis pairs from web text corpus

---

## 🏗️ Architecture Details

### 1. Bidirectional GRU
```python
Premise/Hypothesis → BiGRU (256 units, 5 layers) → LayerNorm → 
Concatenate → BiGRU (128 units, 5 layers) → MaxPool → 
Dense (64 units, ReLU) → Dropout (0.4) → Classifier
```

**Hyperparameters:**
- Learning rate: 1e-5
- Optimizer: Adam
- Epochs: 10
- Dropout: 0.4

**Key Feature:** Token-level representation through final hidden layer outputs rather than just final token state.

### 2. Variational Siamese Autoencoder (VSAE)
```python
Premise/Hypothesis → BiLSTM Encoder → Masked Mean Pooling →
Variational Layer (μ, σ²) → Reparameterization (z) →
Feature Combination (|z₁-z₂|; z₁⊙z₂) → MLP Classifier
                                      ↓
                            Reconstruction Decoder
```

**Loss Function:**
```
L = L_cls + β·L_KL + γ·L_rec
```

**Ablation Study Results (γ sweep):**
- γ = 0.0: Test F1 = 0.690 (baseline, no reconstruction)
- γ = 0.3: Val F1 = 0.754 (best validation performance)
- Finding: Moderate reconstruction (γ ≈ 0.3) enhances class balance without impairing generalization

### 3. Self-Attention Transformer
```python
GloVe Embeddings → Positional Encoding → LayerNorm →
Transformer Encoder (4 heads, 2 layers) → 
Attention-Weighted Pooling → Classifier
```

**Hyperparameters:**
- Attention heads: 4
- Encoder layers: 2
- Feed-forward dim: 512
- Learning rate: 8e-5
- Weight decay: 0.01
- Gradient clipping: 0.5

**Attention Pooling with Numerical Stability:**
```python
α_i = exp(s_i - s_max) / Σ exp(s_j - s_max)
```

---

## 🔍 Ablation Studies

### Study 1: Self-Attention vs Cross-Attention

| Configuration | Val Acc | Test Acc | F1-Score | Epochs |
|--------------|---------|----------|----------|--------|
| **Self-Attention** (Main) | 67.6% | **69.1%** | **0.654** | 8 |
| Cross-Attention (Ablation) | 65.3% | 62.8% | 0.622 | 9 |

**Key Insights:**
- Self-attention converges faster and generalizes better
- Cross-attention's additional complexity doesn't translate to improved performance with limited training data
- Joint processing of premise-hypothesis pairs through self-attention captures semantic relationships more effectively than explicit cross-attention with separate encoders

### Study 2: VSAE Reconstruction Weight (γ)

Testing γ ∈ {0.0, 0.1, 0.3, 0.5} with fixed β:

**Findings:**
- **Validation:** Macro-F1 improved steadily up to γ = 0.3, indicating moderate reconstruction promotes balanced latent representations
- **Test:** Macro-F1 peaked at γ = 0.0, suggesting stronger reconstruction slightly reduces generalization
- **Optimal trade-off:** γ ≈ 0.3 balances semantic regularization with discriminative accuracy

---

## 🛠️ Technical Implementation

### Preprocessing Pipeline
1. **Tokenization:** spaCy library
2. **Normalization:** Lowercase conversion
3. **Outlier Removal:** 2.5th-97.5th percentile filtering based on token length
4. **Embeddings:** GloVe-Twitter-100 (pretrained on 2B tweets)
5. **Padding:** Dynamic padding to max premise/hypothesis length
6. **OOV Handling:** Special token for vocabulary gaps

### Model Training
- **Framework:** PyTorch
- **Optimizer:** Adam / AdamW
- **Early Stopping:** Validation accuracy monitoring
- **Regularization:** Dropout, Layer Normalization, Weight Decay
- **GPU:** CUDA-enabled training

---

## 📈 Performance Analysis

### Confusion Matrix Insights
All models exhibit conservative prediction strategies influenced by class imbalance:

- **BiGRU:** High accuracy (70.0%) but lower F1 (0.650) indicates majority class bias
- **VSAE:** Best F1 (0.678) with balanced precision-recall trade-off
- **Transformer:** 46.1% recall on entails vs 84.2% on neutral, reflecting dataset bias

### Attention Visualization
Transformer attention weights show:
- ✅ **Correct predictions:** Strong attention alignment on semantically important tokens
- ❌ **Misclassifications:** Over-attention to lexical overlap without capturing semantic neutrality

---

## 📂 Repository Structure

```
nlp-project/
├── data/                    # Dataset files (train, val, test)
├── models/
│   ├── bigru.py            # Bidirectional GRU implementation
│   ├── vsae.py             # Variational Siamese Autoencoder
│   └── transformer.py      # Self-Attention Transformer
├── ablations/
│   ├── cross_attention.py  # Cross-attention variant
│   └── gamma_sweep.py      # VSAE reconstruction weight ablation
├── preprocessing/
│   ├── tokenizer.py        # spaCy tokenization
│   └── embeddings.py       # GloVe embedding loader
├── training/
│   ├── train.py            # Training loop
│   └── evaluate.py         # Evaluation metrics
├── visualization/
│   ├── attention_viz.py    # Attention weight visualization
│   └── confusion_matrix.py # Performance analysis
├── requirements.txt         # Python dependencies
├── NLP_Document.pdf        # Full project report
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
PyTorch 1.12+
spaCy 3.0+
NumPy, Pandas, Matplotlib, Seaborn
```

### Installation
```bash
# Clone repository
git clone https://github.com/MahitGtg/nlp-project.git
cd nlp-project

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download GloVe embeddings
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip -d embeddings/
```

### Training Models
```bash
# Train BiGRU
python training/train.py --model bigru --epochs 10 --lr 1e-5

# Train VSAE
python training/train.py --model vsae --gamma 0.3 --beta 1.0

# Train Transformer
python training/train.py --model transformer --heads 4 --layers 2
```

### Running Ablations
```bash
# Cross-attention ablation
python ablations/cross_attention.py

# VSAE gamma sweep
python ablations/gamma_sweep.py --gamma_values 0.0 0.1 0.3 0.5
```

### Evaluation
```bash
# Test all models
python training/evaluate.py --checkpoint models/best_checkpoint.pth
```

---

## 📊 Results Visualization

### Training Curves
- Training/validation loss and accuracy over epochs
- Early stopping points for optimal generalization
- Self-attention vs cross-attention convergence comparison

### Confusion Matrices
- Per-model classification patterns
- Class-specific precision and recall breakdown
- Identification of systematic misclassification trends

### Attention Heatmaps
- Token-level attention weight distributions
- Correct vs incorrect prediction comparisons
- Semantic vs lexical attention patterns

---

## 🔑 Key Contributions

1. **Comprehensive Architecture Comparison:** Evaluated RNN, VAE, and Transformer approaches on same NLI task
2. **Novel Ablation Studies:** 
   - Self-attention vs cross-attention for NLI
   - VSAE reconstruction weight impact on class balance
3. **Attention Analysis:** Visualized what Transformer learns for textual entailment
4. **Class Imbalance Handling:** Demonstrated F1-score superiority of VSAE for imbalanced datasets

---

## 📝 Limitations and Future Work

### Current Limitations
- Small training set (22,122 samples)
- Significant class imbalance (neutral-heavy)
- Static GloVe embeddings (no task-specific adaptation)
- Binary classification only (excludes contradiction)

### Future Directions
- **Data Augmentation:** Paraphrasing, back-translation, synthetic data generation
- **Contextualized Embeddings:** BERT, RoBERTa for dynamic representations
- **Three-Way Classification:** Extend to entailment/neutral/contradiction
- **Ensemble Methods:** Combine complementary strengths of BiGRU, VSAE, Transformer
- **Few-Shot Learning:** Improve minority class performance with meta-learning

---

## 📚 References

- **GloVe Embeddings:** Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
- **Variational Autoencoders:** Arad et al. (2023) - "Using EfficientNet-B7, VAE, and Siamese Twin Network"
- **Attention Mechanisms:** Vaswani et al. (2017) - "Attention Is All You Need"
- **NLI Benchmarks:** Storks et al. (2020) - "Recent Advances in Natural Language Inference"
- **Uncertainty Quantification:** Li & Yuan (2025) - "Boosting Neural Language Inference via Cascaded Interactive Reasoning"

---

## 👥 Team Contributions

| Name | Student ID | Contribution | Sections |
|------|-----------|--------------|----------|
| **Mitchell Otley** | 23475725 | BiGRU Model, Data Preprocessing | 1, 2.1, 2.4, 3.1, 4 |
| **James Wigfield** | 23334375 | VSAE Model, Reconstruction Ablation | 2.2, 2.6, 3.3, 4 |
| **Mahit Gupta** | 23690265 | Transformer Models, Attention Ablation, Abstract | 2.3, 2.5, 3.2 |

*All team members contributed equally (33% each) to the project's success.*

---

## 📄 License

This project is part of academic coursework for CITS4012 Natural Language Processing at the University of Western Australia.

---

## 📧 Contact

For questions or collaboration opportunities:
- **Mahit Gupta:** mahit.gupta64@gmail.com
- **Project Report:** [NLP_Document.pdf](./NLP_Document.pdf)
- **GitHub:** [github.com/MahitGtg/nlp-project](https://github.com/MahitGtg/nlp-project)

---

## 🎓 Acknowledgments

Special thanks to:
- **CITS4012 Teaching Team** for course guidance and dataset provision
- **University of Western Australia** for computational resources
- **spaCy and PyTorch communities** for excellent documentation
- **GloVe authors** for pretrained embeddings

---

**Last Updated:** February 2026  
**Course:** CITS4012 Natural Language Processing  
**Institution:** University of Western Australia
