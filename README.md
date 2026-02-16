# Parallel Fusion Hybrid Quantum-Classical Convolutional Neural Network (PF-HQCCNN)

This repository contains the source code and experimental results for the paper: **"Parallel Fusion Hybrid Quantum-Classical Architecture for Optimizing Image Feature Preservation in Hilbert Space."**

## 🔬 Overview
In the current NISQ (Noisy Intermediate-Scale Quantum) era, deep learning architectures like U-Net or Vision Transformers (ViT) achieve spatial feature preservation at an immense computational cost (millions of parameters). 

This project proposes a novel **Parallel Fusion Hybrid QCNN (PF-HQCCNN)**. By integrating a parallel quantum branch, data is mapped into a high-dimensional Hilbert space to preserve fine-grained features via Unitary transformations. This creates an entanglement-induced non-factorizable feature map that is immune to the Barren Plateaus problem.

### Key Contributions:
- **Parallel Architecture:** Prevents the classical-to-quantum data bottleneck by operating branches simultaneously.
- **Scaling Law Decoupling:** Uses a bottleneck classical CNN layer to decouple the number of qubits ($n=4$) from the input image resolution ($M \times N$).
- **Barren Plateaus Immunity:** Utilizes a shallow ansatz ($L=2$) to maintain a polynomial gradient variance $\Omega(1/n^{1.5})$.
- **Parameter Efficiency:** Achieves **92.2% test accuracy** on the MNIST dataset using only **7,746 parameters** (a 3.5x reduction compared to classical baselines).

## 📦 Dataset
The experiments in this repository are conducted on the **MNIST database of handwritten digits**. 

- **Automated Download:** The provided PyTorch script uses `torchvision.datasets.MNIST` to download and process the dataset automatically into the `/Data` directory.
- **Manual Download / Source:** If you wish to inspect the dataset manually, you can download the raw files from the official repository provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges:
  👉 [The MNIST Database (Official Link)](http://yann.lecun.com/exdb/mnist/)

## 📊 Experimental Results

### 1. Parameter Efficiency & Accuracy
The proposed PF-HQCCNN significantly reduces the number of trainable parameters while maintaining highly competitive accuracy.

| Model Architecture | Parameters | Test Accuracy |
| :--- | :---: | :---: |
| Classical Baseline CNN | 26,698 | 95.6% |
| Sequential Hybrid CNN | 7,722 | 81.4% |
| **Parallel Fusion Hybrid (Proposed)** | **7,746** | **92.2%** |

### 2. Training Dynamics
The training loss of our model converges smoothly without encountering the vanishing gradient problem (Barren Plateaus), a common issue in deep Variational Quantum Circuits (VQCs).

*(You can upload the `training_comparison.png` to your repo and link it here)*
`![Training Dynamics](real_loss_convergence.png)`

### 3. Confusion Matrix
The model exhibits exceptional complex amplitude classification, clearly distinguishing topologically similar digits (e.g., 4 vs 9, or 3 vs 8) thanks to quantum phase interference.

*(You can upload the `confusion_matrix_Parallel_Fusion_Hybrid.png` to your repo and link it here)*
`![Confusion Matrix](confusion_matrix_Parallel_Fusion_Hybrid.png)`

## ⚙️ Dependencies & Installation
This project is built using PyTorch and PennyLane. To run the code locally or on Google Colab, install the following dependencies:

```bash
pip install torch torchvision
pip install pennylane
pip install matplotlib seaborn scikit-learn
