# 🧪 f-GAN-Based Implementation of Minimax-Optimal Private Sampling

This repository provides a PyTorch implementation of the minimax-optimal private sampling mechanism described in the paper:

> **Exactly Minimax-Optimal Locally Differentially Private Sampling**  
> Hyun-Young Park, Shahab Asoodeh, and Si-Hyeon Lee  
> *NeurIPS 2024*

Our implementation follows the theoretical formulation proposed in the paper but employs the **f-GAN framework** to learn the private sampler for a 2D Gaussian Ring input distribution under a fixed local differential privacy (LDP) constraint.

---

## 🧾 References

### Main Paper
```bibtex
@article{park2024exactly,
  title={Exactly minimax-optimal locally differentially private sampling},
  author={Park, Hyun-Young and Asoodeh, Shahab and Lee, Si-Hyeon},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={10274--10319},
  year={2024}
}
```

### f-GAN Framework
```bibtex
@article{nowozin2016f,
  title={f-gan: Training generative neural samplers using variational divergence minimization},
  author={Nowozin, Sebastian and Cseke, Botond and Tomioka, Ryota},
  journal={Advances in neural information processing systems},
  volume={29},
  year={2016}
}
```

---

## 📂 Repository Structure

```
├── main.py                     # Main training script
├── train.py                    # Training logic using f-GAN
├── visualize_generator.py      # Visualization of learned sampler
├── models/
│   ├── generator.py            # Generator architecture
│   └── discriminator.py        # Discriminator architecture
├── utils/
│   ├── f_divergences.py        # Definitions of f-divergences
│   └── gaussian_ring.py        # Data generation: Gaussian Ring
```

---

## ⚙️ Setup Instructions

This project requires Python 3.8+ and PyTorch. You can set up the environment with:

```bash
pip install torch
```

All other dependencies are standard and should work in a basic Python environment.

---

## 🚀 How to Run

### 1. Train the Private Sampler

```bash
python main.py
```

This trains a minimax-optimal private sampler under ε-LDP for a **2D Gaussian Ring** input distribution using an f-GAN objective.

---

### 2. Visualize the Results

```bash
python visualize_generator.py
```

This script visualizes the learned private sampler compared to the original data distribution.

---

## 📬 Contact

For questions or collaboration inquiries:

**Hrad Ghoukasian**  
✉️ ghoukash@mcmaster.ca  
🔗 [GitHub](https://github.com/hradghoukasian)

---

## 🙏 Acknowledgments

This repository is an independent implementation based on theoretical work by Park, Asoodeh, and Lee (2024). The codebase draws inspiration from the f-GAN framework by Nowozin et al. (2016).
