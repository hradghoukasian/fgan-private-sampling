# ### writefile visualize_generator.py

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from models.generator import Generator
# from utils.gaussian_ring import evaluate_gaussian_ring_pdf

# torch.manual_seed(42)
# np.random.seed(42)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# gen = Generator().to(device)
# gen.load_state_dict(torch.load("generator.pth", map_location=device))
# gen.eval()

# with torch.no_grad():
#     z = torch.randn(10000, 2).to(device)
#     samples = gen(z).cpu().numpy()

# x = np.linspace(-2, 2, 300)
# y = np.linspace(-2, 2, 300)
# X, Y = np.meshgrid(x, y)
# Z = evaluate_gaussian_ring_pdf(X, Y)

# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# axes[0].contourf(X, Y, Z, levels=100, cmap="plasma")
# axes[0].set_title("True Input Density")

# sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, cmap="plasma", ax=axes[1])
# axes[1].set_title("Generated KDE")

# plt.tight_layout()
# plt.savefig("comparison.png")
# plt.show()

### writefile visualize_generator.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from models.generator import Generator
from utils.gaussian_ring import evaluate_gaussian_ring_pdf

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# Load generator
gen = Generator().to(device)
gen.load_state_dict(torch.load("generator.pth", map_location=device))
gen.eval()

# Sample from generator
with torch.no_grad():
    z = torch.randn(10000, 2).to(device)
    samples = gen(z).cpu().numpy()

# Compute input (true) density
x = np.linspace(-2, 2, 300)
y = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(x, y)
Z_true = evaluate_gaussian_ring_pdf(X, Y)

# KDE for generator output
kde = gaussian_kde(samples.T, bw_method='scott')
Z_kde = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].contourf(X, Y, Z_true, levels=100, cmap="plasma")
axes[0].set_title("True Input Density")

axes[1].contourf(X, Y, Z_kde, levels=100, cmap="plasma")
axes[1].set_title("Generated KDE (scipy)")

for ax in axes:
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.savefig("comparison.pdf")
plt.show()
