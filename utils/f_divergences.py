import torch
import torch.nn.functional as F
import math

def get_f_divergence(name):
    eps = 1e-6  # for numerical stability

    if name == "kl":
        gf = lambda v: v
        f_star = lambda t: torch.exp(t - 1)

    elif name == "tv":
        # output activation  g_f : ℝ → (-½, ½)
        gf = lambda v: 0.5 * torch.tanh(v)

        f_star = lambda t: torch.where((t >= -0.5) & (t <= 0.5), t, 1e6)

    elif name == "rev_kl":
        gf = lambda v: -torch.exp(-v)
        f_star = lambda t: -1.0 - torch.log(torch.clamp(-t, min=eps))

    elif name == "js":
        gf = lambda v: torch.log(torch.tensor(2.0, device=v.device)) - F.softplus(-v)
        f_star = lambda t: -torch.log(torch.clamp(2.0 - torch.exp(t), min=eps))

    elif name == "gan":
        gf = lambda v: -F.softplus(-v)
        f_star = lambda t: -torch.log(torch.clamp(1.0 - torch.exp(t), min=eps))

    elif name == "pearson":
        gf = lambda v: v
        f_star = lambda t: 0.25 * t**2 + t

    elif name == "sqh":
        gf = lambda v: 1.0 - torch.exp(-v)
        f_star = lambda t: t / torch.clamp(1.0 - t, min=eps)

    elif name == "neyman":
        gf = lambda v: 1.0 - torch.exp(-v)
        f_star = lambda t: 2.0 - 2.0 * torch.sqrt(torch.clamp(1.0 - t, min=eps))

    elif name == "clip":
        # Convex conjugate of the indicator function over [b, b * e^ε]
        epsilon = 1
        c1 = 0  # replace with actual value
        c2 = 3.7724  # from previous computation
        b = (c2 - c1) / ((math.exp(epsilon) - 1) * (1 - c1) + (c2 - c1))
        
        low  = -5
        high = 5

        # output activation: ℝ → [low, high]
        # gf = lambda v: low + (high - low) * torch.sigmoid(v)
        gf = lambda v: v
        f_star = lambda u: torch.where(u >= 0, u * (1/b), u * (1/(b * math.exp(epsilon))))

    else:
        raise NotImplementedError(f"Unknown divergence: {name}")

    return gf, f_star
