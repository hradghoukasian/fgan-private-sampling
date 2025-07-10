# import torch
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import seaborn as sns
# from models.generator import Generator
# from models.discriminator import Discriminator
# from utils.gaussian_ring import sample_gaussian_ring, sample_from_h_rejection, sample_from_h_importance
# from utils.f_divergences import get_f_divergence


# def train_fgan(f_divergence="js", steps=10000, batch_size=256, device="cuda" if torch.cuda.is_available() else "cpu"):
#     gen = Generator().to(device)

#     # Discriminator for main f-divergence (e.g., JS)
#     disc_main = Discriminator().to(device)
#     opt_d_main = optim.Adam(disc_main.parameters(), lr=1e-4)

#     # Discriminator for clipping divergence
#     disc_clip = Discriminator().to(device)
#     opt_d_clip = optim.Adam(disc_clip.parameters(), lr=1e-4)

#     opt_g = optim.Adam(gen.parameters(), lr=1e-4)

#     # f-divergence functions
#     gf_main, f_star_main = get_f_divergence(f_divergence)
#     gf_clip, f_star_clip = get_f_divergence("clip")

#     for step in range(steps):
#         # Sample real and fake
#         real = torch.tensor(sample_gaussian_ring(batch_size)).to(device)
#         z = torch.randn(batch_size, 2).to(device)
#         fake = gen(z).detach()

#         # --- Update Discriminator for main divergence ---
#         v_real_main = disc_main(real)
#         v_fake_main = disc_main(fake)

#         d_loss_main = -gf_main(v_real_main).mean() + f_star_main(gf_main(v_fake_main)).mean()
#         opt_d_main.zero_grad()
#         d_loss_main.backward()
#         opt_d_main.step()

#         # --- Update Discriminator for clipping divergence ---
#         # In place of "real", you can optionally use a known sampler from h, but using real is acceptable proxy
#         real_h = torch.tensor(sample_from_h_importance(batch_size), dtype=torch.float32).to(device)
#         v_real_clip = disc_clip(real_h)

#         v_fake_clip = disc_clip(fake)

#         d_loss_clip = -gf_clip(v_real_clip).mean() + f_star_clip(gf_clip(v_fake_clip)).mean()
#         opt_d_clip.zero_grad()
#         d_loss_clip.backward()
#         opt_d_clip.step()

#         # --- Update Generator (combined objectives) ---
#         z = torch.randn(batch_size, 2).to(device)
#         fake = gen(z)

#         # g_loss_main = -gf_main(disc_main(fake)).mean()
#         # g_loss_clip = -gf_clip(disc_clip(fake)).mean()

#         g_loss_main = -f_star_main(gf_main(disc_main(fake))).mean()
#         g_loss_clip = -f_star_clip(gf_clip(disc_clip(fake))).mean()
#         g_loss = g_loss_main + g_loss_clip

#         opt_g.zero_grad()
#         g_loss.backward()
#         opt_g.step()

#         if step % 1000 == 0:
#             print(f"Step {step}: D_main {d_loss_main.item():.4f}, D_clip {d_loss_clip.item():.4f}, G_main {g_loss_main.item():.4f}, G_Clip {g_loss_clip.item():.4f}")

#     print("Training finished. Saving generator model...")
#     torch.save(gen.state_dict(), "generator.pth")

# train.py
import torch
import torch.optim as optim
import numpy as np
import math
from scipy.stats import gaussian_kde

from models.generator import Generator
from models.discriminator import Discriminator
from utils.gaussian_ring import (
    sample_gaussian_ring,
    sample_from_h_importance,
    evaluate_h_density,
)
from utils.f_divergences import get_f_divergence


def train_fgan(
    f_divergence: str = "js",
    steps: int = 10_000,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_clip_steps: int = 1,          # ← how many D_clip updates per G update
    lambda_clip: float = 1,      # ← weight on clipping loss
):
    gen = Generator().to(device)

    # Discriminators (same architecture; no change)
    disc_main = Discriminator().to(device)
    disc_clip = Discriminator().to(device)

    # Optimisers
    opt_d_main = optim.Adam(disc_main.parameters(), lr=1e-4)
    opt_d_clip = optim.Adam(disc_clip.parameters(), lr=1e-4)
    opt_g = optim.Adam(gen.parameters(), lr=1e-4)

    # f-divergence helpers
    gf_main, f_star_main = get_f_divergence(f_divergence)
    gf_clip, f_star_clip = get_f_divergence("clip")

    # constants for logging
    sigma = 0.5
    epsilon = 5
    c1, c2 = 0, 3.772454
    b = (c2 - c1) / ((math.exp(epsilon) - 1) * (1 - c1) + (c2 - c1))
    lower, upper = 1 / (b * math.exp(epsilon)), 1 / b
    log_lower, log_upper = math.log(lower), math.log(upper)

    for step in range(steps):
        # ------------------------------------------------------------------ #
        # 1) update main discriminator once
        # ------------------------------------------------------------------ #
        real_p = torch.tensor(sample_gaussian_ring(batch_size), device=device)
        z = torch.randn(batch_size, 2, device=device)
        fake = gen(z).detach()

        v_real_main = disc_main(real_p)
        v_fake_main = disc_main(fake)

        d_loss_main = -gf_main(v_real_main).mean() + f_star_main(
            gf_main(v_fake_main)
        ).mean()

        opt_d_main.zero_grad()
        d_loss_main.backward()
        opt_d_main.step()

        # ------------------------------------------------------------------ #
        # 2) update clip-discriminator n_clip_steps times
        # ------------------------------------------------------------------ #
        for _ in range(n_clip_steps):
            real_h = torch.tensor(
                sample_from_h_importance(batch_size), dtype=torch.float32, device=device
            )
            v_real_clip = disc_clip(real_h)
            v_fake_clip = disc_clip(fake)  # fake still detached

            d_loss_clip = -gf_clip(v_real_clip).mean() + f_star_clip(
                gf_clip(v_fake_clip)
            ).mean()

            opt_d_clip.zero_grad()
            d_loss_clip.backward()
            opt_d_clip.step()

        # ------------------------------------------------------------------ #
        # 3) generator update
        # ------------------------------------------------------------------ #
        z = torch.randn(batch_size, 2, device=device)
        fake = gen(z)

        g_loss_main = -f_star_main(gf_main(disc_main(fake))).mean()
        g_loss_clip = -f_star_clip(gf_clip(disc_clip(fake))).mean()

        g_loss = g_loss_main + lambda_clip * g_loss_clip

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        # ------------------------------------------------------------------ #
        # 4) logging every 1000 steps
        # ------------------------------------------------------------------ #
        if step % 1000 == 0 and step != 0:
            print(
                f"\nStep {step}: "
                f"D_main {d_loss_main.item():.4f}, "
                f"D_clip {d_loss_clip.item():.4f}, "
                f"G_main {g_loss_main.item():.4f}, "
                f"G_clip {g_loss_clip.item():.4f}"
            )

            with torch.no_grad():
                #  — estimate q on 50 000 generator samples for smoother KDE
                z_eval = torch.randn(50_000, 2, device=device)
                gen_samps = gen(z_eval).cpu().numpy()
                kde = gaussian_kde(gen_samps.T, bw_method=0.3)

                #  — test on 10 000 samples from P
                eval_samps = sample_gaussian_ring(10_000, sigma=sigma)
                q_vals = kde(eval_samps.T)

                h_vals = evaluate_h_density(
                    torch.tensor(eval_samps, dtype=torch.float32), sigma=sigma
                ).numpy()

                ratios = (h_vals + 1e-8) / (q_vals + 1e-8)
                within = np.logical_and(
                    np.log(ratios) >= log_lower, np.log(ratios) <= log_upper
                )
                pct = 100 * np.mean(within)

                print(
                    f"=> h/q in [{lower:.3f}, {upper:.3f}]: {pct:.2f}% "
                    f"(min {ratios.min():.3f}, max {ratios.max():.3f})"
                )

    # save final generator
    torch.save(gen.state_dict(), "generator.pth")
    print("Training finished. Generator saved as generator.pth")


# if __name__ == "__main__":
#     train_fgan()
