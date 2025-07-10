from train import train_fgan

if __name__ == "__main__":
    train_fgan(f_divergence="js", steps=25000)
