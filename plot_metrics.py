import os
import json
from config import LOGS_DIR

def plot_history():
    os.makedirs(LOGS_DIR, exist_ok=True)

    r_file = os.path.join(LOGS_DIR, "retriever_history.json")
    g_file = os.path.join(LOGS_DIR, "generator_history.json")

    r_hist = []
    g_hist = []

    if os.path.exists(r_file):
        with open(r_file, "r") as f:
            r_hist = json.load(f)

    if os.path.exists(g_file):
        with open(g_file, "r") as f:
            g_hist = json.load(f)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        if r_hist:
            epochs = [x["epoch"] for x in r_hist]
            losses = [x["loss"] for x in r_hist]
            plt.plot(epochs, losses, label="Retriever Loss", color="blue", marker="o")

        if g_hist:
            epochs = [x["epoch"] for x in g_hist]
            losses = [x["loss"] for x in g_hist]
            plt.plot(epochs, losses, label="Generator Loss", color="green", marker="s")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History (PyTorch Local RAG)")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(LOGS_DIR, "training_loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[OK] Saved training loss plot to: {plot_path}")
    except ImportError:
        print("[!] Matplotlib not installed. Generating text summary log instead...")
        summary_path = os.path.join(LOGS_DIR, "training_loss_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=== TRAINING LOSS SUMMARY ===\n\n")
            if r_hist:
                f.write("--- Retriever Training History ---\n")
                for entry in r_hist:
                    f.write(f"Epoch {entry['epoch']:02d}: Loss = {entry['loss']:.4f}\n")
            if g_hist:
                f.write("\n--- Generator Training History ---\n")
                for entry in g_hist:
                    f.write(f"Epoch {entry['epoch']:02d}: Loss = {entry['loss']:.4f}\n")
        print(f"[OK] Saved text loss summary to: {summary_path}")

if __name__ == "__main__":
    plot_history()
