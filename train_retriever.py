import argparse
from src.retriever.train_retriever import run_retriever_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch Transformer Dense Retriever")
    parser.add_argument("--debug", action="store_true", help="Run quick debug pass")
    parser.add_argument("--overfit-test", action="store_true", help="Run overfitting check on small subset")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--preset", type=str, default="SMALL", choices=["SMALL", "MEDIUM", "LARGE"])
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()

    run_retriever_training(
        debug=args.debug,
        overfit_test=args.overfit_test,
        resume=args.resume,
        preset=args.preset,
        epochs=args.epochs
    )
