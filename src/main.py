"""
main.py

Entry point script that wires CLI arguments and runs the full
model selection and stacking pipeline defined in modeling.ModelTrainer.
"""

from pathlib import Path
import argparse

from modeling import ModelTrainer

import random
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="House Prices Advanced Regression Techniques"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="SalePrice",
        help="Name of the target column",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./dataset/train.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory for logs, models and results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top models to keep before tuning",
    )
    parser.add_argument(
        "--trials-model",
        type=int,
        default=30,
        help="Number of Optuna trials per single model",
    )
    parser.add_argument(
        "--trials-stack",
        type=int,
        default=40,
        help="Number of Optuna trials for stacking",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV folds for model evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_global_seed(args.random_state)

    trainer = ModelTrainer(
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=str(args.output_dir),
    )

    # 1. Load data
    trainer.load_data(str(args.data_path))

    # 2. Train test split
    trainer.split_data()

    # 3. Build feature preprocessing pipeline
    trainer.build_preprocessing()

    # 4. Run full model selection and stacking pipeline
    trainer.run_full_model_selection_and_stacking(
        top_k=args.top_k,
        n_trials_model=args.trials_model,
        n_trials_stack=args.trials_stack,
        cv_splits=args.cv_splits,
    )


if __name__ == "__main__":
    main()
