from __future__ import annotations

import argparse
import json

from etri_human_challenge.sequence_lite import train_sequence_lite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "tcn"])
    parser.add_argument("--window-size", default=7, type=int)
    parser.add_argument("--epochs", default=8, type=int)
    args = parser.parse_args()

    scores = train_sequence_lite(model_type=args.model_type, window_size=args.window_size, epochs=args.epochs)
    print(json.dumps({"status": "ok", "model_type": args.model_type, "mean_log_loss": scores["mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

