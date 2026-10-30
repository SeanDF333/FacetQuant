import argparse

from .predict import add_predict_arguments, predict
from .train import add_train_arguments, train
from .evaluate import add_evaluate_arguments, evaluate


def main():
    parser = argparse.ArgumentParser(prog="facetquant")
    commands = parser.add_subparsers(dest="command", required=True)
    add_train_arguments(commands.add_parser("train", help="Train on the Zenodo CHiMP detector dataset"))
    add_predict_arguments(commands.add_parser("predict", help="Measure crystal areas in images"))
    add_evaluate_arguments(commands.add_parser("evaluate", help="Evaluate on the published validation CSVs"))
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
