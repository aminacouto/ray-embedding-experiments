"""Cria dados sintéticos reproduzíveis para validar e comparar o pipeline."""
import argparse
import csv
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use um inteiro positivo.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/texts.csv"))
    args = parser.parse_args()
    topics = ["parallel computing", "data processing", "vector search", "machine learning"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text"])
        for i in range(args.rows):
            writer.writerow([i, f"Document {i} discusses {topics[i % len(topics)]} and its practical applications."])
    print(f"Criados {args.rows} textos em {args.output}")


if __name__ == "__main__":
    main()
