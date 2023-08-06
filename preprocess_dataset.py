import argparse
import csv
import hashlib
from random import shuffle

import jsonlines
import numpy as np
import tiktoken


def read_jsonl_file(file_path: str):
    """Load a JSONL file and return a list of all records."""
    records = []
    with jsonlines.open(file_path) as reader:
        for record in reader:
            records.append(record)
    return records


def normalize_text(text: str):
    """Normalize the text."""
    # Replaces escaped double quotes with single quotes
    text = text.replace('\\"', "'")
    # Replaces newline characters with a space
    text = text.replace("\n", " ")
    return text


def generate_hash_id(text: str):
    """Generate a hash ID for the record."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 10 ** 8


def char_token_ratio(example, tokenizer):
    """Compute character/token ratio of the record with tokenizer."""
    tokens = tokenizer.encode(example)
    ratio = len(example) / len(tokens)
    return ratio


def check_uniques(data):
    """Check for duplicates in the dataset."""
    seen = set()
    unique_data = [x for x in data if x["id"] not in seen and not seen.add(x["id"])]
    return unique_data


def preprocess_dataset(file_path: str, tokenizer_model: str):
    data = read_jsonl_file(file_path)
    processed_data = []
    tokenizer = tiktoken.get_encoding(tokenizer_model)

    for record in data:
        prompt = normalize_text(record["prompt"])
        completion = normalize_text(record["completion"])

        # generate unique id based on file hash
        unique_id = generate_hash_id(record["prompt"] + record["completion"])

        # calculate sizes
        prompt_size = len(prompt)
        completion_size = len(completion)

        # calculate line stats for completion
        lines = completion.split(" ")
        line_sizes = [len(line) for line in lines]
        min_line_size = np.min(line_sizes)
        max_line_size = np.max(line_sizes)
        mean_line_size = np.mean(line_sizes)

        # compute character/token ratio
        ratio = char_token_ratio(record["completion"], tokenizer)

        processed_record = {
            "id": unique_id,
            "prompt": prompt,
            "completion": completion,
            "prompt_size": prompt_size,
            "completion_size": completion_size,
            "min_line_size": min_line_size,
            "max_line_size": max_line_size,
            "mean_line_size": mean_line_size,
            "ratio": ratio,
        }

        processed_data.append(processed_record)

    processed_data = check_uniques(processed_data)

    return processed_data


def write_to_csv(data, output_file):
    """Write the processed data to a CSV file."""
    shuffle(data)  # Shuffle data
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=data[0].keys(), quotechar='"', quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    """Main function to read arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for Hugging Face model training."
    )
    parser.add_argument("-i", "--input", help="Input JSONL file path", required=True)
    parser.add_argument("-o", "--output", help="Output CSV file path", required=True)
    parser.add_argument(
        "-t", "--tokenizer", help="Tokenizer model to be used", required=True
    )
    args = parser.parse_args()

    processed_data = preprocess_dataset(args.input, args.tokenizer)
    write_to_csv(processed_data, args.output)


if __name__ == "__main__":
    main()
