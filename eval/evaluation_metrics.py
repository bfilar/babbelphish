import os
import time

import numpy as np
import openai
from datasets import load_dataset
from openai.error import APIConnectionError

import bleu

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_test_dataset(dataset_name="bfilar/babbelphish"):
    """Loads the test dataset from Hugging Face and returns a DataFrame"""
    dataset = load_dataset(dataset_name)
    df = dataset.data["test"].to_pandas()
    return df[["prompt", "completion"]]


def pass_at_k(num_answers, correct_answers, k):
    """
    Computes the Pass@k metric.
    """
    if num_answers - correct_answers < k:
        return 1.0
    return 1.0 - np.prod(
        1.0 - k / np.arange(num_answers - correct_answers + 1, num_answers + 1)
    )


def predict(
    prompt_str,
    k,
    model_name="curie:ft-sublime-security-2023-08-05-00-34-40",
    retries=3,
    sleep_time=1,
):
    """Makes prediction using the OpenAI API"""
    attempt = 0
    while attempt < retries:
        try:
            response = openai.Completion.create(
                model=model_name,
                max_tokens=128,
                temperature=0.3,
                stop=["\n"],
                prompt=f"{prompt_str.lower()} ->",
                n=k,
            )
            return [
                choice["text"].lstrip(" -> ").rstrip() for choice in response["choices"]
            ]
        except APIConnectionError:
            attempt += 1
            if attempt < retries:
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            raise e


def run_test(df, k):
    """Computes pass@k and BLEU score on a given dataset"""
    total_samples = len(df)
    correct_samples = 0
    total_bleu_score = 0

    for _, row in df.iterrows():
        correct_completion = row["completion"].lower()
        completions = predict(row["prompt"], k)
        completions = [completion.lower() for completion in completions]

        if correct_completion in completions:
            correct_samples += 1

        bleu_score, _, _, _, _, _ = bleu.compute_bleu(
            [[correct_completion.split()]],
            [completions[0].split()],
            max_order=4,
            smooth=True,
        )
        total_bleu_score += bleu_score

    pass_at_k_score = pass_at_k(total_samples, correct_samples, k)
    avg_bleu_score = total_bleu_score / total_samples

    return pass_at_k_score, avg_bleu_score


if __name__ == "__main__":
    df = load_test_dataset()
    pass_at_k_score, avg_bleu_score = run_test(df, k=3)
    print(f"Pass@3: {pass_at_k_score}")
    print(f"BLEU: {avg_bleu_score}")
