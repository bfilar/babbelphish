# BabbelPhish
> Accelerating Adoption of Domain-Specific Languages with Large Language Models

## Getting Started
To install the required libraries, use:

```
pip install -r requirements.txt
```

Getting Started
Clone the repository:
```
git clone https://github.com/bfilar/babbelphish.git
cd babbelphish
```
Next, you need to set your OpenAI API key as an environment variable:

```
export OPENAI_API_KEY='your-api-key-here'
```

## Fine-Tuning a Model with finetune.py
The finetune.py script provides a complete workflow to fine-tune an OpenAI model on a custom dataset. In this specific use case, we are fine-tuning the Curie model on the Sublime Security BabbelPhish dataset, but the code can be easily adapted to other datasets and models.

### Usage
Set the OpenAI API Key: Ensure your OpenAI API key is set as an environment variable or modify the script to include it directly.

```
export OPENAI_API_KEY='your-api-key-here'
```
1. Load and Prepare Data: The script loads the dataset and preprocesses the prompts and completions.
2. Convert to JSONL and Upload: The data is converted to JSONL format and uploaded to OpenAI.
3. Fine-Tune the Model: The script initiates the fine-tuning process using the uploaded data.
4. Monitor Fine-Tuning Events: You can track the progress and events of the fine-tuning process through the OpenAI Dashboard or modify the script to print events to the console.

To run the script, simply execute:
```
python finetune.py
```

## MQL Tokenizer Webapp
The MQL Tokenizer Webapp is a Flask-based web application that allows users to input a piece of text and tokenize it using different tokenizers, including GPT-2, GPT-3, GPT-4, and a custom tokenizer trained on MQL dataset. The tokenization results, alongside some useful tokenizer metrics, are displayed in a user-friendly way.

### Setup & Installation
To get started, you will need to install the required packages. This can be done by running:
```
pip install -r requirements.txt
```
Next, navigate to the webapp directory:
```
cd tokenizer_webapp
```

Then, start the Flask application:
```
python app.py
```
Your application should now be running at http://127.0.0.1:5000/.

### Usage
To use the webapp, simply enter a piece of text into the text area and select the tokenizer you would like to use. Click the 'Submit' button to tokenize the text. The results will be displayed below, showing each token, its position in the sentence, and its corresponding color. You can also see the total number of tokens, unique tokens, and average tokens per sentence.

To switch to a different tokenizer, simply select the desired tokenizer and click 'Submit' again. You can also reset the form by clicking the 'Reset' button.

Again, you will want to customize this to fit your specific application, including providing more detailed instructions or screenshots if necessary.

## Training a Custom GPT Tokenizer
You can train a custom GPT tokenizer on your dataset by using the `train_tokenizer.py` script. This script reads a dataset from a JSONL file, trains a new tokenizer based on a base model, and saves the new tokenizer to a directory of your choice.

### Usage
The `train_tokenizer.py` script accepts the following command-line arguments:

- `--base_model`: The model to use as the base for training the new tokenizer. (Default: `"gpt2"`)
- `--dataset_path`: The path to the dataset file in JSONL format.
- `--vocab_size`: The vocabulary size for the new tokenizer. (Default: `35000`)
- `--output_dir`: The directory where the new tokenizer will be saved. (Default: `"mql-tokenizer"`)

### Example
To train a tokenizer from the gpt2 model using a dataset in `data/` and a vocabulary size of 35000, saving the result in mql-tokenizer, you would run:

```{bash}
python train_tokenizer.py --base_model gpt2 --dataset_path data/mql_prompts_prepared.jsonl --vocab_size 35000 --output_dir mql-tokenizer
```
After the script finishes, the new tokenizer will be saved in the mql-tokenizer directory. You can then use this tokenizer for other tasks.

## Tokenizer Metrics
The Tokenizer Metrics script allows you to analyze the performance and characteristics of different tokenizers. It calculates various metrics such as Out-of-Vocabulary (OOV) rate, tokenization granularity, information loss, token type ratio, and reversibility. This tool is especially useful when comparing different tokenizers or fine-tuning a custom tokenizer.

### Usage
The Tokenizer Metrics script can be run from the command line as follows:
```
python tokenizer_metrics.py --tokenizer_path "tokenizer_directory" --dataset_path "data/mql_prompts_prepared.jsonl"
```

In the above command, replace "tokenizer_directory" with the path to your tokenizer and "data/mql_prompts_prepared.jsonl" with the path to your JSONL dataset.

The script will then output the calculated metrics to the console, for example:

```
OOV rate: 0.30
Tokenization granularity: 0.37
Information loss: 0.00
Token type ratio: 0.012
Reversibility: 0.00
```

These metrics provide valuable insights into the tokenizer's performance on the provided dataset.


## Evaluation Metrics

### Usage

Next, navigate to the evaluation directory:
```
cd eval
```
You can run the `evaluation_metrics.py` script via:
```
python evaluation_metrics.py
```

This script will load the babbelphish test dataset and perform a pass@k and BLEU score evaluation using a fine-tuned language model from OpenAI.

### Metrics
The evaluation_metrics.py script evaluates the performance of the model using two metrics:

_pass@k_: This measures the proportion of test cases where the correct answer is within the model's top 'k' predictions.

_BLEU_: This compares the model's output to reference translations and scores the similarity, providing an indication of how closely the model's output matches the expected output. It is computed on the top-ranked prediction.
