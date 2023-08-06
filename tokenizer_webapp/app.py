import tiktoken
from flask import Flask, render_template, request
from transformers import GPT2Tokenizer

app = Flask(__name__)

colors = ["purple", "green", "yellow", "red", "blue"]


def get_tokenizer_stats(token_tuples):
    """Get tokenizer statistics"""
    total_tokens = len(token_tuples)
    total_characters = sum(
        len(token[0]) for token in token_tuples
    )  # token[0] is the text of the token
    return {
        "Total tokens": total_tokens,
        "Total characters": total_characters,
        "Average token length": round(total_characters / total_tokens, 2),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    tokens = []
    stats = {}
    tokenizer_type = "gpt2"

    if request.method == "POST":
        text = request.form["text"]
        tokenizer_type = request.form["tokenizer"]

        if tokenizer_type == "gpt2":
            enc = GPT2Tokenizer.from_pretrained("mql-tokenizer")
        elif tokenizer_type == "gpt3":
            enc = tiktoken.get_encoding("p50k_base")
        elif tokenizer_type == "gpt4":
            enc = tiktoken.get_encoding("cl100k_base")

        token_ids = enc.encode(text)
        for i, token_id in enumerate(token_ids):
            token_text = enc.decode([token_id])
            tokens.append((token_text, i % 5))

        stats = get_tokenizer_stats(tokens)

    return render_template(
        "index.html",
        text=text,
        tokens=tokens,
        stats=stats,
        tokenizer_type=tokenizer_type,
    )


if __name__ == "__main__":
    app.run(debug=True)
