# Tokenizer Web Application

This web application is a visual aid for understanding the effect of tokenization on large language models, especially for code generation tasks. Different tokenizers can result in different behaviors, and the choice of tokenizer can have a significant impact on your model's performance.

## Features

1. Choice between Custom GPT-2, Standard GPT-3, and GPT-4 tokenizers.
2. Input any MQL and see how it's tokenized by the chosen model.
3. Visual representation of tokens and their corresponding types.
4. Detailed statistics including token count, unique tokens, and average token length.

## How to Run the App

The application is built using Flask. To run it, follow the steps below:

1. Clone this repository.
2. Install the necessary dependencies with `pip install -r requirements.txt`.
3. Run the app with `python app.py`.

This will start the server, and you can access the application in your web browser at `http://localhost:5000`.

## License

This project is [MIT](./LICENSE) licensed.
