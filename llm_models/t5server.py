from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the fine-tuned T5 model and tokenizer
model_path = '/vol/bitbucket/wz1620/t5/t5-small-revdict/checkpoint-250000'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

@app.route('/generate-text', methods=['POST'])
def generate_text():
    input_text = request.json['text']

    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate multiple outputs
   
    output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=10, 
    no_repeat_ngram_size=2, 
    num_return_sequences=10, 
    early_stopping=True
)

    # Decode the generated text
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Return the generated text as a response
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run()
