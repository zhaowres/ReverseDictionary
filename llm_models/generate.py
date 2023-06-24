from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import csv

def t5(sentences):
    # Load the fine-tuned T5 model and tokenizer '/vol/bitbucket/wz1620/t5/t5-small-revdict/checkpoint-250000'
    model_path = "../t5_models/t5-base-revdict/checkpoint-235000"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    task_prefix = "solve: "

    inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

    # input_ids = tokenizer.encode(inputs, return_tensors='pt')

    # Generate multiple outputs
    output = model.generate(
        input_ids=inputs["input_ids"],
        max_length=50, 
        num_beams=120, 
        no_repeat_ngram_size = 2,m
        num_return_sequences=100, 
        early_stopping=True
    )

    # Decode and print the generated text
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    print("Generated Text: ", generated_text)
    n = 100 
    generated_text = [generated_text[i:i + n] for i in range(0, len(generated_text), n)]
    return generated_text

test_set_paths = ["../data/data_test_500_rand1_seen.json", "../data/data_test_500_rand1_unseen.json", "../data/data_desc_c.json"]
def evaluate_test(ground_truth, prediction):
    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0.
    length = len(ground_truth)
    pred_rank = []
    for i in range(length):
        try:
            pred_rank.append(prediction[i][:].index(ground_truth[i]))
        except:
            pred_rank.append(1000)
        if ground_truth[i] in prediction[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    accu_1 += 1
    return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))

def evaluate():
   

    test_sets = ["seen", "unseen", "description"]
    for i,test_set in enumerate(test_set_paths):
    
        inputs = []
        words = []
        with open(test_set) as f:
            data = json.load(f)
            for point in data:
                inputs.append(point['definitions'])
                words.append(point['word'])
            
        predictions = t5(inputs)
        with open(f'../results/llm/t5_base_results_.csv', 'w') as results:
            writer = csv.writer(results)
            writer.writerow(['Description', 'Solution', 'Prediction rank', 'Predictions'])

            total = len(words)
            correct = 0
            rank = [0] * total
            for i, value in enumerate(predictions):
                for j, value2 in enumerate(value):
                    if (value2 == words[i]):
                        correct += 1
                        rank[i] = j+1


                writer.writerow([inputs[i], words[i], rank[i], predictions[i]])
            
            writer.writerow(evaluate_test(words, predictions))

t5(["a feeling of excitement", "a musical instrument with strings"])
evaluate()
