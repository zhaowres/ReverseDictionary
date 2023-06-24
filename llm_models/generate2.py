import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import json
import csv
import numpy as np

#models: "t5-small-revdict/checkpoint-240000" "t5-base-revdict/checkpoint-240000" "t5-large-revdict/checkpoint-240000"
model_name = "t5-base-revdict/checkpoint-240000"
model_dir = f"../t5-models/{model_name}"

def generate(inputs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    mini_inputs = [inputs[i:i + 1] for i in range(0, len(inputs), 1)] 
    output = []
    for inp in mini_inputs:
        max_input_length = 128

        inp = tokenizer([sentence for sentence in inp], max_length=max_input_length, return_tensors="pt", padding=True)

        # generate text for each batch
        predictions = model.generate(    
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"], 
            num_beams = 100,
            num_return_sequences = 100,

            # do_sample=True,

            num_beam_groups=4,
            diversity_penalty = 1.0
            )

        generated_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        generated_text = [generated_text[x:x+100] for x in range(0, len(generated_text), 100)]
        output.extend(generated_text)
    return output


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
    return np.median(pred_rank), accu_1/length*100, accu_10/length*100, accu_100/length*100, np.sqrt(np.var(pred_rank))

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
            
        predictions = generate(inputs)
        with open(f'../results/llm/t5_large_{test_sets[i]}_diverse_beam_results.csv', 'w') as results:
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
                        break


                writer.writerow([inputs[i], words[i], rank[i], predictions[i]])
            
            writer.writerow(evaluate_test(words, predictions))

print(generate(["the sweet liquid stored inside fruit that is healthy and delicious to drink"]))
evaluate()