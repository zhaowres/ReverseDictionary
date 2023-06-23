import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import json
import csv

model_name = "t5-base-revdict/checkpoint-235000"
model_dir = f"../t5-models/{model_name}"

def generate(inputs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    max_input_length = 128

    prefix = "solve: "
            
    inputs = tokenizer([prefix + sentence for sentence in inputs], return_tensors="pt", padding=True)

    # generate text for each batch
  
    predictions = model.generate(    
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], 
                    num_beams = 100,
                    num_return_sequences = 100,
                    )

    generated_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)

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
            
        predictions = generate(inputs)
        with open(f'../results/llm/{model_name}_{test_set}_results_.csv', 'w') as results:
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

evaluate()