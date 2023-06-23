import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import json
import csv
import torch

model_name = "t5-base-revdict/checkpoint-235000"
model_dir = f"/vol/bitbucket/wz1620/_githubrepo/ReverseDictionary/t5_models/{model_name}"

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
    print(generated_text)
    return generated_text

def generate_b(location):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    max_input_length = 128

    prefix = "solve: "

    inputs = []
    with open(location) as f:
        data = json.load(f)
        for point in data:
            inputs.append(point['definitions'])

            
    inputs = tokenizer([prefix + sentence for sentence in inputs], return_tensors="pt", padding=True)

 
    # generate text for each batch
  
    predictions = model.generate(    
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], 
                    num_beams = 100,
                    num_return_sequences = 100,
                    )

    generated_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    print(generated_text)
    return generated_text

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

generate_b("/vol/bitbucket/wz1620/_githubrepo/ReverseDictionary/data/dd.json")