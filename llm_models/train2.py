import os
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/wz1620/.cache'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/wz1620/.cache'

import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_checkpoint = "t5-large"

def finetune_t5(model_name):

    datasets = load_dataset('json', data_files={"train": "../data/data_train.json", "validation": "../data/data_dev.json"})
    datasets = datasets.remove_columns(["lexnames", "root_affix", "sememes"])

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    max_input_length = 128
    max_target_length = 32
    prefix = "solve: "

    def preprocess_data(examples):
        inputs = [prefix + text for text in examples["definitions"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["word"], max_length=max_target_length, padding=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = datasets.map(preprocess_data, batched=True)

    batch_size = 8
    model_dir = f"../t5-models/{model_name}_revdict"

    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=20000,
        save_strategy="steps",
        save_steps=20000,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        fp16=True,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # Function that returns an untrained model to be trained
    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

finetune_t5(model_checkpoint)