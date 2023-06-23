import os
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/wz1620/.cache'
os.environ['HF_DATASETS_CACHE'] = '/vol/bitbucket/wz1620/.cache'


from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch

def train_t5_model(model_type = "t5-small"):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset('json', data_files={"train": "../data/data_train.json", "validation": "../data/data_dev.json"})
    dataset = dataset.remove_columns(["lexnames", "root_affix", "sememes"])


    prefix = "solve: "
    max_input_length = 32
    max_target_length = 4
    # prefix + 
    def preprocess_data(examples):
        model_inputs = tokenizer(
            [prefix + sequence for sequence in examples['definitions']],
            padding="longest",
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )

        # encode the targets
        target_encoding = tokenizer(
            examples['word'],
            padding="longest",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    args = Seq2SeqTrainingArguments(
        f"../t5models/{model_name}-revdict",
        save_total_limit = 2,
        save_strategy = "no",
        load_best_model_at_end=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    # trainer.save_model()