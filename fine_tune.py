import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Function to load the dataset with retries
def load_dataset_with_retries(dataset_name, config_name, retries=5, delay=10):
    for i in range(retries):
        try:
            return load_dataset(dataset_name, config_name)
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(delay)
    raise Exception("Failed to load dataset after multiple retries")

# Use the function to load the dataset
dataset = load_dataset_with_retries("cnn_dailymail", "3.0.0")

# Use a subset for testing
small_train_dataset = dataset["train"].select(range(1000))  # First 1000 samples
small_val_dataset = dataset["validation"].select(range(100))  # First 100 samples
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = small_val_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,  # Use fewer epochs for testing
    predict_with_generate=True,
    fp16=False,  # Set to True if you have a GPU with FP16 support
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
