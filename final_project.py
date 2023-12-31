import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.nn import BCEWithLogitsLoss

file_name = 'movieplots.csv'
data_frame = pd.read_csv(file_name)

genre_labels = ['western', 'comedy', 'short', 'film', 'drama', 'adventure', 'fantasy', 'sports', 'horror', 'crime', 'historical', 'biography', 'action', 'romantic', 'mystery', 'romance', 'war', 'spy', 'animated', 'family', 'thriller', 'comedydrama', 'musical', 'science', 'fiction', 'noir', 'animation', 'scifi', 'suspense', 'social', 'anime']

genre_set = set()
for genre_list in data_frame['Genre'].str.split():
    for g in genre_list:
        if g in genre_labels:
            genre_set.add(g)

genre2id = {genre: idx for idx, genre in enumerate(genre_labels)}
id2genre = {idx: genre for genre, idx in genre2id.items()}
num_labels = len(genre_labels)

def encode_genres(genres):
    genre_vector = [0] * num_labels
    for genre in genres:
        if genre in genre2id:
            genre_vector[genre2id[genre]] = 1
    return genre_vector

def prepare_data(row):
    genres = [g for g in row['Genre'].split() if g in genre_labels]
    if not genres:
        return None 
    genre_vector = encode_genres(genres)
    return {'text': row['Plot'], 'labels': genre_vector}

processed_data = data_frame.apply(prepare_data, axis=1).dropna()
data_list = processed_data.to_list()

dataset = Dataset.from_list(data_list)
small_dataset = dataset.select(range(1000)) 

model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_output['labels'] = examples['labels']
    return tokenized_output

tokenized_dataset = small_dataset.map(tokenize_function, batched=True)

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,  
    per_device_train_batch_size=16,  
    num_train_epochs=2,  
    weight_decay=0.01,
    evaluation_strategy="no",
    gradient_accumulation_steps=2  
)

trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save Model
trainer.save_model("distilbert-for-movie-genre")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-for-movie-genre",
                                                           num_labels=num_labels)

# # Load Model
tokenizer = AutoTokenizer.from_pretrained("distilbert-for-movie-genre")

def predict_genres(text, tokenizer, model, threshold=0.3):

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = (torch.sigmoid(logits) > threshold).int()

    predicted_genres = [id2genre[i] for i, label in enumerate(predictions[0]) if label]

    return predicted_genres

test_plot = "Mary Pickford plays Priscilla an unemployed maid who finds work at a farm. There she meets a no-good peddler who starts flirting with her and makes her fall in love with him. He runs up a gambling bill and asks her to help him pay his debts or he won't be able to marry her."

predicted_genres = predict_genres(test_plot, tokenizer, model)
print("Predicted Genres:", predicted_genres)
