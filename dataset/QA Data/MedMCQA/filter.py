from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Step 1: Load the MedMCQA dataset
dataset = load_dataset("medmcqa")

# Step 2: Define cardiovascular-related keywords
cardiology_keywords = [
    "cardio", "heart", "hypertension", "arrhythmia", "ischemia", 
    "vascular", "aorta", "myocardial", "angina", "ecg", "atrial", "ventricular"
]

# Step 3: Filter function
def is_cardiology(example):
    subject = example.get("subject_name")
    topic = example.get("topic_name")
    subject = subject.lower() if isinstance(subject, str) else ""
    topic = topic.lower() if isinstance(topic, str) else ""
    return any(kw in subject or kw in topic for kw in cardiology_keywords)

# Step 4: Apply filter to train/validation/test sets with tqdm
def filter_with_progress(dataset_split, split_name):
    indices = []
    for i, example in enumerate(tqdm(dataset_split, desc=f"Filtering {split_name}")):
        if is_cardiology(example):
            indices.append(i)
    return dataset_split.select(indices)

cardio_train = filter_with_progress(dataset["train"], "train")
cardio_valid = filter_with_progress(dataset["validation"], "validation")
cardio_test = filter_with_progress(dataset["test"], "test")

# Step 5: Combine and convert to pandas DataFrame
df_train = pd.DataFrame(cardio_train)
df_valid = pd.DataFrame(cardio_valid)
df_test = pd.DataFrame(cardio_test)

df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

# Step 6: Save to CSV and JSONL
df_all.to_json("./medmcqa_cardiovascular_subset.jsonl", orient="records", lines=True, force_ascii=False)

print(f"Extracted {len(df_all)} cardiovascular-related MCQs and saved to 'medmcqa_cardiovascular_subset.csv' and '.jsonl'")