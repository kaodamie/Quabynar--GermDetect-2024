#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import json
import re

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
# max_len = 128


 # Define a Dataset class for BERT
class ExtremismDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


# In[2]:


# Function to train a BERT model for a single annotator
def train_bert_model(train_loader, val_loader):
    model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Training {annotator} Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f'Annotator:{annotator} Epoch {epoch + 1} Validation Accuracy: {val_accuracy}')
        print(f'Annotator:{annotator} Epoch {epoch + 1} Validation Precision: {val_precision}')
        print(f'Annotator:{annotator} Epoch {epoch + 1} Validation Recall: {val_recall}')
        print(f'Annotator:{annotator} Epoch {epoch + 1} Validation F1 Score: {val_f1}')

    return model


# In[3]:


# 'load data'
# Function to read the JSONL file and process the data
def process_jsonl_file(file_path):
    data = []
    
    # Read the JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    
    # Determine if the file is for training or test based on the presence of specific keys
    is_training = 'annotations' in data[0] and 'user' in data[0]['annotations'][0]

    # Prepare lists to hold the processed data
    textid = []
    texts = []
    annotators = []
    labels = [] if is_training else None

    # Process each entry in the JSONL file
    for entry in data:
        text_id = entry['id']
        text = entry['text']
        
        if is_training:
            for annotation in entry['annotations']:
                annotator = annotation['user']
                label = annotation['label']
                
                textid.append(text_id)
                texts.append(text)
                annotators.append(annotator)
                labels.append(label)
        else:
            for annotator in entry['annotators']:
                textid.append(text_id)
                texts.append(text)
                annotators.append(annotator)

    # Create a DataFrame
    if is_training:
        df = pd.DataFrame({
            'id': textid,
            'text': texts,
            'annotator': annotators,
            'label': labels
        })
    else:
        df = pd.DataFrame({
            'id': textid,
            'text': texts,
            'annotator': annotators
        })
    
    return df

# Example usage
# train_df = process_jsonl_file('path_to_train_file.jsonl')
# test_df = process_jsonl_file('path_to_test_file.jsonl')

# Function to sort data into separate DataFrames for each annotator
def sort_data_by_annotator(df):
    annotator_groups = df.groupby('annotator')
    
    # Dictionary to hold DataFrames for each annotator
    annotator_dfs = {}
    
    for annotator, group in annotator_groups:
        annotator_dfs[annotator] = group.reset_index(drop=True)
    
    return annotator_dfs


def preprocess_labels(df):
    df['label_text'] = df['label'].apply(lambda x: re.sub(r'\d+', '', x).strip())
    df['label'] = df['label'].apply(lambda x: int(re.search(r'\d+', x).group()))
    return df

# Save the DataFrame to a CSV file (optional)
# df.to_csv('annotator_labels.csv', index=False)

# Display the DataFrame


# In[4]:


# Define the path to the JSONL file
file_train = r"C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\germeval-competition-traindev.jsonl"
file_test =r"C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\germeval-competition-test.jsonl"

# Process the file and create the DataFrame
df = process_jsonl_file(file_train)
df2 = process_jsonl_file(file_test)
# Sort the data by annotator
annotator_dfs = sort_data_by_annotator(df)
annotator_dfs2=sort_data_by_annotator(df2)

for d in annotator_dfs:    
    preprocess_labels(annotator_dfs[d])
    annotator_dfs[d].pop('annotator')
    

for d in annotator_dfs2:    
    # preprocess_labels(annotator_dfs[d])
    annotator_dfs2[d].pop('annotator')


# In[ ]:


#test set has the same annotators as the training set
print('training_set:')
for annotator in annotator_dfs.keys():
    print(annotator)
print('test_set:')
for annotator in annotator_dfs2.keys():
    print(annotator)


# In[33]:


# Train and evaluate each annotator model
for annotator, df in annotator_dfs.items():
    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.1, random_state=42
    )
    
    # Create DataLoader for training and validation
    train_dataset = ExtremismDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_dataset = ExtremismDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    # Create test DataLoader (without labels)
    test_dataset = ExtremismDataset_test(test_df[annotator]['text'], tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# In[ ]:


print(test_loader.items())


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#test set renamed
test_df= annotator_dfs2.copy()

# Create dictionaries to store predictions
all_preds = defaultdict(list)
all_text_ids = defaultdict(list)

# Dictionary to store metrics for each annotator
annotator_metrics = {}

# Train and evaluate each annotator model
for annotator, df in annotator_dfs.items():
    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create DataLoader for training and validation
    train_dataset = ExtremismDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_dataset = ExtremismDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # Create test DataLoader (without labels)
    test_dataset = ExtremismDataset(test_df[annotator]['text'], tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Train the model
    model = train_bert_model(train_loader, val_loader)

    # Evaluate the model on the test set
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Evaluating {annotator}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            test_preds.extend(preds)
            
    # Store predictions
    annotator_dfs2[annotator]['predictions'] = test_preds
    all_preds[annotator].extend(test_preds)
    all_text_ids[annotator].extend(test_df[annotator]['id'])
    


# In[6]:


train_dfs= annotator_dfs.copy()
# Evaluate the model on the training set
for annotator, df in train_dfs.items():

    training_dataset = ExtremismDataset(train_dfs[annotator]['text'], tokenizer=tokenizer)
    training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)
    model.eval()
    train_preds = []
    with torch.no_grad():
        for batch in tqdm(training_loader, desc=f'Evaluating {annotator}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            train_preds.extend(preds)
            
    # Store predictions
    train_dfs[annotator]['predictions'] = train_preds


# In[85]:


train_result


# In[7]:


# Training data evaluation metrics

# Initialize the final data structure
final_data = {}

# Iterate through each DataFrame in the dictionary
for annotator, df in train_dfs.items():
    for index, row in df.iterrows():
        text_id = row['id']
        if text_id not in final_data:
            final_data[text_id] = {
                'text': row['text'],
                'annotators': [],
                'labels': [],
                'predictions': []
            }
        final_data[text_id]['annotators'].append(annotator)
        final_data[text_id]['labels'].append(row['label'])
        final_data[text_id]['predictions'].append(row['predictions'])

# Convert the final data structure into a DataFrame
train_result = pd.DataFrame([{
    'text_id': text_id,
    'text': data['text'],
    'annotators': data['annotators'],
    'labels': data['labels'],
    'predictions': data['predictions']
} for text_id, data in final_data.items()])

# majority vote function
from collections import Counter
#Function to compute majority vote
def majority_vote(labels):
    count = Counter(labels)
    most_common_label, most_common_count = count.most_common(1)[0]
    # if most_common_count > len(labels) / 2:
    return most_common_label
    # else:
    #     return None

# Adding new columns
train_result['bin_maj'] = train_result['predictions'].apply(lambda labels: 1 if majority_vote(labels)  != 0 else 0)
train_result['bin_one'] = train_result['predictions'].apply(lambda labels: 1 if any(label != 0 for label in labels) else 0)
train_result['bin_all'] = train_result['predictions'].apply(lambda labels: 1 if all(label != 0 for label in labels) else 0)
train_result['multi_maj'] = train_result['predictions'].apply(lambda labels: majority_vote(labels) if majority_vote(labels) is not None else labels[0])
train_result['disagree_bin'] = train_result['predictions'].apply(lambda labels: 1 if len(set(label != 0 for label in labels)) > 1 else 0)

# Adding new columns
train_result['labels_bin_maj'] = train_result['labels'].apply(lambda labels: 1 if majority_vote(labels)  != 0 else 0)
train_result['labels_bin_one'] = train_result['labels'].apply(lambda labels: 1 if any(label != 0 for label in labels) else 0)
train_result['labels_bin_all'] = train_result['labels'].apply(lambda labels: 1 if all(label != 0 for label in labels) else 0)
train_result['labels_multi_maj'] = train_result['labels'].apply(lambda labels: majority_vote(labels) if majority_vote(labels) is not None else labels[0])
train_result['labels_disagree_bin'] = train_result['labels'].apply(lambda labels: 1 if len(set(label != 0 for label in labels)) > 1 else 0)

# print(preds_df)
# train_result


# Calculate metrics
bin_maj_accuracy = accuracy_score(train_result['labels_bin_maj'], train_result['bin_maj'])
bin_maj_precision = precision_score(train_result['labels_bin_maj'], train_result['bin_maj'], average='weighted')
bin_maj_recall = recall_score(train_result['labels_bin_maj'], train_result['bin_maj'], average='weighted')
bin_maj_f1 = f1_score(train_result['labels_bin_maj'], train_result['bin_maj'], average='weighted')

print(f'bin_maj_Accuracy: {bin_maj_accuracy}')
print(f'bin_maj_Precision: {bin_maj_precision}')
print(f'bin_maj_Recall: {bin_maj_recall}')
print(f'bin_maj_F1 Score: {bin_maj_f1}')

print()
# metrics for bin_one
bin_one_accuracy = accuracy_score(train_result['labels_bin_one'], train_result['bin_one'] )
bin_one_precision = precision_score(train_result['labels_bin_one'], train_result['bin_one'], average='weighted')
bin_one_recall = recall_score(train_result['labels_bin_one'], train_result['bin_one'], average='weighted')
bin_one_f1 = f1_score(train_result['labels_bin_one'], train_result['bin_one'], average='weighted')
print(f'bin_one_accuracy: {bin_one_precision}')
print(f'bin_one_precision: {bin_one_recall}')
print(f'bin_one_recall: {bin_one_f1}')
print(f'bin_one_F1 Score: {bin_one_f1}')
print()

# metrics for bin_all
bin_all_accuracy = accuracy_score(train_result['labels_bin_all'], train_result['bin_all'])
bin_all_precision = precision_score(train_result['labels_bin_all'], train_result['bin_all'], average='weighted')
bin_all_recall = recall_score(train_result['labels_bin_all'], train_result['bin_all'], average='weighted')
bin_all_f1 = f1_score(train_result['labels_bin_all'], train_result['bin_all'], average='weighted')
print(f'bin_all_accuracy: {bin_all_precision}')
print(f'bin_all_precision: {bin_all_recall}')
print(f'bin_all_recall: {bin_all_f1}')
print(f'bin_all_F1 Score: {bin_all_f1}')
print()
# metrics for multi_maj
multi_maj_accuracy = accuracy_score(train_result['labels_multi_maj'], train_result['multi_maj'] )
multi_maj_precision = precision_score(train_result['labels_multi_maj'], train_result['multi_maj'], average='weighted')
multi_maj_recall = recall_score(train_result['labels_multi_maj'], train_result['multi_maj'], average='weighted')
multi_maj_f1 = f1_score(train_result['labels_multi_maj'], train_result['multi_maj'], average='weighted')
print(f'multi_maj_accuracy: {multi_maj_precision}')
print(f'multi_maj_precision: {multi_maj_recall}')
print(f'multi_maj_recall: {multi_maj_f1}')
print(f'multi_maj_F1 Score: {multi_maj_f1}')
print()
# metrics for disagree_bin
disagree_bin_accuracy = accuracy_score(train_result['labels_disagree_bin'], train_result['disagree_bin'])
disagree_bin_precision = precision_score(train_result['labels_disagree_bin'], train_result['disagree_bin'], average='weighted')
disagree_bin_recall = recall_score(train_result['labels_disagree_bin'], train_result['disagree_bin'], average='weighted')
disagree_bin_f1 = f1_score(train_result['labels_disagree_bin'], train_result['disagree_bin'], average='weighted')
print(f'disagree_bin_accuracy: {disagree_bin_precision}')
print(f'disagree_bin_precision: {disagree_bin_recall}')
print(f'disagree_bin_recall: {disagree_bin_f1}')
print(f'disagree_bin_F1 Score: {disagree_bin_f1}')


# In[130]:


train_dfs


# In[8]:


# Initialize the final data structure
final_data = {}

# Iterate through each DataFrame in the dictionary
for annotator, df in annotator_dfs2.items():
    for index, row in df.iterrows():
        text_id = row['id']
        if text_id not in final_data:
            final_data[text_id] = {
                'text': row['text'],
                'annotators': [],
                'predictions': []
            }
        final_data[text_id]['annotators'].append(annotator)
        final_data[text_id]['predictions'].append(row['predictions'])

# Convert the final data structure into a DataFrame
result = pd.DataFrame([{
    'id': text_id,
    'text': data['text'],
    'annotators': data['annotators'],
    'predictions': data['predictions']
} for text_id, data in final_data.items()])

# print(result)
result


# In[93]:


df2


# In[95]:


len(df2['id'].unique())


# In[9]:


from collections import Counter
#Function to compute majority vote
def majority_vote(labels):
    count = Counter(labels)
    most_common_label, most_common_count = count.most_common(1)[0]
    # if most_common_count > len(labels) / 2:
    return most_common_label
    # else:
    #     return None

# Define a function to convert multi-maj prediction
def convert_label(label):
    label_map = {0: "0-Kein", 1: "1-Gering", 2: "2-Vorhanden", 3: "3-Stark", 4: "4-Extrem"}
    return label_map.get(label, label)

# Adding new columns
result['bin_maj'] = result['predictions'].apply(lambda labels: 1 if majority_vote(labels)  != 0 else 0)
result['bin_one'] = result['predictions'].apply(lambda labels: 1 if any(label != 0 for label in labels) else 0)
result['bin_all'] = result['predictions'].apply(lambda labels: 1 if all(label != 0 for label in labels) else 0)
result['multi_maj'] = result['predictions'].apply(lambda labels: majority_vote(labels) if majority_vote(labels) is not None else labels[0])
result['disagree_bin'] = result['predictions'].apply(lambda labels: 1 if len(set(label != 0 for label in labels)) > 1 else 0)


# Convert multi_maj labels
result['multi_maj'] = result['multi_maj'].apply(convert_label)
# print(preds_df)
result


# In[14]:


multi_maj = lambda predictions: (lambda count: (lambda max_count: [label for label, freq in count.items() if freq == max_count][0])
                                 (max(count.values()))) (Counter(predictions))

result['multi_maj2'] = result['predictions'].apply(multi_maj)

# Convert multi_maj labels
result['multi_maj2'] = result['multi_maj2'].apply(convert_label)
result


# In[22]:


# Check if the 'multi_maj' column is equal to 'multi_maj2' column for all rows
if (result['multi_maj'] == result['multi_maj2']).all()==True:
    print(True)
else:
    print(False)


# In[21]:


# Function to predict the majority label with specific rules for evaluation
def multi_maj(labels):
    count = Counter(labels)
    most_common = count.most_common()
    max_count = most_common[0][1]
    
    # Get all labels with the highest count
    top_labels = [label for label, count in most_common if count == max_count]
    
    # If there is a clear majority, return it
    if len(top_labels) == 1:
        return top_labels[0]
    # If there is no clear majority, return any of the highest occurring labels that is not 0
    for label in top_labels:
        if label != 0:
            return label
    # If all the top labels are 0, return 0
    return 0


result['multi_maj2'] = result['predictions'].apply(multi_maj)

# Convert multi_maj labels
result['multi_maj2'] = result['multi_maj2'].apply(convert_label)
result


# In[124]:


Counter(result.iloc[4]['predictions']).most_common(1)[0]


# In[30]:


result.tail(20)


# In[10]:


# Pop the columns
submission_1 = result[['id', 'bin_maj', 'bin_one','bin_all','multi_maj','disagree_bin']]

# Save to TSV file
tsv_file_path = r'C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\Bert_german_1.tsv'
submission_1.to_csv(tsv_file_path, sep='\t', index=False)


# In[ ]:





# In[12]:


sub2_result=result.copy()

# Initialize the new columns
sub2_result['dist_bin_0'] = 0
sub2_result['dist_bin_1'] = 0
sub2_result['dist_multi_0'] = 0
sub2_result['dist_multi_1'] = 0
sub2_result['dist_multi_2'] = 0
sub2_result['dist_multi_3'] = 0
sub2_result['dist_multi_4'] = 0

# Calculate the distributions
for index, row in sub2_result.iterrows():
    labels = row['predictions']
    total_labels = len(labels)
    
    # Binary distribution
    dist_bin_0 = labels.count(0) / total_labels
    dist_bin_1 = sum(1 for label in labels if label != 0) / total_labels
    
    # Multi-score distribution
    dist_multi_0 = labels.count(0) / total_labels
    dist_multi_1 = labels.count(1) / total_labels
    dist_multi_2 = labels.count(2) / total_labels
    dist_multi_3 = labels.count(3) / total_labels
    dist_multi_4 = labels.count(4) / total_labels
    
    # Assign values to the DataFrame
    sub2_result.at[index, 'dist_bin_0'] = dist_bin_0
    sub2_result.at[index, 'dist_bin_1'] = dist_bin_1
    sub2_result.at[index, 'dist_multi_0'] = dist_multi_0
    sub2_result.at[index, 'dist_multi_1'] = dist_multi_1
    sub2_result.at[index, 'dist_multi_2'] = dist_multi_2
    sub2_result.at[index, 'dist_multi_3'] = dist_multi_3
    sub2_result.at[index, 'dist_multi_4'] = dist_multi_4
sub2_result


# In[13]:


# Pop the columns
submission_2 = sub2_result[['id', 'dist_bin_0', 'dist_bin_1','dist_multi_0','dist_multi_1','dist_multi_2','dist_multi_3','dist_multi_4']]

# Save to TSV file
tsv_file_path = r'C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\Bert_german_sub2.tsv'
submission_2.to_csv(tsv_file_path, sep='\t', index=False)

