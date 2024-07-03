#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install openai
from openai import OpenAI
import pandas as pd
from collections import defaultdict
import json
import os
import re

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


# In[2]:


# Define the path to the JSONL file
file_train = r"C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\germeval-competition-traindev.jsonl"
file_test =r"C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\germeval-competition-test.jsonl"
# files/germeval-competition-test.jsonl
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


# In[3]:


client = OpenAI(api_key = 'sk-proj-e0qJ5dbTiqsqDxu9VzggT3BlbkFJRIxLl5vAdZyxOCdMDSZK')

def format_few_shot_prompt(examples, max_examples=10):
    prompt = "The following texts are in German. Assign a label from 0 (not offensive) to 4 (most offensive):\n\n"
    for example in examples[:max_examples]:
        prompt += f"Text: {example['text']}\nLabel: {example['label']}\n\n"
    return prompt

# def truncate_text(text, max_length=512):
#     return text if len(text) <= max_length else text[:max_length] + "..."

def generate_prediction(text, few_shot_prompt, max_tokens=100):
   # text = truncate_text(text)  # Truncate the text to avoid exceeding the context length
    prompt = few_shot_prompt + f"Text: {text}\nLabel: "
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  ###3.5 turno
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        stop=["\n"],
        temperature=0.3
    )
    prediction = response.choices[0].message.content.strip()
    
    # Ensure prediction is a valid integer between 0 and 5
    try:
        prediction = int(prediction)
        if prediction < 0 or prediction > 4:
            prediction = None
    except ValueError:
        prediction = None

    return prediction


# In[5]:


# Initialize a new dictionary to store predictions
predictions_dict = {}

# Loop through each annotator
for annotator, examples_df in annotator_dfs.items():
    # Create a few-shot prompt for the current annotator
    few_shot_prompt = format_few_shot_prompt(examples_df.to_dict('records'))
    
    # Initialize a list to store predictions
    predictions = []
    
    # Generate predictions for each new text
    for text_record in annotator_dfs2[annotator].to_dict('records'):
        text = text_record['text']
        prediction = generate_prediction(text, few_shot_prompt)
        predictions.append(prediction)
    
    # Add predictions to the new column in the new_texts DataFrame
    annotator_dfs2[annotator]['prediction'] = predictions

# Combine all annotators' DataFrames into a single DataFrame for ease of analysis
combined_predictions = pd.concat(annotator_dfs2.values(), keys=annotator_dfs2.keys()).reset_index(level=0).rename(columns={'level_0': 'annotator'})

# Print the combined predictions DataFrame
combined_predictions


# In[ ]:


# for annotator, dfs in annotator_dfs2.items():
#     for i in range(len(annotator_dfs2[annotator]['prediction'])):
#    # annotator_dfs2[annotator]['prediction']= annotator_dfs2[annotator]['prediction']==5
#         if annotator_dfs2[annotator]['prediction'][i]==5:
#            print(annotator_dfs2[annotator]['prediction'][i])


# In[11]:


for a, dfs in annotator_dfs2.items():
    annotator_dfs2[a]=annotator_dfs2[a].fillna(0)


# In[13]:


# Initialize the final data structure for test set
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
        final_data[text_id]['predictions'].append(row['prediction'])

# Convert the final data structure into a DataFrame
result = pd.DataFrame([{
    'id': text_id,
    'text': data['text'],
    'annotators': data['annotators'],
    'predictions': data['predictions']
} for text_id, data in final_data.items()])

# print(result)
result


# In[14]:


from collections import Counter
#Function to compute majority vote
def majority_vote(labels):
    count = Counter(labels)
    most_common_label, most_common_count = count.most_common(1)[0]
    # if most_common_count > len(labels) / 2:
    return most_common_label
    # else:
    #     return None

multi_maj = lambda predictions: (lambda count: (lambda max_count: [label for label, freq in count.items() if freq == max_count][0])
                                 (max(count.values()))) (Counter(predictions))

#result['multi_maj2'] = result['predictions'].apply(multi_maj)
# Define a function to convert multi-maj prediction
def convert_label(label):
    label_map = {0: "0-Kein", 1: "1-Gering", 2: "2-Vorhanden", 3: "3-Stark", 4: "4-Extrem"}
    return label_map.get(label, label)

# Adding new columns
result['bin_maj'] = result['predictions'].apply(lambda labels: 1 if majority_vote(labels)  != 0 else 0)
result['bin_one'] = result['predictions'].apply(lambda labels: 1 if any(label != 0 for label in labels) else 0)
result['bin_all'] = result['predictions'].apply(lambda labels: 1 if all(label != 0 for label in labels) else 0)
# result['multi_maj'] = result['predictions'].apply(lambda labels: majority_vote(labels) if majority_vote(labels) is not None else random.choice(labels))
result['multi_maj'] = result['predictions'].apply(multi_maj)
result['disagree_bin'] = result['predictions'].apply(lambda labels: 1 if len(set(label != 0 for label in labels)) > 1 else 0)


# Convert multi_maj labels
result['multi_maj'] = result['multi_maj'].apply(convert_label)
# print(preds_df)
result


# In[15]:


# Pop the columns
submission_1 = result[['id', 'bin_maj', 'bin_one','bin_all','multi_maj','disagree_bin']]

# Save to TSV file
tsv_file_path = r'C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\germeval_aproach_LLAMA3_5.tsv'
submission_1.to_csv(tsv_file_path, sep='\t', index=False)


# In[16]:


sub2_result=result.copy()

# Initialize the new columns
sub2_result['dist_bin_0'] = 0
sub2_result['dist_bin_1'] = 0
sub2_result['dist_multi_0'] = 0
sub2_result['dist_multi_1'] = float(0)
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
    dist_multi_1 = float(labels.count(1) / total_labels)
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


# In[17]:


# Pop the columns
submission_2 = sub2_result[['id', 'dist_bin_0', 'dist_bin_1','dist_multi_0','dist_multi_1','dist_multi_2','dist_multi_3','dist_multi_4']]

# Save to TSV file
tsv_file_path = r'C:\Users\kaoda\Desktop\PHD\Dataset-GERMEVAL 2024\Germeval_sub2LLAMA3.tsv'
submission_2.to_csv(tsv_file_path, sep='\t', index=False)

