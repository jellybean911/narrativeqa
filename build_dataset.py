import json
import csv
from datasets import load_dataset
import pandas as pd


json_list = []
with open('my-dataset-validation.json', "r") as f:
    for jsonObj in f:
        json_Dict = json.loads(jsonObj)
        json_list.append(json_Dict)
        if len(json_list) == 10:
            break


print(len(json_list))

# data_file = open('narrativeqa_dataset_train.csv', 'a', newline='', encoding='utf8')
# csv_writer = csv.writer(data_file)
# csv_writer.writerow(['id','text','question','answers'])


qa_pairs = []
window_size = 7600 # 32000 for 8k tokens, 7600 for 2k tokens (leave some space tor question and answers) each token is ~ 4 chars
stride = 512 # the overlap between each slice ~ 128 tokens
for json_obj in json_list:
    context = json_obj["document"]["text"]
    question = json_obj["question"]["text"]
    answer = json_obj["answers"][1]["text"]
    # generated sliced context 
    for start_idx in range(0, len(context), (window_size-stride)):
        end_idx = min((start_idx+window_size), len(context))
        context_slice = "".join(context[start_idx:end_idx]) 
        qa_pairs.append("<|endoftext|>" + "[context]: " + context_slice + "\n[question]: " + question + "\n[answer]: " + answer + "\n###\n"+ "<|endoftext|>")
    # csv_writer.writerow([json_obj["document"]["id"], json_obj["document"]["text"], json_obj["question"], json_obj["answers"]])

train_df = pd.DataFrame(qa_pairs,columns=['text'])
    
#drops na items if there are any
train_df = train_df.dropna()
train_df.to_csv("narrativeqa-validation_2k_sample.csv")
# dataset = load_dataset('csv', data_files={'train': '/home/jie/narrativeqa/narrativeqa_dataset_train.csv'})
# for split, data in dataset.items():
#     data.to_json(f"narrativeqa-{split}.json")