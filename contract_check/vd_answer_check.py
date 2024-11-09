import re
import concurrent.futures
import json
from tqdm import tqdm 
import pdb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from vd_oai_model_interface import chat_gpt_text_completion
from prompt_templates import *
from data import data_preprocess

# def load_dataset(path,index):
#     data_name = os.listdir(path)[index]
#     path = os.path.join(path, data_name)
#     with open(path, 'r') as f:
#         data_lines = f.readlines()
#     return data_lines

def check_answer(line):
    code = line['code']
    messages = [
        {'role':'system','content':STEM_CORRECT_SYSTEM},
        {'role':'user','content':STEM_CORRECT_USER.format(code=code)}
        ]
    while True:
        try:
            answer = chat_gpt_text_completion(messages=messages)
        except Exception as e:
            answer = ''

        if answer == '':
            print('reques again!~')
            continue
        else:
            start_index = answer.find('{')
            end_index = answer.rfind('}')
            json_str = answer[start_index:end_index + 1]
            try:
                json_str = json.loads(json_str)
            except:
                continue
            # pdb.set_trace()
            line.update(json_str)
            break
    return line

if __name__ == '__main__':
    # path = 'VD-data/data/train_data_class_public.jsonl'
    # path = 'VD-data/data/val_data_class_public.jsonl'
    path = 'VD-data/data/test_public.json'
    code_source = 'Solidity_test'
    project_name = 'Filtered_DS'
    datalines = data_preprocess.output_datasets2(path)

    with open(f'./{project_name}/{code_source}.jsonl', 'a+', encoding='utf-8') as f:
        with concurrent.futures.ThreadPoolExecutor(10) as executor:
            futures = [executor.submit(check_answer, item) for _,item in datalines.items()]
            results = []
            for ind, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
                f.write(json.dumps(future.result(), ensure_ascii=False) + '\n')