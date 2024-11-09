import json
import re
import pdb

def load_file(path):
    with open(path,'r') as f:
        datalines = f.readlines()
    return datalines

def extract_label(text):
    """
    Extracts the label from the given text. The label is expected to be in the format 'The label is [label]',
    where [label] can be either 'safe' or 'vulnerable'.

    Parameters:
    text (str): The text from which to extract the label.

    Returns:
    str: The extracted label ('safe', 'vulnerable', or 'Label not found' if not found).
    """
    # Regular expression pattern to find the label
    text = text.lower()
    pattern = r"the label is (safe|vulnerable)"

    # Search the text for the pattern
    match = re.search(pattern, text)

    # Extract and return the label if found, otherwise return 'Label not found'
    if match:
        return match.group(1)
    else:
        print("Label not found")
        if "is safe" in text:
            return "safe"
        elif "to be safe" in text:
            return "safe"
        else:
            return "vulnerable"

def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)


def find_code_function_remove_space_comments(text):
    text = remove_comments_and_docstrings(text)
    matches = re.findall(r"```Solidiy\n(.*?)\n```", text, re.DOTALL)
    # print(matches)
    # Remove new lines in the found code
    cleaned_matches = [match.replace("\n", " ") for match in matches]

    # Replace the original code with the cleaned code in the text
    # for original, cleaned in zip(matches, cleaned_matches):
    #     text = text.replace(original, cleaned)
    return cleaned_matches

def output_datasets1(path):
    datas = {}
    datalines = load_file(path)
    for line in datalines:
        line_dict = json.loads(line)
        id = line_dict['id']
        if id in datas.keys():
            continue
        label = extract_label(line_dict['completion'])
        if label not in ['safe','vulnerable']:
            print('error')
            continue
        # pdb.set_trace()
        code = find_code_function_remove_space_comments(line_dict['prompt'])
        if len(code) == 0:
            print('error: not match')
            continue
        datas[id] = {'label':label,'code':code[0]}

    return datas

def output_datasets2(path):
    datas = {}

    with open(path, 'r') as f:
        f = json.loads(f.read())
    
    # pdb.set_trace()

    for id, v in f.items():
    
        if id in datas.keys():
            continue
        label = v['ground_truth_label']
        if label not in ['safe','vulnerable']:
            print('error')
            continue
        # pdb.set_trace()
        code = find_code_function_remove_space_comments(v['input_prompts_list'][0])
        if len(code) == 0:
            print('error: not match')
            continue
        datas[id] = {'label':label,'code':code[0]}

    return datas