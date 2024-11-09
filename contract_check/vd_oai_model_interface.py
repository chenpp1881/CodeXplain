# from kconf.get_config import get_json_config
import pdb
import requests
import random
import json
from openai import AzureOpenAI
import itertools

api_keys = []
key_iterator = itertools.cycle(api_keys)

def chat_gpt_text_completion(messages):
    model,api_key,url = next(key_iterator)
    try:
        client = AzureOpenAI(
            azure_endpoint = url,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        if response != "":
            if response.choices[0].finish_reason == "length":
                return ""
            return response.choices[0].message.content
    except Exception as e:
        # print("请求失败", e)
        return ""