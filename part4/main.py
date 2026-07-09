import os
import json
import re
import joblib
import requests
import pandas as pd
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

OUTPUT_DIR = "part3/output"
load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
MODEL_PATH = f"{OUTPUT_DIR}/best_model.pkl"
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4.1-mini"

def has_pii(text):
    email = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_no = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    return bool(re.search(email, text) or re.search(phone_no, text))

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    
    if has_pii(user_prompt):
        print("Input blocked - PII detected")
        return None
    
    headers = {
        "Authorization": f"Bearer {API_KEY}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(URL, headers=headers, json=payload)
    if response.status_code != 200:
        print("Status code: ", response.status_code)
        print(response.text)
        return None
    return response.json()["choices"][0]["message"]["content"]

def main():
    system_prompt = """
    You are a cybersecurity assistant.

    Return ONLY valid JSON.

    Do not use markdown.
    Do not use ```json.
    Do not explain anything.
    Do not add any text before or after the JSON.

    The response MUST exactly match this schema:

    {
    "prediction_label": "string",
    "confidence_level": "string",
    "top_reason": "string",
    "second_reason": "string",
    "recommended_actions": "string"
    }
    """
    print("====== Test case 1 - With PII ======")
    user_prompt = """
    Email: abc123@gmail.com
    
    Prediction: 
    Phishing
    Probability: 
    0.97
    """
    response = call_llm(system_prompt, user_prompt)
    print("====== Test Case 2 - Without PII ======")
    user_prompt = """
    Email features:
    urls: 3
    contains_http: 1
    uppercase_count: 25
    word_count: 120
    
    Prediction: Phishing
    Probability: 0.99
    """
    response = call_llm(system_prompt, user_prompt)
    if response is None:
        return
    schema = {
        "type": "object",
        "properties": {
            "prediction_label": {"type": "string"},
            "confidence_level": {"type": "string"},
            "top_reason": {"type": "string"},
            "second_reason": {"type": "string"},
            "recommended_actions": {"type": "string"}
        },
        "required": [
            "prediction_label",
            "confidence_level",
            "top_reason",
            "second_reason",
            "recommended_actions"   
        ]
    }
    try:
        result = json.loads(response.strip())
    except json.JSONDecodeError as e:
        print("JSON Decode Error: ", e)
        return
    try:
        validate(instance=result, schema=schema)
        print("Schema validation passed")
    except ValidationError as e:
        print("Schema validation failed")
        print(e)
    print(json.dumps(result, indent=4))
    
if __name__ == "__main__":
    main()
