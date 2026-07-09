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
feature_names = joblib.load(f"{OUTPUT_DIR}/feature_names.pkl")

#print(model)

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

def extract_email_features(sender_domain,urls,subject_length,word_count,uppercase_count,special_characters,question_marks,exclamation_marks,contains_http):
    return {
        "sender_domain": sender_domain,
        "urls": urls,
        "subject_length": subject_length,
        "word_count": word_count,
        "uppercase_count": uppercase_count,
        "special_characters": special_characters,
        "question_marks": question_marks,
        "exclamation_marks": exclamation_marks,
        "contains_http": contains_http,
    }

def predict_email(features):
    """
    Predict the phishing probability using trained model
    """
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    # Numeric features
    numeric_features = [
        "urls",
        "subject_length",
        "word_count",
        "uppercase_count",
        "special_characters",
        "question_marks",
        "exclamation_marks",
        "contains_http"
    ]
    for feature in numeric_features:
        input_df.loc[0, feature] = features.get(feature, 0)

    # One-hot encoded sender domain
    column = f"sender_domain_{features['sender_domain']}"
    if column in input_df.columns:
        input_df.loc[0, column] = 1
    elif "sender_domain_other" in input_df.columns:
        input_df.loc[0, "sender_domain_other"] = 1
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    pred_class = "Phishing" if pred == 1 else "Legitimate"
    confidence = prob[pred]
    if confidence >= 0.90:
        confidence_level = "High"
    elif confidence >= 0.70:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    return pred_class, confidence, confidence_level


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
    Rules:
    - Use the Machine Learning Prediction exactly as provided.
    - Use the Confidence Level exactly as provided.
    - Explain the prediction using the supplied email features.
    """
    email_features = extract_email_features(
        sender_domain="other",
        urls=3,
        subject_length=45,
        word_count=120,
        uppercase_count=25,
        special_characters=20,
        question_marks=1,
        exclamation_marks=2,
        contains_http=1,
    )
    prediction, confidence, confidence_level = predict_email(email_features)
    print(f"ML Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    user_prompt = f"""
    Analyze the following phishing detection result.

    Email Features:
    {json.dumps(email_features, indent=2)}

    Machine Learning Prediction:
    {prediction}

    Prediction Probability:
    {confidence:.4f}

    Confidence Level:
    {confidence_level}
    """
    response = call_llm(system_prompt, user_prompt)
    if response is None:
        print("LLM request failed")
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
    os.makedirs("part4/output", exist_ok=True)

    with open("part4/output/llm_response.json", "w") as f:
        json.dump(result, f, indent=4)
    print("LLM response saved to part4/output/llm_response.json")
    
if __name__ == "__main__":
    main()
