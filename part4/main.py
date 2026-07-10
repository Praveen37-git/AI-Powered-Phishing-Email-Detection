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
    test_inputs = [
        extract_email_features(
            sender_domain="other",
            urls=3,
            subject_length=45,
            word_count=120,
            uppercase_count=25,
            special_characters=20,
            question_marks=1,
            exclamation_marks=2,
            contains_http=1,
        ),
        extract_email_features(
            sender_domain="gmail.com",
            urls=0,
            subject_length=22,
            word_count=70,
            uppercase_count=1,
            special_characters=3,
            question_marks=0,
            exclamation_marks=0,
            contains_http=0,
        ),
        extract_email_features(
            sender_domain="google.com",
            urls=1,
            subject_length=35,
            word_count=90,
            uppercase_count=5,
            special_characters=8,
            question_marks=1,
            exclamation_marks=1,
            contains_http=1,
        )
    ]
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
    fallback = {
        "prediction_label": None,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "recommended_actions": None
    }
    os.makedirs("part4/output/json", exist_ok=True)
    os.makedirs("part4/output/raw", exist_ok=True)
    for i, email_features in enumerate(test_inputs, start=1):
        print(f"\n====== Test case {i} ======")
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
    
        response_temp0 = call_llm(system_prompt, user_prompt, temperature=0)
        response_temp07 = call_llm(system_prompt, user_prompt, temperature=0.7)
        if response_temp0 is None or response_temp07 is None:
            print("LLM request failed")
            continue
    
        try:
            result1 = json.loads(response_temp0.strip())
        except json.JSONDecodeError as e:
            print("JSON Decode Error: ", e)
            result1 = fallback.copy()
        try:
            result2 = json.loads(response_temp07.strip())
        except json.JSONDecodeError as e:
            print("JSON Decode Error: ", e)
            result2 = fallback.copy()
            
        try:
            validate(instance=result1, schema=schema)
            print("Validation (Temp 0): PASS")
        except ValidationError as e:
            print("Temp0 validation failed")
            result1 = fallback.copy()
        try:
           validate(instance=result2, schema=schema)
           print("Validation (Temp 0.7): PASS")
        except ValidationError as e:
            print("Temp0.7 validation failed")
            result2 = fallback.copy()
            
        print("\n====== Temperature = 0 ======")
        print(json.dumps(result1, indent=4))
        print("\n====== Temperature = 0.7 ======")
        print(json.dumps(result2, indent=4))
        print("-" * 60)



        with open(f"part4/output/json/llm_response_{i}_temp0.json", "w", encoding="utf-8") as f:
            json.dump(result1, f, indent=4)
        with open(f"part4/output/json/llm_response_{i}_temp07.json", "w", encoding="utf-8") as f:
            json.dump(result2, f, indent=4)
        print(f"Saved responses for Test Case {i}")
        
        with open(f"part4/output/raw/raw_temp0_{i}.txt", "w", encoding="utf-8") as f:
            f.write(response_temp0)
        with open(f"part4/output/raw/raw_temp07_{i}.txt", "w", encoding="utf-8") as f:
            f.write(response_temp07)
    print("\n====== PII Guardrail Test ======")

    pii_prompt = """
    Email: john.doe@gmail.com
    Phone: 9876543210

    Prediction: Phishing
    Probability: 0.98
    """

    response = call_llm(system_prompt, pii_prompt)

    if response is None:
        print("PII guardrail working successfully.")
    print("\nAll test cases completed successfully.")
        
    
if __name__ == "__main__":
    main()
