# Part 4 – LLM Integration

## Overview

In this part, the trained Random Forest model from Part 3 is integrated with an LLM (OpenRouter GPT-4.1 Mini) to generate human-readable phishing explanations. The ML model predicts whether an email is phishing or legitimate, while the LLM explains the prediction in structured JSON format.

---

## Features

- Loaded the trained ML pipeline (`best_model.pkl`)
- Generated phishing predictions and confidence scores
- Assigned confidence levels (High / Medium / Low)
- Integrated OpenRouter LLM API
- Used prompt engineering with system and user prompts
- Enforced JSON-only responses
- Validated responses using JSON Schema
- Added PII detection for email addresses and phone numbers
- Blocked prompts containing PII before sending to the LLM
- Compared deterministic (`temperature=0`) and creative (`temperature=0.7`) outputs
- Tested three sample emails
- Saved JSON and raw LLM responses

---

## Test Results

### Test Case 1
- Prediction: **Phishing**
- Confidence: **95% (High)**

### Test Case 2
- Prediction: **Legitimate**
- Confidence: **96.5% (High)**

### Test Case 3
- Prediction: **Phishing**
- Confidence: **71.5% (Medium)**

---

## Temperature Comparison

Two temperatures were tested:

- **Temperature = 0** produced consistent and deterministic responses.
- **Temperature = 0.7** produced slightly more varied wording while preserving the prediction.

---

## JSON Schema

Each LLM response follows this schema:

```json
{
  "prediction_label": "",
  "confidence_level": "",
  "top_reason": "",
  "second_reason": "",
  "recommended_actions": ""
}
```

All responses were successfully validated using the `jsonschema` library.

---

## PII Guardrail

A regex-based guardrail detects:

- Email addresses
- Phone numbers

If PII is detected, the request is blocked before being sent to the LLM.

Example:

```
Input blocked - PII detected
PII guardrail working successfully.
```

---

## Conclusion

The ML model provides phishing predictions, while the LLM generates structured explanations and recommendations. JSON validation, PII filtering, and temperature comparison improve the reliability and safety of the generated responses.