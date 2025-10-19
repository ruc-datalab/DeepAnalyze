import requests, json
import os

url_base = os.getenv("STEM_LLM_JUDGE_URL")
url = url_base.rstrip("/") + "/v1/chat/completions"

prompt = (
    f"User: ### Question: What is 1+1\n\n"
    f"### Ground Truth Answer: 2\n\n"
    f"### Student Answer: It is 2.\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    'If the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
)

payload = {
    "model": "TIGER-Lab/general-verifier",
    "messages": [{"role": "user", "content": prompt}],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0.0,
}

resp = requests.post(url, json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()
print(data)
