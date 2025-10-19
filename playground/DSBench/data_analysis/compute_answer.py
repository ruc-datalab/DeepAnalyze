import json

from tqdm import tqdm
import os
from openai import OpenAI

client = OpenAI(api_key="")

samples = []
with open("./data.json", "r") as f:
    for line in f:
        samples.append(eval(line.strip()))


def evaluate_prediction(client, question, answer, prediction):
    prompt = (
        f"Please judge whether the generated answer is right or wrong. We require that the correct answer "
        f"to the prediction gives a clear answer, not just a calculation process or a disassembly of ideas. "
        f"The question is {question}. The true answer is \n {answer}. \n The predicted answer is \n {prediction}.\n "
        f"If the predicted answer is right, please output True. Otherwise output Flase. "
        f"Don't output any other text content. You only can output True or False."
    )
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    # print(prompt)
    # print(response.choices[0].message.content)
    # exit()
    return response.choices[0].message.content


def read_txt(path):
    with open(path, "r") as f:
        return f.read()


save_path = "./save_process"
model = "gpt-3.5-turbo-0125"
# model = 'gpt-4o-2024-05-13'
# model = 'llama-3-8b-instruct'
# model = 'gpt-3.5-turbo-0125-autoagent'
# model = 'gpt-4o-2024-05-13-autoagent'
# model = 'llava-v1.5-13b'
# model = 'llama3-autoagent'

results = []
save_f = open(os.path.join(save_path, model, "results.json"), "w")
save_process = open(os.path.join(save_path, model, "results_process.json"), "w")

for sample in tqdm(samples):
    result = []
    if len(sample["questions"]) > 0:
        # print(sample['id'])
        predicts = []
        with open(os.path.join(save_path, model, sample["id"] + ".json"), "r") as f:
            for line in f:
                predicts.append(eval(line.strip()))

        questions = []
        for id, question_name in enumerate(tqdm(sample["questions"])):
            question = read_txt(
                os.path.join("./data", sample["id"], question_name + ".txt")
            )
            pre = predicts[id]
            try:
                if not model.endswith("autoagent"):
                    ans = evaluate_prediction(
                        client, question, str(sample["answers"][id]), pre["response"]
                    )
                else:
                    ans = evaluate_prediction(
                        client, question, str(sample["answers"][id]), pre["summary"]
                    )
            except Exception as e:
                print(e)
                ans = "False"
            # print(result)
            if not model.endswith("autoagent"):
                process = [
                    sample["id"],
                    ans,
                    str(sample["answers"][id]),
                    pre["response"][:],
                ]
            else:
                process = [
                    sample["id"],
                    ans,
                    str(sample["answers"][id]),
                    pre["summary"][:],
                ]

            result.append(ans)
            json.dump(process, save_process)
            save_process.write("\n")
            save_process.flush()
    json.dump(result, save_f)
    save_f.write("\n")
    save_f.flush()
    results += result

save_f.close()
save_process.close()

results_c = []
for i, result in enumerate(results):
    if "true" in result.lower():
        results_c.append(True)
    else:
        results_c.append(False)

idx = 0
score4cha = []
for sample in tqdm(samples):
    if len(sample["questions"]) > 0:
        score_ = sum(results_c[idx : idx + len(sample["questions"])]) / len(
            sample["questions"]
        )
        idx += len(sample["questions"])
        score4cha.append(score_)
print(f"Accuracy for each challenge is {score4cha}")


acc = sum(results_c) / len(results_c)
print(f"Accuracy for all the {len(results_c)} questions is {acc}")
