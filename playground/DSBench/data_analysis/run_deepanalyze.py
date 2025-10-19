import os
import re
import json
import base64
import tiktoken
import time
import pandas as pd
import openai
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re
import contextlib
import io
import sys
from pathlib import Path
import random
import traceback
import os
import json

# Model limits and cost configurations
MODEL_LIMITS = {
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "deepseek-v3-250324": 128_000,
    "deepseek-r1-250528": 128_000,
    "DeepAnalyz-8B": 128_000,
}

MODEL_COST_PER_INPUT = {
    "gpt-3.5-turbo-0125": 0.0000005,
    "gpt-4-turbo-2024-04-09": 0.00001,
    "gpt-4o-2024-05-13": 0.000005,
    "deepseek-v3-250324": 0.000005,
}

MODEL_COST_PER_OUTPUT = {
    "gpt-3.5-turbo-0125": 0.0000015,
    "gpt-4-turbo-2024-04-09": 0.00003,
    "gpt-4o-2024-05-13": 0.000015,
    "deepseek-v3-250324": 0.000015,
}


def get_gpt_res(text, image, model, client):

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    payload = {
        "model": "/fs/fast/u2023000922/zhangshaolei/checkpoints/rl0906.step28.0929/export/global_step_33/policy/",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4000,
        "add_generation_prompt": False,
    }

    # Send request to vLLM API
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()  # Raise an error for bad status codes
    response_data = response.json()
    print(response_data)
    return response_data


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def find_jpg_files(directory):
    jpg_files = [
        file
        for file in os.listdir(directory)
        if file.lower().endswith(".jpg") or file.lower().endswith(".png")
    ]
    return jpg_files if jpg_files else None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def find_excel_files(directory):
    excel_files = [
        file
        for file in os.listdir(directory)
        if (
            file.lower().endswith("xlsx")
            or file.lower().endswith("xlsb")
            or file.lower().endswith("xlsm")
        )
        and "answer" not in file.lower()
    ]
    return excel_files if excel_files else None


def read_excel(file_path):
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets


def dataframe_to_text(df):
    text = df.to_string(index=False)
    return text


def combine_sheets_text(sheets):
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text


def read_txt(path):
    with open(path, "r") as f:
        return f.read()


def truncate_text(text, max_tokens=128000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        text = " ".join(tokens[-max_tokens:])
    return text


def process_question(question, text, image, model, client, sample_id, index):
    prompt = (
        text
        + f"The questions are detailed as follows.\n{question}"
        + "\n\nYour response should end with 'Answer: <results>'"
    )
    start = time.time()
    try:
        # if True:
        response = get_gpt_res(prompt, image, model, client)
        cost = 0  # Cost calculation commented out as in original code

        # thinking = getattr(response.choices[0].message, "reasoning_content", "")
        # answer = response.choices[0].message.content.strip()
        answer = response["choices"][0]["message"]["content"].strip()

        output_text = answer
        print(output_text)

        return {
            "index": index,  # Store the original index
            "id": sample_id,
            "model": response["model"],
            "input": response["usage"]["prompt_tokens"],
            "output": response["usage"]["completion_tokens"],
            "cost": cost,
            "time": time.time() - start,
            "prompt": prompt,
            "response": output_text,
        }
    except Exception as e:
        return {
            "index": index,  # Store the original index
            "id": sample_id,
            "model": model,
            "input": 0,
            "output": 0,
            "cost": 0,
            "time": time.time() - start,
            "prompt": prompt,
            "response": "",
            "status": "error",
            "error_message": str(e),
        }


def main():
    client = OpenAI(
        api_key="",
        base_url="http://localhost:8000/v1",
    )

    tokens4generation = 4000
    model = "DeepAnalyz-8B"
    model_name = "DeepAnalyz-8B"
    data_path = "./data/"
    total_cost = 0
    encoding = tiktoken.encoding_for_model("gpt-4-turbo-2024-04-09")

    # Load samples from data.json
    samples = []
    with open("./data.json", "r") as f:
        for line in f:
            samples.append(eval(line.strip()))

    for id in tqdm(range(len(samples))):
        sample = samples[id]
        save_path = os.path.join("./save_process", f"{model_name}")
        output_file = os.path.join(save_path, sample["id"] + ".json")

        # Skip if output file already exists
        if os.path.exists(output_file):
            continue

        if len(sample["questions"]) > 0:
            image = find_jpg_files(os.path.join(data_path, sample["id"]))
            if image:
                image = os.path.join(data_path, sample["id"], image[0])

            excel_content = ""
            excels = find_excel_files(os.path.join(data_path, sample["id"]))
            if excels:
                for excel in excels:
                    excel_file_path = os.path.join(data_path, sample["id"], excel)
                    sheets = read_excel(excel_file_path)
                    combined_text = combine_sheets_text(sheets)
                    excel_content += f"The excel file {excel} is: {combined_text}"

                excel_content = encoding.decode(
                    encoding.encode(excel_content)[
                        tokens4generation - MODEL_LIMITS[model] :
                    ]
                )

            introduction = read_txt(
                os.path.join(data_path, sample["id"], "introduction.txt")
            )
            questions = []
            for question_name in sample["questions"]:
                questions.append(
                    read_txt(
                        os.path.join(data_path, sample["id"], question_name + ".txt")
                    )
                )

            text = ""
            if excel_content:
                text += f"The workbook is detailed as follows. {excel_content}\n"
            text += f"The introduction is detailed as follows.\n{introduction}\n"
            answers = []

            # Parallel processing of questions using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(questions)) as executor:
                future_to_question = {
                    executor.submit(
                        process_question,
                        question,
                        text,
                        image,
                        model,
                        client,
                        sample["id"],
                        idx,
                    ): question
                    for idx, question in enumerate(questions)
                }
                for future in as_completed(future_to_question):
                    answer = future.result()
                    answers.append(answer)
                    total_cost += answer["cost"]
                    print("Total cost: ", total_cost)

            # Sort answers by their original index to match questions order
            answers.sort(key=lambda x: x["index"])

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(output_file, "w") as f:
                for answer in answers:
                    # Remove the index field before saving to maintain original output format
                    answer_without_index = {
                        k: v for k, v in answer.items() if k != "index"
                    }
                    json.dump(answer_without_index, f)
                    f.write("\n")


if __name__ == "__main__":
    main()
