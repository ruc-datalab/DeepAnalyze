import re
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import sys
from openai import OpenAI
import json
from time import sleep


THINK_START, THINK_END = "<Analyze>", "</Analyze>"
SQL_START, SQL_END = "<Code>", "</Code>"
SOLUTION_START, SOLUTION_END = "<Answer>", "</Answer>"
OBS_START, OBS_END = "<Execute>", "</Execute>"


# NOTE: bring back reward
def verify_format_and_extract(output: str):
    if output.count(SOLUTION_START) != 1:
        return False, None, None, None
    pre_solution, tail = output.split(SOLUTION_START, 1)

    if tail.count(SOLUTION_END) != 1:
        return False, None, None, None

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(Analyze|Code|Execute)\b", solution_text, re.I):
        return False, None, None, None

    thoughts = re.findall(r"<Analyze>(.*?)</Analyze>", output, re.S)
    if not thoughts:
        return False, None, None, None

    for m in re.finditer(r"</Execute>", pre_solution, re.I):
        rest = pre_solution[m.end() :].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None, None

    return True, thoughts, solution_text.strip(), None


def execute_sql_single(db_file, sql):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        conn.close()
        # print('Successfully executed')
        return db_file, sql, execution_res, 1
    except Exception:
        # print(f"Error executing SQL: {e}, db file: {db_file}")
        conn.rollback()
        conn.close()
        return db_file, sql, None, 0


def execute_sql_wrapper_single(db_file, sql, timeout, output_str):
    try:
        res = func_timeout(timeout, execute_sql_single, args=(db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"SQL:\n{sql}\nTime Out!")
        print("-" * 30)
        res = (db_file, sql, None, 0)
    except Exception:
        # print(f"Error executing SQL: {e}, db_file: {db_file}")
        res = (db_file, sql, None, 0)

    # Append the output to the tuple
    if isinstance(res, tuple):
        res = res + (output_str,)

    return res


def extract_tableqa_answer(text):
    """Extract the last answer from text"""
    last_pos = text.rfind("<Answer>")
    if last_pos == -1:
        return None
    remaining_text = text[last_pos:]
    pattern = r"<Answer>(.*?)</Answer>"
    match = re.search(pattern, remaining_text, re.DOTALL)
    result = ""
    if match:
        result = match.group(1).strip()
    if "Answer:" in result:
        result = result.split("Answer:")[1].strip()
    return result


def extract_answer(text):
    """Extract the last answer from text"""
    last_pos = text.rfind("<Answer>")
    if last_pos == -1:
        return None
    remaining_text = text[last_pos:]
    pattern = r"<Answer>(.*?)</Answer>"
    match = re.search(pattern, remaining_text, re.DOTALL)
    result = text
    if match:
        result = match.group(1).strip()
    return result


def compute_tableqa_score_single(completion, reference):
    try:
        answer = extract_tableqa_answer(completion)
        if answer == reference:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Unexpected error: {e}; Setting reward as 0")
        return 0


PROMPT = """
You are a strict data science evaluation assistant.

Compare the following **gold** and **predicted** solutions for data science task. Your job is to determine if the predicted solution is correct.

You must only give a score of "1" if:
- The predicted solution **matches the answer in the gold solution** exactly.

Instructions:
- You may provide internal reasoning or explanation before giving your final judgment.
- Your final judgment must appear as a separate line at the end of your response, in the format:

### Final Score: 1

or

### Final Score: 0

Do not include any explanation after the final score.
"""


def llm_as_judgement_accuracy1(completion, reference, client, model):
    if extract_answer(completion) is None:
        return 0.0

    message = (
        PROMPT
        + f"\n\nGOLD SOLUTION:\n{reference}\n\nPREDICTED SOLUTION:\n{extract_answer(completion)}\n\nAnswer:"
    )
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": message}]
        )
        reply = response.choices[0].message.content.strip()

        # Try to parse score from "### Final Score: x"
        match = re.search(r"### Final Score:\s*([01](?:\.0)?)", reply)
        if match:
            print(
                extract_answer(completion),
                "++++++++++++",
                reference,
                "++++++",
                float(match.group(1)),
            )
            return float(match.group(1))

        # Fallback: raw "1" or "0"
        if reply.strip() in {"1", "0"}:
            print(
                extract_answer(completion),
                "++++++++++++",
                reference,
                "++++++",
                float(reply.strip()),
            )
            return float(reply.strip())

        print(f"Unrecognized reward output: {reply}")
        return 0.0

    except Exception as e:
        print(f"LLM Judge error: {type(e).__name__}: {e}")
        return 0.0


ACC_PROMPT = """You are a data science evaluation assistant. Your task is to carefully evaluate the accuracy of the model's response using the provided data science question [QUESTION], the corresponding ground-truth solution [REFERENCE SOLUTION], and the model's response [PREDICTED SOLUTION]. After the evaluation, assign a single score for accuracy on a scale from 0 to 5.

### [REFERENCE SOLUTION]:
{question} 

### [REFERENCE SOLUTION]:
{ref}

### [PREDICTED SOLUTION]:
{answer}

Please evaluate step by step, and finally output the result strictly in this format: `SCORE: <assign_score>`."""


def llm_as_judgement_accuracy(completion, reference, question, client, model):
    if extract_answer(completion) is None:
        return 0.0

    message = ACC_PROMPT.format(
        question=question, ref=reference, answer=extract_answer(completion)
    )
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": message}]
        )
        reply = response.choices[0].message.content.strip()

        match = re.search(r"SCORE\s*:\s*(\d+)", reply, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 5

        print(f"Unrecognized reward output: {reply}")
        return 0.0

    except Exception as e:
        print(f"LLM Judge error: {type(e).__name__}: {e}")
        return 0.0


ANALYZE_PROMPT = """You are a data science evaluation assistant. Your task is to carefully evaluate the quality of the model's response using the provided data science question [QUESTION], the corresponding ground-truth solution [REFERENCE SOLUTION], and the model's response [PREDICTED SOLUTION], focusing on the analysis process. 

After evaluation, assign a single score for the analysis quality of [PREDICTED SOLUTION] on a scale from **1 (adequate)** to **5 (exceptional)**, using the following guidelines:
- **1 (Adequate):** The analysis demonstrates a basic level of reasoning; partially correct but limited in depth and scope, with several gaps or missing considerations.
- **2 (Moderate):** The analysis is generally understandable and somewhat logical, but still incomplete or uneven; key aspects may be underdeveloped or superficial.
- **3 (Good):** The analysis is reasonably correct, coherent, and covers the main points; some important details may be missing, but the reasoning is mostly sound.
- **4 (Strong):** The analysis is clear, well-structured, and logically consistent; it addresses most aspects with only minor omissions or imperfections.
- **5 (Exceptional):** The analysis is fully correct, rigorous, and comprehensive; reasoning is highly detailed, deeply coherent, and accounts for all relevant aspects and edge cases.

### [REFERENCE SOLUTION]:
{question} 

### [REFERENCE SOLUTION]:
{ref}

### [PREDICTED SOLUTION]:
{answer}

Find some weaknesses, and finally output the result in this format: `SCORE: <assign_score>`."""


def llm_as_judgement_analyze(completion, reference, question, client, model):
    if extract_answer(completion) is None:
        return 0.0

    message = ANALYZE_PROMPT.format(question=question, ref=reference, answer=completion)
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": message}]
        )
        reply = response.choices[0].message.content.strip()

        match = re.search(r"SCORE\s*:\s*(\d+)", reply, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 5

        print(f"Unrecognized reward output: {reply}")
        return 0.0

    except Exception as e:
        print(f"LLM Judge error: {type(e).__name__}: {e}")
        return 0.0


OPENDOMAIN_PROMPT = """You are a data science evaluation assistant. Here's a generated data science report based on the user instruction and provided data. Your task is to comprehensively evaluate the quality of a generated data science report and its analytical process, based on the provided user instruction [INSTRUCTION], the data thumbnail [DATA THUMBNAIL], and the generated report [GENERATED REPORT].

You should assess the report across the following five dimensions, each scored on a scale from 1 (lowest) to 5 (highest). Please use the detailed guidelines below to calibrate your evaluation:

- **Usefulness**: Does the report successfully extract and highlight key insights from the data?
    - **1**: Provides a few relevant insights at a basic level, but misses many important aspects.
    - **2**: Some useful insights, but often superficial or incomplete.
    - **3**: Reasonably useful, capturing the main points with noticeable gaps.
    - **4**: Strong and clear insights with only minor omissions.
    - **5**: Outstandingly insightful, comprehensive, and precisely aligned with the instruction.

- **Richness**: Does the analysis explore the data from diverse and meaningful perspectives?
    - **1**: Covers a minimal range of perspectives with limited depth.
    - **2**: Some exploration but overly narrow or shallow.
    - **3**: Moderate exploration with several perspectives, though uneven in depth.
    - **4**: Rich analysis with broad and reasonably deep perspectives.
    - **5**: Exceptionally rich, broad, and profound exploration from multiple meaningful angles.

- **Soundness**: Is the analytical process well-structured, accurate, and logically planned?
    - **1**: Adequate but contains some flaws or unclear logic.
    - **2**: Generally correct but with noticeable weaknesses or errors.
    - **3**: Mostly sound with a coherent structure, though not fully rigorous.
    - **4**: Strong logical flow and accuracy, with only minor issues.
    - **5**: Perfectly rigorous, precise, and fully coherent analytical process.

- **Interpretability**: Does the report provide sufficient intermediate outputs to make the reasoning process transparent?
    - **1**: Minimal interpretability; reasoning is partially understandable.
    - **2**: Some explanation provided but key steps remain unclear.
    - **3**: Reasonably interpretable with several intermediate steps shown.
    - **4**: Clear and well-supported reasoning with good step-by-step evidence.
    - **5**: Fully transparent and highly readable reasoning, with complete and well-structured intermediate outputs.

- **Readability**: Is the final report presented in a polished academic style and does it fulfill the instruction?
    - **1**: Adequately structured, but lacks polish or academic tone.
    - **2**: Mostly understandable but inconsistent in style or weakly aligned with the instruction.
    - **3**: Clear and acceptable structure, partially academic but not fully polished.
    - **4**: Well-written in an academic style, fulfilling the instruction with only minor issues.
    - **5**: Exceptionally polished, professional academic style, fully aligned and highly effective.

### [INSTRUCTION]:
{instruction} 

### [DATA THUMBNAIL]:
{thumbnail}

### [PREDICTED SOLUTION]:
{answer}

First find some weaknesses, and finally return your evaluation in the following JSON format:
    ```json
    {{
    "usefulness": <score from 1 to 5>,
    "richness": <score from 1 to 5>,
    "soundness": <score from 1 to 5>,
    "interpretability": <score from 1 to 5>,
    "readability": <score from 1 to 5>,
    }}
    ```
"""


def llm_as_judgement_opendomain(completion, reference, question, client, model):
    if extract_answer(completion) is None:
        return 0.0

    message = OPENDOMAIN_PROMPT.format(
        instruction=question, thumbnail=reference, answer=completion
    )
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": message}]
        )
        reply = response.choices[0].message.content.strip()

        reply_match = re.search(r"```(?:json)?(.*?)```", reply, re.DOTALL)
        score_str = reply_match.group(1).strip() if reply_match else response
        score_dict = json.loads(score_str)
        reward = {
            key: float(score_dict[key]) / 5
            for key in [
                "usefulness",
                "richness",
                "soundness",
                "interpretability",
                "readability",
            ]
        }
        return reward

    except Exception as e:
        print(f"LLM Judge error: {type(e).__name__}: {e}")
        return 0.0


def check_valid_code_block(text: str) -> bool:
    code_blocks = re.findall(r"<Code>(.*?)</Code>", text, re.DOTALL)

    if not code_blocks:
        return True

    for block in code_blocks:
        block = block.strip()
        if not (block.startswith("```python") and block.endswith("```")):
            return False

    return True
