import json
import re
import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict


def extract_sql(solution_str):
    """Extract SQL query from model response, supporting standard format and tag-wrapped format"""
    if solution_str is None:
        return None

    # First check if there are <answer> tags and extract content
    answer_tag_pattern = r"<Answer>(.*?)</Answer>"
    answer_tag_match = re.search(answer_tag_pattern, solution_str, re.DOTALL)
    if answer_tag_match:
        # If <answer> tags are found, only process content inside the tags
        solution_str = answer_tag_match.group(1).strip()

    # Now process the extracted content or original content

    # Try to extract SQL from after Answer marker
    answer_pattern = r"Answer:\s*```sql\s*(.*?)\s*(?:```|$)"
    answer_matches = re.findall(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].strip()

    # If no explicit Answer marker, try to extract from Markdown code blocks
    md_pattern = r"```sql\s*(.*?)\s*```"
    md_matches = re.findall(md_pattern, solution_str, re.DOTALL)
    if md_matches:
        return md_matches[-1].strip()

    # Try to extract from top-level code blocks
    code_pattern = r"```\s*(.*?)\s*```"
    code_matches = re.findall(code_pattern, solution_str, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()

    # Check if it contains SELECT
    if "SELECT" in solution_str.upper():
        # Try to extract complete SQL statement
        sql_line_pattern = r"SELECT\s+.*?FROM\s+.*?(WHERE\s+.*?)?(?:;|$)"
        sql_line_matches = re.findall(
            sql_line_pattern, solution_str, re.IGNORECASE | re.DOTALL
        )
        if sql_line_matches:
            return sql_line_matches[-1].strip()

    return None


def normalize_sql(sql_string):
    """Normalize SQL query string for more lenient matching"""
    if sql_string is None:
        return ""

    # Basic cleaning
    sql_string = " ".join(sql_string.split()).lower()
    sql_string = sql_string.replace("'", "").replace('"', "")  # Remove quotes

    # Handle semicolons
    sql_string = sql_string.rstrip(";")

    # Normalize column names - replace underscores with spaces
    sql_string = sql_string.replace("_", " ")

    # Normalize aggregation functions
    agg_funcs = ["count", "sum", "avg", "min", "max"]
    for func in agg_funcs:
        pattern = r"{}[ ]*\(([^)]+)\)".format(func)
        replacement = r"{} \1".format(func)
        sql_string = re.sub(pattern, replacement, sql_string)

    return sql_string


def parse_sql(query, columns, verbose=False):
    """Parse SQL query and extract structured information, with enhanced support for column names with spaces and multiple conditions"""
    if query is None:
        return {
            "agg": 0,
            "conds": {"column_index": [], "condition": [], "operator_index": []},
            "human_readable": "",
            "sel": -1,
        }

    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", "OP"]

    # Create a copy of column names to avoid modifying the original list
    columns = list(columns)

    # Create column name mapping dictionary for fuzzy matching
    col_map = {}
    col_map_lower = {}  # For case-insensitive matching
    for i, col in enumerate(columns):
        # Normalize column name (lowercase, remove spaces and parentheses)
        norm_col = (
            col.lower().strip().replace("(", "").replace(")", "").replace(" ", "")
        )
        col_map[norm_col] = i
        col_map_lower[col.lower().strip()] = i  # Only lowercase conversion, keep spaces

    # Extract SELECT target column
    select_match = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE | re.DOTALL)
    select_clause = select_match.group(1).strip() if select_match else None

    if select_clause is None:
        return {
            "agg": 0,
            "conds": {"column_index": [], "condition": [], "operator_index": []},
            "human_readable": query,
            "sel": -1,
        }

    # Process aggregation function
    agg = 0
    select_col = select_clause
    for i, op in enumerate(agg_ops[1:], 1):
        pattern = rf"{op}\s*\((.*?)\)"
        match = re.search(pattern, select_clause, re.IGNORECASE)
        if match:
            agg = i
            select_col = match.group(1).strip()
            break

    # Try to match column name
    sel = -1
    # 1. Direct matching
    if select_col in columns:
        sel = columns.index(select_col)
    # 2. Case-insensitive matching
    elif select_col.lower() in col_map_lower:
        sel = col_map_lower[select_col.lower()]
    # 3. Normalized matching
    else:
        select_col_norm = (
            select_col.lower()
            .strip()
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "")
        )
        if select_col_norm in col_map:
            sel = col_map[select_col_norm]

    # Extract WHERE conditions
    conds = {"column_index": [], "condition": [], "operator_index": []}
    where_match = re.search(r"WHERE\s+(.*?)(?:;|$)", query, re.IGNORECASE | re.DOTALL)

    if where_match:
        conditions_str = where_match.group(1).strip()
        # Process AND or OR connected conditions - better handling of connection symbols
        conditions = re.split(
            r"\s+AND\s+|\s+OR\s+", conditions_str, flags=re.IGNORECASE
        )

        for cond in conditions:
            # Remove leading and trailing spaces
            cond = cond.strip()
            if not cond:
                continue

            # For each condition, try to extract column name, operator, and value
            # Try various operators first
            matched = False
            for op_idx, op in enumerate(cond_ops):
                # Use regex to precisely match operator
                pattern = r"(.+?)\s*" + re.escape(op) + r"\s*(.+)"
                match = re.search(pattern, cond)

                if match:
                    matched = True
                    col_name = match.group(1).strip()
                    value = match.group(2).strip().strip("'").strip('"')

                    # Try to match column name
                    col_idx = -1

                    # Try direct matching
                    if col_name in columns:
                        col_idx = columns.index(col_name)
                    # Try case-insensitive matching
                    elif col_name.lower() in col_map_lower:
                        col_idx = col_map_lower[col_name.lower()]
                    # Try normalized matching
                    else:
                        # Compare without spaces and symbols
                        col_norm = col_name.lower().replace(" ", "").replace("_", "")
                        for i, col in enumerate(columns):
                            col_clean = col.lower().replace(" ", "").replace("_", "")
                            if col_clean == col_norm:
                                col_idx = i
                                break

                        # Substring matching
                        if col_idx < 0:
                            for i, col in enumerate(columns):
                                if (
                                    col.lower() in col_name.lower()
                                    or col_name.lower() in col.lower()
                                ):
                                    col_idx = i
                                    break

                    # If column index is found, add to conditions
                    if col_idx >= 0:
                        conds["column_index"].append(col_idx)
                        conds["condition"].append(value)
                        conds["operator_index"].append(op_idx)
                    elif verbose:  # Only show warning in verbose mode
                        print(
                            f"Warning: Could not match column name '{col_name}' to header list: {columns}"
                        )

                    break  # Exit loop after finding operator

    return {"agg": agg, "conds": conds, "human_readable": query, "sel": sel}


def score_sql(parsed_sql, correct_sql):
    """
    Compare predicted SQL with correct SQL's structured information
    Return 1 point only if all components match exactly, otherwise 0 points
    """
    # Check if selected column matches
    if parsed_sql["sel"] != correct_sql["sel"]:
        return 0

    # Check if aggregation function matches
    if parsed_sql["agg"] != correct_sql["agg"]:
        return 0

    parsed_conds = parsed_sql["conds"]
    correct_conds = correct_sql["conds"]

    # Check if condition column indices, values, and operators match exactly (regardless of order)
    if sorted(parsed_conds["column_index"]) != sorted(correct_conds["column_index"]):
        return 0

    # Sort condition lists and compare one by one
    sorted_parsed = sorted(
        zip(
            parsed_conds["column_index"],
            parsed_conds["condition"],
            parsed_conds["operator_index"],
        )
    )
    sorted_correct = sorted(
        zip(
            correct_conds["column_index"],
            correct_conds["condition"],
            correct_conds["operator_index"],
        )
    )

    for i in range(len(sorted_parsed)):
        # Check column index
        if sorted_parsed[i][0] != sorted_correct[i][0]:
            return 0

        # Check operator
        if sorted_parsed[i][2] != sorted_correct[i][2]:
            return 0

        # Compare condition values with more lenient matching
        parsed_value = str(sorted_parsed[i][1]).lower().strip()
        correct_value = str(sorted_correct[i][1]).lower().strip()

        # Normalize condition values, replace underscores with spaces
        parsed_value = parsed_value.replace("_", " ")
        correct_value = correct_value.replace("_", " ")

        # Remove possible quotes and parentheses
        parsed_value = (
            parsed_value.replace("'", "")
            .replace('"', "")
            .replace("(", "")
            .replace(")", "")
        )
        correct_value = (
            correct_value.replace("'", "")
            .replace('"', "")
            .replace("(", "")
            .replace(")", "")
        )

        if parsed_value != correct_value:
            return 0

    # All components match, return 1 point
    return 1.0


def compute_score(solution_str, ground_truth, table, ans):
    """
    Calculate WikiSQL score:
    1. If SQL statements are exactly the same, score 1
    2. If SQL statements differ, but all structured components match exactly, score 1
    3. Otherwise, score 0

    Args:
        solution_str: Model-generated answer
        ground_truth: True SQL query string
        table: Table structure dictionary containing header field
        ans: Structured representation of the true answer

    Returns:
        Score (0 or 1)
    """
    # Extract predicted SQL
    predicted_sql = extract_sql(solution_str=solution_str)

    # Randomly print debug info
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Ground Truth SQL: {ground_truth}")
        print(f"Generation SQL: {predicted_sql}")
        print(f"Solution string: {solution_str}")

    if predicted_sql is None:
        if do_print:
            print(f"No SQL query found")
        return 0

    # Check for exact SQL text match
    predicted_sql_normalize = normalize_sql(predicted_sql)
    ground_truth_sql_normalize = normalize_sql(ground_truth)

    if predicted_sql_normalize == ground_truth_sql_normalize:
        if do_print:
            print(
                f"Correct SQL query: PreSQL: {predicted_sql}, GoldSQL: {ground_truth}"
            )
        return 1.0

    # SQL text doesn't match, check structured information
    predicted_answer = parse_sql(predicted_sql, table["header"], verbose=do_print)
    final_score = score_sql(predicted_answer, ans)

    if do_print:
        if final_score == 1.0:
            print(f"Structurally correct: Components match exactly")
        else:
            print(f"Incorrect: Components do not match")
        print(f"Predicted: {predicted_answer}")
        print(f"Expected: {ans}")

    return final_score


def load_test_data_headers(test_data_file):
    """
    Load all questions and their corresponding header information from the test data file
    Returns a dictionary with question text as keys and header lists as values
    """
    headers_by_question = {}
    try:
        # Try to read JSON objects line by line
        with open(test_data_file, "r", encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse single-line JSON
                    item = json.loads(line)
                    question = item.get("question", "")
                    if "table" in item and "header" in item["table"]:
                        headers_by_question[question] = item["table"]["header"]
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: JSON parsing error at line {line_num}, skipped: {str(e)}"
                    )

        print(
            f"Successfully loaded headers for {len(headers_by_question)} questions from test data file"
        )
    except Exception as e:
        print(f"Error loading test data file: {str(e)}")

    return headers_by_question


def evaluate_single_item(item, default_header=None):
    """Evaluate a single JSON item"""
    # Extract required information
    model_answer = item.get("model_answer", "")
    truth_sql = item.get("truth_sql", "")
    truth_answer = item.get("truth_answer", {})
    question = item.get("question", "")  # Get question for inferring column names

    # Use provided header, if not provided use a default generic header
    if default_header:
        header_columns = default_header
    else:
        # Use a very generic default header
        header_columns = [
            "column_0",
            "column_1",
            "column_2",
            "column_3",
            "column_4",
            "column_5",
            "column_6",
            "column_7",
            "column_8",
            "column_9",
        ]

    # Ensure header length can cover required column indices
    if truth_answer.get("sel", -1) >= len(header_columns):
        header_columns.extend(
            [
                "column_" + str(i)
                for i in range(len(header_columns), truth_answer["sel"] + 1)
            ]
        )

    for idx in truth_answer.get("conds", {}).get("column_index", []):
        if idx >= len(header_columns):
            header_columns.extend(
                ["column_" + str(i) for i in range(len(header_columns), idx + 1)]
            )

    table = {"header": header_columns}

    # Calculate score
    score = compute_score(model_answer, truth_sql, table, truth_answer)

    return {
        "id": item.get("id", ""),
        "score": score,
        "prediction": extract_sql(model_answer),
        "prompt": item.get("prompt", ""),
        "full_response": item.get("model_answer", ""),
        "ground_truth": truth_sql,
        "header": header_columns,  # Add header for debugging
        "question": question,  # Add question for debugging
    }


def process_json_file(json_file, output_file=None, test_data_file=None):
    """Process JSON file containing WikiSQL evaluation items"""
    try:
        # Load JSON data
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load header information from test data file
        headers_by_question = {}
        if test_data_file:
            headers_by_question = load_test_data_headers(test_data_file)
            if not headers_by_question:
                print(
                    f"Warning: Failed to load header information from test data file, will use default headers"
                )

        # Determine data format
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # If it's a single item, place in a list
            items = [data]
        else:
            print(f"Error: Unsupported JSON format")
            return

        # Evaluation results
        results = []
        total_score = 0

        # Evaluate each item
        for item in items:
            # Get the question for this item
            question = item.get("question", "")

            # Try to get headers for this question
            item_headers = None
            if question in headers_by_question:
                item_headers = headers_by_question[question]

            # Evaluate this item
            result = evaluate_single_item(item, item_headers)
            results.append(result)
            total_score += result["score"]

        # Calculate overall accuracy
        accuracy = total_score / len(results) if results else 0

        # Generate evaluation report
        report = {
            "overall_accuracy": accuracy,
            "total_items": len(results),
            "correct_items": total_score,
            "results": results,
        }

        # Save results
        if output_file:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        # Print results
        print(f"\nEvaluation Results:")
        print(f"Overall accuracy: {accuracy:.4f} ({int(total_score)}/{len(results)})")
        print(f"Evaluated items: {len(results)}")
        if output_file:
            print(f"Detailed results saved to: {output_file}")

        return report

    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model answers for WikiSQL questions"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to JSON file containing predictions",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file for evaluation results",
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for the project (optional)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/wikisql/wikisql_test.json",
        help="Path to WikiSQL test data file for accurate header information",
    )

    args = parser.parse_args()

    test_path = os.path.join(args.base_path, args.test_data)

    # Process with new parameter names
    process_json_file(args.results_file, args.output_file, test_path)


if __name__ == "__main__":
    main()
