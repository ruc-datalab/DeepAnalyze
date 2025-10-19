COT_PROMPT_TEMPLATE = """Based on the table below, please answer the question, the answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>here is your answer</answer>

#Task:

Table:
{table}

Question: {question}
"""


COT_PROMPT_WIKISQL_TEMPLATE = """Given the following table:
{table}
Answer the question: {question}.

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final SQL answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer> SELECT Name FROM table WHERE Age > 30 </answer>
"""


COT_PROMPT_TATQA_TEMPLATE = """<|im_start|>system
You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
<|im_end|>
<|im_start|>user
Below is an instruction that describes a question answering task in the finance domain, provided with an input table and relevant text as context.
Please answer the given question based on the table and text.
### Instruction
Given a table and a list of texts in the following, what is the answer to the question?
### Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.
### Table:
{table}
### Text:
{text}
### Question:
{question}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>Lake Palas Tuzla</answer>

<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>
"""


COT_PROMPT_HYBRIDQA_TEMPLATE = """<|im_start|>system
You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
<|im_end|>
<|im_start|>user
Please answer the given question based on the table and text. You should reach a short-form answer after reasoning.

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

### Table:
{table}

### Related Text Information:
{text}

### Question:
{question}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>Lake Palas Tuzla</answer>

<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>
"""

COT_PROMPT_TABFACT_TEMPLATE = """
Task:
You will predict the claim's label based on the given claim and table. There are 2 possible labels: SUPPORTS, REFUTES. SUPPORTS means the claim is supported by the table. REFUTES means the claim is refuted by the table.
## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

Table:
{table}

Claim:
{claim}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>SUPPORTS</answer>
"""


COT_PROMPT_MMQA_TEMPLATE = """
"You are an expert at multi-table question answering, you need to extract answers based on the given multi-hop question and given two tables. 
Please only output the results without any other words in the format of: {{"Answers": List}}. You should think step-by-step to reach the final answer.

## The given multi-hop question:
{question}
## The given table 1:
{table1}
## The given table 2:
{table2}
"""

COT_PROMPT_MULTHIERTT_TEMPLATE = """
Task:
You are an expert at multi-table question answering, you need to extract answers based on the given multi-hop question, tables and text.
## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## The question:
{question}
## The given tables:
{table}
## Table Description:
{table_description}
## Text:
{text}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>Lake Palas Tuzla</answer>
"""

COT_PROMPT_FINQA_TEMPLATE = """Below is an task that describes a question answering task in the finance domain, provided with an input table and relevant text as context. 
Please answer the given question based on the table and text.

## Task:
Given a table and a list of texts in the following, what is the answer to the question?

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

### pre_text:
{pre_text}

### Table:
{table}

### post_text:
{post_text}

### Question:
{question}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>96.7</answer>
"""

COT_PROMPT_SCITAB_TEMPLATE = """
## Task:
You will predict the claim's label based on the given claim and table. There are 2 possible labels: SUPPORTS, REFUTES. SUPPORTS means the claim is supported by the table. REFUTES means the claim is refuted by the table.
You should think step-by-step to reach the final answer.

### Table:
{table}

###  Claim:
{claim}

### Response
Your response should end with `Answer: xxx` (answer to the question).
"""

COT_PROMPT_OTTQA_TEMPLATE = """Please answer the given question based on the table and text. You should reach a short-form answer after reasoning.
### Task:
Given a table and a list of texts in the following, what is the answer to the question?
## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

### Table:
{table}

### Text:
{text}

### Question:
{question}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>96.7</answer>
"""


COT_PROMPT_HITAB_TEMPLATE = """
Task:
You are capable of effectively identifying the hierarchical structure of the table. Based on the provided table, please provide the answer to the question.

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## Table:
{table}

## Question:
{question}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>Paris</answer>
"""


COT_PROMPT_FEVEROUS_TEMPLATE = """
Task:
You will predict the claim's label based on the given claim and table (text). There are 3 possible labels: SUPPORTS, REFUTES, and NOT ENOUGH INFO. SUPPORTS means the claim is supported by the table or text. REFUTES means the claim is refuted by the table or text. NOT ENOUGH INFO means the table (text) does not provide enough information to make the decision.

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## Table:
{table}

## Text:
{text}

## Claim:
{claim}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>SUPPORTS</answer>
"""


COT_PROMPT_TOTTO_TEMPLATE = """You are given a table in one of the following formats: html, plain text, or json.
The table is accompanied by a context.
The context is in the form of page title, table title, section title, and a few sentences taken from the same section as the table.
The table has some highlighted cells. These cells are wrapped with <hl> and </hl> tags.
Your job is to use the table and the context to produce a sentence.
The sentence must have the following characteristics:
- it must be structured in a way that allows the reader to understand it without seeing the table
- it can't contain lists
- it must be only a single sentence long
- it should focus on the content of the highlighted cells wrapped with <hl> and </hl> tags
- it must not explicitly mention that some cells are highlighted

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## TABLE:
{table}

## CONTEXT:
page title: {page_title}
section title: {section_title}
section text: {section_text} 

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.</answer>
"""


COT_PROMPT_FETAQA_TEMPLATE = """You are given a table in one of the following formats: html, plain text, or json.
The table is accompanied by a context.
The context is in the form of page title, table title, section title.
The table has some highlighted cells. These cells are wrapped with <hl> and </hl> tags.
Your task is to answer the question based on the table and the text content, and the answer should be a single sentence.
The sentence must have the following characteristics:
- it must be able to answer what the question asks
- it can't contain lists
- it must be only a single sentence long
- it should focus on the content of the highlighted cells wrapped with <hl> and </hl> tags
- it must not explicitly mention that some cells are highlighted

## Notes:
- Output the reasoning process inside a single pair of <think>...</think> tags.
- Output the final answer inside <answer>...</answer> tags.
- Think through your reasoning process carefully, solve Question step by step.

## TABLE:
{table}

## CONTEXT:
page title: {page_title}
section title: {section_title}

## Question:
{question}

## Output Format Example

<think>
Step-by-step reasoning here...
</think>

<answer>Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.</answer>
"""


END_TO_END_PROMPT_TATQA_TEMPLATE = """Below is an instruction that describes a question answering task in the finance domain, paired with an input table and its
relevant text that provide further context. The given question is relevant to the table and text. Generate an appropriate
answer to the given question.
### Instruction
Given a table and a list of texts in the following, what is the answer to the question? Please predict the answer and store
it in a variable named ‘{answer}‘. If there are multiple values, separate them using the ’#’ symbol. If the value of the
‘{answer}‘ is numerical, predict its scale and store it in a variable named ‘{scale}‘. The value of ‘{scale}‘ can be one of
the following: ‘none‘, ‘percent‘, ‘thousand‘, ‘million‘, or ‘billion‘. For non-numerical values, set the value of ‘{scale}‘
to ’none’. Finally, present the final answer in the format of "The answer is: {answer} #### and its corresponding scale is:
{scale}"
### Table
{table}
### Text
{text}
### Question
{question}
### Response
The answer is: {gold_answer} #### and its corresponding scale is: {gold_scale}

"""


STEP_WISE_PROMPT_TATQA_TEMPLATE = """Below is an instruction that describes a question answering task in the finance domain, paired with an input table and its
relevant text that provide further context. The given question is relevant to the table and text. Generate an appropriate
answer to the given question.
### Instruction
Given a table and a list of texts in the following, answer the question posed using the following five-step process:
1. Step 1: Predict the type of question being asked. Store this prediction in the variable ‘{question_type}‘. The value of
‘{question_type}‘ can be one of the following:‘Single span‘, ‘Multiple spans‘, ‘Count‘, or ‘Arithmetic‘.
2. Step 2: Extract the relevant strings or numerical values from the provided table or texts. Store these pieces of evidence
in the variable ‘{evidence}‘. If there are multiple pieces of evidence, separate them using the ’#’ symbol.
3. Step 3: if the ‘{question_type}‘ is ‘Arithmetic‘, formulate an equation using values stored in ‘{evidence}‘. Store this
equation in the variable ‘{equation}‘. For all other question types, set the value of {equation} to ’N.A.’.
4. Step 4: Predict or calculate the answer based on the question type, evidence and equation. Store it in the variable
‘{answer}‘. If there are multiple values, separate them using the ’#’ symbol.
5. Step 5: If the value of the ‘{answer}‘ is numerical, predict its scale and store it in a variable named ‘{scale}‘. The
value of ‘{scale}‘ can be one of the following: ‘none‘, ‘percent‘, ‘thousand‘, ‘million‘, or ‘billion‘. For non-numerical
values, set the value of ‘{scale}‘ to ’none’.
Please organize the results in the following table:
| step | output |
| 1 | {question_type} |
| 2 | {evidence} |
| 3 | {equation} |
| 4 | {answer} |
| 5 | {scale} |
Finally, present the final answer in the format: "The answer is: {answer} #### and its corresponding scale is: {scale}"
### Table
{table}
### Text
{text}
### Question
{question}
### Response
| step | output |
| 1 | {gold_question_type} |
| 2 | {gold_evidence} |
| 3 | {gold_equation} |
| 4 | {gold_answer} |
| 5 | {gold_scale} |
The answer is: {gold_answer} #### and its corresponding scale is: {gold_scale}

"""

MACT_PROMPT_TATQA_TEMPLATE = """Solve a table question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and
Action can be four types:
(1) Retrieve[cells], which retrieves certain cell(s) from the table and returns the retrieved cells in string format.
(2) Look up[information], which looks up the information in the context (if any) and returns the information in string format.
(3) Calculate[formular/instruction], which carries out calculations based on the formular, or the instruction and returns the calculated results.
(4) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
Table:
| | 2019 | | 2018
In thousands | $ | % | $ | %
Drinkable Kefir other than ProBugs | $ 71,822 | 77% | $ 78,523 | 76%
Cheese | 11,459 | 12% | 11,486 | 11%
Cream and other | 4,228 | 4% | 5,276 | 5%
ProBugs Kefir | 2,780 | 3% | 2,795 | 3%
Other dairy | 1,756 | 2% | 3,836 | 4%
Frozen Kefir (a) | 1,617 | 2% | 1,434 | 1%
Net Sales | $ 93,662 | 100% | $ 103,350 | 100%
Context: Paragraph 1: Our product categories are: Paragraph 2: Drinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and
types, including low fat, non-fat, whole milk, protein, and BioKefir (a 3.5 oz. kefir with additional probiotic cultures). Paragraph 3: Europeanstyle soft cheeses, including farmer cheese in resealable cups. Paragraph 4: Cream and other, which consists primarily of cream, a byproduct of
making our kefir. Paragraph 5: ProBugs, a line of kefir products designed for children. Paragraph 6: Other Dairy, which includes Cupped Kefir and
Icelandic Skyr, a line of strained kefir and yogurt products in resealable cups. Paragraph 7: Frozen Kefir, available in soft serve and pint-size
containers. Paragraph 8: Lifeway has determined that it has one reportable segment based on how our chief operating decision maker manages the
business and in a manner consistent with the internal reporting provided to the chief operating decision maker. The chief operating decision
maker, who is responsible for allocating resources and assessing our performance, has been identified collectively as the Chief Financial Officer,
the Chief Operating Officer, the Chief Executive Officer, and Chairperson of the board of directors. Substantially all of our consolidated
revenues relate to the sale of cultured dairy products that we produce using the same processes and materials and are sold to consumers through a
common network of distributors and retailers in the United States. Paragraph 9: Net sales of products by category were as follows for the years
ended December 31: Paragraph 10: (a) Includes Lifeway Kefir Shop sales Paragraph 11: Significant Customers – Sales are predominately to companies
in the retail food industry located within the United States. Two major customers accounted for approximately 22% and 21% of net sales for the
years ended December 31, 2019 and 2018, respectively. Two major customers accounted for approximately 17% of accounts receivable as of December
31, 2019 and 2018. Our ten largest customers as a group accounted for approximately 57% and 59% of net sales for the years ended December 31, 2019
and 2018, respectively.
Question: What is the change in the net sales for cheese between 2018 and 2019?
Thought 1: I need to retrive the net sales for cheese for 2018 and 2019.
Action 1: Retrieve[net sales for cheese for 2018 and 2019]
Observation 1: | 2019 | 2018 |
| 11459 | 11486 |
Thought 2: I need to substract the net sales for cheese for 2019 from 2018, which results in the formular: [11459 - 11486]
Action 2: Calculate[11459 - 11486]
Observation 2: -27
Thought 3: The question asks for change in the net sales for cheese between 2018 and 2019. The answer is in the observation 2, which is -27. The
unit/scale of the number is thousand, therefore the answer is -27 thousands.
Action 3: Finish[-27 thousands]
Table:
| | Three Months Ended | | % Variation |
| December 31, 2019 | September 29, 2019 | December 31, 2018 | Sequential | Year-Over-Year
| | | (Unaudited, in millions) | |
Automotive and Discrete Group (ADG) | $924 | $894 | $967 | 3.3% | (4.5)%
Analog, MEMS and Sensors Group (AMS) | 1,085 | 968 | 988 | 12.1 | 9.9
Microcontrollers and Digital ICs Group (MDG) | 742 | 688 | 689 | 7.9 | 7.6
Others | 3 | 3 | 4 | — | —
Total consolidated net revenues | $2,754 | $2,553 | $2,648 | 7.9% | 4.0%
Context: Paragraph 1: On a sequential basis, ADG revenues were up 3.3%, driven by an increase in volumes of approximately 8%, partially offset by
a decrease in average selling prices of approximately 5%, mostly attributable to product mix. Paragraph 2: AMS revenues increased 12.1% driven by
Analog and Imaging products. AMS increase was due to an increase of approximately 5% in average selling prices, entirely due to product mix, and
to higher volumes of approximately of 7%. Paragraph 3: MDG revenues increased by 7.9%, mainly driven by Microcontrollers, due to both higher
product mix, and higher volumes of approximately 2%. Paragraph 4: On a year-over-year basis, fourth quarter net revenues increased
by 4.0%. ADG revenues decreased 4.5% compared to the year-ago quarter on lower revenues in both Automotive and Power Discrete. The decrease was
entirely due to lower average selling prices of approximately 4%, while volumes remained substantially flat. The decrease in average selling
prices was a combination of less favorable product mix and lower selling prices. Paragraph 5: AMS fourth quarter revenues grew 9.9% year-overyear, mainly driven by Analog and Imaging. The increase was entirely due to higher average selling prices of approximately 18%, entirely
attributable to product mix,
partially offset by lower volumes of approximately 8%. MDG fourth quarter revenues increased by 7.6%, mainly driven by Microcontrollers. The
increase was due to higher average selling prices of approximately 9%,
entirely due to improved product mix.
Question: What led to increase in the revenue of ADG on sequential basis?
Thought 1: I need to look up what led to increase in the revenue of ADG on sequential basis in the context.
Action 1: Look up[what led to increase in the revenue of ADG on sequential basis]
Observation 1: On a sequential basis, ADG revenues were up 3.3%, driven by an increase in volumes of approximately 8%, partially offset by a
decrease in average selling prices of approximately 5%, mostly attributable to product mix.
Thought 2: The question asks what led to increase in the revenue of ADG on sequential basis, the answer is in the observation 1, which is: an
increase in volumes of approximately 8%, partially offset by a decrease in average selling prices of approximately 5%, mostly attributable to
product mix. There is no unit or scale needed as the answer is not a number.
Action 2: Finish[an increase in volumes of approximately 8%, partially offset by a decrease in average selling prices of approximately 5%, mostly
attributable to product mix.]
Table:
| March 31, |
| 2019 | 2018
Raw materials | $74.5 | $26.0
Work in process | 413.0 | 311.8
Finished goods | 224.2 | 138.4
Total inventories | $711.7 | $476.2
Context: Paragraph 1: Inventories Paragraph 2: The components of inventories consist of the following (in millions): Paragraph 3: Inventories are
valued at the lower of cost and net realizable value using the first-in, first-out method. Inventory impairment charges establish a new cost basis
for inventory and charges are not subsequently reversed to income even if circumstances later suggest that increased carrying amounts are
recoverable.
Question: What was the percentage change in total inventories between 2018 and 2019?
Thought 1: I need to retrieve the total inventories of 2018 and 2019 respectively.
Action 1: Retrieve[total inventories of 2018 and 2019]
Observation 1: | 2019 | 2018 |
| $711.7 | $476.2 |
Thought 2: I need to calculate the percentage using the formular: [((711.7-476.2)/476.2)*100]
Action 2: Calculate[((711.7-476.2)/476.2)*100]
Observation 2: 49.45.
Thought 3: The question asks for percentage change of total inventories between 2018 and 2019. The answer is in the observation 2, which is 49.45.
The unit/scale of the number is percentage, therefore the answer is 49.45%
Action 3: Finish[49.45%]
(END OF EXAMPLES)
Now generating the Thought, Action, Observation for the following instance:
Table:
{table}
Context: {context}
Question: {question}
{memory}


"""
