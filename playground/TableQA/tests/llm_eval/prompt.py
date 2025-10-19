WIKITQ_EVAL = """
Here is the original question, the correct answer, and the candidate answer. Please evaluate whether the correct answer and the candidate answer are consistent. 

# Examples:
-
Question: What is the capital of France?
Candidate Answer: The capital of France is Paris
Correct Answer: Paris is the capital of France
Consistent: Yes
-
Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 5000 m
Consistent: Yes
-
Question: How many people live in the city?
Candidate Answer: 1 million
Correct Answer: 1000000
Consistent: Yes
-
Question: What is the date today?
Candidate Answer: 2023-10-01
Correct Answer: October 1, 2023
Consistent: Yes
-
Question: How many pages are in the book?
Candidate Answer: 300 pages
Correct Answer: 300
Consistent: Yes
-
Question: What is the temperature in Paris?
Candidate Answer: 25°C
Correct Answer: 77°F
Consistent: No
-
Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 10 km
Consistent: No
-
Question: What is the date today?
Candidate Answer: 2023-10-01
Correct Answer: 2023-11-01
Consistent: No
-
Question: in which three consecutive years was the record the same?
Candidate Answer: 1971,1972,1976
Correct Answer: 2004|2005|2006
Consistent: No

# YOUR TASK
Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

Question: {question}
Candidate Answer: {candidate_answer}
Correct Answer: {correct_answer}
Consistent:
"""
