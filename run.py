from deepanalyze_gemini import DeepAnalyzeGemini
import os

# It is recommended to set the API key as an environment variable
# export GEMINI_API_KEY="your_api_key"
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

prompt = """# Instruction
Generate a data science report.

# Data
File 1:
{"name": "bool.xlsx", "size": "4.8KB"}
File 2:
{"name": "person.csv", "size": "10.6KB"}
File 3:
{"name": "disabled.xlsx", "size": "5.6KB"}
File 4:
{"name": "enlist.csv", "size": "6.7KB"}
File 5:
{"name": "filed_for_bankrupcy.csv", "size": "1.0KB"}
File 6:
{"name": "longest_absense_from_school.xlsx", "size": "16.0KB"}
File 7:
{"name": "male.xlsx", "size": "8.8KB"}
File 8:
{"name": "no_payment_due.xlsx", "size": "15.6KB"}
File 9:
{"name": "unemployed.xlsx", "size": "5.6KB"}
File 10:
{"name": "enrolled.csv", "size": "20.4KB"}"""

workspace = "example/student_loan/"

deepanalyze = DeepAnalyzeGemini(
    model_name="gemini-flash-latest",
    api_key=api_key,
)
answer = deepanalyze.generate(prompt, workspace=workspace)
print(answer["reasoning"])
