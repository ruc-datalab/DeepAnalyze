from deepanalyze import DeepAnalyzeVLLM

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

workspace = "/home/u2023000922/zhangshaolei/deepanalyze_dev/example/student_loan/"


deepanalyze = DeepAnalyzeVLLM(
    "/fs/fast/u2023000922/zhangshaolei/checkpoints/deepanalyze-8b/"
)
answer = deepanalyze.generate(prompt, workspace=workspace)
print(answer["reasoning"])
