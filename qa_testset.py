from giskard.rag import KnowledgeBase, QATestset, evaluate, generate_testset
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_faithfulness
from giskard.rag.question_generators import complex_questions, double_questions
import giskard as giskard
import json
import os
import pandas as pd
from giskard.llm import set_embedding_model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] =  OPENAI_API_KEY
giskard.llm.set_embedding_model("text-embedding-3-small") #api_key=OPENAI_API_KEY)
giskard.llm.set_embedding_model("text-embedding-3-large") #api_key=OPENAI_API_KEY)

giskard.llm.set_llm_model("gpt-4o-mini")
import csv


# Define input and output file paths
input_file = "data/meta.jsonl"
output_file = "knowledge_base.csv"

# Open the JSONL file and the output CSV file
with open(input_file, "r", encoding="utf-8") as jsonl_file, open(output_file, "w", newline="", encoding="utf-8") as csv_file:
    # Read the first line to get the keys (columns)
    first_line = json.loads(jsonl_file.readline().strip())
    fieldnames = list(first_line.keys())  # Get existing fields
    # Add empty columns 'subtitle' and 'author'
    fieldnames.extend(["subtitle", "author"])

    # Initialize CSV writer
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row

    # Write the first line with the new empty fields
    first_line["subtitle"] = ""
    first_line["author"] = ""
    writer.writerow(first_line)

    # Process the remaining lines
    for line in jsonl_file:
        json_obj = json.loads(line.strip())  # Parse each JSON object
        # Add empty fields to match the new columns
        json_obj["subtitle"] = ""
        json_obj["author"] = ""
        writer.writerow(json_obj)  # Write to CSV

# Load the CSV into a DataFrame
df = pd.read_csv("knowledge_base.csv")

# Create the knowledge base
knowledge_base = KnowledgeBase.from_pandas(df, columns=["main_category", "description"])
print("Knowledge Base: ", knowledge_base)

# Generate a testset
testset = generate_testset(
    knowledge_base,
    question_generators=[complex_questions, double_questions],
    num_questions=2,
    agent_description="Product descriptions",
)
print("Generated Testset: ", testset)
# Save the generated testset
testset.save("my_testset.jsonl")

# You can easily load it back
from giskard.rag import QATestset

loaded_testset = QATestset.load("my_testset.jsonl")
# Convert it to a pandas dataframe
df = loaded_testset.to_pandas()
print(df)