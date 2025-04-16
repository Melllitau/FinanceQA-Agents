import os
import json
import concurrent.futures
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import openai


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()

MODEL_NAME = "gpt-4o"
INPUT_DIRECTORY = ""
OUTPUT_DIRECTORY = ""
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

BATCH_SIZE = 10
MAX_TOKENS = 300
TEMPERATURE = 0.0


def build_messages(answer, response):
    return [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator of answer alignment. "
                "You must reply with only 'yes' or 'no'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Given the following reference answer:\n\n"
                f"\"{answer}\"\n\n"
                f"And this given response:\n\n"
                f"\"{response}\"\n\n"
                f"Does the given response align with and satisfy the reference answer, even if written differently or in a more elaborate, wordy format? "
                f"Reply strictly with 'yes' or 'no'. "
                f"You must answer 'yes' if the given response clearly points to the same final answer as the reference, even if it includes additional explanation, steps, or phrasing differences. "
                f"For mathematical results, you must tolerate small numerical differences caused by rounding—answers that differ by up to ±1 should still be considered correct and marked as 'yes'."
            )
        }
    ]


def fetch_judgement(answer, response):
    try:
        messages = build_messages(answer, response)
        api_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = api_response.choices[0].message.content.strip()
        is_correct = content.lower().startswith("yes")
        return {
            "isCorrect": is_correct,
            "judge_response": content
        }
    except Exception as e:
        return {
            "isCorrect": False,
            "judge_response": f"Error: {str(e)}"
        }


def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        print(f"Skipping {filepath.name}: not a list of items.")
        return

    def process_item(item):
        return fetch_judgement(item["answer"], item["response"])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_item, items), total=len(items), desc=f"Processing {filepath.name}"))

    correct = 0
    incorrect = 0
    for item, result in zip(items, results):
        item["isCorrect"] = result["isCorrect"]
        item["judge_response"] = result["judge_response"]
        if result["isCorrect"]:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0

    final_output = {
        "summary": {
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(accuracy, 4)
        },
        "results": items
    }

    output_path = Path(OUTPUT_DIRECTORY) / filepath.name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Saved output to {output_path}")


def main():
    input_files = list(Path(INPUT_DIRECTORY).glob("*.json"))
    print(f"Found {len(input_files)} JSON files.")

    for file in input_files:
        process_file(file)

if __name__ == "__main__":
    main()
