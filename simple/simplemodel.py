import asyncio
import argparse
import json
from datasets import load_dataset
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import eco2ai


def create_llm(model_name: str) -> Ollama:
    return Ollama(model=model_name, request_timeout=120.0)


async def process_dataset(llm: Ollama, dataset_name: str, split: str, output_file: str):
    dataset = load_dataset(dataset_name, split=split)
    results = []

    conceptual_questions = dataset.filter(lambda x: x["question_type"] == "conceptual")

    for item in conceptual_questions:
        question = item["question"]
        answer = item.get("answer", "")
        question_type = item.get("question_type", "")

        try:
            messages = [
                ChatMessage(
                    role="system",
                    content="You are a financial assistant that answers user questions clearly and concisely.",
                ),
                ChatMessage(role="user", content=question),
            ]
            response = llm.chat(messages)
            response_text = response.message.content
        except Exception as e:
            response_text = f"Error during generation: {e}"

        results.append({
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "response": response_text
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def main(model_name: str, dataset_name: str, split: str, output_file: str):
    llm = create_llm(model_name)
    await process_dataset(llm, dataset_name, split, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Nome do modelo Ollama")
    parser.add_argument("--dataset", type=str, default="AfterQuery/FinanceQA", help="Nome do dataset da Hugging Face")
    parser.add_argument("--split", type=str, default="test", help="Split do dataset (train, test, validation)")
    parser.add_argument("--output", type=str, default=None, help="(Opcional) Nome do arquivo JSON de saída")

    args = parser.parse_args()
    safe_model_name = args.model.replace(":", "_")
    output_file = args.output or f"{safe_model_name}_output_simple.json"

    tracker = eco2ai.Tracker(
        project_name="Itau Agents",
        experiment_description=f"Inferência com modelo {args.model} sem agente.",
        file_name=f"{safe_model_name}_simple_eco.csv"
    )
    tracker.start()

    asyncio.run(main(args.model, args.dataset, args.split, output_file))

    tracker.stop()
