import asyncio
import argparse
import json
from datasets import load_dataset
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
import eco2ai


def create_agent(model_name: str) -> ReActAgent:
    code_spec = CodeInterpreterToolSpec()
    tools = code_spec.to_tool_list()
    llm = Ollama(model=model_name, request_timeout=120.0)
    
    return ReActAgent(
            tools=tools,
            llm=llm,
            system_prompt = '''You are a financial assistant that answers user questions using available tools.
For any question that involves mathematical calculations—no matter how simple—you must always use the code interpreter to perform the computation.
Do not perform math in your head or directly in the response. All calculations must go through the code interpreter.'''

        )


async def process_dataset(agent: ReActAgent, dataset_name: str, split: str, output_file: str):
    dataset = load_dataset(dataset_name, split=split)
    results = []

    conceptual_questions = dataset.filter(lambda x: x["question_type"] == "conceptual")

    for item in conceptual_questions:
        
        question = item["question"]
        answer = item.get("answer", "")
        question_type = item.get("question_type", "")

        try:
            response = await agent.run(user_msg=question)
            tool_call =  str(response.tool_calls)
        except Exception as e:
            response = f"Error during generation: {e}"
            tool_call = ["Error during generation: {e}"]
        
        results.append({
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "response": str(response),
            "tool_call": tool_call
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

async def main(model_name: str, dataset_name: str, split: str, output_file: str):
    agent = create_agent(model_name)
    await process_dataset(agent, dataset_name, split, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Nome do modelo Ollama")
    parser.add_argument("--dataset", type=str, default="AfterQuery/FinanceQA", help="Nome do dataset da Hugging Face")
    parser.add_argument("--split", type=str, default="test", help="Split do dataset (train, test, validation)")
    parser.add_argument("--output", type=str, default=None, help="(Opcional) Nome do arquivo JSON de saída")

    args = parser.parse_args()

    safe_model_name = args.model.replace(":", "_")
    output_file = args.output or f"{safe_model_name}_output.json"

    tracker = eco2ai.Tracker(
        project_name="Itau Agents",
        experiment_description=f"Inferência com modelo {args.model}",
        file_name=f"{safe_model_name}_agent_eco.csv"
    )
    tracker.start()

    asyncio.run(main(args.model, args.dataset, args.split, output_file))

    tracker.stop()
