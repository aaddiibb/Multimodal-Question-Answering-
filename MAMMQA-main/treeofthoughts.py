import uuid
import os
import json
import asyncio
import re
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)

load_dotenv()  # Loads GOOGLE_API_KEY into os.environ

# Prompt with clear instructions for JSON-only output
TREE_OF_THOUGHTS_SYS_PROMPT = """
You are an expert problem-solving agent. Follow these steps:

1. Analyze and break down the user's question.
2. Generate multiple thoughts with reasoning.
3. Self-evaluate each thought (score 0.1–1.0).
4. Synthesize a final answer.
5. Finally, respond with a JSON object with exactly two keys: "thought" (string) and "evaluation" (number), and nothing else.

Response_format:
{
  "thought": "<your_thought>",
  "evaluation": <your_evaluation>
}

The JSON object must be valid and parsable. Do not include any other text or explanations.
"""

class Thought(BaseModel):
    thought: str
    evaluation: Optional[float] = Field(None, description="Score between 0.1 and 1.0")

# Helper to fix missing commas between JSON keys
def fix_missing_commas(s: str) -> str:
    # Adds commas between adjacent quoted keys (e.g. "value" "key": => "value", "key":)
    return re.sub(r'(")\s*(")', r'\1,\2', s)

# Robust JSON parser with cleanup
def safe_json_parse(raw: str) -> dict:
    # Remove markdown-style code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Fix double double-quotes ("" => ")
    cleaned = cleaned.replace('""', '"')

    # Fix missing commas
    cleaned = fix_missing_commas(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after cleanup:\n{cleaned}") from e

class TotAgent:
    def __init__(self, client, model_name, max_loops: int = None):
        self.id = uuid.uuid4().hex
        self.max_loops = max_loops

        self.agent = Agent(
            name=f"ToT-Agent-{self.id}",
            instructions=TREE_OF_THOUGHTS_SYS_PROMPT,
            model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            ),
        )

    async def run(self, task: dict) -> Thought:
        prompt = (
            f"<context>{task.get('text', '')}</text>\n"
            f"<tables>{task.get('tables', '')}</tables>\n"
            f"<imagecaptions>{task.get('captions', '')}</imagecaptions></context>"
            f"Question: {task['question']}\n\n"
        )

        result = await Runner.run(self.agent, prompt)
        raw = result.final_output

        data = safe_json_parse(raw)
        return Thought(**data)

# Async entry point
async def main(task: dict):
    agent = TotAgent(client=task["client"], model_name=task["model_name"], max_loops=2)
    thought = await agent.run(task)
    return thought

# Script runner
if __name__ == "__main__":
    from UnifiedQADataLoader import UnifiedQADataLoader

    dataloader = UnifiedQADataLoader(
        dataset_type="multimqa",
        dev_file="./data/MultiModalQA/endgame_dev_filtered_data.json",
        tables_file="./data/MultiModalQA/MMQA_tables.jsonl",
        texts_file="./data/MultiModalQA/MMQA_texts.jsonl",
        images_file="./data/MultiModalQA/MMQA_images.jsonl",
        images_base_url="./data/MultiModalQA/final_dataset_images",
        captions_file="./data/MultiModalQA/MultiModelQA_Captions.json",
        encode_images=False
    )

    client = AsyncOpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    sample_input = dataloader.get_agent_inputs(10)
    task_payload = {
        "client": client,
        "model_name": "gemini-1.5-flash-8b",
        "question": sample_input["question"],
        "text": sample_input.get("text"),
        "tables": sample_input.get("tables"),
        "captions": sample_input.get("captions"),
    }

    print("Inputs are okay…")
    result = asyncio.run(main(task_payload))
    print(result.model_dump_json(indent=2))
