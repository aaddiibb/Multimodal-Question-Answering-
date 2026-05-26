import uuid
import json
import os
from typing import Optional, Dict, Any
from treeofthoughts import TotAgent
# from loguru import logger
from dotenv import load_dotenv
import asyncio

load_dotenv()

class ToTDFSAgent:
    """
    A class to perform Depth-First Search (DFS) using the TotAgent, with pruning based on evaluation scores.

    Methods:
        dfs(state: dict, step: int = 0) -> Optional[Dict[str, Any]]:
            Performs DFS with pruning and returns the final thought dict.
    """

    def __init__(
        self,
        agent: TotAgent,
        threshold: float,
        max_loops: int,
        prune_threshold: float = 0.5,
        number_of_agents: int = 3,
        autosave_on: bool = True,
        id: str = uuid.uuid4().hex,
        *args,
        **kwargs,
    ):
        self.id = id
        self.agent = agent
        self.threshold = threshold
        self.max_loops = max_loops
        self.prune_threshold = prune_threshold
        self.number_of_agents = number_of_agents
        self.autosave_on = autosave_on

        self.all_thoughts: list[Dict[str, Any]] = []
        self.pruned_branches: list[Dict[str, Any]] = []

        # Ensure agent knows loop limit
        self.agent.max_loops = max_loops

    def dfs(self, state: Dict[str, Any], step: int = 0, depth: int = 0) -> Optional[Dict[str, Any]]:
        # logger.info(f"Starting DFS at depth {depth} for state: {state!r} (step {step})")
        if step >= self.max_loops:
            return None
        

        # Generate candidate thoughts sequentially
        candidates: list[Dict[str, Any]] = []
        for _ in range(self.number_of_agents):
            thought_model = asyncio.run(self.agent.run(state))
            candidates.append({
                "thought": thought_model.thought,
                "evaluation": thought_model.evaluation or 0.0,
                "depth": depth  # Add depth information
            })

        # Sort by evaluation (ascending)
        candidates.sort(key=lambda t: t["evaluation"])

        # Explore branches
        for thought in candidates:
            ev = thought["evaluation"]
            if ev > self.prune_threshold:
                self.all_thoughts.append(thought)
                # Prepare next state with updated question text
                next_state = {"question": thought["thought"], "text": state.get("text", "")}
                # logger.info(f"Depth {depth}: Exploring new branch with thought: {thought['thought']} (evaluation: {ev})")
                deeper = self.dfs(next_state, step + 1, depth + 1)  # Increased depth
                if deeper and deeper.get("evaluation", 0) > self.threshold:
                    return deeper
            else:
                self._prune_thought(thought)

        # logger.info(f"Finished DFS at depth {depth} for state: {state!r} (no branch passed threshold)")
        return None

    def _prune_thought(self, thought: Dict[str, Any]):
        self.pruned_branches.append({
            "thought": thought["thought"],
            "evaluation": thought["evaluation"],
            "depth": thought["depth"],  # Store depth of pruned thought
            "reason": "Below prune threshold"
        })

    def run(self, task: Dict[str, Any], *args, **kwargs) -> str:
        # Initial DFS pass
        result = self.dfs(task, *args, **kwargs)

        # Chain additional passes if needed
        for i in range(1, self.max_loops):
            if not result:
                break
            next_state = {"question": result["thought"], "text": task.get("text", "")}
            result = self.dfs(next_state, step=i, depth=i)

        # Sort collected thoughts and assemble tree
        self.all_thoughts.sort(key=lambda t: t["evaluation"])
        tree = {
            "final_thoughts": self.all_thoughts,
            "pruned_branches": self.pruned_branches,
            "highest_rated_thought": self.all_thoughts[-1] if self.all_thoughts else None,
        }
        json_string = json.dumps(tree, indent=4)

        if self.autosave_on:
            self._save_to_file(
                folder="tree_of_thoughts_runs",
                filename=f"tree_of_thoughts_run_{self.id}.json",
                content=json_string
            )

        return json_string

    def _save_to_file(self, folder: str, filename: str, content: str) -> None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def main(task: Dict[str, Any]):
    # Initialize the ToT agent
    tot_agent = TotAgent(max_loops=2,model_name=task["model_name"],client=task["client"])

    # Initialize the DFS wrapper
    dfs_agent = ToTDFSAgent(
        agent=tot_agent,
        threshold=0.8,
        max_loops=2,
        prune_threshold=0.5,
        number_of_agents=3,
        autosave_on=False
    )

    # Run and print the JSON tree of thoughts
    result_json = dfs_agent.run(task)
    # print(result_json)
    return result_json


if __name__ == "__main__":
    from UnifiedQADataLoader import UnifiedQADataLoader
    from agents import AsyncOpenAI

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

    # dataloader = UnifiedQADataLoader(
    #     dataset_type="manymqa",
    #     dev_file="./data/ManyModalQA/ManyModalQAData/official_aaai_split_dev_data.json",
    #     tables_file="./data/MultiModalQA/MMQA_tables.jsonl",
    #     texts_file="./data/MultiModalQA/MMQA_texts.jsonl",
    #     images_file="./data/MultiModalQA/MMQA_images.jsonl",
    #     images_base_url="./data/ManyModalQA/ManyModalImages",
    #     captions_file="./data/ManyModalQA/ManyModelQA_Captions.json",
    #     encode_images=False
    # )


    client = AsyncOpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    sample_input = dataloader.get_agent_inputs(10)
    task_payload = {
        "client": client,
        "model_name" : "gemini-1.5-flash-8b",
        "question": sample_input["question"],
        "text": sample_input.get("text"),
        "tables": sample_input.get("tables"),
        "captions": sample_input.get("captions"),
    }

    main(task_payload)
