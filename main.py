import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from dotenv import load_dotenv
from numpy.distutils.lib2def import DATA_RE

from common_utils import user_select_prompt, user_select_models
from constants import AVAILABLE_MODELS, CONFIG_BASE_DIR
from subgraph_comparisons import compare_prompt_subgraphs, compare_token_subgraphs, IntersectionMetrics
from error_comparisons import run_error_ranking, run_error_analysis, analyze_condition

load_dotenv()


class Task:
    PROMPT_SUBGRAPH_COMPARE = "prompt"
    TOKEN_SUBGRAPH_COMPARE = "token"


# ================= Setup Variables ==================
CONFIG_NAMES = os.getenv("CONFIG_NAME", "").split(",")
CONFIG_DIR = os.getenv("CONFIG_DIR", "")
MODEL = "gemma-2-2b"
SUBMODEL = AVAILABLE_MODELS[MODEL][0]
DATA_PATH = "~/data/spar-memory/neuronpedia/"
OUTPUT_PATH = "output"
TOP_K = 4
RUN_ERROR_ANALYSIS = True
PROMPT_IDS_TRIPLETS = ["mem", "gen1", "gen2"]
PROMPT_ID_PAIRS = ["main", "baseline"]


def run_for_config(config_dir: Path, config_name: str) -> tuple[Optional[dict], Optional[dict]]:
    config_path = config_dir / f"{config_name}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load prompts from config
    main_prompt = config["MAIN_PROMPT"]
    diff_prompts = config.get("DIFF_PROMPTS", [])
    sim_prompts = config.get("SIM_PROMPTS", [])
    token_of_interest = config.get("TOKEN_OF_INTEREST")
    selected_task = config.get("TASK")

    if diff_prompts or sim_prompts:
        assert not token_of_interest or selected_task, ("Both TOKEN_OF_INTEREST and DIFF_PROMPTS/SIM_PROMPTS supplied. "
                                                        "Must specify what task to perform.")
        selected_task = selected_task if selected_task else Task.PROMPT_SUBGRAPH_COMPARE

    elif token_of_interest:
        selected_task = selected_task if selected_task else Task.TOKEN_SUBGRAPH_COMPARE

    base_save_path = os.path.expanduser(DATA_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODEL)
    graph_dir = os.path.join(base_save_path, "graphs")

    prompt = user_select_prompt(prompt_default=main_prompt, graph_dir=graph_dir)
    prompt_ids = PROMPT_IDS_TRIPLETS if len(diff_prompts) == 3 else PROMPT_ID_PAIRS
    prompt2ids = {p: (prompt_ids[index] if len(prompt_ids) > index else p)
                  for index, p in enumerate([prompt] + diff_prompts)}
    error_analysis_results = None
    if selected_task == Task.PROMPT_SUBGRAPH_COMPARE:
        print(f"\nStarting run for {config_name} with model {model} and submodel {submodel}")
        print("\n".join([f"{p_id}: {p}" if p_id != p else p for p, p_id in prompt2ids.items()]))
        print("==================================")

        metrics = compare_prompt_subgraphs(main_prompt=prompt, diff_prompts=diff_prompts, sim_prompts=sim_prompts,
                                           model=model, submodel=submodel, graph_dir=graph_dir, debug=True)
        metrics = {prompt2ids.get(p, p): res for p, res in metrics.items()}
        print("\n".join([f"{prompt} {[f'{metric:.3f}' for metric in metric_vals]}"
                         for prompt, metric_vals in metrics.items()]))
        if diff_prompts and RUN_ERROR_ANALYSIS:
            error_analysis_results = run_error_ranking(prompt1=prompt, prompt2=diff_prompts[-1], model=model,
                                                       submodel=submodel, graph_dir=graph_dir)
        print("==================================\n")
    elif selected_task == Task.TOKEN_SUBGRAPH_COMPARE:
        print(f"\nStarting run with model {model} and submodel {submodel}"
              f"\nPrompt1: '{prompt}'\n"
              f"\nToken of interest: {token_of_interest}.\n")

        metrics = compare_token_subgraphs(main_prompt=prompt, token_of_interest=token_of_interest, model=model,
                                          submodel=submodel,
                                          graph_dir=graph_dir, tok_k_outputs=TOP_K)
    else:
        raise NotImplementedError("Unknown task")
    return metrics, error_analysis_results


if __name__ == "__main__":
    assert CONFIG_NAMES or CONFIG_DIR, "Must provide config names or config dir!"
    config_dir = Path(CONFIG_BASE_DIR)
    if CONFIG_DIR:
        config_dir = config_dir / CONFIG_DIR
    if not CONFIG_NAMES or CONFIG_NAMES[0].lower().strip() == 'all':
        CONFIG_NAMES = [f.stem for f in config_dir.glob("*.json")]
    all_results = {"config_name": [], "prompt_type": [],
                   **{metric_name: [] for metric_name in IntersectionMetrics._fields}}
    error_pair_results = []
    for config in CONFIG_NAMES:
        config = config.strip()
        results, error_analysis_results = run_for_config(config_dir, config)
        results: dict[str, IntersectionMetrics]
        error_pair_results.append(error_analysis_results)
        for i, (prompt, metrics) in enumerate(results.items()):
            all_results["config_name"].append(config)
            all_results["prompt_type"].append(prompt)
            for metric_name, metric in zip(IntersectionMetrics._fields, metrics):
                all_results[metric_name].append(metric)

    results_df = pd.DataFrame(all_results)
    results_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUT_PATH, results_dir, f"overlap-metrics.csv"))

    if all(error_pair_results):
        stats, deltas = analyze_condition(error_pair_results,
                                          save_path=Path(OUTPUT_PATH) / results_dir / "error-results.csv")
