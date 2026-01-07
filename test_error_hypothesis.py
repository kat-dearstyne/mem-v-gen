"""
Test error hypothesis: Compare base model logits with CLT replacement logits.

Measures accuracy between original model outputs and outputs when using
transcoder-reconstructed MLP activations.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.constants import DATA_PATH, PROMPT_IDS_MEMORIZED, PROMPT_ID_BASELINE, SUBMODELS, MODEL
from src.analysis.config_analysis.config_analyzer import ConfigAnalyzer
from src.analysis.config_analysis.config_replacement_model_accuracy_step import ConfigReplacementModelAccuracyStep
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyzer import CrossConfigAnalyzer
from src.neuronpedia_manager import NeuronpediaManager, GraphConfig
from src.utils import load_json
from utils import get_env_bool

CONFIG_DIR = Path("configs")

OUTPUT_DIR = Path("output/test_error_hypothesis")
load_dotenv()

ANALYZE_RESULTS = get_env_bool("ANALYZE_RESULTS", True)

def run_analysis_for_configs(config_dir: Path = CONFIG_DIR) -> dict:
    """
    Run accuracy analysis across all configs and conditions.

    Args:
        config_dir: Directory containing config JSON files

    Returns:
        Dictionary structured as {config_name: {condition: AccuracyResult}}
    """
    prompt_ids = PROMPT_IDS_MEMORIZED if len(config_dir.parents) == 1 else PROMPT_ID_BASELINE
    results = {}

    config_files = sorted(config_dir.glob("*.json"))
    base_save_path = os.path.expanduser(DATA_PATH)
    graph_dir = os.path.join(base_save_path, "graphs")

    graph_config = GraphConfig(model=MODEL, submodel=SUBMODELS[0])
    neuronpedia_manager = NeuronpediaManager(graph_dir=graph_dir, config=graph_config)

    for config_path in config_files:
        config_name = config_path.stem
        config = load_json(config_path)

        print(f"\nProcessing config: {config_name}")
        main_prompt = config["MAIN_PROMPT"]
        diff_prompts = config.get("DIFF_PROMPTS", [])
        sim_prompts = config.get("SIM_PROMPTS", [])
        prompt2ids = {p: (prompt_ids[index] if len(prompt_ids) > index else p)
                      for index, p in enumerate([main_prompt] + diff_prompts + sim_prompts)}
        analyzer = ConfigAnalyzer(neuronpedia_manager, prompts={p_id: p for p, p_id in prompt2ids.items()})
        memorized_completion = config.get("MEMORIZED_COMPLETION", None)
        results[config_name] = analyzer.run([SupportedConfigAnalyzeStep.REPLACEMENT_MODEL],
                                            conditions=[cond for cond in prompt_ids if cond != "rephrased"],
                                            memorized_completion=memorized_completion)

    return results

def main():
    results = run_analysis_for_configs(CONFIG_DIR)
    ConfigReplacementModelAccuracyStep.save_results(results, OUTPUT_DIR)

    return results


if __name__ == "__main__":
    if ANALYZE_RESULTS:
        analyzer = CrossConfigAnalyzer({}, save_path= Path("output/test_error_hypothesis"))
        result = analyzer.run()
    else:
        main()
