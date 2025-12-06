DEFAULT_SAVE_DIR = "./data/neuronpedia/"
CONFIG_BASE_DIR = "configs"
SAVE_ACT_DENSITIES_FILENAME = "neuronpedia_{model}_{submodel}_feature_act_densities.csv"
SAVE_FEATURE_FILENAME = "neuronpedia_{layer}_{index}_feature.csv"
NEURONPEDIA_API_KEY_ENV_VAR="NEURONPEDIA_API_KEY"
AVAILABLE_MODELS = {
    "gemma-2-2b": ["gemmascope-transcoder-16k", "clt-hp"],
    "qwen3-4b": ["transcoder-hp"]
}