"""
Main entry point for running L0 analysis with Gemma Scope 2 CLTs.
"""
import os

from run_l0 import run_l0_for_all_configs
from src.utils import get_env_list

if __name__ == "__main__":
    results = run_l0_for_all_configs(
        config_dir="l0",
        model_variant="1b-it",
        clt_config="width_262k_l0_medium",
        config_names=get_env_list("CONFIG_NAME")
    )
