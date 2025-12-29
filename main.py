import os
from pathlib import Path

from dotenv import load_dotenv

from constants import OUTPUT_DIR
from overlap_analysis import run_for_all_configs, analyze_overlap
from utils import get_env_bool, get_env_list, get_env_int

load_dotenv()

# ================= Setup Variables ==================
CONFIG_NAMES = get_env_list("CONFIG_NAME")
CONFIG_DIR = os.getenv("CONFIG_DIR", "")
RUN_ERROR_ANALYSIS = get_env_bool("RUN_ERROR_ANALYSIS", False)
ANALYSIS_DIRS = get_env_list("ANALYSIS_DIRS")
RUN_FINAL_ANALYSIS = get_env_bool("RUN_FINAL_ANALYSIS", False) and ANALYSIS_DIRS
SUBMODEL_NUM = get_env_int("SUBMODEL_NUM", default=0)

if __name__ == "__main__":
    if RUN_FINAL_ANALYSIS:
        assert len(ANALYSIS_DIRS) >= 2, f"Expected at least 2 dirs for analysis. Got {len(ANALYSIS_DIRS)}"
        dirs = [Path(OUTPUT_DIR) / d.strip() for d in ANALYSIS_DIRS if d.strip()]
        analyze_overlap(dirs=dirs, save_dir=dirs[0] / "final")
    else:
        run_for_all_configs(CONFIG_NAMES, CONFIG_DIR, RUN_ERROR_ANALYSIS,
                            submodel_num=SUBMODEL_NUM)
