import os
from pathlib import Path

from dotenv import load_dotenv

from constants import OUTPUT_DIR
from overlap_analysis import run_for_all_configs, analyze_overlap

load_dotenv()


# ================= Setup Variables ==================
CONFIG_NAMES = os.getenv("CONFIG_NAME", "").split(",")
CONFIG_DIR = os.getenv("CONFIG_DIR", "")
RUN_ERROR_ANALYSIS = os.getenv("RUN_ERROR_ANALYSIS", "false").lower() == "true"
ANALYSIS_DIRS =  os.getenv("ANALYSIS_DIRS", "").split(",")
RUN_FINAL_ANALYSIS = os.getenv("RUN_FINAL_ANALYSIS", "false").lower() == "true" and ANALYSIS_DIRS
SUBMODEL_NUM = int(os.getenv("SUBMODEL_NUM", "0"))


if __name__ == "__main__":
    if RUN_FINAL_ANALYSIS:
        assert len(ANALYSIS_DIRS) >= 2, f"Expected at least 2 dirs for analysis. Got {len(ANALYSIS_DIRS)}"
        dirs = [Path(OUTPUT_DIR) / d.strip() for d in ANALYSIS_DIRS if d.strip()]
        analyze_overlap(dirs=dirs, save_dir=dirs[0] / "final")
    else:
        run_for_all_configs(CONFIG_NAMES, CONFIG_DIR, RUN_ERROR_ANALYSIS,
                            submodel_num=SUBMODEL_NUM)
