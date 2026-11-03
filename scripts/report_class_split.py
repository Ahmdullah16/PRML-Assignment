from pathlib import Path
from .report_holdout import build_reports

if __name__ == "__main__":
    build_reports(Path("results/class_6_3_3_seed42"), Path("reports/class_6_3_3_seed42"))
