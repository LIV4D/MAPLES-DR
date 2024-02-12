__all__ = ["configure", "Dataset", "export_test_set", "export_train_set", "load_test_set", "load_train_set"]

import os

if os.environ.get("MAPLES-DR_SILENT_IMPORT", None) is None:
    from rich.console import Console

    console = Console()
    console.print(
        "[b]Thanks for using MAPLES-DR![/b]\n"
        "  When using this dataset in academic works,\n"
        "  please cite: [u cyan link=https://arxiv.org/abs/2402.04258]https://arxiv.org/abs/2402.04258[/u cyan link]"
    )

from .dataset import Dataset
from .quick_api import configure, export_test_set, export_train_set, load_test_set, load_train_set
