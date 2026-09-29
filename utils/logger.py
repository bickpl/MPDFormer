import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = None

    def log(self, row):
        row = {key: _to_python(value) for key, value in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        extra = [key for key in row if key not in self.fieldnames]
        if extra:
            old_rows = []
            if self.path.exists():
                with self.path.open("r", newline="", encoding="utf-8") as handle:
                    old_rows = list(csv.DictReader(handle))
            self.fieldnames.extend(extra)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
                writer.writerow(row)
            return

        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)


def _to_python(value):
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    except Exception:
        pass
    return value

