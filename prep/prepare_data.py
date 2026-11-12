"""Prepare balanced complaint routing data from one local CFPB export."""

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import io
from pathlib import Path
import random
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "CCDB"
OUTPUT_DIR = ROOT / "data" / "processed"
NARRATIVE = "Consumer complaint narrative"
PRODUCT = "Product"
SAMPLE_SIZE = 600
TRAIN_SIZE = 480
MAX_CHARS = 1200
SEED = 42

PRODUCT_ALIASES = {
    "Credit reporting": (
        "Credit reporting or other personal consumer reports",
        "Credit reporting, credit repair services, or other personal consumer reports",
    ),
    "Debt collection": (),
    "Credit card": ("Credit card or prepaid card",),
    "Checking or savings account": ("Bank account or service",),
    "Mortgage": (),
    "Student loan": (),
    "Money transfer": (
        "Money transfers",
        "Money transfer, virtual currency, or money service",
        "Prepaid card",
    ),
}
PRODUCT_TO_LABEL = {
    product.casefold(): label
    for label, aliases in PRODUCT_ALIASES.items()
    for product in (label, *aliases)
}


@contextmanager
def open_export(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield stream
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".csv")
            ]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one CSV inside {path}; found {len(members)}.")
            with archive.open(members[0]) as source:
                with io.TextIOWrapper(source, encoding="utf-8-sig", newline="") as stream:
                    yield stream
    else:
        raise ValueError(f"Expected a .csv or .zip export: {path}")


def load_complaints(path):
    by_label = {label: [] for label in PRODUCT_ALIASES}
    unmapped = Counter()
    label_by_text = {}
    conflicts = set()
    with open_export(path) as stream:
        reader = csv.DictReader(stream)
        missing = {PRODUCT, NARRATIVE} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            narrative = " ".join((row[NARRATIVE] or "").split())
            if not narrative:
                continue
            product = " ".join((row[PRODUCT] or "").split())
            label = PRODUCT_TO_LABEL.get(product.casefold())
            if label is None:
                unmapped[product or "<empty>"] += 1
                continue
            if len(narrative) > MAX_CHARS:
                # Cut at a word boundary so redaction markers stay intact.
                narrative = narrative[: MAX_CHARS + 1].rpartition(" ")[0]
                if not narrative:
                    continue
            previous_label = label_by_text.setdefault(narrative, label)
            if previous_label != label:
                conflicts.add(narrative)

    for narrative, label in label_by_text.items():
        if narrative not in conflicts:
            by_label[label].append((narrative, label))
    print("Unmapped products among rows with narratives:")
    if not unmapped:
        print("  (none)")
    for product, count in sorted(unmapped.items()):
        print(f"  {product}: {count}")
    print(f"Distinct cleaned narratives dropped for conflicting labels: {len(conflicts)}")
    return by_label


def prepare(path, output_dir):
    by_label = load_complaints(path)
    print("Usable rows before sampling:")
    for label, rows in by_label.items():
        print(f"  {label}: {len(rows)}")
    if any(len(rows) < SAMPLE_SIZE for rows in by_label.values()):
        raise ValueError(f"Need at least {SAMPLE_SIZE} usable rows in every class. No CSVs were written.")

    rng = random.Random(SEED)
    train, validation = [], []
    for rows in by_label.values():
        sample = rng.sample(rows, SAMPLE_SIZE)
        train.extend(sample[:TRAIN_SIZE])
        validation.extend(sample[TRAIN_SIZE:])
    all_rows = train + validation
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("support_tickets.csv", all_rows),
        ("train.csv", train),
        ("val.csv", validation),
    ):
        rng.shuffle(rows)
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("text", "label"))
            writer.writerows(rows)
        counts = Counter(label for _, label in rows)
        print(f"{filename} ({len(rows)} rows):")
        for label in PRODUCT_ALIASES:
            print(f"  {label}: {counts[label]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to a local CFPB CSV or ZIP.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        path = args.input
        if path is None:
            exports = [
                entry for entry in RAW_DIR.glob("*")
                if entry.is_file() and entry.suffix.lower() in {".csv", ".zip"}
            ]
            if len(exports) != 1:
                raise ValueError(f"Place one CSV or ZIP in {RAW_DIR}, or select one with --input.")
            path = exports[0]
        prepare(path, args.output_dir)
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
