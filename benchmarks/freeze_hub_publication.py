"""Create the immutable local approval manifest. This does not access the Hub."""

import argparse
from pathlib import Path

from hub_publication import file_table, freeze_manifest, size_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files-report", type=Path, required=True)
    args = parser.parse_args()
    if args.files_report.exists():
        raise FileExistsError(args.files_report)
    manifest = freeze_manifest(args.models, args.selection, args.output)
    lines = [
        "# PUBLIC Hub publication approval inventory",
        "",
        manifest["approval_gate"],
        "",
        *size_tables(manifest),
        *file_table(manifest),
        "",
    ]
    args.files_report.write_text("\n".join(lines))
    print(
        f"Prepared {len(manifest['repositories'])} PUBLIC repos, {manifest['total_bytes']} bytes. No Hub writes."
    )


if __name__ == "__main__":
    main()
