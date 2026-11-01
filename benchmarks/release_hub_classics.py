"""After release v2.0.0, render cards in a new local payload view. Never upload."""

import argparse
from pathlib import Path

from hub_publication import file_table, size_tables
from hub_release_cards import prepare_release, write_card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    manifest, receipt = prepare_release(args.manifest, args.output, args.manifest_sha256)
    write_card(
        args.output / "publication-inventory.md",
        "\n".join(
            [
                "# PUBLIC release payload inventory",
                "",
                "Option B: approved content, repository list and licenses. Upload remains a coordinator action.",
                "Model files and all non-README payload hashes equal the basis manifest.",
                "",
                *size_tables(manifest),
                *file_table(manifest),
                "",
            ]
        ),
    )
    print(
        f"Prepared 30 local release payloads. Manifest SHA256: {receipt['publication_manifest_sha256']}. No Hub writes."
    )


if __name__ == "__main__":
    main()
