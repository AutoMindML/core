import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from automind.experiments.protocol import (
    DatasetManifest,
    ResearchProtocol,
    validate_dataset_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automind-experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "dry-run", "run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("protocol", type=Path)
        command.add_argument("--dataset-root", type=Path)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("output_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        from automind.experiments.synthea_pilot import SyntheaPilotRunner

        runs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.output_root.glob("seed_*/run_*/metrics.json")
        ]
        print(
            json.dumps(
                SyntheaPilotRunner.summarize(runs), indent=2, sort_keys=True
            )
        )
        return 0
    protocol = ResearchProtocol.load(args.protocol)
    protocol_root = args.protocol.parent
    manifest_path = (protocol_root / protocol.dataset_manifest).resolve()
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if args.dataset_root is not None:
        validate_dataset_files(manifest, args.dataset_root)
    if args.command == "validate":
        payload = {
            "status": "valid",
            "protocol_fingerprint": protocol.fingerprint(),
            "dataset": manifest.dataset_id,
            "files_checked": args.dataset_root is not None,
        }
    elif args.command == "dry-run":
        payload = {**protocol.dry_run(), "dataset": manifest.dataset_id}
    else:
        if args.dataset_root is None:
            raise ValueError("run/resume requires --dataset-root")
        from automind.experiments.synthea_pilot import SyntheaPilotRunner

        payload = SyntheaPilotRunner(protocol, args.dataset_root).run(
            resume=args.command == "resume"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
