import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from automind.experiments.progress import ProgressReporter, stderr_renderer
from automind.experiments.protocol import (
    DatasetManifest,
    NoviceComparisonProtocol,
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
    replay = commands.add_parser("replay-v2")
    replay.add_argument("completion", type=Path)
    replay.add_argument("output_root", type=Path)
    replay.add_argument("--metadata", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress = ProgressReporter(stderr_renderer())
    if args.command == "replay-v2":
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("replay destination must be empty")
        raw = json.loads(args.completion.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        matches = [
            item for item in items
            if item.get("kind") == "direct_code"
            and item.get("status") == "succeeded"
        ]
        if not matches:
            raise ValueError("saved artifact has no direct-code completion")
        from automind.experiments.codegen import (
            DirectCodeHarness,
            PodmanSandboxExecutor,
            default_sandbox_profile,
            syntax_check,
        )
        from automind.experiments.orchestration import _extract_code

        code = _extract_code(matches[-1]["response"]["content"])
        syntax_check(code)
        metadata = None
        if args.metadata is not None:
            metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        else:
            adjacent = args.completion.parent / "direct_code_metadata.json"
            if adjacent.is_file():
                metadata = json.loads(adjacent.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(
                "replay requires original metadata: pass --metadata or place "
                "direct_code_metadata.json beside the completion artifact"
            )
        dataset = metadata.get("dataset")
        target = (
            dataset.get("target")
            if isinstance(dataset, dict)
            else metadata.get("target")
        )
        if not isinstance(target, str) or not target:
            raise ValueError(
                "replay metadata must declare a non-empty target column"
            )
        profile = default_sandbox_profile()
        probe = DirectCodeHarness(
            PodmanSandboxExecutor(profile), policy=profile.policy
        ).probe(
            code,
            args.output_root / "direct_code_probe",
            metadata,
            target,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": probe.status,
            "probe": probe.as_dict(),
            "provider_calls": 0,
            "completion_sha256": matches[-1].get("completion_sha256"),
            "metadata_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        (args.output_root / "replay.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if probe.status == "succeeded" else 1
    if args.command == "summarize":
        from automind.experiments.synthea_pilot import SyntheaPilotRunner

        v2 = list(args.output_root.glob("seed_*/run_*/result.json"))
        if v2:
            dimensions: dict[str, int] = {}
            for path in v2:
                result = json.loads(path.read_text(encoding="utf-8"))
                for name, condition in result.get("conditions", {}).items():
                    fields = {
                        "condition": name,
                        "status": condition.get("status", "unknown"),
                        "stage": condition.get("stage", "none"),
                        "reason": condition.get("reason", "none"),
                        "phase": condition.get("phase", "none"),
                    }
                    for key, value in fields.items():
                        dimension = f"{key}={value}"
                        dimensions[dimension] = dimensions.get(dimension, 0) + 1
            payload = {
                "protocol_version": 2,
                "runs": len(v2),
                "condition_status_stage_reason_phase_counts": dimensions,
                "failure_stage_counts": {
                    key.split("=", 1)[1]: value
                    for key, value in dimensions.items()
                    if key.startswith("stage=") and not key.endswith("=none")
                },
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

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
    raw_protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol = (
        NoviceComparisonProtocol.model_validate(raw_protocol)
        if raw_protocol.get("schema_version") == 2
        else ResearchProtocol.model_validate(raw_protocol)
    )
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
        if (
            isinstance(protocol, NoviceComparisonProtocol)
            and "direct_code" in protocol.conditions
        ):
            from automind.experiments.codegen import (
                PodmanSandboxExecutor,
                default_sandbox_profile,
            )

            if (
                protocol.sandbox_backend != "podman"
                or protocol.sandbox_profile != "podman-automind-py310-v1"
            ):
                payload["direct_code_ready"] = False
                payload["sandbox_readiness"] = (
                    "a supported sandbox backend and profile are required"
                )
            else:
                try:
                    profile = default_sandbox_profile()
                    PodmanSandboxExecutor(profile).preflight(profile.policy)
                except RuntimeError as error:
                    payload["direct_code_ready"] = False
                    payload["sandbox_readiness"] = str(error)
                else:
                    payload["direct_code_ready"] = True
                    payload["sandbox_readiness"] = "ready"
    else:
        if args.dataset_root is None:
            raise ValueError("run/resume requires --dataset-root")
        try:
            if isinstance(protocol, NoviceComparisonProtocol):
                from automind.experiments.orchestration import (
                    NoviceComparisonStudy,
                    OccupiedRunRootError,
                )

                try:
                    payload = NoviceComparisonStudy(
                        protocol, args.dataset_root, progress=progress
                    ).run(resume=args.command == "resume")
                except OccupiedRunRootError as error:
                    import sys

                    print(f"Experiment not started: {error}", file=sys.stderr)
                    return 2
            else:
                from automind.experiments.synthea_pilot import (
                    SyntheaPilotRunner,
                )

                payload = SyntheaPilotRunner(
                    protocol, args.dataset_root, progress=progress
                ).run(resume=args.command == "resume")
        except KeyboardInterrupt:
            import sys

            print(
                "Experiment interrupted; "
                f"current={progress.summary()}; partial artifacts were "
                "preserved and can be resumed.",
                file=sys.stderr,
            )
            return 130
    print(json.dumps(payload, indent=2, sort_keys=True))
    if (
        args.command in {"run", "resume"}
        and isinstance(protocol, NoviceComparisonProtocol)
        and any(
            condition.get("status") == "failed"
            for run in payload["runs"]
            for condition in run["conditions"].values()
        )
    ):
        return 1
    return 0
