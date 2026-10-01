# Podman direct-code sandbox

The version 2 `direct_code` condition executes generated Python through a disposable
Podman container. The sandbox policy disables network access, withholds host secrets and
workspace mounts, limits resources, and separates read-only fit state from writable
outputs. The experiment runner performs a preflight check against the pinned image and
runtime profile before execution.

## Prerequisites

Install a supported Podman release for your host and create a machine with enough CPU,
memory, disk, cgroup v2, and seccomp support for the configured profile. Verify the host
with:

```powershell
podman info
```

The `podman-automind-py310-v1` sandbox profile requires the `crun` runtime and pins
the image `localhost/automind-sandbox` by digest. The build tag below is
`localhost/automind-sandbox:py310-v1`; do not edit a protocol to accept an unreviewed
image.

## Build the pinned image

From `automind/`, build the repository's sandbox image with the reproducible timestamp
used by the profile:

```powershell
podman build --pull=never --timestamp 0 `
  -t localhost/automind-sandbox:py310-v1 `
  -f sandbox/Containerfile .

podman image inspect localhost/automind-sandbox:py310-v1 `
  --format '{{.Digest}}'
```

Compare the resulting digest with `default_sandbox_profile()` in
`src/automind/experiments/codegen.py`. A mismatch requires reviewing the image inputs
and freezing a new profile before research execution.

## Check isolation

From the repository root, run the real-container suite only when Podman is available:

```powershell
just experiment-sandbox-check
```

The suite checks the positive pandas fit/transform path, network isolation, absence of
host secrets and workspace mounts, read-only state, cleanup, timeouts, output limits,
target preservation, row identity, and schema rules. It makes container and temporary
output changes on the local experiment host.

## Run a direct-code protocol

The retained preset is
`../src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json`.
Validate the protocol and inspect readiness before any live request:

```powershell
uv run --locked --group dev --no-sync automind-experiment dry-run `
  src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json `
  --dataset-root src/automind/data/csv/synthea_covid19_10k
```

A live run calls the configured LLM and executes generated code in the sandbox, so it
can consume model resources and write under the protocol output root. Use the root
`just experiment-run-live` and `just experiment-resume-live` recipes or the package CLI
after reviewing the dry-run output.

## Interruption and cleanup

Press Ctrl-C once to request interruption. The CLI records an interruption marker,
cleans up the active container, and leaves completed observations for a matching resume.
Resume restarts an interrupted observation from its beginning; individual LLM requests
and holdout rows are not checkpointed.

When no experiment is running, the Podman machine may be stopped and restarted with the
host's normal Podman commands. Rerun the isolation suite after a runtime or image change
and before the next live batch.
