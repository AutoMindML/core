# Podman direct-code sandbox

This procedure prepares the OCI sandbox used by the version-2 `direct_code`
research condition. It was verified on Windows 11 with Podman client 5.8.3,
Podman server 5.8.7, a rootless WSL2 machine, cgroup v2, `crun`, and seccomp.
Other hosts must pass the same integration tests before running an experiment.

## Install Podman on Windows

Install the official Red Hat package through WinGet:

```powershell
winget install --id RedHat.Podman --exact --source winget `
  --accept-package-agreements --accept-source-agreements
```

Open a new PowerShell session, then initialize and start a rootless machine:

```powershell
podman machine init --cpus 2 --memory 4096 --disk-size 30
podman machine start
podman info
```

`podman info` must report cgroup v2 with the CPU, memory, and PID controllers,
and seccomp must be enabled. Do not continue if those controls are unavailable.

## Build and verify the research image

Run these commands from `core/automind`. The timestamp is fixed so the local
OCI image has the protocol-pinned digest:

```powershell
podman build --pull=never --timestamp 0 `
  -t localhost/automind-sandbox:py310-v1 `
  -f sandbox/Containerfile .

podman image inspect localhost/automind-sandbox:py310-v1 `
  --format '{{.Digest}}'
```

The expected digest is:

```text
sha256:686146376d8afa0abc8eec0f44245c0e6eab6c63f456c146b8863bc515ba73df
```

The executor fails preflight if this digest or the configured Podman major
version differs. Do not edit the profile to accept an unreviewed image; rebuild,
review the dependency change, run the isolation suite, and freeze a new profile.

## Prove the isolation boundary

From `core/`, run:

```powershell
just experiment-sandbox-check
```

Or, from `core/automind`, run the underlying command directly:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  src/automind/tests/test_podman_sandbox.py -v
```

The tests exercise a real container and verify the positive pandas fit/transform
path, disabled network access, absent host secrets and workspace mounts,
read-only root and transform state, fit-state persistence, timeout cleanup, and
explicit output-overflow reporting. Unit tests separately verify row identity,
target preservation, one-row holdout transformation, duplicate-column rejection,
and profile validation.

The sandbox is not ready merely because `podman info` or a hello-world container
succeeds. Do not run the pilot or confirmatory matrix unless this integration
suite passes on the experiment host with the pinned image.

## Stop and restart

The machine may be stopped when experiments are not running:

```powershell
podman machine stop
```

Restart it with `podman machine start` and rerun the isolation suite before the
next frozen experiment batch.
