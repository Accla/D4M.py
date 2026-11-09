# Podman Test Runner Plan

## Goal

Provide a reproducible, Podman-compatible test environment for D4M.py. The
environment must install SuiteSparse:GraphBLAS so the `python-graphblas`
backend can run alongside the existing SciPy sparse backend.

Database integration tests are out of scope: they require a separately
configured Accumulo/Graphulo instance and remain opt-in.

## Current State

- The project is packaged with `setup.py` and has no container configuration.
- Core tests are in `test/test_assoc.py`, `test/test_util.py`, and
  `test/test_sparse.py`.
- `_sparse.py` imports the `graphblas` Python package directly, and its
  selected backend is currently `scipy.sparse`.
- `pytest` is required by the tests but is not declared in `setup.py`.
- The `pygraphblas` optional dependency in `setup.py` is distinct from the
  `graphblas` package imported by the current sparse abstraction.

## Proposed Files

1. `Containerfile.test`
   - Use a Linux base image with stable Python development packages, such as
     `python:3.11-slim`.
   - Set noninteractive package installation and unbuffered Python output.
   - Install build tooling and SuiteSparse:GraphBLAS development libraries.
   - Install the checkout and test dependencies.
   - Default to `pytest test` as the container command.

2. `requirements-test.txt`
   - Pin or bound test-only dependencies: `pytest`, `numpy`, `scipy`, and
     `python-graphblas`.
   - Keep runtime requirements separate unless the project is migrated to
     `pyproject.toml` in a later change.

3. `scripts/test-container.sh`
   - Use `podman` by default, with an optional `CONTAINER_RUNTIME` override
     for Docker-compatible environments.
   - Build a local test image from `Containerfile.test`.
   - Bind-mount the repository at `/workspace` so the container tests the
     current checkout without copying build artifacts into the image.
   - Run as the invoking user's UID and GID when supported, avoiding
     root-owned files on the host.
   - Forward additional arguments to `pytest`, for example
     `scripts/test-container.sh test/test_sparse.py -q`.

4. Optional `.containerignore`
   - Exclude virtual environments, Git metadata, caches, notebooks'
     checkpoints, and local test artifacts from the build context.

## SuiteSparse:GraphBLAS Installation

1. Download SuiteSparse:GraphBLAS from the upstream GitHub releases at
   <https://github.com/DrTimothyAldenDavis/GraphBLAS>. Pin the initial runner
   to release `v10.5.2` (the current latest release), and record the archive
   checksum in the Containerfile or an adjacent version file.
2. Build the pinned release in a dedicated builder stage using CMake and a C
   compiler. Install the library and headers to a defined prefix such as
   `/opt/suitesparse-graphblas`.
3. Copy only the required shared library, headers, and license information
   into the final test image. Set the dynamic linker path so `libgraphblas` is
   discoverable at runtime without relying on distribution-provided GraphBLAS
   packages.
4. Install `python-graphblas` after the system library is present. Verify the
   package discovers SuiteSparse:GraphBLAS by importing `graphblas` and
   creating a small matrix before running D4M tests.

### Compatibility Note

The current `python-graphblas` release requires the `suitesparse-graphblas`
Python binding package with a `<10` native-library constraint. Its installed
wheel bundles SuiteSparse:GraphBLAS 9.4.5, so it cannot currently exercise the
container's upstream v10.5.2 library. The image still builds and includes
v10.5.2 as required; enabling a v10-backed Python GraphBLAS test run requires
an upstream-compatible Python binding release and further D4M adapter work.

## Test Matrix

Run each of the following in a clean container:

1. SciPy backend
   - Leave `D4M._sparse.selected_splinalg_library` set to `scipy.sparse`.
   - Run `pytest test/test_util.py test/test_assoc.py test/test_sparse.py`.

2. GraphBLAS backend
   - Select `python-graphblas` without leaving a source-tree modification.
     Implement this through a supported environment variable or a pytest
     fixture/parameter in a follow-up change; do not edit `_sparse.py` with
     shell text replacement.
   - Run the same core suite.
   - Mark or skip backend operations that are not yet implemented with a clear
     reason, then track each skip as a GraphBLAS parity item.

3. Database tests
   - Keep `test/_test_db.py` excluded from the default runner.
   - Add a separately documented integration profile only after an Accumulo
     and Graphulo container environment is defined.

## Implementation Order

1. Verify supported Python, SciPy, and `python-graphblas` versions against
   the current source and targeted SuiteSparse:GraphBLAS release.
2. Add `requirements-test.txt` and a multi-stage `Containerfile.test`.
3. Build the image with `podman build -f Containerfile.test -t d4m-test .`.
4. Add the runner script and execute the SciPy suite through Podman.
5. Add a controlled backend-selection mechanism and execute the GraphBLAS
   suite through Podman.
6. Resolve compatibility failures, extend sparse-layer tests where behavior is
   uncovered, and document the canonical local command in `README.md`.
7. Add CI using a container-capable Linux runner after the local runner is
   stable.

## Acceptance Criteria

- `podman build -f Containerfile.test -t d4m-test .` completes from a clean
  checkout.
- `podman run --rm d4m-test` runs the non-database suite successfully.
- The runner script succeeds on rootless Podman and leaves no root-owned files
  in the checkout.
- The container imports `graphblas` and exercises SuiteSparse:GraphBLAS.
- Both SciPy and GraphBLAS backend test runs are explicit, reproducible, and
  reported separately.
- The default test path does not require database credentials, Graphulo JARs,
  or a running Accumulo service.

## Verification Results

On the ARM64 Podman test machine, `podman build -f Containerfile.test -t
d4m-test .` completed successfully. GraphBLAS v10.5.2 required `-O2` during
the build because GCC 12 produced an internal compiler error while compiling
one generated factory kernel at its default Release optimization level.

The default container command completed successfully: `534 passed` in 1.03s.
This run uses the default SciPy backend. The GraphBLAS backend is selectable
with `D4M_SPARSE_BACKEND=python-graphblas`, but is not yet a passing test
configuration because `_sparse.py` and `assoc.py` still contain SciPy-specific
operations. Its parity work remains a follow-up.
