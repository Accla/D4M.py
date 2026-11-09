#!/bin/sh
set -eu

container_runtime=${CONTAINER_RUNTIME:-podman}
image=${D4M_TEST_IMAGE:-d4m-test}
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

"$container_runtime" build \
    --file "$root/Containerfile.test" \
    --tag "$image" \
    "$root"

exec "$container_runtime" run --rm \
    --userns=keep-id \
    --volume "$root:/workspace" \
    --workdir /workspace \
    --env "D4M_SPARSE_BACKEND=${D4M_SPARSE_BACKEND:-scipy.sparse}" \
    "$image" \
    python -m pytest "$@"
