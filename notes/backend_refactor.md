# Sparse Backend Refactor Plan

## Goal

Replace the mixed SciPy and Python-GraphBLAS branches in `D4M/_sparse.py` with
one implementation class per backend. The public functional API must remain
available as `D4M._sparse` so existing callers, especially `D4M/assoc.py`, do
not need to know which backend is active.

The target layout is:

```
D4M/
  _sparse/
    __init__.py
    scipy_backend.py
    graphblas_backend.py
```

`D4M/_sparse.py` will be removed as part of the refactor. Python resolves
`import D4M._sparse` to the new package, so the existing import path remains
stable.

## Current API And Constraints

`D4M._sparse` currently provides a single functional abstraction over
`scipy.sparse` and `python-graphblas`. Its 30 public operations are used
throughout `D4M/assoc.py`; callers also rely on these module attributes:

- `selected_splinalg_library`, populated from `D4M_SPARSE_BACKEND` at import
  time.
- `supported_splinalg_libraries`.
- `Sparse_Matrix`, used in type annotations and runtime `isinstance` checks.

The refactor must preserve all function names, signatures, default values, and
documented result-format behavior. In particular, SciPy operations currently
canonicalize or return COO matrices in specific places, while GraphBLAS has no
CSR/CSC/COO storage-format distinction. The backend layer must preserve these
semantic differences behind the existing API instead of forcing either library
to imitate implementation details it does not support.

The current GraphBLAS backend is incomplete and does not pass the full D4M
suite. This refactor should faithfully move its present behavior first; it
should not silently change parity scope. Correctness fixes discovered while
extracting code should be covered by focused tests and documented separately
from mechanical relocation.

## Design

### Backend Classes

Create `ScipyBackend` in `scipy_backend.py` and `GraphBLASBackend` in
`graphblas_backend.py`. Each class implements the complete current functional
interface as instance methods:

- Construction and conversion: `init`, `from_coo`, `from_dense`, `empty`,
  `astype`, `to_coo`, `is_coo_format`, `to_csr`, `to_csc`, `to_array`,
  `to_scipy`, `sum_duplicates`, `eliminate_zeros`, and `copy`.
- Algebra and comparison: `add`, `matmul`, `ewise_mult`, `nnz`, `sparse_sum`,
  `logical`, `transpose`, `coo_canonicalize`, and `sparse_equal`.
- Inspection and selection: `get_data`, `get_dtype`, `get_shape`,
  `get_coo_params`, `get_csr_params`, `get_csc_params`, `getitem_rows`,
  `getitem_cols`, `getvalue`, and `update_data`.

Each class exposes a `sparse_matrix_type` class or instance attribute. It is
`gb.Matrix` for GraphBLAS and the existing SciPy sparse-array union/type tuple
for SciPy. `__init__.py` uses this attribute to publish the backwards-compatible
`Sparse_Matrix` alias.

Do not add a broad conversion layer between the libraries. Backend methods
receive and return their backend's native matrix type. `to_scipy` remains the
explicit cross-backend conversion operation already present in the facade.

### Optional Shared Contract

Add a small internal `Protocol` or abstract base class only if it can express
the full interface without weakening useful type information. It belongs in
`D4M/_sparse/__init__.py` or a new internal module only if needed. The initial
implementation should favor two explicit, similarly ordered method lists over
metaprogramming or dynamic method generation.

The contract should define behavioral requirements, not merely names:

- `from_coo` returns canonical COO data for SciPy and a native GraphBLAS matrix
  for GraphBLAS.
- `convert_to_coo` is honored by SciPy algebra methods and intentionally has no
  storage-format effect for GraphBLAS.
- `copy=False` preserves the existing non-copy behavior wherever the backend
  permits it.
- Mutating methods (`sum_duplicates`, `eliminate_zeros`) return `None`.
- Getter tuple ordering remains exactly as today. `get_coo_params` returns
  `(row, col, data)`, while CSR/CSC getters return `(data, indices, indptr)`.

### Facade And Backend Selection

`D4M/_sparse/__init__.py` becomes the only module that reads
`D4M_SPARSE_BACKEND`. It will:

1. Define `supported_splinalg_libraries` with the existing backend identifiers.
2. Read and validate `D4M_SPARSE_BACKEND`, preserving the current default of
   `scipy.sparse` and current `ValueError` behavior for unsupported values.
3. Instantiate exactly one selected backend at module import time.
4. Publish `selected_splinalg_library` and `Sparse_Matrix` for compatibility.
5. Define the existing module-level functions as thin delegations to that
   backend instance, with the same names and signatures.

Keep the selected backend object private, for example `_backend`. Do not expose
backend switching at runtime in this change: existing behavior selects once at
import time, and runtime switching would invalidate `Sparse_Matrix` checks and
permit mixed native matrix types in one process.

The facade wrappers should be explicit rather than generated with `getattr` or
`__getattr__`. Explicit wrappers retain signatures, docstrings, IDE discovery,
and a stable public API. Move the existing operation docstrings to the facade
where they describe the abstraction; backend-only format/performance notes
belong with the relevant backend method.

### Imports And Dependency Boundaries

- `scipy_backend.py` imports NumPy and `scipy.sparse`, but not `graphblas`.
- `graphblas_backend.py` imports NumPy and `graphblas`, but not
  `scipy.sparse`, except if GraphBLAS's existing `to_scipy` integration requires
  it for a return annotation.
- `__init__.py` imports the selected implementation lazily after environment
  validation. This avoids importing the unused native backend merely because
  the package is imported.
- Keep public type annotations compatible with the project Python versions.
  Avoid importing optional backend-specific symbols into package scope unless
  that backend is selected.

The project currently imports `graphblas` unconditionally before the backend
selection logic. During implementation, verify whether GraphBLAS remains a
required dependency for normal SciPy installs. If it should be optional, make
the GraphBLAS import lazy and raise a clear error only when
`D4M_SPARSE_BACKEND=python-graphblas` is selected. Coordinate any dependency
metadata change with the existing `setup.py` optional-dependency design; do not
bundle it into the mechanical extraction without a tested install path.

## Implementation Steps

1. Inventory the current facade and callers.
   - Record every public function and its signature from `_sparse.py`.
   - Search for direct uses of `selected_splinalg_library` and `Sparse_Matrix`.
   - Capture current SciPy behavior with the existing core suite before moving
     code.

2. Establish backend-focused tests before extraction.
   - Expand `test/test_sparse.py` so every facade operation has at least one
     assertion for output values, shape, dtype where relevant, and return
     format/copy semantics where observable.
   - Parameterize test expectations by selected backend without importing the
     other backend's matrix type as the expected result type.
   - Run each backend in a separate pytest process because selection occurs at
     module import time.
   - Add focused regression cases for GraphBLAS boolean row/column selection
     and `get_data`, which were recently adapted to current GraphBLAS APIs.

3. Add `ScipyBackend`.
   - Move each SciPy branch into a method without changing algorithm or result
     format.
   - Keep canonicalization, CSR/CSC conversion choices, zero elimination, and
     tolerance logic exactly as they work today.
   - Make the SciPy matrix runtime type suitable for the existing
     `isinstance(..., _sparse.Sparse_Matrix)` checks in `assoc.py`.

4. Add `GraphBLASBackend`.
   - Move each GraphBLAS branch into a corresponding method, retaining native
     methods such as `dup`, `to_coo`, `to_csr`, `to_csc`, and expression
     materialization with `.new()` for indexing.
   - Preserve no-op behavior for GraphBLAS operations that have no native
     equivalent (`sum_duplicates`, `eliminate_zeros`, and storage-format
     conversion).
   - Correct extraction mistakes only with tests. For example, verify argument
     ordering in `update_data` against `get_coo_params` and construct the
     replacement matrix with `new_data`, not the old values, if that is the
     intended behavior.

5. Replace the module with the package facade.
   - Add `D4M/_sparse/__init__.py` and explicit delegating wrappers.
   - Remove `D4M/_sparse.py` in the same commit; a module and package with the
     same import name must not coexist.
   - Confirm `import D4M._sparse as _sparse` works unchanged in `assoc.py`,
     tests, and an installed wheel.

6. Verify behavior and packaging.
   - Run `python -m pytest test/test_util.py test/test_assoc.py
     test/test_sparse.py` with the default SciPy backend.
   - Run the same command in the existing container with
     `D4M_SPARSE_BACKEND=python-graphblas`; record existing parity failures
     separately from refactor regressions.
   - Run the containerized default suite through `scripts/test-container.sh`
     and verify the GitHub Actions Docker invocation still uses the image's
     default command.
   - Build/install the package from a clean tree and import `D4M.assoc` to
     confirm `setuptools.find_packages()` includes `D4M._sparse`.

## Stage 1 Inventory And Baseline

Inventory captured before extraction:

- The facade has 33 public functions (the earlier count of 30 was incorrect).
- Compatibility attributes are `selected_splinalg_library`,
  `supported_splinalg_libraries`, and `Sparse_Matrix`.
- `D4M/assoc.py` uses `Sparse_Matrix` in three annotations and three runtime
  `isinstance` checks. Tests inspect `selected_splinalg_library` in
  `test/test_sparse.py` and `test/test_assoc.py`. No caller outside
  `_sparse.py` currently reads `supported_splinalg_libraries`.
- `assoc.py` accesses all facade operations except `from_dense` directly;
  `from_dense` remains part of the public facade and must be preserved.

The public callable argument signatures are:

```text
init(sparse_like, *args, **kwargs)
from_coo(row, col, val, shape=None, dtype=None)
from_dense(np_mat, missing_value=0, dtype=None)
empty()
astype(sparse_mat, dtype)
to_coo(sparse_mat, copy=True)
is_coo_format(sparse_mat)
to_csr(sparse_mat, copy=True)
to_csc(sparse_mat, copy=True)
to_array(sparse_mat)
to_scipy(sparse_mat)
sum_duplicates(sparse_mat)
eliminate_zeros(sparse_mat)
copy(sparse_mat)
add(sparse_mat_A, sparse_mat_B, convert_to_coo=True)
matmul(sparse_mat_A, sparse_mat_B, convert_to_coo=True)
ewise_mult(sparse_mat_A, sparse_mat_B, convert_to_coo=True)
nnz(sparse_mat)
sparse_sum(sparse_mat, dimension=None)
logical(sparse_mat, copy=False)
transpose(sparse_mat, copy=False)
coo_canonicalize(sparse_mat, copy=False)
sparse_equal(sparse_mat_A, sparse_mat_B, rtol=1e-05, atol=1e-08)
get_data(sparse_mat)
get_dtype(sparse_mat)
get_shape(sparse_mat)
get_coo_params(sparse_mat)
get_csr_params(sparse_mat)
get_csc_params(sparse_mat)
getitem_rows(sparse_mat, sub_row)
getitem_cols(sparse_mat, sub_col)
getvalue(sparse_mat, row_index, col_index)
update_data(sparse_mat, new_data)
```

Default SciPy baseline, run in the repository test container with
`D4M_SPARSE_BACKEND=scipy.sparse`:

```text
python -m pytest test/test_util.py test/test_assoc.py test/test_sparse.py
534 passed, 49 warnings
```

## Stage 2 Backend Test Coverage

`test/test_sparse.py` now exercises every public facade operation without
importing either backend library directly. Shared behavior is asserted through
the facade, while SciPy storage formats and observable copy behavior are
checked only where applicable. Focused GraphBLAS regression coverage includes
boolean row/column selection and canonical `get_data` ordering.

The unchanged GraphBLAS core-suite baseline before expanding the tests was:

```text
446 passed, 88 failed, 50 warnings
```

Three of those failures came from the previous `test_from_coo` test indexing a
`None` shape, rather than from production behavior. The expanded suite marks
five known GraphBLAS facade incompatibilities as expected failures: `init`, the
three `sparse_sum` cases, and `update_data`. These document the current parity
boundary without bundling correctness fixes into test establishment.

Stage 2 verification results are:

```text
D4M_SPARSE_BACKEND=scipy.sparse
567 passed, 49 warnings

D4M_SPARSE_BACKEND=python-graphblas (test/test_sparse.py)
32 passed, 5 xfailed

D4M_SPARSE_BACKEND=python-graphblas (core suite)
477 passed, 85 failed, 5 xfailed, 50 warnings
```

## Definition Of Done

- `D4M/_sparse.py` is replaced by the `_sparse` package containing
  `__init__.py`, `scipy_backend.py`, and `graphblas_backend.py`.
- Backend-specific library calls exist only in their respective backend module,
  aside from selection and type aliases in the facade.
- Every former facade function remains importable with the same signature and
  public semantics.
- `D4M/assoc.py` requires no backend conditionals and continues using the
  facade unchanged.
- The default SciPy container suite remains passing.
- GraphBLAS test results are no worse than the baseline recorded before the
  refactor, and any behavior changes are covered by tests and documented.
- The selected backend, valid environment values, and missing-dependency errors
  are deterministic and clearly reported.
