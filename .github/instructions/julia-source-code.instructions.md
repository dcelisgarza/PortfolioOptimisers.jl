---
applyTo: "src/**/*.jl"
---

# Julia Source Code Guidelines for PortfolioOptimisers.jl

## Estimator, Algorithm, and Result Roles

These three abstract hierarchies form the backbone of the library. Understanding their distinct roles is critical for correct design.

- **Estimators** (`<: AbstractEstimator`):
  - User-facing. Compose algorithms and/or other estimators as fields.
  - Are the entry points for computation — all high-level API functions accept estimators.
  - May consume data (e.g., a returns matrix) and produce results or transformed data.
  - Example: `Covariance`, `EmpiricalPrior`, `Denoise`.

- **Algorithms** (`<: AbstractAlgorithm`):
  - Internal dispatch mechanism. Never called directly from user-facing APIs.
  - Modify or specialise the behaviour of an estimator they are stored in.
  - Must not contain data — only parameters that tune an algorithm's behaviour.
  - Example: `FullMoment`, `SemiMoment`, `SpectralDenoise`, `Newton`.

- **Results** (`<: AbstractResult`):
  - Returned by functions that consume estimators when the output is complex enough to warrant its own type (e.g., contains multiple arrays, metadata).
  - Can themselves be passed as inputs to further computations — functions must dispatch on both estimator and result types where this makes sense.
  - Example: `LowOrderPrior`, `Clusters`, `OptimisationResult`.

## Type Definitions

- **Abstract types**:

  - Always prefix with `Abstract` (e.g., `AbstractCovarianceEstimator`).
  - Include comprehensive docstrings explaining their role in the type hierarchy.
  - List related types in the `# Related` section.
  - When subtypes must implement specific methods, document this in an `# Interfaces` section (see the docstring guide).

- **Struct types**:

  - Use `@concrete` from `ConcreteStructs.jl` — it auto-generates type parameters so `struct MyType{T1, T2}` boilerplate is not needed.
  - Use `DocStringExtensions.TYPEDEF` in the docstring header for struct types.
  - All fields must be documented using inline `"$(field_dict[:key])"` strings and reflected in the `# Fields` section via `$(DocStringExtensions.FIELDS)`.
  - Call `@define_pretty_show(TypeName)` immediately after any new struct that should display nicely in the REPL (all estimators, algorithms, and results).
  - Every source file must end with an `export` statement listing all public symbols defined in that file.

## Capability Catalogue (required for every new public-facing addition)

Every concrete type on the Choice Surface, and every exported function, is listed in the **Capability Catalogue** — the user-facing inventory of what the package can do (see ADR 0040).

**When you add a new type or exported function, you must also place it in `docs/capability_catalogue.jl`.** This is not optional bookkeeping: it is enforced by `test/test_26_docs.jl`, which fails if any name on the Choice Surface is absent, and CI runs that test on every PR touching `src/`. A concrete type the package declares is on the surface when it is a leaf subtype of `AbstractEstimator`, of `AbstractAlgorithm` or of `AbstractCovarianceEstimator`, or when it is an export under its own name; a Result and an error are subtracted. `choice_surface_names` in `docs/generate_capability_catalogue.jl` states that rule once.

- **New type on the choice surface** — add a `Cap(:YourType)` to the group it belongs to. Pick the group by the *job it does*, not the file it lives in.
- **New type the library constructs for itself** — a marker or other internal type no caller ever writes is not a choice, so list it in `NOT_A_CHOICE` with the reason `:internal` instead of cataloguing it. It keeps its docstring and its API page.
- **New exported function** — either add a `Cap`, mention it in a section's `Prose` (a prose `@ref` counts as catalogued), or, if it is genuinely not a user-facing capability, list it in `NOT_A_FEATURE` with a reason: `:alias`, `:base_overload`, `:trait`, or `:internal`.
- **Do not write a description.** Each entry's one-line description is taken from the first sentence of its docstring at build time, so there is exactly one description of every type in the repo. Pass `label` only where the docstring genuinely reads worse in a bullet (for example when a group's children would all repeat the same prefix).
- **Removing an export or a type?** Also remove its `NOT_A_FEATURE` or `NOT_A_CHOICE` entry — both checks run in both directions and a stale exemption fails too.

Because descriptions come from docstrings, the docstring summary convention below is load-bearing for this page, not just for the API reference.

## Constructor Pattern

The library uses a strict **inner/outer constructor split** for all structs:

- **Inner constructor** (positional arguments, type assertions, validation):
  - Accepts concrete positional arguments (no keyword args).
  - Performs all validation using `@argcheck` or shared `assert_*` helpers.
  - Calls `new{typeof(arg1), typeof(arg2), ...}(arg1, arg2, ...)` explicitly.

- **Outer constructor** (keyword arguments, default values):
  - Accepts keyword arguments with default values.
  - Performs any additional validation that requires comparing multiple fields (and cannot be done in the inner constructor because the struct does not yet exist).
  - Delegates to the inner constructor.

```julia
@concrete struct MyEstimator <: AbstractEstimator
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:oow])"
    w
    function MyEstimator(ce::StatsBase.CovarianceEstimator, w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)      # shared validation helper
        return new{typeof(ce), typeof(w)}(ce, w)
    end
end
function MyEstimator(;
                     ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(),
                     w::Option{<:ObsWeights} = nothing)
    return MyEstimator(ce, w)
end
```

- **Prefer shared validation helpers** over inline `@argcheck` where they exist:
  - `assert_nonempty_nonneg_finite_val(x, sym)` — non-empty, non-negative, finite.
  - `assert_nonempty_gt0_finite_val(x, sym)` — non-empty, positive, finite.
  - `assert_nonempty_finite_val(x, sym)` — non-empty, finite.
  - `assert_matrix_issquare(A, sym)` — square matrix.

## Type Aliases and `Option{T}`

- **`Option{T}`**: Alias for `Union{Nothing, T}`. Use it for any field or argument that can be absent.

  ```julia
  function foo(x::Option{<:VecNum} = nothing) ...
  ```

- **Common type aliases** defined in `01_Base.jl` — always prefer them over writing out the full union/abstract type:

  | Alias | Meaning |
  | --- | --- |
  | `VecNum` | `AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}` |
  | `MatNum` | `AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}` |
  | `ArrNum` | `AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}` |
  | `VecInt` | `AbstractVector{<:Integer}` |
  | `Num_VecNum` | `Union{<:Number, <:VecNum}` |
  | `VecNum_MatNum` | `Union{<:VecNum, <:MatNum}` |
  | `ObsWeights` | `StatsBase.AbstractWeights` |
  | `Option{T}` | `Union{Nothing, T}` |

## Union Type Aliases and Dispatch Groups

When multiple abstract subtypes share a common interface, define a `const` union alias and dispatch on it:

```julia
const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                  <:AbstractLowOrderPriorEstimator_AF}
```

This avoids duplicating method definitions. A union alias is a **dispatch alias**, and its docstring states what the alias groups and why the group exists. Which sections that docstring carries is stated by [`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) § *Section Structure for Aliases*, which is the Authority for it.

## Docstrings

[`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) is the Authority for every docstring rule. It states which sections each kind of unit carries, what each section holds, the dictionaries in `01_Base.jl` that a description interpolates from, the mathematical notation, and the `jldoctest` blocks. Read it before you write a docstring, and change a docstring rule there and nowhere else.

An **alias** is the case to check first. Its docstring carries a different set of sections from the unit it names, the set differs by kind of alias, and `test/test_26_docs.jl` reds the file over a section outside that set. See § *Section Structure for Aliases*.

## Immutability and `Accessors.jl`

- The library **never uses mutable structs**. All types are immutable.
- When a field of an existing struct instance must be changed, use `Accessors.jl`:

  ```julia
  using Accessors: @set
  new_obj = @set obj.field = new_value
  ```

- Functions must be as pure as possible — avoid side effects and global state.

## Input Validation

- **Always use `@argcheck` for validation**:

    ```julia
    @argcheck !isempty(v) IsEmptyError("v cannot be empty")
    @argcheck all(isfinite, v) DomainError("v must be finite")
    @argcheck size(A, 1) == size(B, 1) DimensionMismatch("A and B must have same rows")
    ```

- **Common validation patterns**:

  - Empty checks: `@argcheck !isempty(x) IsEmptyError(...)`
  - Nothing checks: `@argcheck !isnothing(x) IsNothingError(...)`
  - Finite checks: `@argcheck all(isfinite, x) DomainError(...)`
  - Dimension checks: `@argcheck size(A) == size(B) DimensionMismatch(...)`

- **Prefer shared helpers over inline `@argcheck`** — see constructor section above.

## Multiple Dispatch

- **Prefer multiple dispatch over conditionals** for algorithm/method selection.
- Define specific methods for different type combinations rather than using if/else on types.
- Use abstract types and union aliases in method signatures for flexibility.
- Functions that accept either an estimator or a result must dispatch on both:

  ```julia
  function prior(pe::AbstractPriorEstimator, X::MatNum, ...) ... end
  function prior(pr::AbstractPriorResult, args...; kwargs...) = pr  # passthrough
  ```

## Return Type Annotations

- See `.github/instructions/julia-return-types.instructions.md` for full guidelines.
- Key rule: annotate functions whose return type is always the same concrete type (e.g., `-> Nothing` for assertion helpers, `-> ConcreteEstimatorType` for factory methods).

## Code Organization

- **File naming**: Source files are prefixed numerically to indicate load order (e.g., `01_Base.jl`).
- **Module structure**: Each submodule focuses on a specific domain (moments, risk, priors, etc.).
- **Type hierarchy**: Subtype the appropriate abstract type (`AbstractEstimator`, `AbstractAlgorithm`, `AbstractResult`).
- **Exports**: Every source file ends with an `export` line listing all public symbols it defines. Do not export internal helpers.

## Composability

- Design estimators and algorithms to be composable.
- Accept other estimators/algorithms as parameters when appropriate.
- Return result types that encapsulate outcomes for easy chaining.
- Implement `factory(estimator, w::ObsWeights)` to propagate observation weights through composed estimators.
- Implement `port_opt_view(estimator, i)` and `obs_weights_view(estimator, i)` to support windowed or cross-validated slicing.

## Error Handling

- Use custom exception types (subtype `PortfolioOptimisersError`):

  - `IsEmptyError` for empty collections.
  - `IsNothingError` for unexpected `nothing` values.
  - `IsNonFiniteError` for non-finite numbers.
  - `DimensionMismatch` for size mismatches.
  - `DomainError` for out-of-range values.
