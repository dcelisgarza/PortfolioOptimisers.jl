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
  - Example: `Full`, `Semi`, `SpectralDenoise`, `Newton`.

- **Results** (`<: AbstractResult`):
  - Returned by functions that consume estimators when the output is complex enough to warrant its own type (e.g., contains multiple arrays, metadata).
  - Can themselves be passed as inputs to further computations — functions must dispatch on both estimator and result types where this makes sense.
  - Example: `LowOrderPrior`, `ClustersResult`, `OptimisationResult`.

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

This avoids duplicating method definitions. Always document the alias with a docstring explaining which types it groups and why.

## Docstring Documentation Dictionaries

Four dictionaries in `01_Base.jl` provide standardised descriptions for arguments, fields, returns, and validation. **Always interpolate from these dictionaries** in docstrings instead of writing ad-hoc descriptions. This ensures consistency across the entire library.

- `arg_dict` — argument descriptions (e.g., `$(arg_dict[:ce])`).
- `field_dict` — field descriptions, derived from `arg_dict` by stripping the parameter name prefix (e.g., `"$(field_dict[:ce])"`).
- `val_dict` — validation rules (e.g., `$(val_dict[:oow])`).
- `ret_dict` — return value descriptions (e.g., `$(ret_dict[:sigma])`).
- `math_dict` — LaTeX mathematical notation descriptions (e.g., `$(math_dict[:tgt])`).

If a parameter, field, or return value does not yet have an entry, add it to the appropriate dictionary in `01_Base.jl` before writing the docstring.

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
- Implement `moment_view(estimator, i)` / `prior_view(estimator, i)` to support windowed or cross-validated slicing.

## Error Handling

- Use custom exception types (subtype `PortfolioOptimisersError`):

  - `IsEmptyError` for empty collections.
  - `IsNothingError` for unexpected `nothing` values.
  - `IsNonFiniteError` for non-finite numbers.
  - `DimensionMismatch` for size mismatches.
  - `DomainError` for out-of-range values.

## Related Types References

- Always include a `# Related` section in docstrings.
- Link to abstract parent types, related estimators, algorithms, and result types.
- Use the `[@ref]` syntax: ``[`TypeName`](@ref)``.

## Examples in Docstrings

- Use `jldoctest` blocks for testable examples when feasible.
- Provide simple, clear examples that demonstrate basic usage.
- For complex features, provide multiple examples showing different use cases.
