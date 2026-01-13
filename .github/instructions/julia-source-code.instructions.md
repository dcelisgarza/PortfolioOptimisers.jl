---
applyTo: "src/**/*.jl"
---

# Julia Source Code Guidelines for PortfolioOptimisers.jl

## Docstring Requirements

  - **All public types, functions, and macros must have docstrings**.
  - Docstrings should include:
    
      + Brief description of purpose.
      + `# Arguments` section with detailed parameter descriptions.
      + `# Returns` section describing what is returned.
      + `# Details` section for implementation notes (optional but recommended).
      + `# Related` section with links to related types/functions using `[@ref]` syntax.
      + Code examples using `jldoctest` blocks where appropriate.

  - **Docstring format**:
    
    ```julia
    """
        function_name(arg1, arg2; kwarg1=default)
    
    Brief description of what the function does.
    
    # Arguments
    
      - `arg1::Type1`: Description of arg1.
      - `arg2::Type2`: Description of arg2.
      - `kwarg1::Type3 = default`: Description of kwarg1.
    
    # Returns
    
      - `ReturnType`: Description of return value.
    
    # Details
    
      - Additional implementation details.
      - Algorithm notes or references.
    
    # Related
    
      - [`RelatedFunction`](@ref)
      - [`RelatedType`](@ref)
    
    # Examples
    
    ```jldoctest
    julia> result = function_name(1, 2)
    expected_output
    ```
    """
    function function_name(arg1, arg2; kwarg1=default)
        # implementation
    end
    ```

## Type Definitions

  - **Abstract types**:
    
      + Always prefix with `Abstract` (e.g., `AbstractCovarianceEstimator`).
      + Include comprehensive docstrings explaining their role in the type hierarchy.
      + List related types in the `# Related` section.

  - **Struct types**:
    
      + Use parametric types for flexibility: `struct MyType{T1, T2}`.
      + Include validation in constructors using `@argcheck` from ArgCheck.jl.
      + Use `@define_pretty_show(TypeName)` after struct definition for consistent pretty-printing.
      + All fields should be documented in the docstring.

## Input Validation

  - **Always use `@argcheck` for validation**:
    
    ```julia
    @argcheck !isempty(v) IsEmptyError("v cannot be empty")
    @argcheck all(isfinite, v) DomainError("v must be finite")
    @argcheck size(A, 1) == size(B, 1) DimensionMismatch("A and B must have same rows")
    ```

  - **Common validation patterns**:
    
      + Empty checks: `@argcheck !isempty(x) IsEmptyError(...)`
      + Nothing checks: `@argcheck !isnothing(x) IsNothingError(...)`
      + Finite checks: `@argcheck all(isfinite, x) DomainError(...)`
      + Dimension checks: `@argcheck size(A) == size(B) DimensionMismatch(...)`

## Multiple Dispatch

  - **Prefer multiple dispatch over conditionals** for algorithm/method selection.
  - Define specific methods for different type combinations rather than using if/else on types.
  - Use abstract types in method signatures for flexibility.

## Code Organization

  - **File naming**: Source files are prefixed numerically to indicate load order (e.g., `01_Base.jl`).
  - **Module structure**: Each submodule focuses on a specific domain (moments, risk, priors, etc.).
  - **Type hierarchy**: Subtype the appropriate abstract type (`AbstractEstimator`, `AbstractAlgorithm`, `AbstractResult`).

## Composability

  - Design estimators and algorithms to be composable.
  - Accept other estimators/algorithms as parameters when appropriate.
  - Return result types that encapsulate outcomes for easy chaining.

## Error Handling

  - Use custom exception types (subtype `PortfolioOptimisersError`):
    
      + `IsEmptyError` for empty collections.
      + `IsNothingError` for unexpected `nothing` values.
      + `IsNonFiniteError` for non-finite numbers.
      + `DimensionMismatch` for size mismatches.
      + `DomainError` for out-of-range values.

## Related Types References

  - Always include a `# Related` section in docstrings.
  - Link to abstract parent types, related estimators, algorithms, and result types.
  - Use the `[@ref]` syntax: `` [`TypeName`](@ref) ``.

## Examples in Docstrings

  - Use `jldoctest` blocks for testable examples when feasible.
  - Provide simple, clear examples that demonstrate basic usage.
  - For complex features, provide multiple examples showing different use cases.
