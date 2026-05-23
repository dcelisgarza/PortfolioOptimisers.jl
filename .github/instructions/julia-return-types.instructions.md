---
applyTo: "src/**/*.jl"
---

# Julia Return Type Annotation Guidelines for PortfolioOptimisers.jl

Julia's type inference is excellent, so return type annotations (`::ReturnType`) are not needed everywhere. They are most valuable when they:

1. Serve as documentation — making the return type explicit when reading a function definition.
2. Catch programming errors — Julia will throw a `TypeError` if the actual return does not match the annotation.
3. Enforce a contract — particularly for library-facing interfaces.

## When to Annotate

### Always annotate

- **Validation/assertion helpers** that always return `nothing`:

  ```julia
  function assert_nonempty_nonneg_finite_val(x, sym)::Nothing
      ...
      return nothing
  end
  ```

- **Passthrough/no-op methods** where the return is the same type as an argument:

  ```julia
  function prior(pr::AbstractPriorResult, args...; kwargs...)::AbstractPriorResult
      return pr
  end
  ```

- **Factory methods** when the return type is the exact same concrete type as the first argument:

  ```julia
  function factory(ce::GeneralCovariance, w::ObsWeights)::GeneralCovariance
      return GeneralCovariance(; ce = factory(ce.ce, w), w = w)
  end
  ```

- **Boolean predicate helpers** that always return `Bool`:

  ```julia
  function has_pretty_show_method(::Any)::Bool
      return false
  end
  ```

### Annotate when helpful for clarity

- Functions in a clearly typed dispatch chain where the concrete return type is stable and non-obvious from the name.
- Functions that return a union type and the annotation is narrower than what inference produces.

## When NOT to Annotate

- Functions whose return type is polymorphic or depends on input types in complex ways (let Julia infer).
- Functions that return `JuMP` expressions or model variables (types are opaque/complex).
- Internal helper functions where the call site is always in the same file and context is obvious.
- Anywhere the annotation would force an allocation-inducing type conversion.

## Syntax

```julia
# Preferred form — annotation on the function definition
function foo(x::VecNum)::MatNum
    ...
end

# Also valid for one-liners
bar(x::Option{<:Number})::Nothing = (@argcheck ...; return nothing)
```

## Abstract Return Types

When the return type is an abstract type (e.g., `AbstractPriorResult`), the annotation still adds value as documentation even though Julia cannot use it for specialisation:

```julia
function prior(pe::EmpiricalPrior, X::MatNum; kwargs...)::AbstractPriorResult
    ...
    return LowOrderPrior(...)
end
```

Prefer the most specific concrete type when it is always the same; use the abstract supertype when the concrete type varies by dispatch.

## `Nothing` Returns

All functions that exist solely for side effects (validation, in-place mutation, printing) and explicitly `return nothing` should be annotated `::Nothing`:

```julia
function denoise!(dn::SpectralDenoise, X::MatNum, q::Number)::Nothing
    ...
    return nothing
end
```
