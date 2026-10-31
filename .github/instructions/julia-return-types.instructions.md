---
applyTo: "src/**/*.jl"
---

# Julia Return Type Annotation Guidelines for PortfolioOptimisers.jl

Julia infers return types well, so a function does not need a return type annotation (`::ReturnType`) everywhere. An annotation is most useful when it does one of these:

1. It documents the return type, so a reader of the function definition sees it.
2. It catches a programming error. Julia throws a `TypeError` when the returned value does not match the annotation.
3. It enforces a contract, most of all on an interface that the library exposes.

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

- **Hand-written factory methods** whose return type is the same concrete type as the first argument. A `@propagatable` type gets its `factory` from the macro, so this rule applies only to a method that someone writes by hand:

  ```julia
  function factory(re::LinearModel, w::ObsWeights)::LinearModel
      return LinearModel(; kwargs = (; re.kwargs..., weights = w))
  end
  ```

- **Boolean predicate helpers** that always return `Bool`:

  ```julia
  function has_pretty_show_method(::Any)::Bool
      return false
  end
  ```

### Annotate when helpful for clarity

- A function in a typed dispatch chain whose concrete return type is stable, and which its name does not make obvious.
- A function that returns a union type, when the annotation is narrower than the type that inference gives.

## When NOT to Annotate

- A function whose return type is polymorphic, or depends on the input types in a complex way. Let Julia infer it.
- A function that returns `JuMP` expressions or model variables. Their types are opaque and complex.
- An internal helper whose call sites are all in the same file, where the context makes the type obvious.
- Any place where the annotation forces a type conversion that allocates.

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

When the return type is an abstract type, such as `AbstractPriorResult`, the annotation still documents the function. Julia cannot use it for specialisation.

```julia
function prior(pe::EmpiricalPrior, X::MatNum; kwargs...)::AbstractPriorResult
    ...
    return LowOrderPrior(...)
end
```

Prefer the most specific concrete type when the function always returns that type. Use the abstract supertype when the concrete type varies by dispatch.

## `Nothing` Returns

Annotate with `::Nothing` every function that exists only for a side effect, such as validation, in-place mutation or printing, and that ends with an explicit `return nothing`.

```julia
function record_non_investable_drop!(ledger::AbstractVector, what::AbstractString)::Nothing
    push!(ledger, what)
    return nothing
end
```
