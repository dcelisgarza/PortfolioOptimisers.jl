---
applyTo: 'src/**/*.jl, docs/**/*.md'
---

# Docstring and Documentation Guidelines for PortfolioOptimisers.jl

## General Guidelines

- Look at how other docstrings are implemented and follow similar patterns.
- Write clear and concise documentation.
- Use consistent terminology and style.
- Include code examples where applicable.
- **All public types, functions, and macros must have docstrings.**

## The summary sentence (load-bearing — read this before writing a type docstring)

Every type docstring opens with `$(DocStringExtensions.TYPEDEF)`, a blank line, then a **summary paragraph**. The **first sentence of that paragraph is extracted verbatim** and rendered as the type's one-line description in the [Capability Catalogue](../../docs/capability_catalogue.jl) (ADR 0040), which is the user-facing inventory of everything the package can do.

This is a real contract, not a convention that merely happens to hold. It is what lets the catalogue carry prose without keeping a second, drifting copy of every description. If a docstring loses its summary paragraph, the docs build fails with an error naming the type.

Write the first sentence so it stands alone in a bullet list:

- **Lead with what it does**, in the active voice. `Denoises by setting the smallest \`num_factors\` eigenvalues to zero.` — not `A denoising algorithm that sets...`.
- **Keep it under ~120 characters.** If the idea needs more, put a crisp first sentence and move the detail into a *second* sentence, which still renders on the API page but not in the catalogue. Do not compress by deleting information.
- **Avoid filler openers**: `A flexible container type for...`, `A concrete estimator type for...`. They cost a line and say nothing.
- **Do not append `in \`PortfolioOptimisers.jl\``.** Every docstring in the package is in`PortfolioOptimisers.jl`.
- **Do not put `@ref` links in the first sentence.** The catalogue appends the type's own links after the description, so a link in the summary renders twice. Put cross-references in a second sentence or in `# Related`.
- **Do not open with a display formula.** Inline maths is fine; a full `$...$` equation belongs in `# Mathematical definition`.
- **Never leave a bare `_` outside a code span.** Markdown reads `_` as emphasis and will pair it with the underscore inside a neighbouring `` `snake_case` `` link, eating both and destroying the link. `(f_μ vector)` sitting next to `` [`plot_factor_mu`](@ref) `` rendered as ``(fμ vector … [`plotfactor_`` — a dead link that Documenter cannot resolve and the site builder reports only as a single anonymous `./@ref`. Write `` `f_mu` `` instead.
- **Siblings should not all share a prefix.** If every algorithm in a family starts `Centrality algorithm type for ...`, the catalogue shows that boilerplate eight times over. Say what distinguishes each one.

## Grammar

- Use present tense verbs (is, open) instead of past tense (was, opened).
- Write factual statements and direct commands. Avoid hypotheticals like "could" or "would".
- Use active voice where the subject performs the action.
- Write in third person (one, the user) to keep statements consistent.

## Markdown Guidelines

- Use headings to organise content.
- Use bullet points for lists.
- Include links to related resources.
- Use code blocks for code snippets.

---

## DocStringExtensions.jl Macros

Always use `DocStringExtensions.jl` macros where applicable rather than writing boilerplate manually:

| Macro | Use for |
| --- | --- |
| `$(DocStringExtensions.TYPEDEF)` | Docstring header for any `abstract type` or `struct` |
| `$(DocStringExtensions.TYPEDSIGNATURES)` | Docstring header for internal/private functions (auto-generates signature) |
| `$(DocStringExtensions.FIELDS)` | `# Fields` section body — auto-generates field list from inline field docstrings |
| `$(DocStringExtensions.README)` | Module-level docstring (top of main module file) |

Public functions use a **manually written signature** in the docstring header (not `TYPEDSIGNATURES`), because the manual form allows showing default values and grouping overloads clearly.

---

## Documentation Dictionaries

Four dictionaries in `src/01_Base.jl` provide standardised, consistent descriptions. **Always interpolate from them** instead of writing ad-hoc text.

- `arg_dict` — argument descriptions. Use as `$(arg_dict[:key])` in `# Arguments` sections.
- `field_dict` — field descriptions (derived from `arg_dict`). Use as `"$(field_dict[:key])"` in inline field docstrings inside structs.
- `val_dict` — validation rule descriptions. Use as `$(val_dict[:key])` in `## Validation` sections.
- `ret_dict` — return value descriptions. Use as `$(ret_dict[:key])` in `# Returns` sections.
- `math_dict` — LaTeX mathematical notation. Use as `$(math_dict[:key])` in `# Details` sections.

If a needed key is missing, add it to the appropriate dictionary in `01_Base.jl` before writing the docstring.

### Inline field docstrings

Fields in `@concrete` structs are documented inline using `field_dict`:

```julia
@concrete struct Covariance <: AbstractCovarianceEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:malg])"
    alg
    ...
end
```

---

## Section Structure for Types (abstract and concrete)

### Abstract types

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this abstract type represents.

All concrete subtypes should subtype `MyAbstractType`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype
`MyAbstractType` and implement the following methods:

## Required method name

  - `method_name(x::MyAbstractType, arg::Type) -> ReturnType`: What the method does.

### Arguments

  - `x`: The concrete subtype instance.
  - `arg`: Description.

### Returns

  - `result::ReturnType`: Description.

### Examples

```jldoctest
julia> struct MyConcreteType <: PortfolioOptimisers.MyAbstractType end
...
```

## Related

  - [`ConcreteSubtype1`](@ref)
  - [`related_function`](@ref)
"""
abstract type MyAbstractType <: AbstractEstimator end
````

### Concrete struct types

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this type does.

Optional longer explanation with mathematical notation if needed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyType(;
        field1::Type1 = default1,
        field2::Type2 = default2
    ) -> MyType

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:key1])
  - $(val_dict[:key2])

# Examples

```jldoctest
julia> MyType()
MyType
  field1 ┴ default1
```

## Related

  - [`AbstractMyType`](@ref)
  - [`related_function`](@ref)
"""
@concrete struct MyType <: AbstractMyType
    "$(field_dict[:key1])"
    field1
    "$(field_dict[:key2])"
    field2
    function MyType(field1::Type1, field2::Type2)
        # validation
        return new{typeof(field1), typeof(field2)}(field1, field2)
    end
end
function MyType(; field1::Type1 = default1, field2::Type2 = default2)
    return MyType(field1, field2)
end
````

### `@propagatable` concrete struct types

When a struct is decorated with `@propagatable`, fields carry two orthogonal, stackable tags inside the struct body:

- `@fprop` — **factory propagation**: the field is automatically propagated when [`factory`](@ref) is called on the enclosing struct (see `factory_child` for dispatch rules).
- `@vprop` — **view propagation**: the field is automatically subset when [`port_opt_view`](@ref) is called on the enclosing struct (recursing into composed children, slicing data arrays).

A field may carry neither, either, or both (`@fprop @vprop field`, in either order) — the factory- and view-relevant field sets genuinely diverge (a field can be factory-propagated but view-passthrough, or vice versa). Document each in its own subsection inside `# Constructors`, placed **after** `## Validation` (or directly after "Keywords correspond to the struct's fields." when there is no `## Validation`).

For `@fprop`-tagged fields, add a `## Propagated parameters` subsection listing each field and how it is propagated:

- **Observation-weight fields** (type `ObsWeights`, `Nothing`, or `Option{<:ObsWeights}`): write `` `fieldname`: Replaced with the incoming [`ObsWeights`](@ref). ``
- **Estimator, algorithm, or result fields** (subtypes of `AbstractEstimator`, `AbstractAlgorithm`, or `AbstractResult`): write `` `fieldname`: Recursively updated via [`factory`](@ref). ``

For `@vprop`-tagged fields, add a `## View parameters` subsection listing each field and how it is viewed:

- **Estimator, algorithm, or result fields** (subtypes of `AbstractEstimator`, `AbstractAlgorithm`, or `AbstractResult`): write `` `fieldname`: Recursively viewed via [`port_opt_view`](@ref). ``
- **Data fields** (arrays, scalars, or `Option` thereof): write `` `fieldname`: Sliced to the selected indices via [`port_opt_view`](@ref). ``

List fields in the same order they appear in the struct body, and add [`factory`](@ref) and/or [`port_opt_view`](@ref) to `# Related` to match the tags present.

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this type does.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyType(;
        weight_field::Option{<:ObsWeights} = nothing,
        nested_est::AbstractEstimator = MyEstimator(),
        config::Bool = true
    ) -> MyType

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `nested_est`: Recursively updated via [`factory`](@ref).
  - `weight_field`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `nested_est`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> MyType()
MyType
  nested_est ┼ MyEstimator
  weight_field ┴ nothing
```

# Related

  - [`AbstractMyType`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct MyType <: AbstractMyType
    "$(field_dict[:oow])"
    @fprop weight_field
    "$(field_dict[:nested])"
    @fprop @vprop nested_est
    "$(field_dict[:cfg])"
    config
    function MyType(weight_field::Option{<:ObsWeights}, nested_est::AbstractEstimator,
                    config::Bool)
        assert_nonempty_nonneg_finite_val(weight_field, :weight_field)
        return new{typeof(weight_field), typeof(nested_est), typeof(config)}(weight_field,
                                                                             nested_est,
                                                                             config)
    end
end
function MyType(;
                weight_field::Option{<:ObsWeights} = nothing,
                nested_est::AbstractEstimator = MyEstimator(),
                config::Bool = true)::MyType
    return MyType(weight_field, nested_est, config)
end
````

---

## Section Structure for Functions

Public functions use a manually written signature as the docstring header.

````julia
"""
    function_name(
        arg1::Type1,
        arg2::Type2;
        kwarg1::Type3 = default
    ) -> ReturnType

One-sentence description of what this function does.

Longer explanation if needed.

# Arguments

  - $(arg_dict[:key1])
  - $(arg_dict[:key2])
  - `kwarg1::Type3 = default`: Description.

# Validation

  - $(val_dict[:key])

# Returns

  - $(ret_dict[:key])

# Details

  - Additional implementation notes.

# Examples

```jldoctest
julia> function_name(...)
...
```

## Related

  - [`RelatedType`](@ref)
  - [`related_function`](@ref)
"""
function function_name(arg1::Type1, arg2::Type2; kwarg1::Type3 = default)
    ...
end
````

Internal/private functions may use `$(DocStringExtensions.TYPEDSIGNATURES)` as the header instead of a manually written signature.

---

## The `# Validation` Section

- Include a `## Validation` sub-section in struct docstrings and a `# Validation` section in function docstrings whenever the function or constructor enforces preconditions.
- Use `val_dict` entries for common validation rules.
- For custom validation, describe the condition clearly: `` `x > 0` ``.

---

## `jldoctest` Examples

- All examples in docstrings must use `jldoctest` blocks so they are testable via `julia --doctest`.
- Output of pretty-printed structs must match exactly what `@define_pretty_show` produces.
- For abstract types with an `# Interfaces` section, the `jldoctest` must demonstrate a complete working implementation of the interface.
- Keep examples minimal but complete enough to be useful.

---

## `docs/src/api/` Markdown Files

Each source file `src/SomeFeature.jl` has a corresponding `docs/src/api/SomeFeature.md`. Every public symbol defined in the source file must be listed under an appropriate heading using the Documenter.jl `@docs` block:

````markdown
## My section heading

```@docs
MyType
my_function
MyAbstractType
```
````

When adding a new symbol, also add it to the corresponding API markdown file.

---

## Mathematical Notation

When a type or function has a mathematical formulation, include a `# Mathematical definition` section immediately before `# Fields` (for structs) or before `# Arguments` (for functions).

### LaTeX conventions

| Notation | Use for |
| --- | --- |
| `\boldsymbol{x}` | Vectors (e.g., ``\boldsymbol{w}``, ``\boldsymbol{\mu}``) |
| `\mathbf{A}` | Matrices (e.g., ``\mathbf{\Sigma}``, ``\mathbf{F}``) |
| `\mathbb{R}` | Number domains (e.g., ``\mathbb{R}^N``, ``\mathbb{Z}_{\geq 0}``) |
| `\mathcal{W}` | Sets (e.g., ``\mathcal{W}``, ``\mathcal{K}_{\mathrm{SOC}}``) |
| `\underset{\boldsymbol{w}}{\min}` | Optimisation formulations (not `\min_{\boldsymbol{w}}`) |
| `\intercal` | Transpose (e.g., ``\boldsymbol{w}^\intercal``) |

### `\begin{align}` environment

All math blocks use `\begin{align}...\end{align}` with `&` alignment markers and `\\` line breaks. Each separate equation goes on its own line. Use `\quad` for in-equation spacing only; split distinct equations onto separate lines (never `\qquad` between two equations in the same block).

````julia
# Good — each equation on its own line
```math
\begin{align}
\hat{\boldsymbol{\mu}} &= \frac{1}{T} \sum_{t=1}^{T} \boldsymbol{x}_t\,, \\
\hat{\mathbf{\Sigma}} &= \frac{1}{T-1} \sum_{t=1}^{T}
    (\boldsymbol{x}_t - \hat{\boldsymbol{\mu}})
    (\boldsymbol{x}_t - \hat{\boldsymbol{\mu}})^\intercal\,.
\end{align}
```

# Bad — \qquad to cram two equations on one line
```math
\begin{align}
\hat{\boldsymbol{\mu}} &= \frac{1}{T} \sum_{t=1}^T \boldsymbol{x}_t \qquad
\hat{\mathbf{\Sigma}} = \frac{1}{T-1} \sum_{t=1}^T \ldots
\end{align}
```
````

### `Where:` section

Immediately after each math block (or after the **last** math block when multiple consecutive blocks belong to the same docstring), add a `Where:` bullet list defining every symbol. Use `$(math_dict[:key])` for common variables.

````julia
"""
# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\frac{1}{T} \\sum_{t=1}^{T} \\boldsymbol{x}_t\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\frac{1}{T-1} \\sum_{t=1}^{T}
    (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})
    (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})^\\intercal\\,.
\\end{align}
```

Where:

- ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
- ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
- $(math_dict[:x_t])
- $(math_dict[:T])
"""
````

Key rules:

- One comprehensive `Where:` after the last block is acceptable when multiple blocks appear in the same docstring.
- Every symbol that appears in any block must be defined.
- Interpolate `$(math_dict[:key])` for standardised variables (``T``, ``\boldsymbol{x}_t``, ``\alpha``, etc.).
- If a key is missing from `math_dict`, add it to `src/01_Base.jl` first.

---

## Complete Example

The following is a fully worked example covering abstract types, concrete types, and functions. Use it as a reference when writing docstrings.

````julia
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom processes in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a custom process should subtype `MyAbstractCustomProcess`.

# Interfaces

In order to implement a new custom process that can seamlessly work with the library, subtype `MyAbstractCustomProcess`, ensuring that the structure contains all necessary parameters for the custom process, and implement the following methods:

## Custom process interface

### Functions

- `do_process(pr::MyAbstractCustomProcess, b::Real, c::Integer)`: Performs the custom process.

#### Arguments

- `pr`: Custom process.
- `b`: First argument for the custom process.
- `c`: Second argument for the custom process.

#### Returns

- `nothing`.

### Examples

We can create a dummy custom process as follows:

```jldoctest
julia> struct MyNewCustomProcess{T1, T2} <: PortfolioOptimisers.MyAbstractCustomProcess
           alg::T1
           new_param::T2
           function MyNewCustomProcess(alg::MyAbstractCustomProcessAlgorithm, new_param::Symbol)
               return new{typeof(alg), typeof(new_param)}(alg, new_param)
           end
       end

julia> function MyNewCustomProcess(; alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(), new_param::Symbol = :Foo)
           return MyNewCustomProcess(alg, new_param)
        end

julia> function PortfolioOptimisers.do_process(a::MyNewCustomProcess, b::Real, c::Integer)
          println("new custom process: $b $c $(a.sym)")
          do_algorithm(a.alg, c)
          return nothing
       end

julia> do_process(MyNewCustomProcess(), -0.5, 9)
new custom process: -0.5 9 Foo
algorithm 1: 9
```

# Related

- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
abstract type MyAbstractCustomProcess end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom process algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a custom process algorithms should subtype `MyAbstractCustomProcessAlgorithm`.

# Interfaces

In order to implement a new custom process algorithms that can seamlessly work with the library, subtype `MyAbstractCustomProcessAlgorithm`, ensuring that the structure contains all necessary parameters for the custom process algorithm, and implement the following methods:

## Custom process algorithm interface

### Functions

- `do_algorithm(pra::MyAbstractCustomProcessAlgorithm, c::Integer) -> Integer`: Performs the custom process algorithm and returns the result.

#### Arguments

- `pra`: Custom process algorithm.
- `c`: Argument for the custom process algorithm.

#### Returns

- `res::Integer`: The result of the algorithm.

### Examples

We can create a dummy custom process algorithm as follows:

```jldoctest
julia> struct MyNewCustomProcessAlgorithm{T} <: PortfolioOptimisers.MyAbstractCustomProcessAlgorithm
           new_param::T
           function MyNewCustomProcessAlgorithm(new_param::Symbol)
               return new{typeof(new_param)}(new_param)
           end
       end

julia> function MyNewCustomProcessAlgorithm(; new_param::Symbol = :Bar)
           return MyNewCustomProcessAlgorithm(new_param)
        end

julia> function PortfolioOptimisers.do_algorithm(alg::MyNewCustomProcessAlgorithm, c::Integer)
          println("new algorithm: $c $(alg.new_param)")
          return c + 1
       end

julia> do_algorithm(MyNewCustomProcessAlgorithm(), 3)
new algorithm: 3 Bar
4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
abstract type MyAbstractCustomProcessAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements my custom process algorithm 1.

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyCustomProcessAlgorithm1 <: MyAbstractCustomProcessAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process algorithm 1.

# Arguments

- `alg::MyCustomProcessAlgorithm1`: The algorithm to perform.
- `c::Integer`: The input integer.

# Returns

- `res::Integer`: The result of the algorithm.

# Details

- Multiplies `c` by 2.
- Prints the result with a custom message.
- Returns the result.

```jldoctest
julia> do_algorithm(MyCustomProcessAlgorithm1(), 3)
algorithm 1: 6
6
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
"""
function do_algorithm(::MyCustomProcessAlgorithm1, c::Integer)
    c = c * 2
    println("algorithm 1: $c")
    return c
end
"""
$(DocStringExtensions.TYPEDEF)

Defines my custom process 1.

# Fields

- `alg::MyAbstractCustomProcessAlgorithm`: The algorithm to use.

# Constructors

    MyConcreteCustomProcess1(;
        alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1()
    ) -> MyConcreteCustomProcess1

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MyConcreteCustomProcess1()
MyConcreteCustomProcess1
  alg ┴ MyCustomProcessAlgorithm1()
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyConcreteCustomProcess1{T} <: MyAbstractCustomProcess
    alg::T
    function MyConcreteCustomProcess1(alg::MyAbstractCustomProcessAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function MyConcreteCustomProcess1(;
                                  alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1())
    return MyConcreteCustomProcess1(alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process 1.

# Arguments

- `a::MyConcreteCustomProcess1`: The custom process to perform.
- `b::Real`: The first argument.
- `c::Integer`: The second argument.

# Validation

- `b >= 0`: `b` must be non-negative.

# Returns

- `nothing`.

# Details

- Checks `b >= 0` before performing the process.
- Prints a message using the arguments `b` and `c`.
- Calls `do_algorithm` with the algorithm from `a.alg` and `c`.

# Examples

```jldoctest
julia> do_process(MyConcreteCustomProcess1(), 1.0, 2)
Custom process 1: 1.0 + 2
algorithm 1: 4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`MyConcreteCustomProcess1`](@ref)
- [`do_algorithm`](@ref)
"""
function do_process(a::MyConcreteCustomProcess1, b::Real, c::Integer)
    @argcheck(b >= 0, "b must be non-negative")
    println("Custom process 1: $b + $c")
    do_algorithm(a.alg, c)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validates that `val > 2`.

# Arguments

- `val::Real`: The value to validate.

# Returns

- `nothing`.

# Related

- [`MyConcreteCustomProcess2`](@ref)
"""
function assert_val_value(val::Real)
    @argcheck(val >= 2 * one(eltype(val)), "val must be non-negative")
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Defines my custom process 2.

# Fields

- `alg::MyAbstractCustomProcessAlgorithm`: The algorithm to use.
- `val::Real`: The value to use.

# Constructors

    MyConcreteCustomProcess2(;
        alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(),
        val::Real = 2.0
    ) -> MyConcreteCustomProcess2

Keywords correspond to the struct's fields.

## Validation

- `val` is validated via [`assert_val_value`](@ref).

# Examples

```jldoctest
julia> MyConcreteCustomProcess2()
MyConcreteCustomProcess2
  alg ┼ MyCustomProcessAlgorithm1()
  val ┴ 2.0
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`assert_val_value`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyConcreteCustomProcess2{T1, T2} <: MyAbstractCustomProcess
    alg::T1
    val::T2
    function MyConcreteCustomProcess2(alg::MyAbstractCustomProcessAlgorithm, val::Real)
        return new{typeof(alg), typeof(val)}(alg, val)
    end
end
function MyConcreteCustomProcess2(;
                                  alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(),
                                  val::Real = 2.0)
    assert_val_value(val)
    return MyConcreteCustomProcess2(alg, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process 2.

# Arguments

- `a::MyConcreteCustomProcess2`: The custom process to perform.
- `b::Real`: The first argument.
- `c::Integer`: The second argument.

# Returns

- `nothing`.

# Details

- Prints a message using the arguments `a.val`, `b` and `c`.
- Calls `do_algorithm` with the algorithm from `a.alg` and `c`.

# Examples

```jldoctest
julia> do_process(MyConcreteCustomProcess2(), 1.0, 2)
Custom process 2: 2.0 - 1.0 + 2
algorithm 1: 4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`MyConcreteCustomProcess1`](@ref)
- [`do_algorithm`](@ref)
"""
function do_process(a::MyConcreteCustomProcess2, b::Real, c::Integer)
    println("Custom process 2: $(a.val) - $b + $c")
    do_algorithm(a.alg, c)
    return nothing
end

````
