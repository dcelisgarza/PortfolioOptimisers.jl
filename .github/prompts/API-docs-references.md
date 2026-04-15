---
agent: ask
description: Add docstrings to markdown files.
---

Docstrings should be referenced in a `@docs` codeblock in their respective `*.md` file. For example:

- If the item associated with the docstring is found in `src/folder/myfile.jl`, it should appear in `docs/src/api/myfile.md`.
- If the `*.md` file does not exist, create it.
- If the file exists and has a `@docs` codeblock, add it to the block, otherwise create the codeblock and add the item inside.
- Some functions have multiple methods (overloads). If this is the case, you can disambiguate the methods by using their respective type signatures. For example:

```julia
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Documentation for my_function method int, int.
"""
function my_function(a::Int, b::Int, args...; kwargs...)
  return a - b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Documentation for my_function method (float64, real), float64.
"""
function my_function(a::Tuple{Float64, <:Real}, b::Float64, args...; kwargs...)
  return a[1] + b + a[2]/b
end
```

They would be disambiguated by using their type signatures inside the `@doc` codblock.

```@docs
my_function(a::Int, b::Int, args...; kwargs...)
my_function(a::Tuple{Float64, <:Real}, b::Float64, args...; kwargs...)
```

The same should be done when referencing specific methods of functions. For example:

[`my_function(a::Int, b::Int, args...; kwargs...)`](@ref)
[`my_function(a::Tuple{Float64, <:Real}, b::Float64, args...; kwargs...)`](@ref)

These references can be placed inside standard markdown (including inside docstrings themselves).
