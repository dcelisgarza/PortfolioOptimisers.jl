---
agent: agent
description: Add docstrings to markdown files.
---

List each docstring in a `@docs` codeblock of the `*.md` file that mirrors its source file. For example:

- If the documented item is in `src/folder/myfile.jl`, list it in `docs/src/public_api/folder/myfile.md` or in `docs/src/private_api/folder/myfile.md`, on the side that its classification gives it. A file at the top level of `src/` maps to a page at the top level of each tree.
- If the `*.md` file does not exist, create it.
- If the file exists and has a `@docs` codeblock, add the item to that block. Otherwise, create the codeblock and add the item to it.
- Some functions have more than one method. Tell the methods apart by their type signatures. For example:

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

The `@docs` block tells them apart by their type signatures.

```@docs
my_function(a::Int, b::Int, args...; kwargs...)
my_function(a::Tuple{Float64, <:Real}, b::Float64, args...; kwargs...)
```

A reference to one method of a function uses its signature in the same way. For example:

[`my_function(a::Int, b::Int, args...; kwargs...)`](@ref)
[`my_function(a::Tuple{Float64, <:Real}, b::Float64, args...; kwargs...)`](@ref)

These references work in any Markdown, docstrings included.
