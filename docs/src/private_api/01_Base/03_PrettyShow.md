```@meta
Description = "Pretty printing, private API of PortfolioOptimisers.jl: @define_pretty_show, show_fields, pretty_show_fields, has_pretty_show_method, …"
```

# [Pretty printing: private API](@id private-api-pretty-printing)

The types of `PortfolioOptimisers.jl` hold many fields. The functions below print them one field per line, with the fields of a nested type indented under it. By default the printout hides a field that holds `nothing`, and this documentation shows it. [`set_show_nothing_fields!`](@ref) turns those fields on or off, and a type can overload [`show_fields`](@ref) to hide a field of its own choice.

```@docs
@define_pretty_show
show_fields
pretty_show_fields
has_pretty_show_method
pretty_show_vector_summary
pretty_show_vector_element
pretty_show_vector_body
```
