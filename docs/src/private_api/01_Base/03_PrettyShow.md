```@meta
Description = "Pretty printing, private API of PortfolioOptimisers.jl: @define_pretty_show, show_fields, pretty_show_fields, has_pretty_show_method, …"
```

# Pretty printing: private API

`PortfolioOptimisers.jl`'s types tend to contain quite a lot of information, these functions enable pretty printing so they are easier to interpret. A field that holds `nothing` is hidden by default and shown in this documentation; [`set_show_nothing_fields!`](@ref) is the switch, and [`show_fields`](@ref) is the hook a type overloads to hide a field of its own choice.

```@docs
@define_pretty_show
show_fields
pretty_show_fields
has_pretty_show_method
pretty_show_vector_summary
pretty_show_vector_element
pretty_show_vector_body
```
