```@meta
Description = "Type utilities, public API of PortfolioOptimisers.jl: traverse_concrete_subtypes, concrete_typed_array."
```

# Type utilities

This page lists the general functions, macros and types that the rest of the library uses.

## Utility functions

`traverse_concrete_subtypes` lists the struct types under an abstract type. `concrete_typed_array` converts an array with an abstract element type into an array with a concrete element type. `factory` rebuilds an estimator with the values a fit supplies, such as the prior moments, the observation weights and the previous portfolio weights, and `factory_child` applies it to one field. `@propagatable` defines a struct and writes its `factory` and `port_opt_view` methods from the tags on its fields, which are `@fprop`, `@vprop`, `@pprop`, `@wprop` and `@cprop`. `@forward_properties` writes `getproperty` and `propertynames` for a type from a list of forwarding rules.

```@docs
traverse_concrete_subtypes
concrete_typed_array
```
