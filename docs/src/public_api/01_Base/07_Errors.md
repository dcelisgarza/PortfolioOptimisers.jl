```@meta
Description = "Error types, public API of PortfolioOptimisers.jl: IsNothingError, IsEmptyError, IsNonFiniteError, PropertyPathError, ConflictingArgumentError, …"
```

# Error types

Many of the types defined in `PortfolioOptimisers.jl` make use of extensive data validation to ensure values meet various criteria. This simplifies the implementation of methods, and improves performance and by delegating as many checks as possible to variable instantiation. In cases where validation cannot be performed at variable instantiation, they are performed as soon as possible within functions.

`PortfolioOptimisers.jl` aims to catch potential data validation issues as soon as possible and in an informative manner, in order to do so it makes use of a few custom error types.

```@docs
IsNothingError
IsEmptyError
IsNonFiniteError
PropertyPathError
ConflictingArgumentError
ObservationWeightsError
NonPositiveWealthError
Base.showerror(io::IO, err::PortfolioOptimisersError)
```
