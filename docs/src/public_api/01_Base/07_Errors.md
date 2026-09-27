```@meta
Description = "Error types, public API of PortfolioOptimisers.jl: IsNothingError, IsEmptyError, IsNonFiniteError, PropertyPathError, ConflictingArgumentError, …"
```

# Error types

Most types of `PortfolioOptimisers.jl` check their values in the constructor. A method that receives such an object uses it without checking it again, which keeps the method short and fast. A check that needs data the constructor does not have runs at the start of the first function that has that data.

When a check fails, the library throws one of the error types below. The type names the kind of failure, such as a value that is `nothing`, an empty array or a non-finite number.

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
