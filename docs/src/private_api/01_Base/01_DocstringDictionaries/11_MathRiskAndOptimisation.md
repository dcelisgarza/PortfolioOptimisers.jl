```@meta
Description = "Risk and optimisation notation has no private API in PortfolioOptimisers.jl; its names are in the public API."
```

# Risk and optimisation notation: private API

This file defines no name. It fills [`math_dict`](@ref) with the notation of the risk measures, their JuMP formulations, the penalties, weight finalisation and the meta-optimisers. [`unique_key_dict!`](@ref) refuses a key that another entry holds with a different description.
