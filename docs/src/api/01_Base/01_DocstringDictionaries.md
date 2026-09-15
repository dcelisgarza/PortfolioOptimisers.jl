# Docstring dictionaries

[`src/01_Base/`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/src/01_Base) implements the most basal symbols used in `PortfolioOptimisers.jl`. One file per concept: the docstring dictionaries, the type roots, the pretty-show macro, the `ScopedConfig` holders, the load-time preferences, the message builders, the error hierarchy, the type aliases, the observation weights, the `assert_*` family, `VecScalar`, the `NormError` family, the Kaniadakis logarithm, the partial-fit state seam and the sample buffer the online step folds into.

```@docs
PortfolioOptimisers
```

## Glossaries

In order to standardise the documentation we use a arg_dict of terms.

```@docs
unique_key_dict
arg_dict
val_dict
ret_dict
field_dict
math_dict
err_name_dict
ref_dict
```
