# Deliberately unsupported CV boundaries

Combinatorial and multiple-randomised cross-validation are defined over the returns matrix and cannot drive contiguous input-row windows; wrapping a `Pipeline` inside a meta-optimiser needs an asset view of a universe that is itself fitted state (ADR 0028, "Future expansion"). Each fails with an explanatory error rather than silently doing the wrong thing.

```@docs
Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)
Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)
port_opt_view(pipe::Pipeline, i, args...; kwargs...)
```
