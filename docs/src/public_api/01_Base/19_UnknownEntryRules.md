```@meta
Description = "Unknown entry rules, public API of PortfolioOptimisers.jl: AbstractUnknownEntryRule, EntrywiseUnknown, ZeroUnknown, KindwiseUnknown."
```

# Unknown entry rules

A rule for an entry that the data do not state has two jobs. A factor attribution applies it to an
entry that the factor model does not state, and a union of graphs applies it to a part that has no
graph in force.

```@docs
AbstractUnknownEntryRule
EntrywiseUnknown
ZeroUnknown
KindwiseUnknown
```
