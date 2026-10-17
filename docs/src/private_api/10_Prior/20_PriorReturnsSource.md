```@meta
Description = "Prior Returns Source, private API of PortfolioOptimisers.jl: reader_frames, picked_returns_data, warn_source_fallback, graph_verb_keywords, …"
```

# [Prior Returns Source: private API](@id private-api-prior-returns-source)

Each reader of the returns states its own source in its `x_src` field. A prior reads the returns data in the source that it states. A clustering, phylogeny or centrality estimator reads a prior result or the returns data, and the picker below gives it the matrix of its source. A fit finds the gaps of its plain graph readers reader by reader, and leaves each asset with a gap out of the assets that the optimisation can trade. The last methods say whether a prior of the library gives no loadings, and record the source of the prior in the Prior Result that the prior fits. The [public page](@ref api-prior-returns-source) documents the method that gives the source of each prior of the library.

```@docs
PortfolioOptimisers.reader_frames(::AbstractPriorEstimator, ::AbstractString, ::Bool)
PortfolioOptimisers.reader_frames(::AbstractPriorResult, ::AbstractString, ::Bool)
picked_returns_data
warn_source_fallback
graph_verb_keywords
returns_matrix_picker
plain_graph_reader
graph_gap_remedy
assert_graph_readable
graph_reader_gaps
graph_reader_mask
implied_volatility_picker
centrality_constraints_from
PortfolioOptimisers.refits_regression(::Union{<:EmpiricalPrior, <:LowDimensionEnsemblePrior})
stamp_returns_source
prior_returns_source
```
