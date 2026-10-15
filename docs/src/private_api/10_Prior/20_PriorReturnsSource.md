```@meta
Description = "Prior Returns Source, private API of PortfolioOptimisers.jl: reader_frames, picked_returns_data, warn_source_fallback, graph_verb_keywords, …"
```

# [Prior Returns Source: private API](@id private-api-prior-returns-source)

Each reader of the returns states its own source in its `x_src` field. A prior reads the returns data in the source that it states. A clustering, phylogeny or centrality estimator reads a prior result or the returns data, and the picker below gives it the matrix of its source. A fit finds the gaps of its plain graph readers reader by reader, and leaves each asset with a gap out of the Investable Mask. The last methods give the source of each prior of the library to the frame walk of an optimiser, and record that source in the Prior Result that the prior fits.

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
PortfolioOptimisers.returns_source(::Union{<:EmpiricalPrior, <:FactorPrior, <:HighOrderPriorEstimator, <:HighOrderFactorPriorEstimator, <:BlackLittermanPrior, <:BayesianBlackLittermanPrior, <:FactorBlackLittermanPrior, <:AugmentedBlackLittermanPrior, <:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior, <:OpinionPoolingPrior, <:CrossSectionalFactorPrior, <:LowDimensionEnsemblePrior})
PortfolioOptimisers.refits_regression(::Union{<:EmpiricalPrior, <:LowDimensionEnsemblePrior})
stamp_returns_source
prior_returns_source
```
