```@meta
Description = "Returns Source, public API of PortfolioOptimisers.jl: AbstractReturnsSource, DataSource, ExcessSource, PriorSource, source_frames, returns_source."
```

# Returns Source

Each reader of the returns states the returns that it fits on in its `x_src` field. A reader is a prior, a clustering, phylogeny or centrality estimator, a factor regression, or the step of an [`OnlinePortfolioSelection`](@ref). `x_src = nothing` means that you stated no source. [`DataSource`](@ref) reads the returns data as it arrived. [`ExcessSource`](@ref) reads the returns in excess of the benchmark, `X - B`. [`PriorSource`](@ref) reads the Prior Result. [`AbstractReturnsSource`](@ref) gives the table of what each reader reads at each value. A source of your own implements [`benchmark_relative_returns`](@ref) and [`source_frames`](@ref). A reader of your own states its source with a method of [`returns_source`](@ref).

```@docs
AbstractReturnsSource
DataSource
ExcessSource
PriorSource
source_frames
returns_source(::Any)
```
