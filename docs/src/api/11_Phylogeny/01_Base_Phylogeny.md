# Base Phylogeny

```@docs
AbstractPhylogenyEstimator
AbstractPhylogenyAlgorithm
AbstractPhylogenyResult
PlE_Pl
factory(pl::PlE_Pl, args...; kwargs...)
factory(alg::AbstractPhylogenyAlgorithm, args...; kwargs...)
AbstractSeparationAlgorithm
HopCountAlgorithm
PathLengthAlgorithm
HopCountRule
HopCountValue
PathLengthRule
PathLengthValue
HopCount
PathLength
AbstractSeparationDecayAlgorithm
LinearDecay
ExponentialDecay
ReciprocalDecay
NoDecay
separation_decay
PortfolioOptimisers.assert_separation_decay
```
