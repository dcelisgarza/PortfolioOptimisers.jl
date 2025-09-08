# Smyth-Broby Covariance

```@docs
SmythBroby0
SmythBroby1
SmythBroby2
SmythBrobyGerber0
SmythBrobyGerber1
SmythBrobyGerber2
NormalisedSmythBroby0
NormalisedSmythBroby1
NormalisedSmythBroby2
NormalisedSmythBrobyGerber0
NormalisedSmythBrobyGerber1
NormalisedSmythBrobyGerber2
SmythBrobyCovariance
cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                             <:PortfolioOptimisers.UnNormalisedSmythBrobyCovarianceAlgorithm,
                             <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                             <:PortfolioOptimisers.UnNormalisedSmythBrobyCovarianceAlgorithm,
                             <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
PortfolioOptimisers.BaseSmythBrobyCovariance
PortfolioOptimisers.SmythBrobyCovarianceAlgorithm
PortfolioOptimisers.UnNormalisedSmythBrobyCovarianceAlgorithm
PortfolioOptimisers.NormalisedSmythBrobyCovarianceAlgorithm
PortfolioOptimisers.sb_delta
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby0, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBroby0, <:Any},
                    X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby1, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBroby1, <:Any},
                    X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby2, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBroby2, <:Any},
                    X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber0, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBrobyGerber0,
                                             <:Any}, X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber1, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBrobyGerber1,
                                             <:Any}, X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber2, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBrobyGerber2,
                                             <:Any}, X::AbstractMatrix)
```
