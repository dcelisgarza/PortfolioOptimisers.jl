# Smyth-Broby Covariance

```@docs
PortfolioOptimisers.BaseSmythBrobyCovariance
PortfolioOptimisers.SmythBrobyCovarianceAlgorithm
PortfolioOptimisers.UnNormalisedSmythBrobyCovarianceAlgorithm
PortfolioOptimisers.NormalisedSmythBrobyCovarianceAlgorithm
PortfolioOptimisers.SmythBroby0
PortfolioOptimisers.SmythBroby1
PortfolioOptimisers.SmythBroby2
PortfolioOptimisers.SmythBrobyGerber0
PortfolioOptimisers.SmythBrobyGerber1
PortfolioOptimisers.SmythBrobyGerber2
PortfolioOptimisers.NormalisedSmythBroby0
PortfolioOptimisers.NormalisedSmythBroby1
PortfolioOptimisers.NormalisedSmythBroby2
PortfolioOptimisers.NormalisedSmythBrobyGerber0
PortfolioOptimisers.NormalisedSmythBrobyGerber1
PortfolioOptimisers.NormalisedSmythBrobyGerber2
SmythBrobyCovariance
SmythBrobyCovariance()
PortfolioOptimisers.sb_delta
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby0, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:PortfolioOptimisers.NormalisedSmythBroby0, <:Any},
                    X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby1, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:PortfolioOptimisers.NormalisedSmythBroby1, <:Any},
                    X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby2, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:PortfolioOptimisers.NormalisedSmythBroby2, <:Any},
                    X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber0, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:PortfolioOptimisers.NormalisedSmythBrobyGerber0,
                                             <:Any}, X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber1, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:PortfolioOptimisers.NormalisedSmythBrobyGerber1,
                                             <:Any}, X::AbstractMatrix)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber2, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
PortfolioOptimisers.smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:PortfolioOptimisers.NormalisedSmythBrobyGerber2,
                                             <:Any}, X::AbstractMatrix)
cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:PortfolioOptimisers.UnNormalisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:PortfolioOptimisers.UnNormalisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:PortfolioOptimisers.NormalisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:PortfolioOptimisers.NormalisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
```
