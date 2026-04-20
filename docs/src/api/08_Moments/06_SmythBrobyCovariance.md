# Smyth-Broby Covariance

```@docs
SmythBroby0
SmythBroby1
SmythBroby2
SmythBrobyGerber0
SmythBrobyGerber1
SmythBrobyGerber2
SmythBrobyCovariance
cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
BaseSmythBrobyCovariance
SmythBrobyCovarianceAlgorithm
sb_delta
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby0, <:Any},
                    X::MatNum, mu::ArrNum, sd::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby1, <:Any},
                    X::MatNum, mu::ArrNum, sd::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby2, <:Any},
                    X::MatNum, mu::ArrNum, sd::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber0, <:Any},
                    X::MatNum, mu::ArrNum, sd::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber1, <:Any},
                    X::MatNum, mu::ArrNum, sd::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber2, <:Any},
                    X::MatNum, mu::ArrNum, sd::ArrNum)
```
