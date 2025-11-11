# Smyth-Broby Covariance

```@docs
SmythBroby0
SmythBroby1
SmythBroby2
SmythBrobyGerber0
SmythBrobyGerber1
SmythBrobyGerber2
StandardisedSmythBroby0
StandardisedSmythBroby1
StandardisedSmythBroby2
StandardisedSmythBrobyGerber0
StandardisedSmythBrobyGerber1
StandardisedSmythBrobyGerber2
SmythBrobyCovariance
cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                             <:UnstandardisedSmythBrobyCovarianceAlgorithm,
                             <:Any}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                             <:UnstandardisedSmythBrobyCovarianceAlgorithm,
                             <:Any}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
BaseSmythBrobyCovariance
SmythBrobyCovarianceAlgorithm
UnstandardisedSmythBrobyCovarianceAlgorithm
StandardisedSmythBrobyCovarianceAlgorithm
sb_delta
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby0, <:Any},
                    X::MatNum, mean_vec::ArrNum, std_vec::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBroby0, <:Any},
                    X::MatNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby1, <:Any},
                    X::MatNum, mean_vec::ArrNum, std_vec::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBroby1, <:Any},
                    X::MatNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby2, <:Any},
                    X::MatNum, mean_vec::ArrNum, std_vec::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBroby2, <:Any},
                    X::MatNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber0, <:Any},
                    X::MatNum, mean_vec::ArrNum, std_vec::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBrobyGerber0,
                                             <:Any}, X::MatNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber1, <:Any},
                    X::MatNum, mean_vec::ArrNum, std_vec::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBrobyGerber1,
                                             <:Any}, X::MatNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber2, <:Any},
                    X::MatNum, mean_vec::ArrNum, std_vec::ArrNum)
smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBrobyGerber2,
                                             <:Any}, X::MatNum)
```
