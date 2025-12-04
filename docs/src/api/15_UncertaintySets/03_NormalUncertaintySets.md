# Normal Uncertainty Sets

```@docs
NormalUncertaintySet
ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                      <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::NormalUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                      <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                         <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                         <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                            <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
commutation_matrix
```
