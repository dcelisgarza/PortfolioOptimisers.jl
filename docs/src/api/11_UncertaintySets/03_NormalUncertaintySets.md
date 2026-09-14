# Normal Uncertainty Sets

```@docs
NormalUncertaintySet
ucs(ue::NormalUncertaintySet, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                      <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing,
                                 <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing,
                                 <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                      <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                         <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                         <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any,
                                            <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                            <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing,
                                 <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                 <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                 <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing,
                                    <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                    <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                    <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                       <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
commutation_matrix
choose_scaling_parameter(ue::NormalUncertaintySet, pr::AbstractPriorResult)
normal_mu_error_sample
normal_sigma_error_sample
mu_asymptotic_cov
sigma_asymptotic_cov
mu_normal_box_set
sigma_normal_box_set
normal_box_preamble
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
