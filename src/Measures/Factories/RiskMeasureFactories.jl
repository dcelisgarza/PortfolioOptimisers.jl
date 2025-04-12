function risk_measure_nothing_vec_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real}, ::Any)
    return risk_variable
end
function risk_measure_nothing_vec_factory(::Nothing, prior_variable::AbstractVector{<:Real})
    return prior_variable
end
function risk_measure_nothing_matrix_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real}, ::Any)
    return risk_variable
end
function risk_measure_nothing_matrix_factory(::Nothing,
                                             prior_variable::AbstractMatrix{<:Real})
    return prior_variable
end
function risk_measure_nothing_real_vec_factory(risk_variable::AbstractVector{<:Real},
                                               cluster::AbstractVector)
    return view(risk_variable, cluster)
end
function risk_measure_nothing_real_vec_factory(risk_variable::Union{Nothing, Real},
                                               ::AbstractVector)
    return risk_variable
end
function ucs_factory(::Nothing, ::Nothing)
    return nothing
end
function ucs_factory(::Nothing,
                     prior_ucs::Union{<:AbstractUncertaintySet,
                                      <:AbstractUncertaintySetEstimator})
    return prior_ucs
end
function ucs_factory(risk_ucs::Union{<:AbstractUncertaintySet,
                                     <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end

risks = (Skewness, SemiSkewness, Kurtosis, SemiKurtosis)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), prior::AbstractPriorResult, args...;
                                           kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(r)(; settings = r.settings, ve = r.ve, target = r.target, w = r.w,
                             mu = mu)
             end
             function cluster_risk_measure_factory(r::$(r), prior::AbstractPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(r)(; settings = r.settings, ve = r.ve, target = target, w = r.w,
                             mu = mu)
             end
             function risk_measure_factory(r::$(r), prior::EntropyPoolingResult, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(r)(; settings = r.settings, ve = r.ve, target = r.target, w = w,
                             mu = mu)
             end
             function cluster_risk_measure_factory(r::$(r), prior::EntropyPoolingResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(r)(; settings = r.settings, ve = r.ve, target = target, w = w,
                             mu = mu)
             end
             function risk_measure_factory(r::$(r),
                                           prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any}, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(r)(; settings = r.settings, ve = r.ve, target = r.target, w = w,
                             mu = mu)
             end
             function cluster_risk_measure_factory(r::$(r),
                                                   prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                               <:Any, <:Any,
                                                                               <:Any,
                                                                               <:Any},
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(r)(; settings = r.settings, ve = r.ve, target = target, w = w,
                             mu = mu)
             end
         end)
end

risks = (ThirdCentralMoment, ThirdLowerPartialMoment, FourthCentralMoment,
         FourthLowerPartialMoment, FirstLowerPartialMoment, SemiStandardDeviation)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), prior::AbstractPriorResult, args...;
                                           kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(r)(; settings = r.settings, target = r.target, w = r.w, mu = mu)
             end
             function cluster_risk_measure_factory(r::$(r), prior::AbstractPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(r)(; settings = r.settings, target = target, w = r.w, mu = mu)
             end
             function risk_measure_factory(r::$(r), prior::EntropyPoolingResult, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(r)(; settings = r.settings, target = r.target, w = w, mu = mu)
             end
             function cluster_risk_measure_factory(r::$(r), prior::EntropyPoolingResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(r)(; settings = r.settings, target = target, w = w, mu = mu)
             end
             function risk_measure_factory(r::$(r),
                                           prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any}, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(r)(; settings = r.settings, target = r.target, w = w, mu = mu)
             end
             function cluster_risk_measure_factory(r::$(r),
                                                   prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                               <:Any, <:Any,
                                                                               <:Any,
                                                                               <:Any},
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(r)(; settings = r.settings, target = target, w = w, mu = mu)
             end
         end)
end

risks = (ValueatRisk, ValueatRiskRange, DrawdownatRisk, EqualRiskMeasure,
         RelativeConditionalDrawdownatRisk, RelativeDrawdownatRisk, RelativeMaximumDrawdown,
         UlcerIndex, BrownianDistanceVariance, ConditionalValueatRiskRange,
         GiniMeanDifference, Range, TailGiniRange, ConditionalValueatRisk,
         DistributionallyRobustConditionalValueatRisk, TailGini, WorstRealisation,
         ConditionalDrawdownatRisk, MaximumDrawdown, TurnoverRiskMeasure)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), args...; kwargs...)
                 return r
             end
             function cluster_risk_measure_factory(r::$(r), args...; kwargs...)
                 return r
             end
         end)
end

risks = (AverageDrawdown, RelativeAverageDrawdown)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), args...; kwargs...)
                 return $(r)(; settings = r.settings, w = r.w)
             end
             function risk_measure_factory(r::$(r), prior::EntropyPoolingResult, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 return $(r)(; settings = r.settings, w = w)
             end
             function risk_measure_factory(r::$(r),
                                           prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any}, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 return $(r)(; settings = r.settings, w = w)
             end
             function cluster_risk_measure_factory(r::$(r), args...; kwargs...)
                 return r
             end
         end)
end

risks = (RelativeEntropicDrawdownatRisk, EntropicValueatRisk, EntropicDrawdownatRisk)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), ::Any,
                                           slv::Union{Nothing, <:Solver,
                                                      <:AbstractVector{<:Solver}}, args...;
                                           kwargs...)
                 slv = risk_measure_solver_factory(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, slv = slv)
             end
             function cluster_risk_measure_factory(r::$(r), ::Any, ::Any,
                                                   slv::Union{Nothing, <:Solver,
                                                              <:AbstractVector{<:Solver}},
                                                   args...; kwargs...)
                 return risk_measure_factory(r; slv = slv, kwargs = kwargs)
             end
         end)
end

risks = (RelativeRelativisticDrawdownatRisk, RelativisticValueatRisk,
         RelativisticDrawdownatRisk)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), ::Any,
                                           slv::Union{Nothing, <:Solver,
                                                      <:AbstractVector{<:Solver}}, args...;
                                           kwargs...)
                 slv = risk_measure_solver_factory(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, kappa = r.kappa,
                             slv = slv)
             end
             function cluster_risk_measure_factory(r::$(r), ::Any, ::Any,
                                                   slv::Union{Nothing, <:Solver,
                                                              <:AbstractVector{<:Solver}},
                                                   args...; kwargs...)
                 return risk_measure_factory(r; slv = slv, kwargs = kwargs)
             end
         end)
end

function _get_sk(::Union{<:NegativeQuadraticSemiSkewness, <:NegativeSemiSkewness},
                 prior::HighOrderPriorResult)
    return prior.ssk
end
function _get_sk(::Union{<:NegativeQuadraticSkewness, <:NegativeSkewness},
                 prior::HighOrderPriorResult)
    return prior.sk
end
function _get_V(::Union{<:NegativeQuadraticSemiSkewness, <:NegativeSemiSkewness},
                prior::HighOrderPriorResult)
    return prior.SV
end
function _get_V(::Union{<:NegativeQuadraticSkewness, <:NegativeSkewness},
                prior::HighOrderPriorResult)
    return prior.V
end
function _get_smp(::Union{<:NegativeQuadraticSemiSkewness, <:NegativeSemiSkewness},
                  prior::HighOrderPriorResult)
    return prior.sskmp
end
function _get_smp(::Union{<:NegativeQuadraticSkewness, <:NegativeSkewness},
                  prior::HighOrderPriorResult)
    return prior.skmp
end
risks = (NegativeQuadraticSemiSkewness, NegativeSemiSkewness, NegativeQuadraticSkewness,
         NegativeSkewness)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), prior::HighOrderPriorResult, args...;
                                           kwargs...)
                 sk = risk_measure_nothing_matrix_factory(r.sk, _get_sk(r, prior))
                 V = risk_measure_nothing_matrix_factory(r.V, _get_V(r, prior))
                 return $(r)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
             function risk_measure_factory(r::$(r), ::AbstractLowOrderPriorResult, args...;
                                           kwargs...)
                 sk = risk_measure_nothing_matrix_factory(r.sk, nothing)
                 V = risk_measure_nothing_matrix_factory(r.V, nothing)
                 return $(r)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
             function cluster_risk_measure_factory(::$(r){<:Any, <:Any, Nothing, <:Any},
                                                   ::Nothing, prior::AbstractPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
             end
             function cluster_risk_measure_factory(skew_rm::$(r){<:Any, <:Any,
                                                                 <:AbstractMatrix, <:Any},
                                                   prior::AbstractPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 sk = view(skew_rm.sk, cluster, idx)
                 V = __coskewness(sk, prior.X, skew_rm.mp)
                 if all(iszero.(diag(V)))
                     V[diagind(V)] = I(size(V, 1))
                 end
                 return $(r)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
             function cluster_risk_measure_factory(::$(r){<:Any, <:Any, Nothing, <:Any},
                                                   prior::HighOrderPriorResult{<:Any, <:Any,
                                                                               <:Any, <:Any,
                                                                               <:Any, <:Any,
                                                                               <:AbstractMatrix,
                                                                               <:AbstractMatrix,
                                                                               <:AbstractMatrixProcessingEstimator},
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 sk = view(_get_sk(r, prior), cluster, idx)
                 V = __coskewness(sk, prior.X, _get_smp(r, prior))
                 if all(iszero.(diag(V)))
                     V[diagind(V)] = I(size(V, 1))
                 end
                 return $(r)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
         end)
end

function _get_kt(::SquareRootSemiKurtosis, prior::HighOrderPriorResult)
    return prior.skt
end
function _get_kt(::SquareRootKurtosis, prior::HighOrderPriorResult)
    return prior.kt
end
risks = (SquareRootSemiKurtosis, SquareRootKurtosis)
for r ∈ risks
    eval(quote
             function risk_measure_factory(r::$(r), prior::HighOrderPriorResult, args...;
                                           kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 kt = risk_measure_nothing_matrix_factory(r.kt, _get_kt(r, prior))
                 return $(r)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function risk_measure_factory(r::$(r),
                                           prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any}, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 kt = risk_measure_nothing_matrix_factory(r.kt, _get_kt(r, prior))
                 return $(r)(; settings = r.settings, w = w, mu = mu, kt = kt)
             end
             function risk_measure_factory(r::$(r), prior::AbstractLowOrderPriorResult,
                                           args...; kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 kt = risk_measure_nothing_matrix_factory(r.kt, nothing)
                 return $(r)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function risk_measure_factory(r::$(r), prior::EntropyPoolingResult, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 kt = risk_measure_nothing_matrix_factory(r.kt, nothing)
                 return $(r)(; settings = r.settings, w = w, mu = mu, kt = kt)
             end
             function cluster_risk_measure_factory(r::$(r){<:Any, <:Any, <:Any, <:Nothing},
                                                   prior::AbstractPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
             end
             function cluster_risk_measure_factory(r::$(r){<:Any, <:Any, <:Any,
                                                           <:AbstractMatrix},
                                                   prior::AbstractPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 kt = risk_measure_nothing_matrix_factory(r.kt, nothing, idx)
                 return $(r)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function cluster_risk_measure_factory(r::$(r){<:Any, <:Any, <:Any,
                                                           <:AbstractMatrix},
                                                   prior::EntropyPoolingResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 kt = risk_measure_nothing_matrix_factory(r.kt, nothing, idx)
                 return $(r)(; settings = r.settings, w = w, mu = mu, kt = kt)
             end
             function cluster_risk_measure_factory(r::$(r), prior::HighOrderPriorResult,
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 kt = risk_measure_nothing_matrix_factory(r.kt, _get_kt(r, prior), idx)
                 return $(r)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function cluster_risk_measure_factory(r::$(r),
                                                   prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                               <:Any, <:Any,
                                                                               <:Any,
                                                                               <:Any},
                                                   cluster::AbstractVector, args...;
                                                   kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 kt = risk_measure_nothing_matrix_factory(r.kt, _get_kt(r, prior), idx)
                 return $(r)(; settings = r.settings, w = w, mu = mu, kt = kt)
             end
         end)
end
function risk_measure_factory(rs::AbstractVector{<:OptimisationRiskMeasure}, args...;
                              kwargs...)
    return risk_measure_factory.(rs, args...; kwargs...)
end
function cluster_risk_measure_factory(rs::AbstractVector{<:OptimisationRiskMeasure},
                                      args...; kwargs...)
    return cluster_risk_measure_factory.(rs, args...; kwargs...)
end
