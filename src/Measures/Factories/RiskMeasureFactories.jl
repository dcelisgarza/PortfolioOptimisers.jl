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
function uncertainty_set_factory(::NoUncertaintySet, ::NoUncertaintySet)
    return NoUncertaintySet()
end
function uncertainty_set_factory(::NoUncertaintySet,
                                 prior_uncertainty_set::Union{<:BoxUncertaintySet,
                                                              <:EllipseUncertaintySet})
    return prior_uncertainty_set
end
function uncertainty_set_factory(risk_uncertainty_set::Union{<:BoxUncertaintySet,
                                                             <:EllipseUncertaintySet},
                                 ::Any)
    return risk_uncertainty_set
end

risk_measures = union(subtypes(TargetRiskMeasure), subtypes(TargetHierarchicalRiskMeasure),
                      subtypes(TargetNoOptimisationRiskMeasure))
risk_measures = risk_measures[isstructtype.(risk_measures)]
for risk_measure ∈ risk_measures
    eval(quote
             function risk_measure_factory(r::$(risk_measure); prior::AbstractPriorModel,
                                           kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(risk_measure)(; settings = r.settings, target = r.target, w = r.w,
                                        mu = mu)
             end
             function cluster_risk_measure_factory(r::$(risk_measure);
                                                   prior::AbstractPriorModel,
                                                   cluster::AbstractVector, kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, cluster)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 return $(risk_measure)(; settings = r.settings, target = target, w = r.w,
                                        mu = mu)
             end
         end)
end

risk_measures = union(subtypes(SolverRiskMeasure), subtypes(SolverHierarchicalRiskMeasure))
risk_measures = risk_measures[isstructtype.(risk_measures)]
for risk_measure ∈ risk_measures
    eval(quote
             function risk_measure_factory(r::$(risk_measure);
                                           solvers::Union{Nothing, <:Solver,
                                                          <:AbstractVector{<:Solver}},
                                           kwargs...)
                 solvers = risk_measure_solver_factory(r.solvers, solvers)
                 return $(risk_measure)(; settings = r.settings, alpha = r.alpha,
                                        beta = r.beta, solvers = solvers)
             end
             function cluster_risk_measure_factory(r::$(risk_measure);
                                                   solvers::Union{Nothing, <:Solver,
                                                                  <:AbstractVector{<:Solver}},
                                                   kwargs...)
                 return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
             end
         end)
end

function _get_sk(::Union{<:NegativeQuadraticSemiSkewness, <:NegativeSemiSkewness},
                 prior::HighOrderPriorModel)
    return prior.ssk
end
function _get_sk(::Union{<:NegativeQuadraticSkewness, <:NegativeSkewness},
                 prior::HighOrderPriorModel)
    return prior.sk
end
function _get_V(::Union{<:NegativeQuadraticSemiSkewness, <:NegativeSemiSkewness},
                prior::HighOrderPriorModel)
    return prior.SV
end
function _get_V(::Union{<:NegativeQuadraticSkewness, <:NegativeSkewness},
                prior::HighOrderPriorModel)
    return prior.V
end
function _get_smp(::Union{<:NegativeQuadraticSemiSkewness, <:NegativeSemiSkewness},
                  prior::HighOrderPriorModel)
    return prior.sskmp
end
function _get_smp(::Union{<:NegativeQuadraticSkewness, <:NegativeSkewness},
                  prior::HighOrderPriorModel)
    return prior.skmp
end
risk_measures = subtypes(SkewRiskMeasure)
risk_measures = risk_measures[isstructtype.(risk_measures)]
for risk_measure ∈ risk_measures
    eval(quote
             function _risk_measure_factory(r::$(risk_measure), prior::HighOrderPriorModel)
                 sk = risk_measure_nothing_matrix_factory(r.sk,
                                                          _get_sk($(risk_measure), prior))
                 V = risk_measure_nothing_matrix_factory(r.V,
                                                         _get_V($(risk_measure), prior))
                 return $(risk_measure)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
             function _risk_measure_factory(r::$(risk_measure),
                                            ::AbstractLowOrderPriorModel)
                 sk = risk_measure_nothing_matrix_factory(r.sk, nothing)
                 V = risk_measure_nothing_matrix_factory(r.V, nothing)
                 return $(risk_measure)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
             function cluster_negative_skewness(::$(risk_measure){<:Any, <:Any, Nothing,
                                                                  <:Any}, ::Nothing,
                                                prior::AbstractPriorModel,
                                                cluster::AbstractVector)
                 throw(ArgumentError("Neither the risk measure, nor the prior have the required data."))
             end
             function cluster_negative_skewness(skew_rm::$(risk_measure){<:Any, <:Any,
                                                                         <:AbstractMatrix,
                                                                         <:Any},
                                                prior::AbstractPriorModel,
                                                cluster::AbstractVector)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 sk = view(skew_rm.sk, cluster, idx)
                 V = __coskewness(sk, prior.X, skew_rm.mp)
                 if all(iszero.(diag(V)))
                     V += eps(eltype(sk)) * I
                 end
                 return sk, V
             end
             function cluster_negative_skewness(::$(risk_measure){<:Any, <:Any, Nothing,
                                                                  <:Any},
                                                prior::HighOrderPriorModel{<:Any, <:Any,
                                                                           <:Any, <:Any,
                                                                           <:Any, <:Any,
                                                                           <:AbstractMatrix,
                                                                           <:AbstractMatrix,
                                                                           <:MatrixProcessing},
                                                cluster::AbstractVector)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 sk = view(_get_sk($(risk_measure), prior), cluster, idx)
                 V = __coskewness(sk, prior.X, _get_smp($(risk_measure), prior))
                 if all(iszero.(diag(V)))
                     V += eps(eltype(sk)) * I
                 end
                 return sk, V
             end
             function risk_measure_factory(r::$(risk_measure); prior::AbstractPriorModel,
                                           kwargs...)
                 return _risk_measure_factory(r, prior)
             end
             function cluster_risk_measure_factory(r::$(risk_measure);
                                                   prior::AbstractPriorModel,
                                                   cluster::AbstractVector, kwargs...)
                 sk, V = cluster_negative_skewness(r, prior, cluster)
                 return $(risk_measure)(; settings = r.settings, mp = r.mp, sk = sk, V = V)
             end
         end)
end

function _get_kt(::SquareRootSemiKurtosis, prior::HighOrderPriorModel)
    return prior.skt
end
function _get_kt(::SquareRootKurtosis, prior::HighOrderPriorModel)
    return prior.kt
end
risk_measures = subtypes(KurtosisRiskMeasure)
risk_measures = risk_measures[isstructtype.(risk_measures)]
for risk_measure ∈ risk_measures
    eval(quote
             function _risk_measure_factory(r::$(risk_measure), prior::HighOrderPriorModel)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 kt = risk_measure_nothing_matrix_factory(r.kt,
                                                          _get_kt($(risk_measure), prior))
                 return $(risk_measure)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function _risk_measure_factory(r::$(risk_measure),
                                            prior::AbstractLowOrderPriorModel)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 kt = risk_measure_nothing_matrix_factory(r.kt, nothing)
                 return $(risk_measure)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function _cluster_risk_measure_factory(r::$(risk_measure),
                                                    prior::HighOrderPriorModel,
                                                    cluster::AbstractVector)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 kt = risk_measure_nothing_matrix_factory(r.kt,
                                                          _get_kt($(risk_measure), prior),
                                                          idx)
                 return $(risk_measure)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function _cluster_risk_measure_factory(r::$(risk_measure),
                                                    prior::AbstractLowOrderPriorModel,
                                                    cluster::AbstractVector)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
                 idx = fourth_moment_cluster_index_factory(size(prior.X, 2), cluster)
                 kt = risk_measure_nothing_matrix_factory(r.kt, nothing, idx)
                 return $(risk_measure)(; settings = r.settings, w = r.w, mu = mu, kt = kt)
             end
             function risk_measure_factory(r::$(risk_measure); prior::AbstractPriorModel,
                                           kwargs...)
                 return _risk_measure_factory(r, prior)
             end
             function cluster_risk_measure_factory(r::$(risk_measure);
                                                   prior::AbstractPriorModel,
                                                   cluster::AbstractVector, kwargs...)
                 return _cluster_risk_measure_factory(r, prior, cluster)
             end
         end)
end
