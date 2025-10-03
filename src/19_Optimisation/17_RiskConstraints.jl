function set_risk_upper_bound!(args...)
    return nothing
end
#! Using parameters to set the upper bounds would make things more difficult from a user perspective. Keep an eye on this in case things change in the future. We could simplify solve_mean_risk! and solve_noc! for pareto frontiers, we can define ub as a parameter and update it for subsequent solves.
# Solver(; name = :clarabel2,
#        solver = () -> ParametricOptInterface.Optimizer(MOI.instantiate(Clarabel.Optimizer;
#                                                                        with_cache_type = Float64)),
#        check_sol = (; allow_local = true, allow_almost = true),
#        settings = Dict("verbose" => false, "max_step_fraction" => 0.75))
# https://discourse.julialang.org/t/solver-attributes-and-set-optimizer-with-parametricoptinterface-jl-and-jump-jl/129935/8?u=dcelisgarza
function set_risk_upper_bound!(model::JuMP.Model,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               r_expr::AbstractJuMPScalar,
                               ub::Union{<:AbstractVector, <:Frontier}, key)
    bound_key = Symbol(key, :_ub)
    if !haskey(model, :risk_frontier)
        risk_frontier = @expression(model, risk_frontier,
                                    Pair{Symbol,
                                         Tuple{<:AbstractJuMPScalar,
                                               <:Union{<:AbstractVector, <:Frontier}}}[bound_key => (r_expr,
                                                                                                     ub)])
    else
        risk_frontier = model[:risk_frontier]
        push!(risk_frontier, bound_key => (r_expr, ub))
    end
    return nothing
end
function set_risk_upper_bound!(model::JuMP.Model,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               r_expr::AbstractJuMPScalar, ub::Real, key)
    k = model[:k]
    sc = model[:sc]
    bound_key = Symbol(key, :_ub)
    model[bound_key] = @constraint(model, sc * (r_expr - ub * k) <= 0)
    return nothing
end
function set_risk_expression!(model::JuMP.Model, r_expr::AbstractJuMPScalar, scale::Real,
                              rke::Bool)
    if !rke
        return nothing
    end
    if !haskey(model, :risk_vec)
        @expression(model, risk_vec, Union{AffExpr, QuadExpr}[])
    end
    risk_vec = model[:risk_vec]
    push!(risk_vec, scale * r_expr)
    return nothing
end
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                                    <:RiskBudgeting},
                                         r_expr::AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_variance_risk_bounds_and_expression!(model::JuMP.Model,
                                                  opt::Union{<:MeanRisk,
                                                             <:NearOptimalCentering,
                                                             <:RiskBudgeting,
                                                             <:FactorRiskContribution},
                                                  r_expr_ub::AbstractJuMPScalar,
                                                  ub::Union{Nothing, <:Real,
                                                            <:AbstractVector, <:Frontier},
                                                  key::Symbol, r_expr::AbstractJuMPScalar,
                                                  settings::RiskMeasureSettings)
    set_risk_upper_bound!(model, opt, r_expr_ub, ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation,
                   opt::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                   pr::AbstractPriorResult, args...; kwargs...)
    key = Symbol(:sd_risk_, i)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    sd_risk = model[key] = @variable(model)
    model[Symbol(:csd_risk_soc_, i)] = @constraint(model,
                                                   [sc * sd_risk; sc * G * w] in
                                                   SecondOrderCone())
    return sd_risk, key
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    sd_risk, key = set_risk!(model, i, r, opt, pr, args...; kwargs...)
    set_risk_bounds_and_expression!(model, opt, sd_risk, r.settings, key)
    return nothing
end
function sdp_rc_variance_flag!(::JuMP.Model,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               ::Nothing)
    return false
end
function sdp_rc_variance_flag!(::JuMP.Model,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               ::LinearConstraint)
    return true
end
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool,
                            cplg::Union{Nothing, <:SemiDefinitePhylogeny,
                                        <:IntegerPhylogeny},
                            nplg::Union{Nothing, <:SemiDefinitePhylogeny,
                                        <:IntegerPhylogeny})
    return if rc_flag ||
              haskey(model, :rc_variance) ||
              isa(cplg, SemiDefinitePhylogeny) ||
              isa(nplg, SemiDefinitePhylogeny)
        true
    else
        false
    end
end
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance, pr::AbstractPriorResult,
                            flag::Bool, key::Symbol)
    return if flag
        set_sdp_variance_risk!(model, i, r, pr, key)
    else
        set_variance_risk!(model, i, r, pr, key)
    end
end
function set_sdp_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                                pr::AbstractPriorResult, key::Symbol)
    W = set_sdp_constraints!(model)
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = @expression(model, sigma * W)
    model[key] = @expression(model, tr(sigma_W))
    return model[key] = @expression(model, tr(sigma_W))
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:SOCRiskExpr},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    key_dev = Symbol(:dev_, i)
    dev = model[key_dev] = @variable(model)
    model[Symbol(key_dev, :_soc)] = @constraint(model,
                                                [sc * dev; sc * G * w] in SecondOrderCone())
    return model[key] = @expression(model, dev^2)
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:QuadRiskExpr},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    G = isnothing(r.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(r.sigma).U
    dev = model[Symbol(:dev_, i)] = @variable(model)
    model[Symbol(:cdev_soc_, i)] = @constraint(model,
                                               [sc * dev; sc * G * w] in SecondOrderCone())
    return model[key] = @expression(model, dot(w, sigma, w))
end
function variance_risk_bounds_expr(model::JuMP.Model, i::Any, flag::Bool)
    return if flag
        key = Symbol(:variance_risk_, i)
        model[key], key
    else
        key = Symbol(:dev_, i)
        model[key], key
    end
end
function variance_risk_bounds_val(flag::Bool, ub::Frontier)
    return _Frontier(; N = ub.N, factor = 1, flag = flag)
end
function variance_risk_bounds_val(flag::Bool, ub::AbstractVector)
    return flag ? ub : sqrt.(ub)
end
function variance_risk_bounds_val(flag::Bool, ub::Real)
    return flag ? ub : sqrt(ub)
end
function variance_risk_bounds_val(::Any, ::Nothing)
    return nothing
end
function rc_variance_constraints!(args...)
    return nothing
end
function rc_variance_constraints!(model::JuMP.Model, i::Any, rc::LinearConstraint,
                                  variance_risk::AbstractJuMPScalar)
    sigma_W = model[Symbol(:sigma_W_, i)]
    sc = model[:sc]
    if !haskey(model, :rc_variance)
        @expression(model, rc_variance, true)
    end
    rc_key = Symbol(:rc_variance_, i)
    vsw = vec(diag(sigma_W))
    if !isnothing(rc.A_ineq)
        model[Symbol(rc_key, :_ineq)] = @constraint(model,
                                                    sc * (rc.A_ineq * vsw -
                                                          rc.B_ineq * variance_risk) <= 0)
    end
    if !isnothing(rc.A_eq)
        model[Symbol(rc_key, :_eq)] = @constraint(model,
                                                  sc *
                                                  (rc.A_eq * vsw - rc.B_eq * variance_risk) ==
                                                  0)
    end
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any, r::Variance,
                   opt::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                   pr::AbstractPriorResult,
                   cplg::Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                   nplg::Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                   args...; kwargs...)
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, cplg, nplg)
    key = Symbol(:variance_risk_, i)
    variance_risk = set_variance_risk!(model, i, r, pr, sdp_flag, key)
    rc_variance_constraints!(model, i, rc, variance_risk)
    return variance_risk, sdp_flag
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               cplg::Union{Nothing, <:SemiDefinitePhylogeny,
                                           <:IntegerPhylogeny},
                               nplg::Union{Nothing, <:SemiDefinitePhylogeny,
                                           <:IntegerPhylogeny}, args...; kwargs...)
    if !haskey(model, :variance_flag)
        @expression(model, variance_flag, true)
    end
    variance_risk, sdp_flag = set_risk!(model, i, r, opt, pr, cplg, nplg, args...;
                                        kwargs...)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, sdp_flag)
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance,
                               opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any,
                               ::Any, b1::AbstractMatrix, args...; kwargs...)
    if !haskey(model, :variance_flag)
        @expression(model, variance_flag, true)
    end
    rc = linear_constraints(r.rc, opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    key = Symbol(:variance_risk_, i)
    set_sdp_frc_constraints!(model)
    W = model[:W]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = @expression(model,
                                                        transpose(b1) * sigma * b1 * W)
    variance_risk = model[key] = @expression(model, tr(sigma_W))
    rc_variance_constraints!(model, i, rc, variance_risk)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, true)
    ub = variance_risk_bounds_val(true, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return nothing
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::BoxUncertaintySet, args...)
    if !haskey(model, :Au)
        sc = model[:sc]
        W = model[:W]
        N = size(W, 1)
        @variables(model, begin
                       Au[1:N, 1:N] >= 0, Symmetric
                       Al[1:N, 1:N] >= 0, Symmetric
                   end)
        @constraint(model, cbucs_variance, sc * (Au - Al - W) == 0)
    end
    key = Symbol(:bucs_variance_risk_, i)
    Au = model[:Au]
    Al = model[:Al]
    ub = ucs.ub
    lb = ucs.lb
    ucs_variance_risk = model[key] = @expression(model, tr(Au * ub) - tr(Al * lb))
    return ucs_variance_risk, key
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::EllipseUncertaintySet,
                                sigma::AbstractMatrix)
    sc = model[:sc]
    if !haskey(model, :E)
        W = model[:W]
        N = size(W, 1)
        @variable(model, E[1:N, 1:N], Symmetric)
        @expression(model, WpE, W + E)
        @constraint(model, ceucs_variance, sc * E in PSDCone())
    end
    key = Symbol(:eucs_variance_risk_, i)
    WpE = model[:WpE]
    k = ucs.k
    G = cholesky(ucs.sigma).U
    t_eucs = model[Symbol(:t_eucs, i)] = @variable(model)
    x_eucs, ucs_variance_risk = model[Symbol(:x_eucs, i)], model[key] = @expressions(model,
                                                                                     begin
                                                                                         G *
                                                                                         vec(WpE)
                                                                                         tr(sigma *
                                                                                            WpE) +
                                                                                         k *
                                                                                         t_eucs
                                                                                     end)
    model[Symbol(:ge_soc, i)] = @constraint(model,
                                            [sc * t_eucs; sc * x_eucs] in SecondOrderCone())
    return ucs_variance_risk, key
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; rd::ReturnsResult, kwargs...)
    if !haskey(model, :variance_flag)
        @expression(model, variance_flag, true)
    end
    set_sdp_constraints!(model)
    ucs = r.ucs
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    ucs_variance_risk, key = set_ucs_variance_risk!(model, i, sigma_ucs(ucs, rd; kwargs...),
                                                    sigma)
    set_risk_bounds_and_expression!(model, opt, ucs_variance_risk, r.settings, key)
    return nothing
end
function calc_risk_constraint_target(::LowOrderMoment{<:Any, <:Any, Nothing, <:Any},
                                     w::AbstractVector, mu::AbstractVector, args...)
    return dot(w, mu)
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}, w::AbstractVector, args...)
    return dot(w, r.mu)
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:Real, <:Any}, ::Any,
                                     ::Any, k)
    return r.mu * k
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:flm_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    flm = model[Symbol(:flm_, i)] = @variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    flm_risk = model[key] = if isnothing(wi)
        @expression(model, mean(flm))
    else
        @expression(model, mean(flm, wi))
    end
    model[Symbol(:cflm_mar_, i)] = @constraint(model, sc * ((net_X + flm) .- target) >= 0)
    set_risk_bounds_and_expression!(model, opt, flm_risk, r.settings, key)
    return nothing
end
function set_second_moment_risk!(model::JuMP.Model, ::QuadRiskExpr, ::Any, factor::Real,
                                 second_moment, key::Symbol, args...)
    return model[key] = @expression(model, factor * dot(second_moment, second_moment)),
                        sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::RSOCRiskExpr, i::Any, factor::Real,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 args...)
    net_X = model[:net_X]
    sc = model[:sc]
    tsecond_moment = model[Symbol(keyt, i)] = @variable(model)
    model[Symbol(keyc, i)] = @constraint(model,
                                         [sc * tsecond_moment;
                                          0.5;
                                          sc * second_moment] in RotatedSecondOrderCone())
    return model[key] = @expression(model, factor * tsecond_moment), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SOCRiskExpr, i::Any, factor::Real,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 tsecond_moment::AbstractJuMPScalar)
    return model[key] = @expression(model, factor * tsecond_moment^2), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SqrtRiskExpr, i::Any, factor::Real,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 tsecond_moment::AbstractJuMPScalar)
    factor = sqrt(factor)
    return model[key] = @expression(model, factor * tsecond_moment), factor
end
function second_moment_bound_val(alg::SecondMomentAlgorithm, ub::Frontier, factor::Real)
    return _Frontier(; N = ub.N, factor = inv(factor), flag = isa(alg, SqrtRiskExpr))
end
function second_moment_bound_val(alg::SecondMomentAlgorithm, ub::AbstractVector,
                                 factor::Real)
    return inv(factor) * (isa(alg, SqrtRiskExpr) ? ub : sqrt.(ub))
end
function second_moment_bound_val(alg::SecondMomentAlgorithm, ub::Real, factor::Real)
    return inv(factor) * (isa(alg, SqrtRiskExpr) ? ub : sqrt(ub))
end
function second_moment_bound_val(::Any, ::Nothing, ::Any)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:LowOrderDeviation{<:Any,
                                                                     <:SecondLowerMoment}},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:second_lower_moment_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    bound_key = Symbol(:sqrt_lower_central_moment_, i)
    sqrt_second_lower_moment, second_lower_moment = model[bound_key], model[Symbol(:second_lower_moment_, i)] = @variables(model,
                                                                                                                           begin
                                                                                                                               ()
                                                                                                                               [1:T],
                                                                                                                               (lower_bound = 0)
                                                                                                                           end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    second_lower_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        set_second_moment_risk!(model, r.alg.alg.alg, i, factor, second_lower_moment, key,
                                :tsecond_lower_moment_, :csecond_lower_moment_rsoc_,
                                sqrt_second_lower_moment)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        scaled_second_lower_moment = model[Symbol(:scaled_second_lower_moment_, i)] = @expression(model,
                                                                                                  dot(wi,
                                                                                                      second_lower_moment))
        set_second_moment_risk!(model, r.alg.alg.alg, i, factor, scaled_second_lower_moment,
                                key, :tsecond_lower_moment_, :csecond_lower_moment_rsoc_,
                                sqrt_second_lower_moment)
    end
    model[Symbol(:csqrt_second_central_moment_soc_, i)], model[Symbol(:csecond_lower_moment_mar_, i)] = @constraints(model,
                                                                                                                     begin
                                                                                                                         [sc *
                                                                                                                          sqrt_second_lower_moment
                                                                                                                          sc *
                                                                                                                          second_lower_moment] in
                                                                                                                         SecondOrderCone()
                                                                                                                         sc *
                                                                                                                         ((net_X +
                                                                                                                           second_lower_moment) .-
                                                                                                                          target) >=
                                                                                                                         0
                                                                                                                     end)
    ub = second_moment_bound_val(r.alg.alg.alg, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_lower_moment, ub,
                                             bound_key, second_lower_moment_risk,
                                             r.settings)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:LowOrderDeviation{<:Any,
                                                                     <:SecondCentralMoment}},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:second_central_moment_risk_, i)
    w = model[:w]
    k = model[:k]
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    bound_key = Symbol(:sqrt_second_central_moment_, i)
    sqrt_second_central_moment = model[bound_key] = @variable(model)
    second_central_moment = model[Symbol(:second_central_moment_, i)] = @expression(model,
                                                                                    net_X .-
                                                                                    target)
    sc = model[:sc]
    wi = nothing_scalar_array_factory(r.w, pr.w)
    second_central_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        set_second_moment_risk!(model, r.alg.alg.alg, i, factor, second_central_moment, key,
                                :tsecond_central_moment_, :csecond_central_moment_rsoc_,
                                sqrt_second_central_moment)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        scaled_second_central_moment = model[Symbol(:scaled_second_central_moment_, i)] = @expression(model,
                                                                                                      dot(wi,
                                                                                                          second_central_moment))
        set_second_moment_risk!(model, r.alg.alg.alg, i, factor,
                                scaled_second_central_moment, key, :tsecond_central_moment_,
                                :csecond_central_moment_rsoc_, sqrt_second_central_moment)
    end
    model[Symbol(:csqrt_second_central_moment_soc, i)] = @constraint(model,
                                                                     [sc *
                                                                      sqrt_second_central_moment
                                                                      sc *
                                                                      second_central_moment] in
                                                                     SecondOrderCone())
    ub = second_moment_bound_val(r.alg.alg.alg, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_central_moment, ub,
                                             bound_key, second_central_moment_risk,
                                             r.settings)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:MeanAbsoluteDeviation},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:mad_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    mad = model[Symbol(:mad_, i)] = @variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    mad_risk = model[Symbol(:mad_risk_, i)] = if isnothing(wi)
        @expression(model, mean(2 * mad))
    else
        @expression(model, mean(2 * mad, wi))
    end
    model[Symbol(:cmar_mad_, i)] = @constraint(model, sc * ((net_X + mad) .- target) >= 0)
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, key)
    return nothing
end
function set_wr_risk_expression!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :wr_risk)
        return model[:wr_risk]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    @variable(model, wr_risk)
    @constraint(model, cwr, sc * (wr_risk .+ net_X) >= 0)
    return wr_risk
end
function set_risk_constraints!(model::JuMP.Model, ::Any, r::WorstRealisation,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :wr_risk)
        return nothing
    end
    wr_risk = set_wr_risk_expression!(model, pr.X)
    set_risk_bounds_and_expression!(model, opt, wr_risk, r.settings, :wr_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, ::Any, r::Range,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :range_risk)
        return nothing
    end
    sc = model[:sc]
    wr_risk = set_wr_risk_expression!(model, pr.X)
    net_X = model[:net_X]
    @variable(model, br_risk)
    @expression(model, range_risk, wr_risk - br_risk)
    @constraint(model, cbr, sc * (br_risk .+ net_X) <= 0)
    set_risk_bounds_and_expression!(model, opt, range_risk, r.settings, :range_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:var_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_risk, z_var = model[key], model[Symbol(:z_var_, i)] = @variables(model,
                                                                         begin
                                                                             ()
                                                                             [1:T],
                                                                             (binary = true)
                                                                         end)
    alpha = r.alpha
    wi = nothing_scalar_array_factory(r.w, pr.w)
    if isnothing(wi)
        model[Symbol(:csvar_, i)] = @constraint(model,
                                                sc * (sum(z_var) - alpha * T + s * T) <= 0)
    else
        sw = sum(wi)
        model[Symbol(:csvar_, i)] = @constraint(model,
                                                sc *
                                                (dot(wi, z_var) - alpha * sw + s * sw) <= 0)
    end
    model[Symbol(:cvar_, i)] = @constraint(model,
                                           sc * ((net_X + b * z_var) .+ var_risk) >= 0)
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:MIPValueatRisk},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:var_range_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_risk_l, z_var_l, var_risk_h, z_var_h = model[Symbol(:var_risk_l_, i)], model[Symbol(:z_var_l_, i)], model[Symbol(:var_risk_h_, i)], model[Symbol(:z_var_h_, i)] = @variables(model,
                                                                                                                                                                                     begin
                                                                                                                                                                                         ()
                                                                                                                                                                                         [1:T],
                                                                                                                                                                                         (binary = true)
                                                                                                                                                                                         ()
                                                                                                                                                                                         [1:T],
                                                                                                                                                                                         (binary = true)
                                                                                                                                                                                     end)
    alpha = r.alpha
    beta = r.beta
    wi = nothing_scalar_array_factory(r.w, pr.w)
    if isnothing(wi)
        model[Symbol(:csvar_l_, i)], model[Symbol(:csvar_h_, i)] = @constraints(model,
                                                                                begin
                                                                                    sc *
                                                                                    (sum(z_var_l) -
                                                                                     alpha *
                                                                                     T +
                                                                                     s * T) <=
                                                                                    0
                                                                                    sc *
                                                                                    (sum(z_var_h) -
                                                                                     beta *
                                                                                     T +
                                                                                     s * T) <=
                                                                                    0
                                                                                end)
    else
        sw = sum(wi)
        model[Symbol(:csvar_l_, i)], model[Symbol(:csvar_h_, i)] = @constraints(model,
                                                                                begin
                                                                                    sc *
                                                                                    (dot(wi,
                                                                                         z_var_l) -
                                                                                     alpha *
                                                                                     sw +
                                                                                     s * sw) <=
                                                                                    0
                                                                                    sc *
                                                                                    (dot(wi,
                                                                                         z_var_h) -
                                                                                     beta *
                                                                                     sw +
                                                                                     s * sw) <=
                                                                                    0
                                                                                end)
    end
    model[Symbol(:cvar_, i)] = @constraints(model,
                                            begin
                                                sc *
                                                ((net_X + b * z_var_l) .+ var_risk_l) >= 0
                                                sc *
                                                ((net_X + b * z_var_h) .+ var_risk_h) <= 0
                                            end)
    var_range_risk = model[key] = @expression(model, var_risk_l - var_risk_h)
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings, key)
    return nothing
end
function compute_value_at_risk_z(dist::Normal, alpha::Real)
    return cquantile(dist, alpha)
end
function compute_value_at_risk_z(dist::TDist, alpha::Real)
    d = dof(dist)
    @argcheck(d > 2)
    return cquantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_z(::Laplace, alpha::Real)
    return -log(2 * alpha) / sqrt(2)
end
function compute_value_at_risk_cz(dist::Normal, alpha::Real)
    return quantile(dist, alpha)
end
function compute_value_at_risk_cz(dist::TDist, alpha::Real)
    d = dof(dist)
    @argcheck(d > 2)
    return quantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_cz(::Laplace, alpha::Real)
    return -log(2 * (one(alpha) - alpha)) / sqrt(2)
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any,
                                              <:DistributionValueatRisk},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_factory(alg.mu, pr.mu)
    G = isnothing(alg.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(alg.sigma).U
    w = model[:w]
    sc = model[:sc]
    z = compute_value_at_risk_z(r.alg.dist, r.alpha)
    key = Symbol(:var_risk_, i)
    g_var = model[Symbol(:g_var_, i)] = @variable(model)
    var_risk = model[key] = @expression(model, -dot(mu, w) + z * g_var)
    model[Symbol(:cvar_soc_, i)] = @constraint(model,
                                               [sc * g_var; sc * G * w] in
                                               SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:DistributionValueatRisk},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_factory(alg.mu, pr.mu)
    G = isnothing(alg.sigma) ? get_chol_or_sigma_pm(model, pr) : cholesky(alg.sigma).U
    w = model[:w]
    sc = model[:sc]
    dist = r.alg.dist
    z_l = compute_value_at_risk_z(dist, r.alpha)
    z_h = compute_value_at_risk_cz(dist, r.beta)
    key = Symbol(:var_range_risk_, i)
    g_var = model[Symbol(:g_var_range_, i)] = @variable(model)
    var_range_mu = model[Symbol(:var_range_mu_, i)] = @expression(model, dot(mu, w))
    var_risk_l, var_risk_h = model[Symbol(:var_risk_l_, i)], model[Symbol(:var_risk_h_, i)] = @expressions(model,
                                                                                                           begin
                                                                                                               -var_range_mu +
                                                                                                               z_l *
                                                                                                               g_var
                                                                                                               -var_range_mu +
                                                                                                               z_h *
                                                                                                               g_var
                                                                                                           end)
    var_range_risk = model[key] = @expression(model, var_risk_l - var_risk_h)
    model[Symbol(:cvar_range_soc_, i)] = @constraints(model,
                                                      begin
                                                          [sc * g_var; sc * G * w] in
                                                          SecondOrderCone()
                                                      end)
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:cvar_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var, z_cvar = model[Symbol(:z_cvar_, i)], model[Symbol(:var_, i)] = @variables(model,
                                                                                   begin
                                                                                       ()
                                                                                       [1:T],
                                                                                       (lower_bound = 0)
                                                                                   end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    cvar_risk = model[key] = if isnothing(wi)
        iat = inv(r.alpha * T)
        @expression(model, var + sum(z_cvar) * iat)
    else
        iat = inv(r.alpha * sum(wi))
        @expression(model, var + dot(wi, z_cvar) * iat)
    end
    model[Symbol(:ccvar_, i)] = @constraint(model, sc * ((z_cvar + net_X) .+ var) >= 0)
    set_risk_bounds_and_expression!(model, opt, cvar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRiskRange,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:cvar_range_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_l, z_cvar_l, var_h, z_cvar_h = model[Symbol(:var_l_, i)], model[Symbol(:z_cvar_l_, i)], model[Symbol(:var_h_, i)], model[Symbol(:z_cvar_h_, i)] = @variables(model,
                                                                                                                                                                     begin
                                                                                                                                                                         ()
                                                                                                                                                                         [1:T],
                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                         ()
                                                                                                                                                                         [1:T],
                                                                                                                                                                         (upper_bound = 0)
                                                                                                                                                                     end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    cvar_risk_l, cvar_risk_h = if isnothing(wi)
        iat = inv(r.alpha * T)
        ibt = inv(r.beta * T)
        model[Symbol(:cvar_risk_l_, i)], model[Symbol(:cvar_risk_h_, i)] = @expressions(model,
                                                                                        begin
                                                                                            var_l +
                                                                                            sum(z_cvar_l) *
                                                                                            iat
                                                                                            var_h +
                                                                                            sum(z_cvar_h) *
                                                                                            ibt
                                                                                        end)
    else
        sw = sum(wi)
        iat = inv(r.alpha * sw)
        ibt = inv(r.beta * sw)
        model[Symbol(:cvar_risk_l_, i)], model[Symbol(:cvar_risk_h_, i)] = @expressions(model,
                                                                                        begin
                                                                                            var_l +
                                                                                            dot(wi,
                                                                                                z_cvar_l) *
                                                                                            iat
                                                                                            var_h +
                                                                                            dot(wi,
                                                                                                z_cvar_h) *
                                                                                            ibt
                                                                                        end)
    end
    cvar_range_risk = model[key] = @expression(model, cvar_risk_l - cvar_risk_h)
    model[Symbol(:ccvar_l_, i)], model[Symbol(:ccvar_h_, i)] = @constraints(model,
                                                                            begin
                                                                                sc *
                                                                                ((z_cvar_l +
                                                                                  net_X) .+
                                                                                 var_l) >=
                                                                                0
                                                                                sc *
                                                                                ((z_cvar_h +
                                                                                  net_X) .+
                                                                                 var_h) <=
                                                                                0
                                                                            end)
    set_risk_bounds_and_expression!(model, opt, cvar_range_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalValueatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:drcvar_risk_, i)
    sc = model[:sc]
    w = model[:w]
    net_X = set_net_portfolio_returns!(model, pr.X)
    Xap1 = set_portfolio_returns_plus_one!(model, pr.X)
    T, N = size(pr.X)

    alpha = r.alpha
    b1 = r.l
    radius = r.r

    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))
    lb, tau, s, tu_drcvar, tv_drcvar, u, v = model[Symbol(:lb_, i)], model[Symbol(:tau_, i)], model[Symbol(:s_, i)], model[Symbol(:tu_drcvar_, i)], model[Symbol(:tv_drcvar_, i)], model[Symbol(:u_, i)], model[Symbol(:v_, i)] = @variables(model,
                                                                                                                                                                                                                                             begin
                                                                                                                                                                                                                                                 ()
                                                                                                                                                                                                                                                 ()
                                                                                                                                                                                                                                                 [1:T]
                                                                                                                                                                                                                                                 [1:T]
                                                                                                                                                                                                                                                 [1:T]
                                                                                                                                                                                                                                                 [1:T,
                                                                                                                                                                                                                                                  1:N],
                                                                                                                                                                                                                                                 (lower_bound = 0)
                                                                                                                                                                                                                                                 [1:T,
                                                                                                                                                                                                                                                  1:N],
                                                                                                                                                                                                                                                 (lower_bound = 0)
                                                                                                                                                                                                                                             end)
    model[Symbol(:cu_drcvar_, i)], model[Symbol(:cv_drcvar_, i)], model[Symbol(:cu_drcvar_infnorm_, i)], model[Symbol(:cv_drcvar_infnorm_, i)], model[Symbol(:cu_drcvar_lb_, i)], model[Symbol(:cv_drcvar_lb_, i)] = @constraints(model,
                                                                                                                                                                                                                                  begin
                                                                                                                                                                                                                                      sc *
                                                                                                                                                                                                                                      (b1 *
                                                                                                                                                                                                                                       tau .+
                                                                                                                                                                                                                                       (a1 *
                                                                                                                                                                                                                                        net_X +
                                                                                                                                                                                                                                        vec(sum(u .*
                                                                                                                                                                                                                                                Xap1;
                                                                                                                                                                                                                                                dims = 2)) -
                                                                                                                                                                                                                                        s)) <=
                                                                                                                                                                                                                                      0
                                                                                                                                                                                                                                      sc *
                                                                                                                                                                                                                                      (b2 *
                                                                                                                                                                                                                                       tau .+
                                                                                                                                                                                                                                       (a2 *
                                                                                                                                                                                                                                        net_X +
                                                                                                                                                                                                                                        vec(sum(v .*
                                                                                                                                                                                                                                                Xap1;
                                                                                                                                                                                                                                                dims = 2)) -
                                                                                                                                                                                                                                        s)) <=
                                                                                                                                                                                                                                      0
                                                                                                                                                                                                                                      [i = 1:T],
                                                                                                                                                                                                                                      [sc *
                                                                                                                                                                                                                                       tu_drcvar[i]
                                                                                                                                                                                                                                       sc *
                                                                                                                                                                                                                                       (-view(u,
                                                                                                                                                                                                                                              i,
                                                                                                                                                                                                                                              :) -
                                                                                                                                                                                                                                        a1 *
                                                                                                                                                                                                                                        w)] in
                                                                                                                                                                                                                                      MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                           N)
                                                                                                                                                                                                                                      [i = 1:T],
                                                                                                                                                                                                                                      [sc *
                                                                                                                                                                                                                                       tv_drcvar[i]
                                                                                                                                                                                                                                       sc *
                                                                                                                                                                                                                                       (-view(v,
                                                                                                                                                                                                                                              i,
                                                                                                                                                                                                                                              :) -
                                                                                                                                                                                                                                        a2 *
                                                                                                                                                                                                                                        w)] in
                                                                                                                                                                                                                                      MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                           N)
                                                                                                                                                                                                                                      sc *
                                                                                                                                                                                                                                      (tu_drcvar .-
                                                                                                                                                                                                                                       lb) <=
                                                                                                                                                                                                                                      0
                                                                                                                                                                                                                                      sc *
                                                                                                                                                                                                                                      (tv_drcvar .-
                                                                                                                                                                                                                                       lb) <=
                                                                                                                                                                                                                                      0
                                                                                                                                                                                                                                  end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    drcvar_risk = model[key] = if isnothing(wi)
        @expression(model, radius * lb + mean(s))
    else
        @expression(model, radius * lb + mean(s, wi))
    end
    set_risk_bounds_and_expression!(model, opt, drcvar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalValueatRiskRange,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:drcvar_risk_range_, i)
    sc = model[:sc]
    w = model[:w]
    net_X = set_net_portfolio_returns!(model, pr.X)
    Xap1 = set_portfolio_returns_plus_one!(model, pr.X)
    T, N = size(pr.X)

    alpha = r.alpha
    b1_l = r.l_a
    radius_l = r.r_a

    beta = r.beta
    b1_h = r.l_b
    radius_h = r.r_b

    a1_l = -one(alpha)
    a2_l = -one(alpha) - b1_l * inv(alpha)
    b2_l = b1_l * (one(alpha) - inv(alpha))

    a1_h = -one(beta)
    a2_h = -one(beta) - b1_h * inv(beta)
    b2_h = b1_h * (one(beta) - inv(beta))
    lb_l, tau_l, s_l, tu_drcvar_l, tv_drcvar_l, u_l, v_l, lb_h, tau_h, s_h, tu_drcvar_h, tv_drcvar_h, u_h, v_h = model[Symbol(:lb_l_, i)], model[Symbol(:tau_l_, i)], model[Symbol(:s_l_, i)], model[Symbol(:tu_drcvar_l_, i)], model[Symbol(:tv_drcvar_l_, i)], model[Symbol(:u_l_, i)], model[Symbol(:v_l_, i)], model[Symbol(:lb_h_, i)], model[Symbol(:tau_h_, i)], model[Symbol(:s_h_, i)], model[Symbol(:tu_drcvar_h_, i)], model[Symbol(:tv_drcvar_h_, i)], model[Symbol(:u_h_, i)], model[Symbol(:v_h_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 end)
    model[Symbol(:cu_drcvar_l_, i)], model[Symbol(:cv_drcvar_l_, i)], model[Symbol(:cu_drcvar_infnorm_l_, i)], model[Symbol(:cv_drcvar_infnorm_l_, i)], model[Symbol(:cu_drcvar_lb_l_, i)], model[Symbol(:cv_drcvar_lb_l_, i)], model[Symbol(:cu_drcvar_h_, i)], model[Symbol(:cv_drcvar_h_, i)], model[Symbol(:cu_drcvar_infnorm_h_, i)], model[Symbol(:cv_drcvar_infnorm_h_, i)], model[Symbol(:cu_drcvar_lb_h_, i)], model[Symbol(:cv_drcvar_lb_h_, i)] = @constraints(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                          begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (b1_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tau_l .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (a1_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                vec(sum(u_l .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                s_l)) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (b2_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tau_l .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (a2_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                vec(sum(v_l .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                s_l)) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tu_drcvar_l[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (-view(u_l,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      :) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                a1_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tv_drcvar_l[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (-view(v_l,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      :) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                a2_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (tu_drcvar_l .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               lb_l) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (tv_drcvar_l .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               lb_l) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (b1_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tau_h .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (a1_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                vec(sum(u_h .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                s_h)) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (b2_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tau_h .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (a2_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                vec(sum(v_h .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                s_h)) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [-sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tu_drcvar_h[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (view(u_h,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     :) +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                a1_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [-sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               tv_drcvar_h[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (view(v_h,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     :) +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                a2_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (tu_drcvar_h .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               lb_h) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (tv_drcvar_h .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               lb_h) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                          end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    drcvar_risk_l, drcvar_risk_h = model[Symbol(:drcvar_risk_l_, i)], model[Symbol(:drcvar_risk_h_, i)] = if isnothing(wi)
        @expressions(model, begin
                         radius_l * lb_l + mean(s_l)
                         radius_h * lb_h + mean(s_h)
                     end)
    else
        @expressions(model, begin
                         radius_l * lb_l + mean(s_l, wi)
                         radius_h * lb_h + mean(s_h, wi)
                     end)
    end
    drcvar_risk_range = model[key] = @expression(model, drcvar_risk_l - drcvar_risk_h)
    set_risk_bounds_and_expression!(model, opt, drcvar_risk_range, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:evar_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    t_evar, z_evar, u_evar = model[Symbol(:t_evar_, i)], model[Symbol(:z_evar_, i)], model[Symbol(:u_evar_, i)] = @variables(model,
                                                                                                                             begin
                                                                                                                                 ()
                                                                                                                                 (),
                                                                                                                                 (lower_bound = 0)
                                                                                                                                 [1:T]
                                                                                                                             end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    at = if isnothing(wi)
        model[Symbol(:cevar_, i)] = @constraint(model, sc * (sum(u_evar) - z_evar) <= 0)
        r.alpha * T
    else
        model[Symbol(:cevar_, i)] = @constraint(model, sc * (dot(wi, u_evar) - z_evar) <= 0)
        r.alpha * sum(wi)
    end
    model[Symbol(:cevar_exp_cone_, i)] = @constraint(model, [i = 1:T],
                                                     [sc * (-net_X[i] - t_evar),
                                                      sc * z_evar, sc * u_evar[i]] in
                                                     MOI.ExponentialCone())
    evar_risk = model[Symbol(:evar_risk_, i)] = @expression(model,
                                                            t_evar - z_evar * log(at))
    set_risk_bounds_and_expression!(model, opt, evar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRiskRange,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:evar_risk_range_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    t_evar_l, z_evar_l, u_evar_l, t_evar_h, z_evar_h, u_evar_h = model[Symbol(:t_evar_l_, i)], model[Symbol(:z_evar_l_, i)], model[Symbol(:u_evar_l_, i)], model[Symbol(:t_evar_h_, i)], model[Symbol(:z_evar_h_, i)], model[Symbol(:u_evar_h_, i)] = @variables(model,
                                                                                                                                                                                                                                                                 begin
                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                     (),
                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                     (),
                                                                                                                                                                                                                                                                     (upper_bound = 0)
                                                                                                                                                                                                                                                                     [1:T]
                                                                                                                                                                                                                                                                 end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    at, bt = if isnothing(wi)
        model[Symbol(:cevar_l_, i)], model[Symbol(:cevar_h_, i)] = @constraints(model,
                                                                                begin
                                                                                    sc *
                                                                                    (sum(u_evar_l) -
                                                                                     z_evar_l) <=
                                                                                    0
                                                                                    sc *
                                                                                    (sum(u_evar_h) -
                                                                                     z_evar_h) >=
                                                                                    0
                                                                                end)
        r.alpha * T, r.beta * T
    else
        sw = sum(wi)
        model[Symbol(:cevar_l_, i)], model[Symbol(:cevar_h_, i)] = @constraints(model,
                                                                                begin
                                                                                    sc *
                                                                                    (dot(wi,
                                                                                         u_evar_l) -
                                                                                     z_evar_l) <=
                                                                                    0
                                                                                    sc *
                                                                                    (dot(wi,
                                                                                         u_evar_h) -
                                                                                     z_evar_h) >=
                                                                                    0
                                                                                end)
        r.alpha * sw, r.beta * sw
    end
    model[Symbol(:cevar_exp_cone_l_, i)], model[Symbol(:cevar_exp_cone_h_, i)] = @constraints(model,
                                                                                              begin
                                                                                                  [i = 1:T],
                                                                                                  [sc *
                                                                                                   (-net_X[i] -
                                                                                                    t_evar_l),
                                                                                                   sc *
                                                                                                   z_evar_l,
                                                                                                   sc *
                                                                                                   u_evar_l[i]] in
                                                                                                  MOI.ExponentialCone()
                                                                                                  [i = 1:T],
                                                                                                  [sc *
                                                                                                   (net_X[i] +
                                                                                                    t_evar_h),
                                                                                                   -sc *
                                                                                                   z_evar_h,
                                                                                                   -sc *
                                                                                                   u_evar_h[i]] in
                                                                                                  MOI.ExponentialCone()
                                                                                              end)
    evar_risk_l, evar_risk_h = model[Symbol(:evar_risk_l_, i)], model[Symbol(:evar_risk_h_, i)] = @expressions(model,
                                                                                                               begin
                                                                                                                   t_evar_l -
                                                                                                                   z_evar_l *
                                                                                                                   log(at)
                                                                                                                   t_evar_h -
                                                                                                                   z_evar_h *
                                                                                                                   log(bt)
                                                                                                               end)
    evar_risk_range = model[key] = @expression(model, evar_risk_l - evar_risk_h)
    set_risk_bounds_and_expression!(model, opt, evar_risk_range, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:rlvar_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    alpha = r.alpha
    kappa = r.kappa
    t_rlvar, z_rlvar, omega_rlvar, psi_rlvar, theta_rlvar, epsilon_rlvar = model[Symbol(:t_rlvar_, i)], model[Symbol(:z_rlvar_, i)], model[Symbol(:omega_rlvar_, i)], model[Symbol(:psi_rlvar_, i)], model[Symbol(:theta_rlvar_, i)], model[Symbol(:epsilon_rlvar_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         (),
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                     end)
    ik2 = inv(2 * kappa)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    rlvar_risk = model[key] = if isnothing(wi)
        iat = inv(alpha * T)
        lnk = (iat^kappa - iat^(-kappa)) * ik2
        @expression(model, t_rlvar + lnk * z_rlvar + sum(psi_rlvar + theta_rlvar))
    else
        iat = inv(alpha * sum(wi))
        lnk = (iat^kappa - iat^(-kappa)) * ik2
        @expression(model, t_rlvar + lnk * z_rlvar + dot(wi, psi_rlvar + theta_rlvar))
    end
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    model[Symbol(:crlvar_pcone_a_, i)], model[Symbol(:crlvar_pcone_b_, i)], model[Symbol(:crlvar_, i)] = @constraints(model,
                                                                                                                      begin
                                                                                                                          [i = 1:T],
                                                                                                                          [sc *
                                                                                                                           z_rlvar *
                                                                                                                           opk *
                                                                                                                           ik2,
                                                                                                                           sc *
                                                                                                                           psi_rlvar[i] *
                                                                                                                           opk *
                                                                                                                           ik,
                                                                                                                           sc *
                                                                                                                           epsilon_rlvar[i]] in
                                                                                                                          MOI.PowerCone(iopk)
                                                                                                                          [i = 1:T],
                                                                                                                          [sc *
                                                                                                                           omega_rlvar[i] *
                                                                                                                           iomk,
                                                                                                                           sc *
                                                                                                                           theta_rlvar[i] *
                                                                                                                           ik,
                                                                                                                           -sc *
                                                                                                                           z_rlvar *
                                                                                                                           ik2] in
                                                                                                                          MOI.PowerCone(omk)
                                                                                                                          sc *
                                                                                                                          ((epsilon_rlvar +
                                                                                                                            omega_rlvar -
                                                                                                                            net_X) .-
                                                                                                                           t_rlvar) <=
                                                                                                                          0
                                                                                                                      end)
    set_risk_bounds_and_expression!(model, opt, rlvar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:rlvar_range_risk_, i)
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    alpha = r.alpha
    kappa_a = r.kappa_a
    beta = r.beta
    kappa_b = r.kappa_b
    t_rlvar_l, z_rlvar_l, omega_rlvar_l, psi_rlvar_l, theta_rlvar_l, epsilon_rlvar_l, t_rlvar_h, z_rlvar_h, omega_rlvar_h, psi_rlvar_h, theta_rlvar_h, epsilon_rlvar_h = model[Symbol(:t_rlvar_l_, i)], model[Symbol(:z_rlvar_l_, i)], model[Symbol(:omega_rlvar_l_, i)], model[Symbol(:psi_rlvar_l_, i)], model[Symbol(:theta_rlvar_l_, i)], model[Symbol(:epsilon_rlvar_l_, i)], model[Symbol(:t_rlvar_h_, i)], model[Symbol(:z_rlvar_h_, i)], model[Symbol(:omega_rlvar_h_, i)], model[Symbol(:psi_rlvar_h_, i)], model[Symbol(:theta_rlvar_h_, i)], model[Symbol(:epsilon_rlvar_h_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             (),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             (),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         end)
    ik2_a = inv(2 * kappa_a)
    ik2_b = inv(2 * kappa_b)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    rlvar_risk_l, rlvar_risk_h = model[Symbol(:rlvar_risk_l_, i)], model[Symbol(:rlvar_risk_h_, i)] = if isnothing(wi)
        iat = inv(alpha * T)
        ibt = inv(beta * T)
        lnk_a = (iat^kappa_a - iat^(-kappa_a)) * ik2_a
        lnk_b = (ibt^kappa_b - ibt^(-kappa_b)) * ik2_b
        @expressions(model,
                     begin
                         t_rlvar_l + lnk_a * z_rlvar_l + sum(psi_rlvar_l + theta_rlvar_l)
                         t_rlvar_h + lnk_b * z_rlvar_h + sum(psi_rlvar_h + theta_rlvar_h)
                     end)
    else
        sw = sum(wi)
        iat = inv(alpha * sw)
        ibt = inv(beta * sw)
        lnk_a = (iat^kappa_a - iat^(-kappa_a)) * ik2_a
        lnk_b = (ibt^kappa_b - ibt^(-kappa_b)) * ik2_b
        @expressions(model,
                     begin
                         t_rlvar_l +
                         lnk_a * z_rlvar_l +
                         dot(wi, psi_rlvar_l + theta_rlvar_l)
                         t_rlvar_h +
                         lnk_b * z_rlvar_h +
                         dot(wi, psi_rlvar_h + theta_rlvar_h)
                     end)
    end
    opk_a = one(kappa_a) + kappa_a
    omk_a = one(kappa_a) - kappa_a
    ik_a = inv(kappa_a)
    iopk_a = inv(opk_a)
    iomk_a = inv(omk_a)
    opk_b = one(kappa_b) + kappa_b
    omk_b = one(kappa_b) - kappa_b
    ik_b = inv(kappa_b)
    iopk_b = inv(opk_b)
    iomk_b = inv(omk_b)
    rlvar_range_risk = model[Symbol(:rlvar_range_risk_, i)] = @expression(model,
                                                                          rlvar_risk_l -
                                                                          rlvar_risk_h)
    model[Symbol(:crlvar_l_pcone_a_, i)], model[Symbol(:crlvar_l_pcone_b_, i)], model[Symbol(:crlvar_l_, i)], model[Symbol(:crlvar_h_pcone_a_, i)], model[Symbol(:crlvar_h_pcone_b_, i)], model[Symbol(:crlvar_h_, i)] = @constraints(model,
                                                                                                                                                                                                                                      begin
                                                                                                                                                                                                                                          [i = 1:T],
                                                                                                                                                                                                                                          [sc *
                                                                                                                                                                                                                                           z_rlvar_l *
                                                                                                                                                                                                                                           opk_a *
                                                                                                                                                                                                                                           ik2_a,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           psi_rlvar_l[i] *
                                                                                                                                                                                                                                           opk_a *
                                                                                                                                                                                                                                           ik_a,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           epsilon_rlvar_l[i]] in
                                                                                                                                                                                                                                          MOI.PowerCone(iopk_a)
                                                                                                                                                                                                                                          [i = 1:T],
                                                                                                                                                                                                                                          [sc *
                                                                                                                                                                                                                                           omega_rlvar_l[i] *
                                                                                                                                                                                                                                           iomk_a,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           theta_rlvar_l[i] *
                                                                                                                                                                                                                                           ik_a,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           -z_rlvar_l *
                                                                                                                                                                                                                                           ik2_a] in
                                                                                                                                                                                                                                          MOI.PowerCone(omk_a)
                                                                                                                                                                                                                                          sc *
                                                                                                                                                                                                                                          ((epsilon_rlvar_l +
                                                                                                                                                                                                                                            omega_rlvar_l -
                                                                                                                                                                                                                                            net_X) .-
                                                                                                                                                                                                                                           t_rlvar_l) <=
                                                                                                                                                                                                                                          0
                                                                                                                                                                                                                                          [i = 1:T],
                                                                                                                                                                                                                                          [sc *
                                                                                                                                                                                                                                           -z_rlvar_h *
                                                                                                                                                                                                                                           opk_b *
                                                                                                                                                                                                                                           ik2_b,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           -psi_rlvar_h[i] *
                                                                                                                                                                                                                                           opk_b *
                                                                                                                                                                                                                                           ik_b,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           -epsilon_rlvar_h[i]] in
                                                                                                                                                                                                                                          MOI.PowerCone(iopk_b)
                                                                                                                                                                                                                                          [i = 1:T],
                                                                                                                                                                                                                                          [sc *
                                                                                                                                                                                                                                           -omega_rlvar_h[i] *
                                                                                                                                                                                                                                           iomk_b,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           -theta_rlvar_h[i] *
                                                                                                                                                                                                                                           ik_b,
                                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                                           z_rlvar_h *
                                                                                                                                                                                                                                           ik2_b] in
                                                                                                                                                                                                                                          MOI.PowerCone(omk_b)
                                                                                                                                                                                                                                          sc *
                                                                                                                                                                                                                                          ((epsilon_rlvar_h +
                                                                                                                                                                                                                                            omega_rlvar_h -
                                                                                                                                                                                                                                            net_X) .-
                                                                                                                                                                                                                                           t_rlvar_h) >=
                                                                                                                                                                                                                                          0
                                                                                                                                                                                                                                      end)
    set_risk_bounds_and_expression!(model, opt, rlvar_range_risk, r.settings, key)
    return nothing
end
function set_drawdown_constraints!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :dd)
        return model[:dd]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    @variable(model, dd[1:(T + 1)])
    @constraints(model, begin
                     cdd_start, sc * dd[1] == 0
                     cdd_geq_0, sc * view(dd, 2:(T + 1)) >= 0
                     cdd, sc * (net_X + view(dd, 2:(T + 1)) - view(dd, 1:T)) >= 0
                 end)
    return dd
end
function set_risk_constraints!(model::JuMP.Model, ::Any, r::MaximumDrawdown,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :mdd_risk)
        return nothing
    end
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    @variable(model, mdd_risk)
    @constraint(model, cmdd_risk, sc * (mdd_risk .- view(dd, 2:(T + 1))) >= 0)
    set_risk_bounds_and_expression!(model, opt, mdd_risk, r.settings, :mdd_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::AverageDrawdown,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:add_risk_, i)
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    wi = nothing_scalar_array_factory(r.w, pr.w)
    add_risk = model[Symbol(key)] = if isnothing(wi)
        @expression(model, mean(view(dd, 2:(T + 1))))
    else
        @expression(model, mean(view(dd, 2:(T + 1)), wi))
    end
    set_risk_bounds_and_expression!(model, opt, add_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, ::Any, r::UlcerIndex,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :uci)
        return nothing
    end
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model, cuci_soc, [sc * uci; sc * view(dd, 2:(T + 1))] in SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, uci_risk, r.settings, :uci_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalDrawdownatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:cdar_risk_, i)
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    iat = inv(r.alpha * T)
    dar, z_cdar = model[Symbol(:dar_, i)], model[Symbol(:z_cdar_, i)] = @variables(model,
                                                                                   begin
                                                                                       ()
                                                                                       [1:T],
                                                                                       (lower_bound = 0)
                                                                                   end)
    cdar_risk = model[key] = @expression(model, dar + sum(z_cdar) * iat)
    model[Symbol(:ccdar_, i)] = @constraint(model,
                                            sc * ((z_cdar - view(dd, 2:(T + 1))) .+ dar) >=
                                            0)
    set_risk_bounds_and_expression!(model, opt, cdar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicDrawdownatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:edar_risk_, i)
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    at = r.alpha * T
    t_edar, z_edar, u_edar = model[Symbol(:t_edar_, i)], model[Symbol(:z_edar_, i)], model[Symbol(:u_edar_, i)] = @variables(model,
                                                                                                                             begin
                                                                                                                                 ()
                                                                                                                                 (),
                                                                                                                                 (lower_bound = 0)
                                                                                                                                 [1:T]
                                                                                                                             end)
    edar_risk = model[key] = @expression(model, t_edar - z_edar * log(at))
    model[Symbol(:cedar_, i)], model[Symbol(:cedar_exp_cone_, i)] = @constraints(model,
                                                                                 begin
                                                                                     sc *
                                                                                     (sum(u_edar) -
                                                                                      z_edar) <=
                                                                                     0
                                                                                     [i = 1:T],
                                                                                     [sc *
                                                                                      (dd[i + 1] -
                                                                                       t_edar),
                                                                                      sc *
                                                                                      z_edar,
                                                                                      sc *
                                                                                      u_edar[i]] in
                                                                                     MOI.ExponentialCone()
                                                                                 end)
    set_risk_bounds_and_expression!(model, opt, edar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:rldar_risk_, i)
    sc = model[:sc]
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    alpha = r.alpha
    kappa = r.kappa
    iat = inv(alpha * T)
    ik2 = inv(2 * kappa)
    lnk = (iat^kappa - iat^(-kappa)) * ik2
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    t_rldar, z_rldar, omega_rldar, psi_rldar, theta_rldar, epsilon_rldar = model[Symbol(:t_rldar_, i)], model[Symbol(:z_rldar_, i)], model[Symbol(:omega_rldar_, i)], model[Symbol(:psi_rldar_, i)], model[Symbol(:theta_rldar_, i)], model[Symbol(:epsilon_rldar_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         (),
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                     end)
    rldar_risk = model[key] = @expression(model,
                                          t_rldar +
                                          lnk * z_rldar +
                                          sum(psi_rldar + theta_rldar))
    model[Symbol(:crldar_pcone_a_, i)], model[Symbol(:crldar_pcone_b_, i)], model[Symbol(:crldar_, i)] = @constraints(model,
                                                                                                                      begin
                                                                                                                          [i = 1:T],
                                                                                                                          [sc *
                                                                                                                           z_rldar *
                                                                                                                           opk *
                                                                                                                           ik2,
                                                                                                                           sc *
                                                                                                                           psi_rldar[i] *
                                                                                                                           opk *
                                                                                                                           ik,
                                                                                                                           sc *
                                                                                                                           epsilon_rldar[i]] in
                                                                                                                          MOI.PowerCone(iopk)
                                                                                                                          [i = 1:T],
                                                                                                                          [sc *
                                                                                                                           omega_rldar[i] *
                                                                                                                           iomk,
                                                                                                                           sc *
                                                                                                                           theta_rldar[i] *
                                                                                                                           ik,
                                                                                                                           -sc *
                                                                                                                           z_rldar *
                                                                                                                           ik2] in
                                                                                                                          MOI.PowerCone(omk)
                                                                                                                          sc *
                                                                                                                          ((epsilon_rldar +
                                                                                                                            omega_rldar +
                                                                                                                            view(dd,
                                                                                                                                 2:(T + 1))) .-
                                                                                                                           t_rldar) <=
                                                                                                                          0
                                                                                                                      end)
    set_risk_bounds_and_expression!(model, opt, rldar_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer,
                                                     <:Any},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::HighOrderPrior, args...;
                               kwargs...)
    key = Symbol(:sqrt_kurtosis_risk_, i)
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    kt = isnothing(r.kt) ? pr.kt : r.kt
    N = size(W, 1)
    f = clamp(r.N, 1, N)
    Nf = f * N
    sqrt_kurtosis_risk, x_kurt = model[key], model[Symbol(:x_kurt_, i)] = @variables(model,
                                                                                     begin
                                                                                         ()
                                                                                         [1:Nf]
                                                                                     end)
    A = block_vec_pq(kt, N, N)
    vals_A, vecs_A = eigen(A)
    vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
    Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
    N_eig = length(vals_A)
    for i in eachindex(Bi)
        j = i - 1
        B = reshape(real(complex(sqrt(vals_A[end - j])) * view(vecs_A, :, N_eig - j)), N, N)
        Bi[i] = B
    end
    model[Symbol(:capprox_kurt_soc_, i)], model[Symbol(:capprox_kurt_, i)] = @constraints(model,
                                                                                          begin
                                                                                              [sc *
                                                                                               sqrt_kurtosis_risk
                                                                                               sc *
                                                                                               x_kurt] in
                                                                                              SecondOrderCone()
                                                                                              [i = 1:Nf],
                                                                                              sc *
                                                                                              (x_kurt[i] -
                                                                                               tr(Bi[i] *
                                                                                                  W)) ==
                                                                                              0
                                                                                          end)
    set_risk_bounds_and_expression!(model, opt, sqrt_kurtosis_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, Nothing,
                                                     <:Any},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::HighOrderPrior, args...;
                               kwargs...)
    key = Symbol(:sqrt_kurtosis_risk_, i)
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    kt = isnothing(r.kt) ? pr.kt : r.kt
    sqrt_kurtosis_risk = model[key] = @variable(model)
    L2 = pr.L2
    S2 = pr.S2
    sqrt_sigma_4 = cholesky(S2 * kt * transpose(S2)).U
    zkurt = model[Symbol(:zkurt_, i)] = @expression(model, L2 * vec(W))
    model[Symbol(:ckurt_soc_, i)] = @constraint(model,
                                                [sc * sqrt_kurtosis_risk;
                                                 sc * sqrt_sigma_4 * zkurt] in
                                                SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, sqrt_kurtosis_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(::JuMP.Model, ::Any, ::SquareRootKurtosis,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               pr::LowOrderPrior, args...; kwargs...)
    throw(ArgumentError("SquareRootKurtosis requires a HighOrderPrior, not a $(typeof(pr))."))
end
function set_owa_constraints!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :owa)
        return model[:owa]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    T = size(X, 1)
    @variable(model, owa[1:T])
    @constraint(model, sc * (net_X - owa) == 0)
    return owa
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ExactOrderedWeightsArray},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:owa_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    owa = set_owa_constraints!(model, pr.X)
    ovec = range(1; stop = 1, length = T)
    owa_a, owa_b = model[Symbol(:owa_a_, i)], model[Symbol(:owa_b_, i)] = @variables(model,
                                                                                     begin
                                                                                         [1:T]
                                                                                         [1:T]
                                                                                     end)
    owa_risk = model[key] = @expression(model, sum(owa_a + owa_b))
    owa_w = isnothing(r.w) ? owa_gmd(T) : r.w
    model[Symbol(:cowa_, i)] = @constraint(model,
                                           sc * (owa * transpose(owa_w) -
                                                 ovec * transpose(owa_a) -
                                                 owa_b * transpose(ovec)) in Nonpositives())
    set_risk_bounds_and_expression!(model, opt, owa_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ExactOrderedWeightsArray},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:owa_range_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    owa = set_owa_constraints!(model, pr.X)
    ovec = range(1; stop = 1, length = T)
    owa_a, owa_b = model[Symbol(:owa_range_a_, i)], model[Symbol(:owa_range_b_, i)] = @variables(model,
                                                                                                 begin
                                                                                                     [1:T]
                                                                                                     [1:T]
                                                                                                 end)
    owa_range_risk = model[key] = @expression(model, sum(owa_a + owa_b))
    owa_w1 = isnothing(r.w1) ? owa_tg(T) : r.w1
    owa_w2 = isnothing(r.w2) ? reverse(owa_w1) : r.w2
    owa_w = owa_w1 - owa_w2
    model[Symbol(:cowa_range_, i)] = @constraint(model,
                                                 sc * (owa * transpose(owa_w) -
                                                       ovec * transpose(owa_a) -
                                                       owa_b * transpose(ovec)) in
                                                 Nonpositives())
    set_risk_bounds_and_expression!(model, opt, owa_range_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ApproxOrderedWeightsArray},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:aowa_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    net_X = set_net_portfolio_returns!(model, pr.X)
    owa_p = r.alg.p
    M = length(owa_p)
    owa_t, owa_nu, owa_eta, owa_epsilon, owa_psi, owa_z, owa_y = model[Symbol(:owa_t_, i)], model[Symbol(:owa_nu_, i)], model[Symbol(:owa_eta_, i)], model[Symbol(:owa_epsilon_, i)], model[Symbol(:owa_psi_, i)], model[Symbol(:owa_z_, i)], model[Symbol(:owa_y_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         [1:T],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                         [1:T],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                         [1:T,
                                                                                                                                                                                                                                                                                          1:M]
                                                                                                                                                                                                                                                                                         [1:T,
                                                                                                                                                                                                                                                                                          1:M]
                                                                                                                                                                                                                                                                                         [1:M]
                                                                                                                                                                                                                                                                                         [1:M],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                     end)
    owa_w = isnothing(r.w) ? -owa_gmd(T) : -r.w
    owa_s = sum(owa_w)
    owa_l = minimum(owa_w)
    owa_h = maximum(owa_w)
    owa_d = [norm(owa_w, p) for p in owa_p]
    aowa_risk, neg_owa_z_owa_p, owa_p_o_owa_pm1 = model[key], model[Symbol(:neg_owa_z_owa_p_, i)], model[Symbol(:owa_p_o_owa_pm1_, i)] = @expressions(model,
                                                                                                                                                      begin
                                                                                                                                                          owa_s *
                                                                                                                                                          owa_t -
                                                                                                                                                          owa_l *
                                                                                                                                                          sum(owa_nu) +
                                                                                                                                                          owa_h *
                                                                                                                                                          sum(owa_eta) +
                                                                                                                                                          dot(owa_d,
                                                                                                                                                              owa_y)
                                                                                                                                                          -sc *
                                                                                                                                                          owa_z .*
                                                                                                                                                          owa_p
                                                                                                                                                          sc *
                                                                                                                                                          owa_p ./
                                                                                                                                                          (owa_p .-
                                                                                                                                                           one(eltype(owa_p)))
                                                                                                                                                      end)
    model[Symbol(:ca1_owa_, i)], model[Symbol(:ca2_owa_, i)], model[Symbol(:ca_owa_pcone_, i)] = @constraints(model,
                                                                                                              begin
                                                                                                                  sc *
                                                                                                                  ((net_X -
                                                                                                                    owa_nu +
                                                                                                                    owa_eta -
                                                                                                                    vec(sum(owa_epsilon;
                                                                                                                            dims = 2))) .+
                                                                                                                   owa_t) ==
                                                                                                                  0
                                                                                                                  sc *
                                                                                                                  (owa_z +
                                                                                                                   owa_y -
                                                                                                                   vec(sum(owa_psi;
                                                                                                                           dims = 1))) ==
                                                                                                                  0
                                                                                                                  [i = 1:M,
                                                                                                                   j = 1:T],
                                                                                                                  [neg_owa_z_owa_p[i],
                                                                                                                   owa_psi[j,
                                                                                                                           i] *
                                                                                                                   owa_p_o_owa_pm1[i],
                                                                                                                   sc *
                                                                                                                   owa_epsilon[j,
                                                                                                                               i]] in
                                                                                                                  MOI.PowerCone(inv(owa_p[i]))
                                                                                                              end)
    set_risk_bounds_and_expression!(model, opt, aowa_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ApproxOrderedWeightsArray},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:aowa_range_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    net_X = set_net_portfolio_returns!(model, pr.X)
    owa_p = r.alg.p
    M = length(owa_p)
    owa_l_t, owa_l_nu, owa_l_eta, owa_l_epsilon, owa_l_psi, owa_l_z, owa_l_y, owa_h_t, owa_h_nu, owa_h_eta, owa_h_epsilon, owa_h_psi, owa_h_z, owa_h_y = model[Symbol(:owa_l_t_, i)], model[Symbol(:owa_l_nu_, i)], model[Symbol(:owa_l_eta_, i)], model[Symbol(:owa_l_epsilon_, i)], model[Symbol(:owa_l_psi_, i)], model[Symbol(:owa_l_z_, i)], model[Symbol(:owa_l_y_, i)], model[Symbol(:owa_h_t_, i)], model[Symbol(:owa_h_nu_, i)], model[Symbol(:owa_h_eta_, i)], model[Symbol(:owa_h_epsilon_, i)], model[Symbol(:owa_h_psi_, i)], model[Symbol(:owa_h_z_, i)], model[Symbol(:owa_h_y_, i)] = @variables(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:M]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:M]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:M]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:M],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:M]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1:M]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:M]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [1:M],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 end)
    owa_l_w = isnothing(r.w1) ? -owa_tg(T) : -r.w1
    owa_l_s = sum(owa_l_w)
    owa_l_l = minimum(owa_l_w)
    owa_l_h = maximum(owa_l_w)
    owa_l_d = [norm(owa_l_w, p) for p in owa_p]
    owa_h_w = isnothing(r.w2) ? reverse(owa_l_w) : -r.w2
    owa_h_s = sum(owa_h_w)
    owa_h_l = minimum(owa_h_w)
    owa_h_h = maximum(owa_h_w)
    owa_h_d = [norm(owa_h_w, p) for p in owa_p]
    owa_l_risk, neg_owa_l_z_owa_p, owa_h_risk, neg_owa_h_z_owa_p, owa_p_o_owa_pm1 = model[Symbol(:owa_l_risk_, i)], model[Symbol(:neg_owa_l_z_owa_p_, i)], model[Symbol(:owa_h_risk_, i)], model[Symbol(:neg_owa_h_z_owa_p_, i)], model[Symbol(:owa_p_o_owa_pm1_, i)] = @expressions(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         owa_l_s *
                                                                                                                                                                                                                                                                                         owa_l_t -
                                                                                                                                                                                                                                                                                         owa_l_l *
                                                                                                                                                                                                                                                                                         sum(owa_l_nu) +
                                                                                                                                                                                                                                                                                         owa_l_h *
                                                                                                                                                                                                                                                                                         sum(owa_l_eta) +
                                                                                                                                                                                                                                                                                         dot(owa_l_d,
                                                                                                                                                                                                                                                                                             owa_l_y)
                                                                                                                                                                                                                                                                                         -sc *
                                                                                                                                                                                                                                                                                         owa_l_z .*
                                                                                                                                                                                                                                                                                         owa_p
                                                                                                                                                                                                                                                                                         owa_h_s *
                                                                                                                                                                                                                                                                                         owa_h_t -
                                                                                                                                                                                                                                                                                         owa_h_l *
                                                                                                                                                                                                                                                                                         sum(owa_h_nu) +
                                                                                                                                                                                                                                                                                         owa_h_h *
                                                                                                                                                                                                                                                                                         sum(owa_h_eta) +
                                                                                                                                                                                                                                                                                         dot(owa_h_d,
                                                                                                                                                                                                                                                                                             owa_h_y)
                                                                                                                                                                                                                                                                                         -sc *
                                                                                                                                                                                                                                                                                         owa_h_z .*
                                                                                                                                                                                                                                                                                         owa_p
                                                                                                                                                                                                                                                                                         sc *
                                                                                                                                                                                                                                                                                         owa_p ./
                                                                                                                                                                                                                                                                                         (owa_p .-
                                                                                                                                                                                                                                                                                          one(eltype(owa_p)))
                                                                                                                                                                                                                                                                                     end)
    model[Symbol(:ca1_owa_l_, i)], model[Symbol(:ca2_owa_l_, i)], model[Symbol(:ca_owa_pcone_l_, i)], model[Symbol(:ca1_owa_h_, i)], model[Symbol(:ca2_owa_h_, i)], model[Symbol(:ca_owa_pcone_h_, i)] = @constraints(model,
                                                                                                                                                                                                                      begin
                                                                                                                                                                                                                          sc *
                                                                                                                                                                                                                          ((net_X -
                                                                                                                                                                                                                            owa_l_nu +
                                                                                                                                                                                                                            owa_l_eta -
                                                                                                                                                                                                                            vec(sum(owa_l_epsilon;
                                                                                                                                                                                                                                    dims = 2))) .+
                                                                                                                                                                                                                           owa_l_t) ==
                                                                                                                                                                                                                          0
                                                                                                                                                                                                                          sc *
                                                                                                                                                                                                                          (owa_l_z +
                                                                                                                                                                                                                           owa_l_y -
                                                                                                                                                                                                                           vec(sum(owa_l_psi;
                                                                                                                                                                                                                                   dims = 1))) ==
                                                                                                                                                                                                                          0
                                                                                                                                                                                                                          [i = 1:M,
                                                                                                                                                                                                                           j = 1:T],
                                                                                                                                                                                                                          [neg_owa_l_z_owa_p[i],
                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                           owa_l_psi[j,
                                                                                                                                                                                                                                     i] *
                                                                                                                                                                                                                           owa_p_o_owa_pm1[i],
                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                           owa_l_epsilon[j,
                                                                                                                                                                                                                                         i]] in
                                                                                                                                                                                                                          MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                                                                          sc *
                                                                                                                                                                                                                          ((-net_X -
                                                                                                                                                                                                                            owa_h_nu +
                                                                                                                                                                                                                            owa_h_eta -
                                                                                                                                                                                                                            vec(sum(owa_h_epsilon;
                                                                                                                                                                                                                                    dims = 2))) .+
                                                                                                                                                                                                                           owa_h_t) ==
                                                                                                                                                                                                                          0
                                                                                                                                                                                                                          sc *
                                                                                                                                                                                                                          (owa_h_z +
                                                                                                                                                                                                                           owa_h_y -
                                                                                                                                                                                                                           vec(sum(owa_h_psi;
                                                                                                                                                                                                                                   dims = 1))) ==
                                                                                                                                                                                                                          0
                                                                                                                                                                                                                          [i = 1:M,
                                                                                                                                                                                                                           j = 1:T],
                                                                                                                                                                                                                          [neg_owa_h_z_owa_p[i],
                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                           owa_h_psi[j,
                                                                                                                                                                                                                                     i] *
                                                                                                                                                                                                                           owa_p_o_owa_pm1[i],
                                                                                                                                                                                                                           sc *
                                                                                                                                                                                                                           owa_h_epsilon[j,
                                                                                                                                                                                                                                         i]] in
                                                                                                                                                                                                                          MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                                                                      end)
    aowa_range_risk = model[key] = @expression(model, owa_l_risk + owa_h_risk)
    set_risk_bounds_and_expression!(model, opt, aowa_range_risk, r.settings, key)
    return nothing
end
function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::NormOneConeBrownianDistanceVariance,
                                                     Dt::AbstractMatrix, Dx::AbstractMatrix)
    T = size(Dt, 1)
    sc = model[:sc]
    @constraint(model, cbdvariance_noc[j = 1:T, i = j:T],
                [sc * Dt[i, j]; sc * Dx[i, j]] in MOI.NormOneCone(2))
    return nothing
end
function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::IneqBrownianDistanceVariance,
                                                     Dt::AbstractMatrix, Dx::AbstractMatrix)
    sc = model[:sc]
    @constraints(model, begin
                     cp_bdvariance, sc * (Dt - Dx) in Nonnegatives()
                     cn_bdvariance, sc * (Dt + Dx) in Nonnegatives()
                 end)
    return nothing
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::QuadRiskExpr,
                                                Dt::AbstractMatrix, iT2::Real)
    @expression(model, bdvariance_risk, iT2 * (dot(Dt, Dt) + iT2 * sum(Dt)^2))
    return bdvariance_risk
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::RSOCRiskExpr,
                                                Dt::AbstractMatrix, iT2::Real)
    sc = model[:sc]
    @variable(model, tDt)
    @constraint(model, rsoc_Dt, [sc * tDt;
                                 0.5;
                                 sc * vec(Dt)] in RotatedSecondOrderCone())
    @expression(model, bdvariance_risk, iT2 * (tDt + iT2 * sum(Dt)^2))
    return bdvariance_risk
end
function set_risk_constraints!(model::JuMP.Model, ::Any, r::BrownianDistanceVariance,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :bdvariance_risk)
        return nothing
    end
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    iT2 = inv(T^2)
    ovec = range(1; stop = 1, length = T)
    @variable(model, Dt[1:T, 1:T], Symmetric)
    @expression(model, Dx, net_X * transpose(ovec) - ovec * transpose(net_X))
    bdvariance_risk = set_brownian_distance_risk_constraint!(model, r.alg, Dt, iT2)
    set_brownian_distance_variance_constraints!(model, r.algc, Dt, Dx)
    set_risk_bounds_and_expression!(model, opt, bdvariance_risk, r.settings,
                                    :bdvariance_risk)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                   <:SqrtRiskExpr},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::HighOrderPrior, args...;
                               kwargs...)
    key = Symbol(:nskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    V = isnothing(r.V) ? pr.V : r.V
    G = real(sqrt(V))
    nskew_risk = model[key] = @variable(model)
    model[Symbol(:cnskew_soc_, i)] = @constraint(model,
                                                 [sc * nskew_risk; sc * G * w] in
                                                 SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, nskew_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                   <:QuadRiskExpr},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::HighOrderPrior, args...;
                               kwargs...)
    key = Symbol(:qnskew_risk_, i)
    sc = model[:sc]
    w = model[:w]
    V = isnothing(r.V) ? pr.V : r.V
    G = real(sqrt(V))
    t_qnskew_risk = model[Symbol(:t_qnskew_risk_, i)] = @variable(model)
    model[Symbol(:cqnskew_soc_, i)] = @constraint(model,
                                                  [sc * t_qnskew_risk; sc * G * w] in
                                                  SecondOrderCone())
    qnskew_risk = model[key] = @expression(model, t_qnskew_risk^2)
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, t_qnskew_risk, ub, key)
    set_risk_expression!(model, qnskew_risk, r.settings.scale, r.settings.rke)
    return nothing
end
function set_risk_constraints!(::JuMP.Model, ::Any, ::NegativeSkewness,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               pr::LowOrderPrior, args...; kwargs...)
    throw(ArgumentError("NegativeSkewness requires a HighOrderPrior, not a $(typeof(pr))."))
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:NOCTracking},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = model[:sc]
    k = model[:k]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = @variable(model)
    tracking_risk = model[key] = @expression(model, t_tracking_risk / T)
    tracking = r.tracking
    benchmark = tracking_benchmark(tracking, pr.X)
    tracking_r = model[Symbol(:tracking_r_, i)] = @expression(model, net_X - benchmark * k)
    model[Symbol(:ctracking_r_noc_, i)] = @constraint(model,
                                                      [sc * t_tracking_risk;
                                                       sc * tracking_r] in
                                                      MOI.NormOneCone(1 + T))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:SOCTracking},
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = model[:sc]
    k = model[:k]
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = @variable(model)
    tracking_risk = model[key] = @expression(model, t_tracking_risk / sqrt(T - r.alg.ddof))
    tracking = r.tracking
    benchmark = tracking_benchmark(tracking, pr.X)
    tracking_r = model[Symbol(:tracking_r_, i)] = @expression(model, net_X - benchmark * k)
    model[Symbol(:ctracking_r_soc_, i)] = @constraint(model,
                                                      [sc * t_tracking_risk;
                                                       sc * tracking_r] in
                                                      SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any,
                   r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                              <:IndependentVariableTracking},
                   opt::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                   pr::AbstractPriorResult,
                   cplg::Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                   nplg::Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                   args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    ri = r.r
    wb = r.tracking.w
    w = model[:w]
    k = model[:k]
    te_dw = Symbol(:rte_dw_, i)
    model[te_dw] = @expression(model, w - wb * k)
    tracking_risk = set_risk!(model, te_dw, ri, opt, pr, cplg, nplg, args...)[1]
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any,
                   r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                              <:DependentVariableTracking},
                   opt::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                   pr::AbstractPriorResult,
                   cplg::Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                   nplg::Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                   args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    ri = r.r
    wb = r.tracking.w
    rb = expected_risk(r, wb, pr.X, opt.opt.fees)
    k = model[:k]
    sc = model[:sc]
    te_dw = Symbol(:rte_w_, i)
    tracking_risk = model[Symbol(key, i)] = @variable(model)
    risk_expr = set_risk!(model, te_dw, ri, opt, pr, cplg, nplg, args...)[1]
    dr = model[Symbol(:rdr_, i)] = @expression(model, risk_expr - rb * k)
    model[Symbol(:crter_noc_, i)] = @constraint(model,
                                                [sc * tracking_risk;
                                                 sc * dr] in MOI.NormOneCone(2))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::TurnoverRiskMeasure,
                               opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                          <:RiskBudgeting}, ::AbstractPriorResult, args...;
                               kwargs...)
    key = Symbol(:turnover_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    N = length(w)
    turnover_risk = model[key] = @variable(model)
    benchmark = r.w
    turnover_r = model[Symbol(:turnover_r_, i)] = @expression(model, w - benchmark * k)
    model[Symbol(:cturnover_r_noc_, i)] = @constraint(model,
                                                      [sc * turnover_risk;
                                                       sc * turnover_r] in
                                                      MOI.NormOneCone(1 + N))
    set_risk_bounds_and_expression!(model, opt, turnover_risk, r.settings, key)
    return nothing
end
