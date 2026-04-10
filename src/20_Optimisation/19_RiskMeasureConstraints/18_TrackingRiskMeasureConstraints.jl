"""
    set_risk_constraints!(model, i, r::TrackingRiskMeasure{...,<:L1Tracking}, opt, pr, args...; kwargs...)
    set_risk_constraints!(model, i, r::TrackingRiskMeasure{...,<:Union{L2Tracking,SquaredL2Tracking}}, opt, pr, args...; kwargs...)
    set_risk_constraints!(model, i, r::TrackingRiskMeasure{...,<:LpTracking}, opt, pr, args...; kwargs...)
    set_risk_constraints!(model, i, r::TrackingRiskMeasure{...,<:LInfTracking}, opt, pr, args...; kwargs...)
    set_risk_constraints!(model, i, r::RiskTrackingRiskMeasure{...,<:IndependentVariableTracking}, opt, pr, pl, fees, args...; kwargs...)
    set_risk_constraints!(model, i, r::RiskTrackingRiskMeasure{...,<:DependentVariableTracking}, opt, pr, pl, fees, args...; kwargs...)

Add tracking risk constraints to `model`.

The `L1Tracking` overload uses an L1-norm cone. The `L2Tracking` / `SquaredL2Tracking`
overload uses an SOC. The `LpTracking` overload uses power cones parameterised by `r.alg.p`.
The `LInfTracking` overload uses an infinity-norm cone. The independent-variable overload
shifts the weight vector by a benchmark before delegating to [`set_triv_risk_constraints!`](@ref).
The dependent-variable overload computes a benchmark risk and adds an L1-norm cone on the
risk difference via [`set_trdv_risk_constraints!`](@ref).

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `i`: Constraint index for unique naming.
  - `r`: Tracking risk measure instance.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result containing `X`.
  - `pl`: Optional phylogeny constraints.
  - `fees`: Optional fees structure.

# Related

  - [`set_tracking_risk!`](@ref)
  - [`set_risk_tr_constraints!`](@ref)
  - [`set_triv_risk_constraints!`](@ref)
  - [`set_trdv_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:L1Tracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = model[:sc]
    k = model[:k]
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = JuMP.@variable(model)
    tracking_risk = model[key] = JuMP.@expression(model, t_tracking_risk / T)
    tr = r.tr
    benchmark = tracking_benchmark(tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_r_noc_, i)] = JuMP.@constraint(model,
                                                           [sc * t_tracking_risk;
                                                            sc * tracking_r] in
                                                           JuMP.MOI.NormOneCone(1 + T))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
    set_tracking_risk!(model, r::TrackingRiskMeasure{...,<:L2Tracking}, opt, tracking_risk, key)
    set_tracking_risk!(model, r::TrackingRiskMeasure{...,<:SquaredL2Tracking}, opt, tracking_risk, key)

Finalise the L2 or squared-L2 tracking risk expression and apply bounds.

The `L2Tracking` overload calls [`set_risk_bounds_and_expression!`](@ref) directly with the
SOC variable. The `SquaredL2Tracking` overload squares it and applies a sqrt-converted upper
bound to the original SOC variable.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `r::TrackingRiskMeasure`: Tracking risk measure instance.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `tracking_risk::JuMP.AbstractJuMPScalar`: Normalised tracking-risk SOC variable.
  - `key::Symbol`: Symbol for storing the expression in the model.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_tracking_risk!(model::JuMP.Model,
                            r::TrackingRiskMeasure{<:Any, <:Any, <:L2Tracking},
                            opt::RiskJuMPOptimisationEstimator,
                            tracking_risk::JuMP.AbstractJuMPScalar, key::Symbol)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
function set_tracking_risk!(model::JuMP.Model,
                            r::TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Tracking},
                            opt::RiskJuMPOptimisationEstimator,
                            tracking_risk::JuMP.AbstractJuMPScalar, key::Symbol)
    qtracking_risk = model[Symbol(:sq_, key)] = JuMP.@expression(model, tracking_risk^2)
    ub = variance_risk_bounds_val(false, r.settings.ub)
    set_risk_upper_bound!(model, opt, tracking_risk, ub, key)
    set_risk_expression!(model, qtracking_risk, r.settings.scale, r.settings.rke)
    return qtracking_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any,
                                                      <:Union{<:L2Tracking,
                                                              <:SquaredL2Tracking}},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = model[:sc]
    k = model[:k]
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = JuMP.@variable(model)
    tracking_risk = model[key] = JuMP.@expression(model,
                                                  t_tracking_risk / sqrt(T - r.alg.ddof))
    tr = r.tr
    benchmark = tracking_benchmark(tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_r_soc_, i)] = JuMP.@constraint(model,
                                                           [sc * t_tracking_risk;
                                                            sc * tracking_r] in
                                                           JuMP.SecondOrderCone())
    return set_tracking_risk!(model, r, opt, tracking_risk, key)
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:LpTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    @argcheck(r.alg.p > 1, DomainError)
    key = Symbol(:tracking_risk_, i)
    sc = model[:sc]
    k = model[:k]
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    t_tracking_risk, r_tr = model[Symbol(:t_tracking_risk_, i)], model[Symbol(:r_tracking_risk_, i)] = JuMP.@variables(model,
                                                                                                                       begin
                                                                                                                           ()
                                                                                                                           [1:T]
                                                                                                                       end)
    p_inv = inv(r.alg.p)
    scale = T - r.alg.ddof
    scale = r.alg.p == 3 ? cbrt(scale) : scale^p_inv
    tracking_risk = model[key] = JuMP.@expression(model, t_tracking_risk / scale)
    benchmark = tracking_benchmark(r.tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_r_pnorm_, i)], model[Symbol(:ctracking_r_pnorm_eq_, i)] = JuMP.@constraints(model,
                                                                                                        begin
                                                                                                            [i = 1:T],
                                                                                                            [sc *
                                                                                                             r_tr[i],
                                                                                                             sc *
                                                                                                             t_tracking_risk,
                                                                                                             sc *
                                                                                                             tracking_r[i]] in
                                                                                                            JuMP.MOI.PowerCone(p_inv)
                                                                                                            sc *
                                                                                                            (sum(r_tr) -
                                                                                                             t_tracking_risk) ==
                                                                                                            0
                                                                                                        end)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:LInfTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = model[:sc]
    k = model[:k]
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = JuMP.@variable(model)
    scale = T - r.alg.ddof
    tracking_risk = model[key] = JuMP.@expression(model, t_tracking_risk / scale)
    benchmark = tracking_benchmark(r.tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_infnorm_, i)] = JuMP.@constraint(model,
                                                             [sc * t_tracking_risk;
                                                              sc * tracking_r] in
                                                             JuMP.MOI.NormInfinityCone(1 +
                                                                                       T))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
    set_risk_tr_constraints!(key, model, r::RiskMeasure, opt, pr, pl, fees, args...; kwargs...)
    set_risk_tr_constraints!(key, model, rs::VecRM, opt, pr, pl, fees, args...; kwargs...)

Dispatch to indexed [`set_risk_constraints!`](@ref) for a single measure or iterate over a
vector of measures, using a name prefix `key` for unique constraint naming.

# Arguments

  - `key`: Name prefix for unique constraint symbols.
  - `model::JuMP.Model`: The JuMP optimisation model.
  - `r`: A [`RiskMeasure`](@ref) or a vector of risk measures.
  - `opt::JuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result.
  - `pl`: Optional phylogeny constraints.
  - `fees`: Optional fees structure.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_triv_risk_constraints!`](@ref)
"""
function set_risk_tr_constraints!(key::Any, model::JuMP.Model, r::RiskMeasure,
                                  opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                                  pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                                  kwargs...)
    return set_risk_constraints!(model, Symbol(key, 1), r, opt, pr, pl, fees, args...;
                                 kwargs...)
end
function set_risk_tr_constraints!(key::Any, model::JuMP.Model, rs::VecRM,
                                  opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                                  pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                                  kwargs...)
    for (i, r) in enumerate(rs)
        set_risk_constraints!(model, Symbol(key, i), r, opt, pr, pl, fees, args...;
                              kwargs...)
    end
    return nothing
end
"""
    set_triv_risk_constraints!(model, i, r::RiskMeasure, opt, pr, pl, fees, args...; kwargs...)

Set risk constraints for independent-variable tracking, saving and restoring any global
singleton model state that would conflict with the nested solve.

Stashes existing model-level expressions and constraints (e.g., `net_X`, `dd`, `wr_risk`,
SDP matrices) with `old` prefixes, calls [`set_risk_tr_constraints!`](@ref) with the
`triv_i_` naming prefix, then restores the original state.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `i`: Constraint index for unique naming.
  - `r::RiskMeasure`: Inner risk measure.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result.
  - `pl`: Optional phylogeny constraints.
  - `fees`: Optional fees structure.

# Related

  - [`set_risk_tr_constraints!`](@ref)
  - [`set_trdv_risk_constraints!`](@ref)
"""
function set_triv_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                    opt::RiskJuMPOptimisationEstimator,
                                    pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                                    fees::Option{<:Fees}, args...; kwargs...)
    variance_flag = haskey(model, :variance_flag)
    rc_variance = haskey(model, :rc_variance)
    W = haskey(model, :W)
    Au = haskey(model, :Au)
    E = haskey(model, :E)
    X = haskey(model, :X)
    net_X = haskey(model, :net_X)
    Xap1 = haskey(model, :Xap1)
    ddap1 = haskey(model, :ddap1)
    wr_risk = haskey(model, :wr_risk)
    range_risk = haskey(model, :range_risk)
    dd = haskey(model, :dd)
    mdd_risk = haskey(model, :mdd_risk)
    uci = haskey(model, :uci)
    owa = haskey(model, :owa)
    bdvariance_risk = haskey(model, :bdvariance_risk)
    if variance_flag
        model[:oldvariance_flag] = model[:variance_flag]
        JuMP.unregister(model, :variance_flag)
    end
    if rc_variance
        model[:oldrc_variance] = model[:rc_variance]
        JuMP.unregister(model, :rc_variance)
    end
    if W
        model[:oldW] = model[:W]
        model[:oldM] = model[:M]
        model[:oldM_PSD] = model[:M_PSD]
        JuMP.unregister(model, :W)
        JuMP.unregister(model, :M)
        JuMP.unregister(model, :M_PSD)
    end
    if Au
        model[:oldAu] = model[:Au]
        model[:oldAl] = model[:Al]
        model[:oldcbucs_variance] = model[:cbucs_variance]
        JuMP.unregister(model, :Au)
        JuMP.unregister(model, :Al)
        JuMP.unregister(model, :cbucs_variance)
    end
    if E
        model[:oldE] = model[:E]
        model[:oldWpE] = model[:WpE]
        model[:oldceucs_variance] = model[:ceucs_variance]
        JuMP.unregister(model, :E)
        JuMP.unregister(model, :WpE)
        JuMP.unregister(model, :ceucs_variance)
    end
    if X
        model[:oldX] = model[:X]
        JuMP.unregister(model, :X)
    end
    if net_X
        model[:oldnet_X] = model[:net_X]
        JuMP.unregister(model, :net_X)
    end
    if Xap1
        model[:oldXap1] = model[:Xap1]
        JuMP.unregister(model, :Xap1)
    end
    if ddap1
        model[:oldddap1] = model[:ddap1]
        JuMP.unregister(model, :ddap1)
    end
    if wr_risk
        model[:oldwr_risk] = model[:wr_risk]
        model[:oldcwr] = model[:cwr]
        JuMP.unregister(model, :wr_risk)
        JuMP.unregister(model, :cwr)
    end
    if range_risk
        model[:oldrange_risk] = model[:range_risk]
        model[:oldbr_risk] = model[:br_risk]
        model[:oldcbr] = model[:cbr]
        JuMP.unregister(model, :range_risk)
        JuMP.unregister(model, :br_risk)
        JuMP.unregister(model, :cbr)
    end
    if dd
        model[:olddd] = model[:dd]
        model[:oldcdd_start] = model[:cdd_start]
        model[:oldcdd_geq_0] = model[:cdd_geq_0]
        model[:oldcdd] = model[:cdd]
        JuMP.unregister(model, :dd)
        JuMP.unregister(model, :cdd_start)
        JuMP.unregister(model, :cdd_geq_0)
        JuMP.unregister(model, :cdd)
    end
    if mdd_risk
        model[:oldmdd_risk] = model[:mdd_risk]
        model[:oldcmdd_risk] = model[:cmdd_risk]
        JuMP.unregister(model, :mdd_risk)
        JuMP.unregister(model, :cmdd_risk)
    end
    if uci
        model[:olduci] = model[:uci]
        model[:olduci_risk] = model[:uci_risk]
        model[:oldcuci_soc] = model[:cuci_soc]
        JuMP.unregister(model, :uci)
        JuMP.unregister(model, :uci_risk)
        JuMP.unregister(model, :cuci_soc)
    end
    if owa
        model[:oldowa] = model[:owa]
        model[:oldowac] = model[:owac]
        JuMP.unregister(model, :owa)
        JuMP.unregister(model, :owac)
    end
    if bdvariance_risk
        model[:oldbdvariance_risk] = model[:bdvariance_risk]
        model[:oldDt] = model[:Dt]
        model[:oldDx] = model[:Dx]
        JuMP.unregister(model, :Dt)
        JuMP.unregister(model, :Dx)
        JuMP.unregister(model, :bdvariance_risk)
    end
    risk_expr = set_risk_tr_constraints!(Symbol(:triv_, i, :_), model, r, opt, pr, pl, fees,
                                         args...; kwargs...)

    if (!variance_flag && haskey(model, :variance_flag)) || haskey(model, :oldvariance_flag)
        model[Symbol(:triv_, i, :_variance_flag)] = model[:variance_flag]
        JuMP.unregister(model, :variance_flag)

        if haskey(model, :oldvariance_flag)
            model[:variance_flag] = model[:oldvariance_flag]
            JuMP.unregister(model, :oldvariance_flag)
        end
    end
    if (!rc_variance && haskey(model, :rc_variance)) || haskey(model, :oldrc_variance)
        model[Symbol(:triv_, i, :_rc_variance)] = model[:rc_variance]
        JuMP.unregister(model, :rc_variance)

        if haskey(model, :oldrc_variance)
            model[:rc_variance] = model[:oldrc_variance]
            JuMP.unregister(model, :oldrc_variance)
        end
    end
    if (!W && haskey(model, :W)) || haskey(model, :oldW)
        model[Symbol(:triv_, i, :_W)] = model[:W]
        model[Symbol(:triv_, i, :_M)] = model[:M]
        model[Symbol(:triv_, i, :_M_PSD)] = model[:M_PSD]
        JuMP.unregister(model, :W)
        JuMP.unregister(model, :M)
        JuMP.unregister(model, :M_PSD)

        if haskey(model, :oldW)
            model[:W] = model[:oldW]
            model[:M] = model[:oldM]
            model[:M_PSD] = model[:oldM_PSD]
            JuMP.unregister(model, :oldW)
            JuMP.unregister(model, :oldM)
            JuMP.unregister(model, :oldM_PSD)
        end
    end
    if (!Au && haskey(model, :Au)) || haskey(model, :oldAu)
        model[Symbol(:triv_, i, :_Au)] = model[:Au]
        model[Symbol(:triv_, i, :_Al)] = model[:Al]
        model[Symbol(:triv_, i, :_cbucs_variance)] = model[:cbucs_variance]
        JuMP.unregister(model, :Au)
        JuMP.unregister(model, :Al)
        JuMP.unregister(model, :cbucs_variance)

        if haskey(model, :oldAu)
            model[:Au] = model[:oldAu]
            model[:Al] = model[:oldAl]
            model[:cbucs_variance] = model[:oldcbucs_variance]
            JuMP.unregister(model, :oldAu)
            JuMP.unregister(model, :oldAl)
            JuMP.unregister(model, :oldcbucs_variance)
        end
    end
    if (!E && haskey(model, :E)) || haskey(model, :oldE)
        model[Symbol(:triv_, i, :_E)] = model[:E]
        model[Symbol(:triv_, i, :_WpE)] = model[:WpE]
        model[Symbol(:triv_, i, :_ceucs_variance)] = model[:ceucs_variance]
        JuMP.unregister(model, :E)
        JuMP.unregister(model, :WpE)
        JuMP.unregister(model, :ceucs_variance)

        if haskey(model, :oldE)
            model[:E] = model[:oldE]
            model[:WpE] = model[:oldWpE]
            model[:ceucs_variance] = model[:oldceucs_variance]
            JuMP.unregister(model, :oldE)
            JuMP.unregister(model, :oldWpE)
            JuMP.unregister(model, :oldceucs_variance)
        end
    end
    if (!X && haskey(model, :X)) || haskey(model, :oldX)
        model[Symbol(:triv_, i, :_X)] = model[:X]
        JuMP.unregister(model, :X)

        if haskey(model, :oldX)
            model[:X] = model[:oldX]
            JuMP.unregister(model, :oldX)
        end
    end
    if (!net_X && haskey(model, :net_X)) || haskey(model, :oldnet_X)
        model[Symbol(:triv_, i, :_net_X)] = model[:net_X]
        JuMP.unregister(model, :net_X)

        if haskey(model, :oldnet_X)
            model[:net_X] = model[:oldnet_X]
            JuMP.unregister(model, :oldnet_X)
        end
    end
    if (!Xap1 && haskey(model, :Xap1)) || haskey(model, :oldXap1)
        model[Symbol(:triv_, i, :_Xap1)] = model[:Xap1]
        JuMP.unregister(model, :Xap1)

        if haskey(model, :oldXap1)
            model[:Xap1] = model[:oldXap1]
            JuMP.unregister(model, :oldXap1)
        end
    end
    if (!ddap1 && haskey(model, :ddap1)) || haskey(model, :oldddap1)
        model[Symbol(:triv_, i, :_ddap1)] = model[:ddap1]
        JuMP.unregister(model, :ddap1)

        if haskey(model, :oldddap1)
            model[:ddap1] = model[:oldddap1]
            JuMP.unregister(model, :oldddap1)
        end
    end
    if (!wr_risk && haskey(model, :wr_risk)) || haskey(model, :oldwr_risk)
        model[Symbol(:triv_, i, :_wr_risk)] = model[:wr_risk]
        model[Symbol(:triv_, i, :_cwr)] = model[:cwr]
        JuMP.unregister(model, :wr_risk)
        JuMP.unregister(model, :cwr)

        if haskey(model, :oldwr_risk)
            model[:wr_risk] = model[:oldwr_risk]
            model[:cwr] = model[:oldcwr]
            JuMP.unregister(model, :oldwr_risk)
            JuMP.unregister(model, :oldcwr)
        end
    end
    if (!range_risk && haskey(model, :range_risk)) || haskey(model, :oldrange_risk)
        model[Symbol(:triv_, i, :_range_risk)] = model[:range_risk]
        model[Symbol(:triv_, i, :_br_risk)] = model[:br_risk]
        model[Symbol(:triv_, i, :_cbr)] = model[:cbr]
        JuMP.unregister(model, :range_risk)
        JuMP.unregister(model, :br_risk)
        JuMP.unregister(model, :cbr)

        if haskey(model, :oldrange_risk)
            model[:range_risk] = model[:oldrange_risk]
            model[:br_risk] = model[:oldbr_risk]
            model[:cbr] = model[:oldcbr]
            JuMP.unregister(model, :oldrange_risk)
            JuMP.unregister(model, :oldbr_risk)
            JuMP.unregister(model, :oldcbr)
        end
    end
    if (!dd && haskey(model, :dd)) || haskey(model, :olddd)
        model[Symbol(:triv_, i, :_dd)] = model[:dd]
        model[Symbol(:triv_, i, :_cdd_start)] = model[:cdd_start]
        model[Symbol(:triv_, i, :_cdd_geq_0)] = model[:cdd_geq_0]
        model[Symbol(:triv_, i, :_cdd)] = model[:cdd]
        JuMP.unregister(model, :dd)
        JuMP.unregister(model, :cdd_start)
        JuMP.unregister(model, :cdd_geq_0)
        JuMP.unregister(model, :cdd)

        if haskey(model, :olddd)
            model[:dd] = model[:olddd]
            model[:cdd_start] = model[:oldcdd_start]
            model[:cdd_geq_0] = model[:oldcdd_geq_0]
            model[:cdd] = model[:oldcdd]
            JuMP.unregister(model, :olddd)
            JuMP.unregister(model, :oldcdd_start)
            JuMP.unregister(model, :oldcdd_geq_0)
            JuMP.unregister(model, :oldcdd)
        end
    end
    if (!mdd_risk && haskey(model, :mdd_risk)) || haskey(model, :oldmdd_risk)
        model[Symbol(:triv_, i, :_mdd_risk)] = model[:mdd_risk]
        model[Symbol(:triv_, i, :_cmdd_risk)] = model[:cmdd_risk]
        JuMP.unregister(model, :mdd_risk)
        JuMP.unregister(model, :cmdd_risk)

        if haskey(model, :oldmdd_risk)
            model[:mdd_risk] = model[:oldmdd_risk]
            model[:cmdd_risk] = model[:oldcmdd_risk]
            JuMP.unregister(model, :oldmdd_risk)
            JuMP.unregister(model, :oldcmdd_risk)
        end
    end
    if (!uci && haskey(model, :uci)) || haskey(model, :olduci)
        model[Symbol(:triv_, i, :_uci)] = model[:uci]
        model[Symbol(:triv_, i, :_uci_risk)] = model[:uci_risk]
        model[Symbol(:triv_, i, :_cuci_soc)] = model[:cuci_soc]
        JuMP.unregister(model, :uci)
        JuMP.unregister(model, :uci_risk)
        JuMP.unregister(model, :cuci_soc)

        if haskey(model, :olduci)
            model[:uci] = model[:olduci]
            model[:uci_risk] = model[:olduci_risk]
            model[:cuci_soc] = model[:oldcuci_soc]
            JuMP.unregister(model, :olduci)
            JuMP.unregister(model, :olduci_risk)
            JuMP.unregister(model, :oldcuci_soc)
        end
    end
    if (!owa && haskey(model, :owa)) || haskey(model, :oldowa)
        model[Symbol(:triv_, i, :_owa)] = model[:owa]
        model[Symbol(:triv_, i, :_owac)] = model[:owac]
        JuMP.unregister(model, :owa)
        JuMP.unregister(model, :owac)

        if haskey(model, :oldowa)
            model[:owa] = model[:oldowa]
            model[:owac] = model[:oldowac]
            JuMP.unregister(model, :oldowa)
            JuMP.unregister(model, :oldowac)
        end
    end
    if (!bdvariance_risk && haskey(model, :bdvariance_risk)) ||
       haskey(model, :oldbdvariance_risk)
        model[Symbol(:triv_, i, :_bdvariance_risk)] = model[:bdvariance_risk]
        model[Symbol(:triv_, i, :_Dt)] = model[:Dt]
        model[Symbol(:triv_, i, :_Dx)] = model[:Dx]
        JuMP.unregister(model, :bdvariance_risk)
        JuMP.unregister(model, :Dt)
        JuMP.unregister(model, :Dx)

        if haskey(model, :oldbdvariance_risk)
            model[:bdvariance_risk] = model[:oldbdvariance_risk]
            model[:Dt] = model[:oldDt]
            model[:Dx] = model[:oldDx]
            JuMP.unregister(model, :oldbdvariance_risk)
            JuMP.unregister(model, :oldDt)
            JuMP.unregister(model, :oldDx)
        end
    end
    return risk_expr
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:IndependentVariableTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               kwargs...)
    key = Symbol(:tracking_risk_, i)
    ri = r.r
    wb = r.tr.w
    w = model[:w]
    k = model[:k]
    model[:oldw] = model[:w]
    JuMP.unregister(model, :w)
    model[:w] = JuMP.@expression(model, w - wb * k)
    tracking_risk = set_triv_risk_constraints!(model, i, ri, opt, pr, pl, fees, args...;
                                               kwargs...)
    model[Symbol(:triv_, i, :_w)] = model[:w]
    model[:w] = model[:oldw]
    JuMP.unregister(model, :oldw)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
    set_trdv_risk_constraints!(model, i, r::RiskMeasure, opt, pr, pl, fees, args...; kwargs...)

Set risk constraints for dependent-variable tracking, saving and restoring variance-related
model state.

Stashes existing SDP matrices (`W`, `Au`, `E`) and variance flags, calls
[`set_risk_tr_constraints!`](@ref) with the `trdv_i_` naming prefix, then restores the
original state. This ensures the inner risk measure's variance constraints do not interfere
with the outer model.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `i`: Constraint index for unique naming.
  - `r::RiskMeasure`: Inner risk measure.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result.
  - `pl`: Optional phylogeny constraints.
  - `fees`: Optional fees structure.

# Related

  - [`set_risk_tr_constraints!`](@ref)
  - [`set_triv_risk_constraints!`](@ref)
"""
function set_trdv_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                    opt::RiskJuMPOptimisationEstimator,
                                    pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                                    fees::Option{<:Fees}, args...; kwargs...)
    variance_flag = haskey(model, :variance_flag)
    rc_variance = haskey(model, :rc_variance)
    W = haskey(model, :W)
    Au = haskey(model, :Au)
    E = haskey(model, :E)
    if variance_flag
        model[:oldvariance_flag] = model[:variance_flag]
        JuMP.unregister(model, :variance_flag)
    end
    if rc_variance
        model[:oldrc_variance] = model[:rc_variance]
        JuMP.unregister(model, :rc_variance)
    end
    if W
        model[:oldW] = model[:W]
        model[:oldM] = model[:M]
        model[:oldM_PSD] = model[:M_PSD]
        JuMP.unregister(model, :W)
        JuMP.unregister(model, :M)
        JuMP.unregister(model, :M_PSD)
    end
    if Au
        model[:oldAu] = model[:Au]
        model[:oldAl] = model[:Al]
        model[:oldcbucs_variance] = model[:cbucs_variance]
        JuMP.unregister(model, :Au)
        JuMP.unregister(model, :Al)
        JuMP.unregister(model, :cbucs_variance)
    end
    if E
        model[:oldE] = model[:E]
        model[:oldWpE] = model[:WpE]
        model[:oldceucs_variance] = model[:ceucs_variance]
        JuMP.unregister(model, :E)
        JuMP.unregister(model, :WpE)
        JuMP.unregister(model, :ceucs_variance)
    end

    risk_expr = set_risk_tr_constraints!(Symbol(:trdv_, i, :_), model, r, opt, pr, pl, fees,
                                         args...; kwargs...)

    if (!variance_flag && haskey(model, :variance_flag)) || haskey(model, :oldvariance_flag)
        model[Symbol(:trdv_, i, :_variance_flag)] = model[:variance_flag]
        JuMP.unregister(model, :variance_flag)

        if haskey(model, :oldvariance_flag)
            model[:variance_flag] = model[:oldvariance_flag]
            JuMP.unregister(model, :oldvariance_flag)
        end
    end
    if (!rc_variance && haskey(model, :rc_variance)) || haskey(model, :oldrc_variance)
        model[Symbol(:trdv_, i, :_rc_variance)] = model[:rc_variance]
        JuMP.unregister(model, :rc_variance)

        if haskey(model, :oldrc_variance)
            model[:rc_variance] = model[:oldrc_variance]
            JuMP.unregister(model, :oldrc_variance)
        end
    end
    if (!W && haskey(model, :W)) || haskey(model, :oldW)
        model[Symbol(:trdv_, i, :_W)] = model[:W]
        model[Symbol(:trdv_, i, :_M)] = model[:M]
        model[Symbol(:trdv_, i, :_M_PSD)] = model[:M_PSD]
        JuMP.unregister(model, :W)
        JuMP.unregister(model, :M)
        JuMP.unregister(model, :M_PSD)

        if haskey(model, :oldW)
            model[:W] = model[:oldW]
            model[:M] = model[:oldM]
            model[:M_PSD] = model[:oldM_PSD]
            JuMP.unregister(model, :oldW)
            JuMP.unregister(model, :oldM)
            JuMP.unregister(model, :oldM_PSD)
        end
    end
    if (!Au && haskey(model, :Au)) || haskey(model, :oldAu)
        model[Symbol(:trdv_, i, :_Au)] = model[:Au]
        model[Symbol(:trdv_, i, :_Al)] = model[:Al]
        model[Symbol(:trdv_, i, :_cbucs_variance)] = model[:cbucs_variance]
        JuMP.unregister(model, :Au)
        JuMP.unregister(model, :Al)
        JuMP.unregister(model, :cbucs_variance)

        if haskey(model, :oldAu)
            model[:Au] = model[:oldAu]
            model[:Al] = model[:oldAl]
            model[:cbucs_variance] = model[:oldcbucs_variance]
            JuMP.unregister(model, :oldAu)
            JuMP.unregister(model, :oldAl)
            JuMP.unregister(model, :oldcbucs_variance)
        end
    end
    if (!E && haskey(model, :E)) || haskey(model, :oldE)
        model[Symbol(:trdv_, i, :_E)] = model[:E]
        model[Symbol(:trdv_, i, :_WpE)] = model[:WpE]
        model[Symbol(:trdv_, i, :_ceucs_variance)] = model[:ceucs_variance]
        JuMP.unregister(model, :E)
        JuMP.unregister(model, :WpE)
        JuMP.unregister(model, :ceucs_variance)

        if haskey(model, :oldE)
            model[:E] = model[:oldE]
            model[:WpE] = model[:oldWpE]
            model[:ceucs_variance] = model[:oldceucs_variance]
            JuMP.unregister(model, :oldE)
            JuMP.unregister(model, :oldWpE)
            JuMP.unregister(model, :oldceucs_variance)
        end
    end
    return risk_expr
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:DependentVariableTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               kwargs...)
    key = Symbol(:tracking_risk_, i)
    ri = r.r
    wb = r.tr.w
    rb = expected_risk(factory(ri, pr, opt.opt.slv), wb, pr.X, fees)
    k = model[:k]
    sc = model[:sc]
    tracking_risk = model[key] = JuMP.@variable(model)
    risk_expr = set_trdv_risk_constraints!(model, i, ri, opt, pr, pl, fees, args...;
                                           kwargs...)
    dr = model[Symbol(:rdr_, i)] = JuMP.@expression(model, risk_expr - rb * k)
    model[Symbol(:crtr_noc_, i)] = JuMP.@constraint(model,
                                                    [sc * tracking_risk;
                                                     sc * dr] in JuMP.MOI.NormOneCone(2))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
