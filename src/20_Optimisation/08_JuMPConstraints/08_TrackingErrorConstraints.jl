function set_tracking_error_constraints!(args...; kwargs...)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any, <:NOCTracking},
                                         args...; kwargs...)
    X = pr.X
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tr.tr, X)
    err = tr.err
    T = size(X, 1)
    f = err * T
    t_te = model[Symbol(:t_te_, i)] = JuMP.@variable(model)
    tr = model[Symbol(:te_, i)] = JuMP.@expression(model, net_X - wb * k)
    model[Symbol(:cte_noc_, i)], model[Symbol(:cte_, i)] = JuMP.@constraints(model,
                                                                             begin
                                                                                 [sc * t_te;
                                                                                  sc * tr] in
                                                                                 JuMP.MOI.NormOneCone(1 +
                                                                                                      T)
                                                                                 sc *
                                                                                 (t_te -
                                                                                  f * k) <=
                                                                                 0
                                                                             end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any,
                                                           <:Union{<:SOCTracking,
                                                                   <:SquaredSOCTracking}},
                                         args...; kwargs...)
    X = pr.X
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tr.tr, X)
    err = tr.err
    f = err * sqrt(size(X, 1) - tr.alg.ddof)
    t_te = model[Symbol(:t_te_, i)] = JuMP.@variable(model)
    tr = model[Symbol(:te_, i)] = JuMP.@expression(model, net_X - wb * k)
    model[Symbol(:cte_soc_, i)], model[Symbol(:cte_, i)] = JuMP.@constraints(model,
                                                                             begin
                                                                                 [sc * t_te;
                                                                                  sc * tr] in
                                                                                 JuMP.SecondOrderCone()
                                                                                 sc *
                                                                                 (t_te -
                                                                                  f * k) <=
                                                                                 0
                                                                             end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:IndependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                         args...; kwargs...)
    r = tr.r
    wb = tr.tr.w
    err = tr.err
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    te_dw = Symbol(:te_dw_, i)
    model[:oldw] = model[:w]
    JuMP.unregister(model, :w)
    model[:w] = JuMP.@expression(model, w - wb * k)
    risk_expr = set_triv_risk_constraints!(model, te_dw, r, opt, pr, pl, fees, args...;
                                           kwargs...)
    model[Symbol(:triv_, i, :_w)] = model[:w]
    model[:w] = model[:oldw]
    JuMP.unregister(model, :oldw)
    model[Symbol(:cter_, i)] = JuMP.@constraint(model, sc * (risk_expr - err * k) <= 0)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:DependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                         args...; kwargs...)
    ri = tr.r
    wb = tr.tr.w
    err = tr.err
    rb = expected_risk(factory(ri, pr, opt.opt.slv), wb, pr.X, fees)
    k = model[:k]
    sc = model[:sc]
    key = Symbol(:t_dr_, i)
    t_dr = model[key] = JuMP.@variable(model)
    risk_expr = set_trdv_risk_constraints!(model, key, ri, opt, pr, pl, fees, args...;
                                           kwargs...)
    dr = model[Symbol(:dr_, i)] = JuMP.@expression(model, risk_expr - rb * k)
    model[Symbol(:cter_noc_, i)], model[Symbol(:cter_, i)] = JuMP.@constraints(model,
                                                                               begin
                                                                                   [sc *
                                                                                    t_dr
                                                                                    sc * dr] in
                                                                                   JuMP.MOI.NormOneCone(2)
                                                                                   sc *
                                                                                   (t_dr -
                                                                                    err * k) <=
                                                                                   0
                                                                               end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, pr::AbstractPriorResult,
                                         tres::Tr_VecTr, args...; kwargs...)
    for (i, tr) in enumerate(tres)
        set_tracking_error_constraints!(model, i, pr, tr, args...; kwargs...)
    end
    return nothing
end
