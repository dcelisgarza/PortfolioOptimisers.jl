function set_tracking_error_constraints!(args...; kwargs...)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::TrackingError{<:Any, <:Any, <:NOCTracking},
                                         args...; kwargs...)
    X = pr.X
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(te.tracking, X)
    err = te.err
    T = size(X, 1)
    f = err * T
    t_te = model[Symbol(:t_te_, i)] = @variable(model)
    te = model[Symbol(:te_, i)] = @expression(model, net_X - wb * k)
    model[Symbol(:cte_noc_, i)], model[Symbol(:cte_, i)] = @constraints(model,
                                                                        begin
                                                                            [sc * t_te;
                                                                             sc * te] in
                                                                            MOI.NormOneCone(1 +
                                                                                            T)
                                                                            sc *
                                                                            (t_te - f * k) <=
                                                                            0
                                                                        end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::TrackingError{<:Any, <:Any, <:SOCTracking},
                                         args...; kwargs...)
    X = pr.X
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(te.tracking, X)
    err = te.err
    f = err * sqrt(size(X, 1) - te.alg.ddof)
    t_te = model[Symbol(:t_te_, i)] = @variable(model)
    te = model[Symbol(:te_, i)] = @expression(model, net_X - wb * k)
    model[Symbol(:cte_soc_, i)], model[Symbol(:cte_, i)] = @constraints(model,
                                                                        begin
                                                                            [sc * t_te;
                                                                             sc * te] in
                                                                            SecondOrderCone()
                                                                            sc *
                                                                            (t_te - f * k) <=
                                                                            0
                                                                        end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:IndependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         plg::Option{<:PhCUVecPhC}, fees::Option{<:Fees},
                                         args...; kwargs...)
    r = te.r
    wb = te.tracking.w
    err = te.err
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    te_dw = Symbol(:te_dw_, i)
    model[:oldw] = model[:w]
    unregister(model, :w)
    model[:w] = @expression(model, w - wb * k)
    risk_expr = set_triv_risk_constraints!(model, te_dw, r, opt, pr, plg, fees, args...;
                                           kwargs...)
    model[Symbol(:triv_, i, :_w)] = model[:w]
    model[:w] = model[:oldw]
    unregister(model, :oldw)
    model[Symbol(:cter_, i)] = @constraint(model, sc * (risk_expr - err * k) <= 0)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:DependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         plg::Option{<:PhCUVecPhC}, fees::Option{<:Fees},
                                         args...; kwargs...)
    ri = te.r
    wb = te.tracking.w
    err = te.err
    rb = expected_risk(factory(ri, pr, opt.opt.slv), wb, pr.X, fees)
    k = model[:k]
    sc = model[:sc]
    key = Symbol(:t_dr_, i)
    t_dr = model[key] = @variable(model)
    risk_expr = set_trdv_risk_constraints!(model, key, ri, opt, pr, plg, fees, args...;
                                           kwargs...)
    dr = model[Symbol(:dr_, i)] = @expression(model, risk_expr - rb * k)
    model[Symbol(:cter_noc_, i)], model[Symbol(:cter_, i)] = @constraints(model,
                                                                          begin
                                                                              [sc * t_dr;
                                                                               sc * dr] in
                                                                              MOI.NormOneCone(2)
                                                                              sc *
                                                                              (t_dr -
                                                                               err * k) <=
                                                                              0
                                                                          end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, pr::AbstractPriorResult,
                                         tres::Union{<:AbstractTracking, <:VecTr}, args...;
                                         kwargs...)
    for (i, te) in enumerate(tres)
        set_tracking_error_constraints!(model, i, pr, te, args...; kwargs...)
    end
    return nothing
end
