function tracking_error_benchmark(::Nothing, ::AbstractMatrix)
    return nothing
end
function tracking_error_benchmark(tracking::WeightsTracking, X::AbstractMatrix)
    return calc_net_returns(tracking.w, X, tracking.fees)
end
function tracking_error_benchmark(tracking::ReturnsTracking, args...)
    return tracking.w
end
function set_tracking_error_constraints!(::JuMP.Model, X::AbstractMatrix, ::Nothing)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, X::AbstractMatrix,
                                         tr::TrackingError)
    k = model[:k]
    sc = model[:sc]
    set_net_portfolio_returns!(model, X)
    net_X = model[:net_X]
    wb = tracking_error_benchmark(tr.tracking, X)
    err = tr.err
    @variable(model, t_tr)
    @expression(model, tr, net_X - wb * k)
    @constraints(model, begin
                     ctr_soc, [sc * t_tr; sc * tr] ∈ SecondOrderCone()
                     ctr, sc * t_tr <= sc * err * k * sqrt(size(X, 1) - 1)
                 end)
    return nothing
end
function set_turnover_constraints!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model, tn::Turnover)
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    wi = tn.w
    val = tn.val
    @variable(model, t_tn[1:N])
    @expression(model, tn, w - wi * k)
    @constraints(model,
                 begin
                     ctr_noc[i = 1:N], [sc * t_tn[i]; sc * tn[i]] ∈ MOI.NormOneCone(2)
                     ctr, sc * t_tn <= sc * val * k
                 end)
    return nothing
end
