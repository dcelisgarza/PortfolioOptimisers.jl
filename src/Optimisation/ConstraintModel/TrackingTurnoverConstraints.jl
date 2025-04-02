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
                                         tre::TrackingError)
    k = model[:k]
    sc = model[:sc]
    set_net_portfolio_returns!(model, X)
    net_X = model[:net_X]
    wb = tracking_error_benchmark(tre.tracking, X)
    @variable(model, t_tre)
    @expression(model, tre, net_X - wb * k)
    @constraints(model, begin
                     ctre_soc, sc * [t_tre; tre] ∈ SecondOrderCone()
                     ctre, sc * t_tre <= sc * tre.err * k * sqrt(size(X, 1) - 1)
                 end)
    return nothing
end
function set_turnover_constraints!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model, turnover::Turnover)
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variable(model, t_tr[1:N])
    @expression(model, tr, w - turnover.w * k)
    @constraints(model, begin
                     ctr_noc[i = 1:N], sc * [t_tr[i]; tr[i]] ∈ MOI.NormOneCone(2)
                     ctr, sc * t_tr <= sc * tr.val * k
                 end)
    return nothing
end
