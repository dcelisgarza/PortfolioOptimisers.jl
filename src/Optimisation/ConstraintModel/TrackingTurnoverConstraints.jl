function tracking_error_benchmark(tracking::WeightsTracking, X::AbstractMatrix)
    return calc_net_returns(X, tracking.w, tracking.fees)
end
function tracking_error_benchmark(tracking::ReturnsTracking, args...)
    return tracking.w
end
function set_tracking_error_constraints!(::JuMP.Model, ::NoTrackingError)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, X::AbstractMatrix,
                                         tre::TrackingError{<:Any, <:ReturnsTracking})
    w, k, sc = get_w_k_sc(model)
    get_net_portfolio_returns(model, X)
    net_X = model[:net_X]
    tre.tracking.w
    return nothing
end