abstract type PortfolioReturnType end
function cluster_return_factory(r::PortfolioReturnType; kwargs...)
    return r
end
function set_portfolio_returns!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :X)
        return nothing
    end
    w = model[:w]
    @expression(model, X, X * w)
    return nothing
end
function set_net_portfolio_returns!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :net_X)
        return nothing
    end
    set_portfolio_returns!(model, X)
    X = model[:X]
    fees = model[:fees]
    @expression(model, net_X, X .- fees)
    return nothing
end
function set_model_fees!(model)
    @expression(model, fees, zero(AffExpr))
    return nothing
end
