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
    set_fees!(model)
    X = model[:X]
    fees = model[:fees]
    @expression(model, net_X, X .- fees)
    return nothing
end
function set_fees!(model)
    if haskey(model, :fees)
        return nothing
    end
    @expression(model, fs, zero(AffExpr))
    if haskey(model, :fl)
        add_to_expression!(fs, model[:fl])
    end
    if haskey(model, :fs)
        add_to_expression!(fs, model[:fs])
    end
    if haskey(model, :ffl)
        add_to_expression!(fs, model[:ffl])
    end
    if haskey(model, :ffs)
        add_to_expression!(fs, model[:ffs])
    end
    if haskey(model, :trfs)
        add_to_expression!(fs, model[:trfs])
    end
    return nothing
end
