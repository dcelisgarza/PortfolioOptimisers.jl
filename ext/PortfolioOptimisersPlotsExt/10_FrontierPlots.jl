## ────────────────────────────────────────────────────────────────────────────
## Portfolio dashboard (multi-panel composite)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_portfolio_dashboard(res::OptimisationResult, rd::Pr_RR;
                                                      slv::Option{<:Slv_VecSlv} = nothing,
                                                      ts = 1:size(rd.X, 1),
                                                      nx = 1:size(rd.X, 2),
                                                      r::PortfolioOptimisers.BaseRM_VecBaseRM = Variance(),
                                                      compound::Bool = false,
                                                      N::Option{<:Number} = nothing,
                                                      delta::Number = 1e-6,
                                                      marginal::Bool = false,
                                                      percentage::Bool = true,
                                                      alpha::Number = 0.05,
                                                      kappa::Number = 0.3, rw = nothing,
                                                      sca::Scalariser = SumScalariser(),
                                                      kwargs...)
    _, w, rd, fees, nx = result_investable_view(res, rd, nothing, nx)
    if isa(rd, ReturnsResult)
        nx = isnothing(rd.nx) ? nx : rd.nx
        ts = isnothing(rd.ts) ? ts : rd.ts
    end
    p1 = PortfolioOptimisers.plot_composition(w, nx; N = N)
    p2 = PortfolioOptimisers.plot_portfolio_cumulative_returns(w, rd.X, fees; ts = ts,
                                                               compound = compound)
    # The carrier is handed whole where it is a prior, so a measure with an unstated slot,
    # the default `Variance()` among them, resolves it there; a returns result carries no
    # moment to resolve against and is unwrapped to its matrix.
    p3 = PortfolioOptimisers.plot_risk_contribution(r, w,
                                                    isa(rd, ReturnsResult) ? rd.X : rd,
                                                    fees; nx = nx, N = N, delta = delta,
                                                    marginal = marginal,
                                                    percentage = percentage, sca = sca)
    p4 = PortfolioOptimisers.plot_drawdowns(w, rd.X, fees; slv = slv, ts = ts,
                                            compound = compound, alpha = alpha,
                                            kappa = kappa, rw = rw)
    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Efficient frontier
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_efficient_frontier(res_vec::AbstractVector{<:OptimisationResult},
                                                     pr::Pr_RR;
                                                     x::PortfolioOptimisers.BaseRM_VecBaseRM = Variance(),
                                                     y::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturn(),
                                                     c::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturnRiskRatio(;
                                                                                                                       rk = x,
                                                                                                                       rt = ArithmeticReturn(),
                                                                                                                       rf = 0),
                                                     slv::Option{<:Slv_VecSlv} = nothing,
                                                     fees::Option{<:Fees} = nothing,
                                                     min_risk::Bool = true,
                                                     max_score::Bool = true,
                                                     factory::Bool = true, kwargs...)
    if factory
        x = PortfolioOptimisers.factory(x, pr, slv)
        y = PortfolioOptimisers.factory(y, pr, slv)
        c = PortfolioOptimisers.factory(c, pr, slv)
    end
    w = getproperty.(res_vec, :w)
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    return efficient_frontier_plot(xr, yr, cr, measure_label(x), measure_label(y),
                                   measure_label(c); min_risk = min_risk,
                                   max_score = max_score, kwargs...)
end
function efficient_frontier_plot(xr::VecNum, yr::VecNum, cr::VecNum, xname::AbstractString,
                                 yname::AbstractString, cname::AbstractString;
                                 min_risk::Bool = true, max_score::Bool = true, kwargs...)
    order = sortperm(xr)
    xr_s = xr[order]
    yr_s = yr[order]
    cr_s = cr[order]
    plt = plot(xr_s, yr_s; zcolor = cr_s, line_z = cr_s, title = "Efficient Frontier",
               xlabel = xname, ylabel = yname, colorbar_title = cname, label = nothing,
               linewidth = 2, markershape = :circle, markersize = 4, kwargs...)
    if min_risk
        i = argmin(xr_s)
        scatter!(plt, [xr_s[i]], [yr_s[i]]; label = "Min Risk", markershape = :star5,
                 markersize = 12, color = :blue, legend = true)
    end
    if max_score
        i = argmax(cr_s)
        scatter!(plt, [xr_s[i]], [yr_s[i]]; label = "Max $(cname)", markershape = :star5,
                 markersize = 12, color = :red, legend = true)
    end
    return plt
end
function PortfolioOptimisers.plot_efficient_frontier(res_vec::AbstractVector{<:OptimisationResult},
                                                     ::ReturnsResult;
                                                     x::PortfolioOptimisers.BaseRM_VecBaseRM = Variance(),
                                                     y::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturn(),
                                                     c::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturnRiskRatio(;
                                                                                                                       rk = x,
                                                                                                                       rt = ArithmeticReturn(),
                                                                                                                       rf = 0),
                                                     slv::Option{<:Slv_VecSlv} = nothing,
                                                     fees::Option{<:Fees} = nothing,
                                                     min_risk::Bool = true,
                                                     max_score::Bool = true,
                                                     factory::Bool = true, kwargs...)
    # Each result carries its own prior, on its own investable universe, so each point is
    # scored against the three the result carries, as `plot_measures` scores a vector of
    # results. The returns result is the tag that asks for the results' own priors.
    views = [result_investable_view(res, nothing, fees) for res in res_vec]
    w = getindex.(views, 2)
    pr = getindex.(views, 3)
    fees = getindex.(views, 4)
    n = length(res_vec)
    function per_result(m)
        return factory ? PortfolioOptimisers.factory.(Ref(m), pr, Ref(slv)) : fill(m, n)
    end
    xr = expected_risk.(per_result(x), w, pr, fees)
    yr = expected_risk.(per_result(y), w, pr, fees)
    cr = expected_risk.(per_result(c), w, pr, fees)
    return efficient_frontier_plot(xr, yr, cr, measure_label(x), measure_label(y),
                                   measure_label(c); min_risk = min_risk,
                                   max_score = max_score, kwargs...)
end
function PortfolioOptimisers.plot_efficient_frontier(w::VecVecNum, pr::Pr_RR;
                                                     x::PortfolioOptimisers.BaseRM_VecBaseRM = Variance(),
                                                     y::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturn(),
                                                     c::PortfolioOptimisers.BaseRM_VecBaseRM = ExpectedReturnRiskRatio(;
                                                                                                                       rk = x,
                                                                                                                       rt = ArithmeticReturn(),
                                                                                                                       rf = 0),
                                                     slv::Option{<:Slv_VecSlv} = nothing,
                                                     fees::Option{<:Fees} = nothing,
                                                     min_risk::Bool = true,
                                                     max_score::Bool = true,
                                                     factory::Bool = true, kwargs...)
    if factory
        x = PortfolioOptimisers.factory(x, pr, slv)
        y = PortfolioOptimisers.factory(y, pr, slv)
        c = PortfolioOptimisers.factory(c, pr, slv)
    end
    xr = expected_risk(x, w, pr, fees)
    yr = expected_risk(y, w, pr, fees)
    cr = expected_risk(c, w, pr, fees)
    return efficient_frontier_plot(xr, yr, cr, measure_label(x), measure_label(y),
                                   measure_label(c); min_risk = min_risk,
                                   max_score = max_score, kwargs...)
end
function PortfolioOptimisers.plot_efficient_frontier(res::OptimisationResult, pr::Pr_RR;
                                                     fees::Option{<:Fees} = nothing,
                                                     kwargs...)
    _, w, pr, fees = result_investable_view(res, pr, fees)
    return PortfolioOptimisers.plot_efficient_frontier(isa(w, VecVecNum) ? w : [w], pr;
                                                       fees = fees, kwargs...)
end
function PortfolioOptimisers.plot_efficient_frontier(res::OptimisationResult,
                                                     ::ReturnsResult;
                                                     fees::Option{<:Fees} = nothing,
                                                     kwargs...)
    # The returns result is the tag that asks for the result's own prior.
    _, w, pr, fees = result_investable_view(res, nothing, fees)
    return PortfolioOptimisers.plot_efficient_frontier(isa(w, VecVecNum) ? w : [w], pr;
                                                       fees = fees, kwargs...)
end
