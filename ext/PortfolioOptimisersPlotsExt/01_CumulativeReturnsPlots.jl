## plot_portfolio_cumulative_returns
function PortfolioOptimisers.plot_portfolio_cumulative_returns(net_ret::VecNum_VecVecNum;
                                                               ts::AbstractVector,
                                                               compound::Bool = false,
                                                               kwargs...)
    ret = cumulative_returns(net_ret, compound)
    label = "$(compound ? "Compound" : "Simple") Cumulative Returns"
    return if isa(net_ret, VecNum)
        plot(ts, ret; title = "Portfolio", xlabel = "Date", ylabel = label, legend = false,
             kwargs...)
    else
        plt = plot(ts, ret[1]; title = "Portfolio", xlabel = "Date", ylabel = label,
                   label = "Portfolio 1", legend = true, kwargs...)
        for i in 2:length(net_ret)
            plot!(plt, ts, ret[i]; label = "Portfolio $(i)", kwargs...)
        end
        plt
    end
end
function PortfolioOptimisers.plot_portfolio_cumulative_returns(w::VecNum_VecVecNum,
                                                               X::MatNum,
                                                               fees::Option{<:Fees} = nothing;
                                                               ts::AbstractVector = 1:size(X,
                                                                                           1),
                                                               compound::Bool = false,
                                                               kwargs...)
    return PortfolioOptimisers.plot_portfolio_cumulative_returns(calc_net_returns(w, X,
                                                                                  fees);
                                                                 ts = ts,
                                                                 compound = compound,
                                                                 kwargs...)
end
function PortfolioOptimisers.plot_portfolio_cumulative_returns(w::VecNum_VecVecNum,
                                                               pr::Pr_RR,
                                                               fees::Option{<:Fees} = nothing;
                                                               ts::AbstractVector = 1:size(pr.X,
                                                                                           1),
                                                               compound::Bool = false,
                                                               kwargs...)
    if isa(pr, ReturnsResult)
        ts = isnothing(pr.ts) ? (1:size(pr.X, 1)) : pr.ts
    end
    return PortfolioOptimisers.plot_portfolio_cumulative_returns(w, pr.X, fees; ts = ts,
                                                                 compound = compound,
                                                                 kwargs...)
end
function PortfolioOptimisers.plot_portfolio_cumulative_returns(res::OptimisationResult,
                                                               pr::Pr_RR;
                                                               fees::Option{<:Fees} = nothing,
                                                               compound::Bool = false,
                                                               kwargs...)
    _, w, pr, fees = result_investable_view(res, pr, fees)
    return PortfolioOptimisers.plot_portfolio_cumulative_returns(w, pr, fees;
                                                                 compound = compound,
                                                                 kwargs...)
end
function PortfolioOptimisers.plot_portfolio_cumulative_returns(res::OptimisationResult;
                                                               fees::Option{<:Fees} = nothing,
                                                               compound::Bool = false,
                                                               kwargs...)
    _, w, pr, fees = result_investable_view(res, nothing, fees)
    return PortfolioOptimisers.plot_portfolio_cumulative_returns(w, pr, fees;
                                                                 compound = compound,
                                                                 kwargs...)
end
function PortfolioOptimisers.plot_portfolio_cumulative_returns(pred::Union{<:PredictionResult,
                                                                           <:MultiPeriodPredictionResult};
                                                               compound::Bool = false,
                                                               kwargs...)
    rd = isa(pred, PredictionResult) ? pred.rd : pred.mrd
    ts = isnothing(rd.ts) ? (1:length(isa(rd.X, VecVecNum) ? first(rd.X) : rd.X)) : rd.ts
    return PortfolioOptimisers.plot_portfolio_cumulative_returns(rd.X; ts = ts,
                                                                 compound = compound,
                                                                 kwargs...)
end
function PortfolioOptimisers.plot_portfolio_cumulative_returns(pred::PopulationPredictionResult;
                                                               compound::Bool = false,
                                                               kwargs...)
    plt = plot(; kwargs...)
    for p in pred.pred
        plot!(plt,
              PortfolioOptimisers.plot_portfolio_cumulative_returns(p; compound = compound,
                                                                    kwargs...))
    end
    return plt
end
## plot_asset_cumulative_returns
function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, X::MatNum,
                                                           fees::Option{<:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(X, 1),
                                                           nx::AbstractVector = 1:size(X, 2),
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    net_asset_ret = calc_net_asset_returns(w, X, fees)
    ret = cumulative_returns(net_asset_ret, compound)
    M = size(X, 2)
    N, idx = relevant_assets(w, M, N)
    ret_sorted = view(ret, :, idx)
    nx_sorted = view(nx, idx)

    label_str = "$(compound ? "Compound" : "Simple") Asset Cumulative Returns"
    f = plot(; xlabel = "Date", ylabel = label_str)
    for i in 1:N
        plot!(f, ts, view(ret_sorted, :, i); label = string(nx_sorted[i]))
    end
    if M > N
        rest_idx = finite_columns(X, view(idx, (N + 1):M))
        rest_ret = cumulative_returns(calc_net_returns(view(w, rest_idx),
                                                       view(X, :, rest_idx),
                                                       PortfolioOptimisers.port_opt_view(fees,
                                                                                         rest_idx)),
                                      compound)
        plot!(f, ts, rest_ret; label = "Others")
    end
    plot!(f; legend = :outerright, kwargs...)
    return f
end
function PortfolioOptimisers.plot_asset_cumulative_returns(w::VecNum, pr::Pr_RR,
                                                           fees::Option{<:Fees} = nothing;
                                                           ts::AbstractVector = 1:size(pr.X,
                                                                                       1),
                                                           nx::AbstractVector = 1:size(pr.X,
                                                                                       2),
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    if isa(pr, ReturnsResult)
        ts = isnothing(pr.ts) ? (1:size(pr.X, 1)) : pr.ts
        nx = isnothing(pr.nx) ? (1:size(pr.X, 2)) : pr.nx
    end
    return PortfolioOptimisers.plot_asset_cumulative_returns(w, pr.X, fees; ts = ts,
                                                             nx = nx, compound = compound,
                                                             N = N, kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(res::OptimisationResult,
                                                           pr::Pr_RR;
                                                           fees::Option{<:Fees} = nothing,
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    imsk, w, pr, fees = result_investable_view(res, pr, fees)
    # The per-asset matrix is on the investable universe, where a liquidated asset has no
    # column for its exit charge to land in, so the two carriers are dropped: the asset
    # that left is not drawn, and neither is its charge. The portfolio figure charges it.
    return PortfolioOptimisers.plot_asset_cumulative_returns(w, pr,
                                                             strip_liquidation_carriers(fees,
                                                                                        nothing);
                                                             nx = result_axis_names(imsk,
                                                                                    pr,
                                                                                    nothing),
                                                             compound = compound, N = N,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(res::OptimisationResult;
                                                           fees::Option{<:Fees} = nothing,
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    imsk, w, pr, fees = result_investable_view(res, nothing, fees)
    return PortfolioOptimisers.plot_asset_cumulative_returns(w, pr,
                                                             strip_liquidation_carriers(fees,
                                                                                        nothing);
                                                             nx = result_axis_names(imsk,
                                                                                    pr,
                                                                                    nothing),
                                                             compound = compound, N = N,
                                                             kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(pred::PredictionResult;
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    return PortfolioOptimisers.plot_asset_cumulative_returns(pred.res; compound = compound,
                                                             N = N, kwargs...)
end
function PortfolioOptimisers.plot_asset_cumulative_returns(pred::MultiPeriodPredictionResult;
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    res_vec = pred.res
    X = Vector{eltype(first(res_vec).rd.X)}[]
    ts = Vector{eltype(first(res_vec).rd.ts)}[]
    nx = first(res_vec).rd.nx
    M = length(nx)
    mean_w = zeros(length(first(res_vec).w), M)
    for res in res_vec
        w = res.w
        pr = extract_pr(res)
        fees = extract_fees(res)
        mean_w .+= w
        net_asset_ret = calc_net_asset_returns(w, pr.X, fees,
                                               PortfolioOptimisers.result_investable_mask(res))
        ret = cumulative_returns(net_asset_ret, compound)
        append!(X, vec(ret))
        append!(ts, res.rd.ts)
    end
    mean_w ./= length(res_vec)
    X = reshape(X, length(ts), M)
    N, idx = relevant_assets(mean_w, M, N)
    ret_sorted = view(X, :, idx)
    nx_sorted = view(nx, idx)
    label_str = "$(compound ? "Compound" : "Simple") Asset Cumulative Returns"
    f = plot(; xlabel = "Date", ylabel = label_str)
    for i in 1:N
        plot!(f, ts, view(ret_sorted, :, i); label = string(nx_sorted[i]))
    end
    if M > N
        # The rest is the assets the ranking left out, which is `view(idx, (N + 1):M)` and
        # not the first `N` columns of `X`: `idx` is a ranking, so the two coincide only
        # when the ranking is the identity.
        rest_idx = finite_columns(X, view(idx, (N + 1):M))
        rest_ret = vec(sum(view(X, :, rest_idx); dims = 2))
        plot!(f, ts, rest_ret; label = "Others")
    end
    plot!(f; legend = :outerright, kwargs...)
    return f
end
function PortfolioOptimisers.plot_asset_cumulative_returns(pred::PopulationPredictionResult;
                                                           compound::Bool = false,
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    plt = plot(; kwargs...)
    for p in pred.pred
        plot!(plt,
              PortfolioOptimisers.plot_asset_cumulative_returns(p; compound = compound,
                                                                N = N, kwargs...))
    end
    return plt
end
## plot_benchmark
function PortfolioOptimisers.plot_benchmark(w::ArrNum, rd::ReturnsResult,
                                            fees::Option{<:Fees} = nothing;
                                            compound::Bool = false, kwargs...)
    if isnothing(rd.B)
        throw(ArgumentError("returns data has no benchmark (`B` is `nothing`)"))
    end
    ts = isnothing(rd.ts) ? (1:size(rd.X, 1)) : rd.ts
    nb = rd.nb
    return PortfolioOptimisers.plot_benchmark(w, rd.X, rd.B, fees; ts = ts, nb = nb,
                                              compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(res::OptimisationResult, rd::ReturnsResult;
                                            compound::Bool = false, kwargs...)
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_benchmark(w, rd, fees; compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(pred::PredictionResult; compound::Bool = false,
                                            kwargs...)
    rd = pred.rd
    if isnothing(rd.B)
        throw(ArgumentError("prediction data has no benchmark (`B` is `nothing`)"))
    end
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    ts = isnothing(rd.ts) ? (1:length(ret)) : rd.ts
    nb = rd.nb
    B = isa(rd.B, VecVecNum) ? first(rd.B) : rd.B
    return PortfolioOptimisers.plot_benchmark(ret, B; ts = ts, nb = nb, compound = compound,
                                              kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Benchmark overlay
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_benchmark(w::ArrNum, X::MatNum, B::VecNum_VecVecNum,
                                            fees::Option{<:Fees} = nothing;
                                            ts::AbstractVector = 1:size(X, 1),
                                            nb::Option{<:AbstractVector} = nothing,
                                            compound::Bool = false, kwargs...)
    return PortfolioOptimisers.plot_benchmark(calc_net_returns(w, X, fees), B; ts = ts,
                                              nb = nb, compound = compound, kwargs...)
end
function PortfolioOptimisers.plot_benchmark(net_ret::VecNum, B::VecNum_VecVecNum;
                                            ts::AbstractVector,
                                            nb::Option{<:AbstractVector} = nothing,
                                            compound::Bool = false, kwargs...)
    ret = cumulative_returns(net_ret, compound)
    Bmat = isa(B, VecVecNum) ? hcat(B...) : reshape(B, :, 1)
    Nb = size(Bmat, 2)
    bench_labels = isnothing(nb) ? ["Benchmark $i" for i in 1:Nb] : string.(view(nb, 1:Nb))
    f = plot(ts, ret; label = "Portfolio", linewidth = 2, title = "Portfolio vs Benchmark",
             xlabel = "Date",
             ylabel = "$(compound ? "Compound" : "Simple") Cumulative Returns",
             legend = :outerright, kwargs...)
    for i in 1:Nb
        b_ret = cumulative_returns(vec(view(Bmat, :, i)), compound)
        plot!(f, ts, b_ret; label = bench_labels[i], linewidth = 1.5, linestyle = :dash)
    end
    return f
end
function PortfolioOptimisers.plot_benchmark(pred::MultiPeriodPredictionResult;
                                            compound::Bool = false, kwargs...)
    mrd = pred.mrd
    if isnothing(mrd.B)
        throw(ArgumentError("multi-period prediction data has no benchmark (`B` is `nothing`)"))
    end
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    ts = isnothing(mrd.ts) ? (1:length(ret)) : mrd.ts
    return PortfolioOptimisers.plot_benchmark(ret, mrd.B; ts = ts, nb = mrd.nb,
                                              compound = compound, kwargs...)
end
