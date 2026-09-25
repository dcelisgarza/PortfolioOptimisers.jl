## ────────────────────────────────────────────────────────────────────────────
## Factor attribution
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_attribution_vol_contrib(fa::FactorAttributionResult;
                                                          by_family::Bool = false,
                                                          rd::ReturnsResult = ReturnsResult(),
                                                          nf::Option{<:AbstractVector} = nothing,
                                                          N::Option{<:Number} = nothing,
                                                          kwargs...)
    bd, labels = PortfolioOptimisers.attribution_plot_axis(fa, by_family, rd, nf)
    idx = attribution_plot_rows(bd.vol_contrib, N)
    return bar(view(bd.vol_contrib, idx);
               xticks = (1:length(idx), attribution_plot_labels(labels, idx)),
               xlabel = by_family ? "Family" : "Factor", ylabel = "Volatility contribution",
               title = "Factor Attribution: Volatility", xrotation = 90, legend = false,
               kwargs...)
end
function PortfolioOptimisers.plot_attribution_mu_contrib(fa::FactorAttributionResult;
                                                         by_family::Bool = false,
                                                         rd::ReturnsResult = ReturnsResult(),
                                                         nf::Option{<:AbstractVector} = nothing,
                                                         N::Option{<:Number} = nothing,
                                                         z::Number = 1.96, kwargs...)
    if !(z >= zero(z))
        throw(DomainError(z, "z must be >= 0"))
    end
    bd, labels = PortfolioOptimisers.attribution_plot_axis(fa, by_family, rd, nf)
    idx = attribution_plot_rows(bd.mu_contrib, N)
    err = isnothing(bd.mu_se) ? nothing : z * view(bd.mu_se, idx)
    return bar(view(bd.mu_contrib, idx);
               xticks = (1:length(idx), attribution_plot_labels(labels, idx)), yerror = err,
               xlabel = by_family ? "Family" : "Factor",
               ylabel = "Mean return contribution",
               title = "Factor Attribution: Mean Return", xrotation = 90, legend = false,
               kwargs...)
end
function PortfolioOptimisers.plot_attribution_exposure(fa::FactorAttributionResult;
                                                       by_family::Bool = false,
                                                       rd::ReturnsResult = ReturnsResult(),
                                                       nf::Option{<:AbstractVector} = nothing,
                                                       N::Option{<:Number} = nothing,
                                                       kwargs...)
    bd, labels = PortfolioOptimisers.attribution_plot_axis(fa, by_family, rd, nf)
    idx = attribution_plot_rows(bd.exposure, N)
    err = isnothing(bd.exposure_std) ? nothing : view(bd.exposure_std, idx)
    return bar(view(bd.exposure, idx);
               xticks = (1:length(idx), attribution_plot_labels(labels, idx)), yerror = err,
               xlabel = by_family ? "Family" : "Factor", ylabel = "Exposure",
               title = "Factor Attribution: Exposure", xrotation = 90, legend = false,
               kwargs...)
end
function PortfolioOptimisers.plot_attribution_mu_vs_vol(fa::FactorAttributionResult;
                                                        by_family::Bool = false,
                                                        rd::ReturnsResult = ReturnsResult(),
                                                        nf::Option{<:AbstractVector} = nothing,
                                                        N::Option{<:Number} = nothing,
                                                        kwargs...)
    bd, labels = PortfolioOptimisers.attribution_plot_axis(fa, by_family, rd, nf)
    idx = attribution_plot_rows(bd.vol_contrib, N)
    x = view(bd.vol_contrib, idx)
    y = view(bd.mu_contrib, idx)
    return scatter(x, y;
                   series_annotations = text.(attribution_plot_labels(labels, idx), 8,
                                              :bottom), xlabel = "Volatility contribution",
                   ylabel = "Mean return contribution",
                   title = "Factor Attribution: Return against Risk", legend = false,
                   kwargs...)
end
# The rows one attribution plot draws, largest in magnitude first and then back in axis order, so a
# plot with an `N` shows the rows that matter without reordering the axis it draws.
function attribution_plot_rows(v::VecNum, N::Option{<:Number})
    M = length(v)
    if isnothing(N)
        return 1:M
    end
    if !(N > zero(N))
        throw(DomainError(N, "N must be > 0"))
    end
    n = clamp(ceil(Int, N), 1, M)
    return sort!(partialsortperm(abs.(v), 1:n; rev = true))
end
# The labels of the rows an attribution plot draws.
function attribution_plot_labels(labels, idx)
    return [labels[i] for i in idx]
end
