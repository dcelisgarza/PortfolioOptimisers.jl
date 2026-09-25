## plot_correlation
function PortfolioOptimisers.plot_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                              nx::AbstractVector = 1:size(pr.sigma, 1);
                                              kwargs...)
    return PortfolioOptimisers.plot_correlation(pr.sigma, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(pr::PortfolioOptimisers.AbstractPriorResult,
                                              rd::ReturnsResult; kwargs...)
    nx = isnothing(rd.nx) ? (1:size(pr.sigma, 1)) : rd.nx
    return PortfolioOptimisers.plot_correlation(pr.sigma, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(res::OptimisationResult, rd::ReturnsResult;
                                              kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    return PortfolioOptimisers.plot_correlation(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(res::OptimisationResult; kwargs...)
    pr, nx = result_prior_view(res)
    return PortfolioOptimisers.plot_correlation(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(pred::PredictionResult, rd::ReturnsResult;
                                              kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    return PortfolioOptimisers.plot_correlation(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_correlation(pred::PredictionResult; kwargs...)
    pr, nx = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_correlation(pr, nx; kwargs...)
end
## plot_mu
function PortfolioOptimisers.plot_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                     nx::AbstractVector = 1:length(pr.mu);
                                     N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_mu(pr.mu, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                     rd::ReturnsResult; N::Option{<:Number} = nothing,
                                     kwargs...)
    nx = isnothing(rd.nx) ? (1:length(pr.mu)) : rd.nx
    return PortfolioOptimisers.plot_mu(pr.mu, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(res::OptimisationResult; N::Option{<:Number} = nothing,
                                     kwargs...)
    pr, nx = result_prior_view(res)
    return PortfolioOptimisers.plot_mu(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(res::OptimisationResult, rd::ReturnsResult;
                                     N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    return PortfolioOptimisers.plot_mu(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(pred::PredictionResult; N::Option{<:Number} = nothing,
                                     kwargs...)
    pr, nx = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_mu(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_mu(pred::PredictionResult, rd::ReturnsResult;
                                     N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    return PortfolioOptimisers.plot_mu(pr, nx; N = N, kwargs...)
end
## plot_sigma
function PortfolioOptimisers.plot_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                        nx::AbstractVector = 1:size(pr.sigma, 1);
                                        N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_sigma(pr.sigma, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                        rd::ReturnsResult; N::Option{<:Number} = nothing,
                                        kwargs...)
    nx = isnothing(rd.nx) ? (1:size(pr.sigma, 1)) : rd.nx
    return PortfolioOptimisers.plot_sigma(pr.sigma, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(res::OptimisationResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    return PortfolioOptimisers.plot_sigma(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(res::OptimisationResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(res)
    return PortfolioOptimisers.plot_sigma(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pred::PredictionResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_sigma(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_sigma(pred::PredictionResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    return PortfolioOptimisers.plot_sigma(pr, nx; N = N, kwargs...)
end
## plot_eigenspectrum
function PortfolioOptimisers.plot_eigenspectrum(pr::PortfolioOptimisers.AbstractPriorResult;
                                                reference::Bool = true, kwargs...)
    pr_i, = investable_plot_view(pr)
    return PortfolioOptimisers.plot_eigenspectrum(pr_i.sigma; reference = reference,
                                                  kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pr::PortfolioOptimisers.AbstractPriorResult,
                                                rd::ReturnsResult; reference::Bool = true,
                                                kwargs...)
    T = isnothing(rd.X) ? nothing : size(rd.X, 1)
    pr_i, = investable_plot_view(pr)
    return PortfolioOptimisers.plot_eigenspectrum(pr_i.sigma; N_obs = T,
                                                  reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(res::OptimisationResult;
                                                reference::Bool = true, kwargs...)
    pr, = result_prior_view(res)
    return PortfolioOptimisers.plot_eigenspectrum(pr; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(res::OptimisationResult, rd::ReturnsResult;
                                                reference::Bool = true, kwargs...)
    pr, = result_prior_view(res)
    return PortfolioOptimisers.plot_eigenspectrum(pr, rd; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pred::PredictionResult;
                                                reference::Bool = true, kwargs...)
    pr, = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_eigenspectrum(pr; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_eigenspectrum(pred::PredictionResult, rd::ReturnsResult;
                                                reference::Bool = true, kwargs...)
    pr, = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_eigenspectrum(pr, rd; reference = reference, kwargs...)
end
## plot_prior
function PortfolioOptimisers.plot_prior(pr::PortfolioOptimisers.AbstractPriorResult,
                                        rd::ReturnsResult; N::Option{<:Number} = nothing,
                                        kwargs...)
    nx = isnothing(rd.nx) ? (1:length(pr.mu)) : rd.nx
    return PortfolioOptimisers.plot_prior(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(res::OptimisationResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(res)
    return PortfolioOptimisers.plot_prior(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(res::OptimisationResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    return PortfolioOptimisers.plot_prior(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(pred::PredictionResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_prior(pr, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_prior(pred::PredictionResult, rd::ReturnsResult;
                                        N::Option{<:Number} = nothing, kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    return PortfolioOptimisers.plot_prior(pr, nx; N = N, kwargs...)
end
## plot_coskewness
# The axis names default to `nothing` and resolve after the guard: a default taken off
# `pr.sk` or `pr.kt` would run before the guard and raise a `MethodError` on `size(nothing)`.
function PortfolioOptimisers.plot_coskewness(pr::HighOrderPrior,
                                             nx::Option{<:AbstractVector} = nothing;
                                             kwargs...)
    if isnothing(pr.sk)
        throw(ArgumentError("prior has no coskewness matrix (`sk` is `nothing`)"))
    end
    nx = isnothing(nx) ? (1:size(pr.sk, 1)) : nx
    return PortfolioOptimisers.plot_coskewness(pr.sk, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pr::HighOrderPrior, rd::ReturnsResult;
                                             kwargs...)
    if isnothing(pr.sk)
        throw(ArgumentError("prior has no coskewness matrix (`sk` is `nothing`)"))
    end
    nx = isnothing(rd.nx) ? (1:size(pr.sk, 1)) : rd.nx
    return PortfolioOptimisers.plot_coskewness(pr.sk, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(res::OptimisationResult; kwargs...)
    pr, nx = result_prior_view(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(res::OptimisationResult, rd::ReturnsResult;
                                             kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pred::PredictionResult; kwargs...)
    pr, nx = result_prior_view(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_coskewness(pred::PredictionResult, rd::ReturnsResult;
                                             kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no coskewness available"))
    end
    return PortfolioOptimisers.plot_coskewness(pr, nx; kwargs...)
end
## plot_cokurtosis
function PortfolioOptimisers.plot_cokurtosis(pr::HighOrderPrior,
                                             nx::Option{<:AbstractVector} = nothing;
                                             heatmap::Bool = false, reference::Bool = true,
                                             kwargs...)
    if isnothing(pr.kt)
        throw(ArgumentError("prior has no cokurtosis matrix (`kt` is `nothing`)"))
    end
    nx = isnothing(nx) ? (1:isqrt(size(pr.kt, 1))) : nx
    if heatmap
        return PortfolioOptimisers.plot_cokurtosis(pr.kt, nx; heatmap = true,
                                                   reference = reference, kwargs...)
    end
    pr_i, = investable_plot_view(pr)
    return PortfolioOptimisers.plot_cokurtosis(pr_i.kt, nx; heatmap = false,
                                               reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pr::HighOrderPrior, rd::ReturnsResult;
                                             heatmap::Bool = false, reference::Bool = true,
                                             kwargs...)
    if isnothing(pr.kt)
        throw(ArgumentError("prior has no cokurtosis matrix (`kt` is `nothing`)"))
    end
    nx = isnothing(rd.nx) ? (1:isqrt(size(pr.kt, 1))) : rd.nx
    return PortfolioOptimisers.plot_cokurtosis(pr, nx; heatmap = heatmap,
                                               reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(res::OptimisationResult;
                                             reference::Bool = true, kwargs...)
    pr, nx = result_prior_view(res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, nx; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(res::OptimisationResult, rd::ReturnsResult;
                                             reference::Bool = true, kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, nx; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pred::PredictionResult; reference::Bool = true,
                                             kwargs...)
    pr, nx = result_prior_view(pred.res)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, nx; reference = reference, kwargs...)
end
function PortfolioOptimisers.plot_cokurtosis(pred::PredictionResult, rd::ReturnsResult;
                                             reference::Bool = true, kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    if !(isa(pr, HighOrderPrior))
        throw(ArgumentError("`$(nameof(typeof(pred.res)))` prior is not a `HighOrderPrior`; no cokurtosis available"))
    end
    return PortfolioOptimisers.plot_cokurtosis(pr, nx; reference = reference, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Correlation / covariance heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_correlation(X::MatNum, nx::AbstractVector = 1:size(X, 1);
                                              kwargs...)
    S = copy(X)
    s = LinearAlgebra.diag(S)
    if any(!isone, s)
        s .= sqrt.(s)
        StatsBase.cov2cor!(S, s)
        clim = (-1.0, 1.0)
    else
        clim = extrema(S)
    end
    N = size(S, 1)
    return heatmap(S; xticks = (1:N, nx), yticks = (1:N, nx), xrotation = 90, clim = clim,
                   color = cgrad(:Spectral), yflip = true, title = "Correlation Matrix",
                   colorbar_title = "ρ", kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Expected returns bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_mu(mu::VecNum, nx::AbstractVector = 1:length(mu);
                                     N::Option{<:Number} = nothing, kwargs...)
    M = length(mu)
    N, idx = relevant_assets(mu, M, N)
    sort!(view(idx, 1:N))
    top_idx = view(idx, 1:N)
    return bar(mu[top_idx]; xticks = (1:N, string.(nx[top_idx])),
               title = "Expected Returns", xlabel = "Asset", ylabel = "μ", xrotation = 90,
               legend = false, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Asset volatility bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_sigma(sigma::MatNum,
                                        nx::AbstractVector = 1:size(sigma, 1);
                                        variance::Bool = false,
                                        N::Option{<:Number} = nothing, kwargs...)
    vals = variance ? LinearAlgebra.diag(sigma) : sqrt.(LinearAlgebra.diag(sigma))
    ylabel_str = variance ? "Variance (σ²)" : "Volatility (σ)"
    M = length(vals)
    idx = sortperm(finite_magnitudes(vals); rev = true)
    N_show = isnothing(N) ? M : clamp(ceil(Int, N), 1, M)
    top_idx = idx[1:N_show]
    sort!(top_idx)
    return bar(vals[top_idx]; xticks = (1:N_show, string.(nx[top_idx])),
               title = "Asset Volatility", xlabel = "Asset", ylabel = ylabel_str,
               xrotation = 90, legend = false, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Eigenvalue spectrum
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_eigenspectrum(sigma::MatNum;
                                                N_obs::Option{<:Integer} = nothing,
                                                reference::Bool = true, kwargs...)
    ev = eigvals(Symmetric(sigma))
    sorted_ev = sort(real.(ev); rev = true)
    N = length(sorted_ev)

    f = bar(1:N, sorted_ev; title = "Eigenspectrum", xlabel = "Component",
            ylabel = "Eigenvalue", legend = reference && !isnothing(N_obs), kwargs...)

    if reference && !isnothing(N_obs)
        q = N / N_obs
        s2 = tr(sigma) / N
        λ_plus = s2 * (1 + sqrt(q))^2
        hline!(f, [λ_plus]; label = "MP upper bound (λ₊=$(round(λ_plus; digits=4)))",
               linewidth = 2, color = :red, linestyle = :dash)
        if q < 1
            λ_minus = s2 * (1 - sqrt(q))^2
            hline!(f, [λ_minus]; label = "MP lower bound (λ₋=$(round(λ_minus; digits=4)))",
                   linewidth = 2, color = :orange, linestyle = :dash)
        end
    end
    return f
end
## ────────────────────────────────────────────────────────────────────────────
## Coskewness heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_coskewness(sk::MatNum, nx::AbstractVector = 1:size(sk, 1);
                                             kwargs...)
    N = size(sk, 1)
    N2 = size(sk, 2)
    tick_step = max(1, div(N2, 20))
    col_ticks = collect(1:tick_step:N2)
    col_labels = if N <= 10
        ["$(nx[j])×$(nx[k])" for j in 1:N for k in 1:N][col_ticks]
    else
        string.(col_ticks)
    end
    return heatmap(sk; yticks = (1:N, string.(nx)), xticks = (col_ticks, col_labels),
                   xrotation = 90, color = cgrad(:RdBu; rev = true),
                   clim = finite_symmetric_clim(sk), title = "Coskewness Matrix",
                   colorbar_title = "S̃", yflip = true, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Cokurtosis eigenspectrum / heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cokurtosis(kt::MatNum,
                                             nx::AbstractVector = 1:isqrt(size(kt, 1));
                                             heatmap::Bool = false, reference::Bool = true,
                                             kwargs...)
    if heatmap
        return StatsPlots.heatmap(kt; color = cgrad(:RdBu; rev = true),
                                  clim = finite_symmetric_clim(kt),
                                  title = "Cokurtosis Matrix", colorbar_title = "K̃",
                                  yflip = true, kwargs...)
    else
        ev = sort(real.(eigvals(Symmetric(kt))); rev = true)
        N2 = length(ev)
        f = bar(1:N2, ev; title = "Cokurtosis Eigenspectrum", xlabel = "Component",
                ylabel = "Eigenvalue", legend = reference, kwargs...)
        if reference
            λ_mean = mean(ev)
            hline!(f, [λ_mean]; label = "Mean eigenvalue ($(round(λ_mean; digits=4)))",
                   linewidth = 2, color = :red, linestyle = :dash)
        end
        return f
    end
end
## ────────────────────────────────────────────────────────────────────────────
## Composite prior dashboard
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_prior(pr::PortfolioOptimisers.AbstractPriorResult,
                                        nx::AbstractVector = 1:length(pr.mu);
                                        N::Option{<:Number} = nothing, kwargs...)
    p_mu = PortfolioOptimisers.plot_mu(pr.mu, nx; N = N, title = "Expected Returns",
                                       ylabel = "μ")
    p_sigma = PortfolioOptimisers.plot_sigma(pr.sigma, nx; N = N,
                                             title = "Asset Volatility", ylabel = "σ")
    p_corr = PortfolioOptimisers.plot_correlation(pr.sigma, nx;
                                                  title = "Correlation Matrix")
    return plot(p_mu, p_sigma, p_corr; layout = (1, 3), size = (1800, 500), kwargs...)
end
