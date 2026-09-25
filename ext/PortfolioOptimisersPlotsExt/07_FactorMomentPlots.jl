## plot_factor_loadings
# The factor-space entry points all guard through `assert_prior_regression`: `rr` and `fpr`
# travel together on the carrier, so checking `rr` establishes the whole block. Each supplies
# its own `lead` — the shared diagnosis lives in `prior_regression_remedy`. Optional axis-name
# arguments must therefore default to `nothing` rather than to a size taken off the block, or
# the default argument would be evaluated before the guard could run.
#
# Guarded, they read through `fpr` rather than the flat `f_` names: `pr.fpr.sigma` is the
# idiomatic read (see [`LowOrderPrior`](@ref)), and the flat names' one advantage — returning
# `nothing` instead of throwing on an absent block — is exactly what the guard above has just
# ruled out. It also puts the code in step with these messages, which name `fpr.sigma`.
const NO_FACTOR_BLOCK_HINT = "The prior result supplied carries no factor block: `rr === nothing`, and `rr` and `fpr` are provided together or not at all."
const NO_FACTOR_LOADINGS_LEAD = "`plot_factor_loadings` draws the regression loadings `rr.M`. $NO_FACTOR_BLOCK_HINT Pass the loadings directly as `plot_factor_loadings(M, nx, nf)` if you hold them."
function PortfolioOptimisers.plot_factor_loadings(pr::PortfolioOptimisers.AbstractPriorResult,
                                                  nx::Option{<:AbstractVector} = nothing,
                                                  nf::Option{<:AbstractVector} = nothing;
                                                  kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_LOADINGS_LEAD)
    nx_use = isnothing(nx) ? (1:size(pr.rr.M, 1)) : nx
    nf_use = isnothing(nf) ? (1:size(pr.rr.M, 2)) : nf
    return PortfolioOptimisers.plot_factor_loadings(pr.rr.M, nx_use, nf_use; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pr::PortfolioOptimisers.AbstractPriorResult,
                                                  rd::ReturnsResult; kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_LOADINGS_LEAD)
    nx = isnothing(rd.nx) ? (1:size(pr.rr.M, 1)) : rd.nx
    nf = isnothing(rd.nf) ? (1:size(pr.rr.M, 2)) : rd.nf
    return PortfolioOptimisers.plot_factor_loadings(pr.rr.M, nx, nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(res::OptimisationResult,
                                                  rd::ReturnsResult; kwargs...)
    pr, nx = result_prior_view(res, rd.nx)
    return PortfolioOptimisers.plot_factor_loadings(pr, nx, rd.nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(res::OptimisationResult; kwargs...)
    pr, nx = result_prior_view(res)
    return PortfolioOptimisers.plot_factor_loadings(pr, nx; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pred::PredictionResult, rd::ReturnsResult;
                                                  kwargs...)
    pr, nx = result_prior_view(pred.res, rd.nx)
    return PortfolioOptimisers.plot_factor_loadings(pr, nx, rd.nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_loadings(pred::PredictionResult; kwargs...)
    pr, nx = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_factor_loadings(pr, nx; kwargs...)
end
## plot_factor_sigma
const NO_FACTOR_SIGMA_LEAD = "`plot_factor_sigma` draws the factor covariance `fpr.sigma`. $NO_FACTOR_BLOCK_HINT Pass the factor covariance directly as `plot_factor_sigma(f_sigma, nf)` if you hold it."
function PortfolioOptimisers.plot_factor_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                               nf::Option{<:AbstractVector} = nothing;
                                               kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_SIGMA_LEAD)
    nf_use = isnothing(nf) ? (1:size(pr.fpr.sigma, 1)) : nf
    return PortfolioOptimisers.plot_factor_sigma(pr.fpr.sigma, nf_use; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pr::PortfolioOptimisers.AbstractPriorResult,
                                               rd::ReturnsResult; kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_SIGMA_LEAD)
    nf = isnothing(rd.nf) ? (1:size(pr.fpr.sigma, 1)) : rd.nf
    return PortfolioOptimisers.plot_factor_sigma(pr.fpr.sigma, nf; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(res::OptimisationResult, rd::ReturnsResult;
                                               kwargs...)
    pr, = result_prior_view(res)
    return PortfolioOptimisers.plot_factor_sigma(pr, rd; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(res::OptimisationResult; kwargs...)
    pr, = result_prior_view(res)
    return PortfolioOptimisers.plot_factor_sigma(pr; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pred::PredictionResult, rd::ReturnsResult;
                                               kwargs...)
    pr, = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_factor_sigma(pr, rd; kwargs...)
end
function PortfolioOptimisers.plot_factor_sigma(pred::PredictionResult; kwargs...)
    pr, = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_factor_sigma(pr; kwargs...)
end
## plot_factor_mu
const NO_FACTOR_MU_LEAD = "`plot_factor_mu` draws the factor expected returns `fpr.mu`. $NO_FACTOR_BLOCK_HINT Pass the factor expected returns directly as `plot_factor_mu(f_mu, nf)` if you hold them."
function PortfolioOptimisers.plot_factor_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                            nf::Option{<:AbstractVector} = nothing;
                                            N::Option{<:Number} = nothing, kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_MU_LEAD)
    nf_use = isnothing(nf) ? (1:length(pr.fpr.mu)) : nf
    return PortfolioOptimisers.plot_factor_mu(pr.fpr.mu, nf_use; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pr::PortfolioOptimisers.AbstractPriorResult,
                                            rd::ReturnsResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    PortfolioOptimisers.assert_prior_regression(pr, :pr; lead = NO_FACTOR_MU_LEAD)
    nf = isnothing(rd.nf) ? (1:length(pr.fpr.mu)) : rd.nf
    return PortfolioOptimisers.plot_factor_mu(pr.fpr.mu, nf; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(res::OptimisationResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    pr, = result_prior_view(res)
    return PortfolioOptimisers.plot_factor_mu(pr; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(res::OptimisationResult, rd::ReturnsResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    pr, = result_prior_view(res)
    return PortfolioOptimisers.plot_factor_mu(pr, rd; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pred::PredictionResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    pr, = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_factor_mu(pr; N = N, kwargs...)
end
function PortfolioOptimisers.plot_factor_mu(pred::PredictionResult, rd::ReturnsResult;
                                            N::Option{<:Number} = nothing, kwargs...)
    pr, = result_prior_view(pred.res)
    return PortfolioOptimisers.plot_factor_mu(pr, rd; N = N, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Factor loadings heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_loadings(M::MatNum,
                                                  nx::AbstractVector = 1:size(M, 1),
                                                  nf::AbstractVector = 1:size(M, 2);
                                                  kwargs...)
    Na, Nf = size(M)
    return heatmap(M; xticks = (1:Nf, string.(nf)), yticks = (1:Na, string.(nx)),
                   xrotation = 90, color = cgrad(:RdBu; rev = true),
                   clim = finite_symmetric_clim(M), title = "Factor Loadings",
                   colorbar_title = "β", yflip = true, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Factor covariance heatmap
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_sigma(f_sigma::MatNum,
                                               nf::AbstractVector = 1:size(f_sigma, 1);
                                               kwargs...)
    return PortfolioOptimisers.plot_correlation(f_sigma, nf; title = "Factor Covariance",
                                                colorbar_title = "ρ_f", kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Factor expected returns bar chart
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_mu(f_mu::VecNum,
                                            nf::AbstractVector = 1:length(f_mu);
                                            N::Option{<:Number} = nothing, kwargs...)
    M = length(f_mu)
    N, idx = relevant_assets(f_mu, M, N)
    sort!(view(idx, 1:N))
    top_idx = view(idx, 1:N)
    return bar(f_mu[top_idx]; xticks = (1:N, string.(nf[top_idx])),
               title = "Factor Expected Returns", xlabel = "Factor", ylabel = "f_μ",
               xrotation = 90, legend = false, kwargs...)
end
