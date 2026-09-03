# Shared fixture for test_06c: a synthetic Asset Panel drawn from a known factor model.
# Not a test file (no `test_` prefix) so it is excluded from discovery; included by
# test_06c_synthetic_asset_panel.jl. See test22_setup.jl for the same pattern.
#
# This generator is a toy fixture for one acceptance test, not library functionality, so
# it lives here rather than under `src/`. It has no Documenter-facing docstrings: nothing
# under `docs/` references it, and `sweep/manifest.toml` carries no row for this file.
using Random, Statistics, Distributions, Dates, StatsBase

# Observations per year the annualised parameters below are divided by. The panel's clock
# is a business-day calendar, which is what makes 252 the right divisor.
const SYNTHETIC_PERIODS_PER_YEAR = 252

# Industry labels, each with the weight its membership is sampled at. The weights make
# some industries larger than others, so the one-hot classification has uneven column
# sums, as a real one does. `n_industries` takes the first entries of this tuple.
const SYNTHETIC_INDUSTRIES = (("Real Estate", 390.0), ("Software", 280.0), ("Banks", 270.0),
                              ("Energy", 240.0), ("Capital Goods", 230.0),
                              ("Commercial", 210.0), ("Financials", 200.0),
                              ("Tech Hardware", 170.0), ("Pharma & Biotech", 170.0),
                              ("Health Care", 160.0), ("Materials", 160.0),
                              ("Food & Beverage", 150.0), ("Utilities", 150.0),
                              ("Retail", 140.0), ("Insurance", 140.0),
                              ("Semiconductors", 130.0))

# Style factors, each with its annualised volatility, its annualised mean and its
# autocorrelation, in the column order the loadings and the factor returns take. Market
# beta, momentum and reversal are deliberately absent: beta rides the market factor, and
# momentum and reversal emerge from the realised return path rather than from a drawn
# factor.
const SYNTHETIC_STYLES = (("size", 0.033, 0.005, -0.05), ("value", 0.014, -0.008, 0.01),
                          ("earnings_yield", 0.017, 0.015, 0.06),
                          ("profitability", 0.012, 0.002, 0.02),
                          ("growth", 0.011, -0.001, 0.04),
                          ("investment", 0.010, 0.000, 0.04),
                          ("leverage", 0.013, -0.006, 0.11),
                          ("dividend_yield", 0.012, 0.001, 0.04),
                          ("liquidity", 0.019, 0.007, -0.01),
                          ("volatility", 0.031, 0.003, 0.02))

# Central value of each per-asset characteristic the panel draws around. Each asset
# starts from these values, and its latent traits and its own noise then move it.
# `log_market_cap` and `price` are levels rather than ratios.
const SYNTHETIC_MEDIANS = (log_market_cap = 8.5, price = 42.0, book_to_price = 0.34,
                           earnings_to_price = 0.05, forward_earnings_to_price = 0.06,
                           sales_to_price = 0.45, cash_flow_to_price = 0.07,
                           gross_margin = 0.36, ebitda_margin = 0.18, asset_turnover = 0.57,
                           market_leverage = 0.19, sales_growth = 0.065,
                           capex_intensity = 0.025, daily_turnover = 0.008,
                           dividend_yield = 0.02, forward_dividend_yield = 0.017,
                           short_interest_ratio = 0.02, eps_dispersion_ratio = 0.06,
                           cash_to_assets = 0.10)

# Fraction of active cells each named field is observed on. A field the dictionary does
# not name is observed on `1 - missing_ratio` of its active cells. The four entries at
# `1.0` are the price, the volume, the share count and the market capitalisation, which
# real data populates whenever the asset trades.
const SYNTHETIC_COVERAGE = Dict("adj_close" => 1.0, "adj_volume" => 1.0,
                                "adj_shares_outstanding" => 1.0, "market_cap" => 1.0,
                                "ebitda_ttm" => 0.90, "enterprise_value" => 0.90,
                                "cost_of_revenue_ttm" => 0.90, "eps_ntm" => 0.80,
                                "dps_ntm" => 0.80, "eps_ntm_std" => 0.80)

# Volatilities, tail thickness and component weights of the return process. The market
# and the industry entries set the systematic block, `beta_spread` sets the
# cross-sectional dispersion of the market loading, and `tail_dof` is the Student-t
# degrees of freedom shared by the market shock and the transitory idiosyncratic shock.
# The four weights split the idiosyncratic variance between a transitory shock, a
# persistent momentum component, a mean-reverting pressure component and a predictable
# bearish component, so the total idiosyncratic variance is preserved.
const SYNTHETIC_DYNAMICS = (market_ann_vol = 0.18, market_ann_mean = 0.13,
                            industry_ann_vol = 0.11, industry_ann_vol_spread = 0.04,
                            beta_spread = 0.40, idio_log_sigma = 0.5, tail_dof = 6.0,
                            momentum_autocorr = 0.98, momentum_weight = 0.005,
                            reversal_autocorr = 0.95, reversal_weight = 0.3,
                            signal_autocorr = 0.99, signal_weight = 0.000625,
                            signal_sensitivity = 0.8)

# Shift a vector to zero mean and scale it to unit standard deviation. Uses the
# uncorrected standard deviation, because the vector is the whole cross-section rather
# than a sample of it. A constant vector is only centred, so a characteristic every asset
# shares standardises to zero rather than to a division by zero.
function synthetic_standardise(x::AbstractVector{<:Real})
    m, s = Statistics.mean(x), Statistics.std(x; corrected = false)
    return iszero(s) ? x .- m : (x .- m) ./ s
end

# Simulate stationary autoregressive factor-return paths of order one. The parameters
# are annualised, and the innovation scale is set so the stationary volatility matches
# `ann_vol` whatever the autocorrelation is.
function synthetic_ar1_paths(rng::Random.AbstractRNG, T::Integer,
                             ann_vol::AbstractVector{<:Real},
                             ann_mean::AbstractVector{<:Real}, rho::AbstractVector{<:Real})
    dvol = ann_vol ./ sqrt(SYNTHETIC_PERIODS_PER_YEAR)
    dmean = ann_mean ./ SYNTHETIC_PERIODS_PER_YEAR
    K = length(ann_vol)
    innov = randn(rng, T, K) .* (dvol .* sqrt.(1 .- rho .^ 2))'
    paths = Matrix{Float64}(undef, T, K)
    paths[1, :] = dmean .+ dvol .* randn(rng, K)
    for t in 2:T
        @views paths[t, :] = dmean .+ rho .* (paths[t - 1, :] .- dmean) .+ innov[t, :]
    end
    return paths
end

# Apply an autoregressive recursion of order one down the observation axis, keeping the
# stationary variance at one.
function synthetic_ar1_filter(innov::AbstractMatrix{<:Real}, rho::Real)
    scale = sqrt(1 - rho^2)
    out = Matrix{Float64}(undef, size(innov))
    out[1, :] = view(innov, 1, :)
    for t in 2:size(innov, 1)
        @views out[t, :] = rho .* out[t - 1, :] .+ scale .* innov[t, :]
    end
    return out
end

# Draw unit-variance fat-tailed shocks from a scale mixture of normals: a standard normal
# divided by the square root of a chi-squared variate over its degrees of freedom is a
# Student-t variate, rescaled here to unit variance. `dof` must exceed two, or the
# Student-t variance does not exist.
function synthetic_fat_tailed_normal(rng::Random.AbstractRNG, dof::Real, dims::Integer...)
    z = randn(rng, dims...)
    s = sqrt.(dof ./ rand(rng, Distributions.Chisq(dof), dims...))
    return z .* s .* sqrt((dof - 2) / dof)
end

# Draw the latent traits of every asset, and the per-asset characteristics and factor
# exposures they anchor. Six independent traits -- size, value, quality, risk, growth and
# liquidity -- are time-invariant, and every characteristic is a noisy function of them.
# The exposures are the standardised characteristics, so each style's exposure is a noisy
# proxy of the trait that drives that style's factor return, which is what makes the
# model recoverable from the panel alone.
function synthetic_panel_ratios(rng::Random.AbstractRNG, N::Integer)
    md, dy = SYNTHETIC_MEDIANS, SYNTHETIC_DYNAMICS
    t_size, t_value, t_quality = randn(rng, N), randn(rng, N), randn(rng, N)
    t_risk, t_growth, t_liquidity = randn(rng, N), randn(rng, N), randn(rng, N)
    nse(s::Real) = s .* randn(rng, N)
    log_market_cap = md.log_market_cap .+ 1.4 .* t_size
    base_market_cap = exp.(log_market_cap)
    base_price = exp.(log(md.price) .+ nse(0.85))
    book_to_price = exp.(log(md.book_to_price) .+ 0.55 .* t_value .+ nse(0.45))
    earnings_to_price = md.earnings_to_price .+
                        0.045 .* (0.6 .* t_value .+ 0.6 .* t_quality) .+ nse(0.05)
    forward_earnings_to_price = max.(md.forward_earnings_to_price .+ 0.04 .* t_value .+
                                     0.03 .* t_quality .+ nse(0.03), 0.003)
    sales_to_price = exp.(log(md.sales_to_price) .- 0.3 .* t_size .+ 0.5 .* t_value .+
                          nse(0.5))
    cash_flow_to_price = md.cash_flow_to_price .+ 0.05 .* t_quality .+ nse(0.05)
    gross_margin = clamp.(md.gross_margin .+ 0.18 .* t_quality .+ nse(0.08), 0.03, 0.95)
    ebitda_margin = clamp.(md.ebitda_margin .+ 0.08 .* t_quality .+ nse(0.05), -0.05, 0.6)
    asset_turnover = exp.(log(md.asset_turnover) .- 0.3 .* t_quality .+ nse(0.4))
    market_leverage = clamp.(md.market_leverage .+ 0.13 .* t_value .- 0.05 .* t_quality .+
                             nse(0.08), 0.0, 0.85)
    market_leverage[rand(rng, N) .< 0.18] .= 0.0
    sales_growth = md.sales_growth .+ 0.12 .* t_growth .+ nse(0.06)
    capex_intensity = clamp.(md.capex_intensity .+ 0.012 .* t_growth .+ nse(0.01), 0.0, 0.2)
    base_daily_turnover = exp.(log(md.daily_turnover) .+ 0.4 .* t_liquidity .-
                               0.2 .* t_size .+ nse(0.4))
    payer = rand(rng, N) .< clamp.(0.55 .+ 0.18 .* t_value .+ 0.18 .* t_quality, 0.05, 0.95)
    dividend_yield = ifelse.(payer,
                             clamp.(md.dividend_yield .+ 0.012 .* t_value .+ nse(0.012),
                                    0.0, 0.1), 0.0)
    forward_dividend_yield = ifelse.(payer,
                                     clamp.(md.forward_dividend_yield .+ 0.01 .* t_value .+
                                            nse(0.01), 0.0, 0.1), 0.0)
    cash_to_assets = clamp.(md.cash_to_assets .+ nse(0.05), 0.0, 0.6)
    idio_shape = exp.(0.35 .* t_risk .- 0.15 .* t_size .+
                      dy.idio_log_sigma .* randn(rng, N))
    loadings = hcat(synthetic_standardise(log_market_cap),
                    synthetic_standardise(log.(book_to_price)),
                    synthetic_standardise(earnings_to_price),
                    synthetic_standardise(gross_margin .* asset_turnover),
                    synthetic_standardise(sales_growth),
                    synthetic_standardise(capex_intensity),
                    synthetic_standardise(market_leverage),
                    synthetic_standardise(dividend_yield),
                    synthetic_standardise(log.(base_daily_turnover)),
                    synthetic_standardise(log.(idio_shape)))
    return (; base_market_cap = base_market_cap, base_price = base_price,
            base_shares = base_market_cap ./ base_price, book_to_price = book_to_price,
            earnings_to_price = earnings_to_price,
            forward_earnings_to_price = forward_earnings_to_price,
            sales_to_price = sales_to_price, cash_flow_to_price = cash_flow_to_price,
            gross_margin = gross_margin, ebitda_margin = ebitda_margin,
            asset_turnover = asset_turnover, market_leverage = market_leverage,
            sales_growth = sales_growth, capex_intensity = capex_intensity,
            base_daily_turnover = base_daily_turnover, dividend_yield = dividend_yield,
            forward_dividend_yield = forward_dividend_yield,
            buyback_yield = 0.01 .+ 0.02 .* t_quality .+ nse(0.03),
            short_interest_offset = -0.3 .* t_quality .+ nse(0.4),
            dispersion_offset = nse(0.4), cash_to_assets = cash_to_assets,
            beta = clamp.(1.0 .+ dy.beta_spread .* t_risk, 0.1, 3.0),
            idio_shape = idio_shape, loadings = loadings)
end

# Draw the factor-return series of the panel's three factor families. The market factor
# is fat-tailed and carries a positive mean, the industry factors are zero-mean and
# serially independent, and the style factors are mean-reverting.
function synthetic_panel_factors(rng::Random.AbstractRNG, T::Integer, n_ind::Integer)
    dy = SYNTHETIC_DYNAMICS
    s_vol = [s[2] for s in SYNTHETIC_STYLES]
    s_mean = [s[3] for s in SYNTHETIC_STYLES]
    s_rho = [s[4] for s in SYNTHETIC_STYLES]
    style = synthetic_ar1_paths(rng, T, s_vol, s_mean, s_rho)
    m_vol = dy.market_ann_vol / sqrt(SYNTHETIC_PERIODS_PER_YEAR)
    m_mean = dy.market_ann_mean / SYNTHETIC_PERIODS_PER_YEAR
    market = m_mean .+ m_vol .* synthetic_fat_tailed_normal(rng, dy.tail_dof, T)
    i_vol = abs.(dy.industry_ann_vol .+ dy.industry_ann_vol_spread .* randn(rng, n_ind))
    industry = synthetic_ar1_paths(rng, T, i_vol, zeros(n_ind), zeros(n_ind))
    return (; market = market, industry = industry, style = style,
            market_daily_mean = m_mean,
            style_daily_mean = s_mean ./ SYNTHETIC_PERIODS_PER_YEAR)
end

# Build the total returns from the systematic block and the idiosyncratic block. The
# idiosyncratic level is set from `svr`, so the share of cross-sectional return variance
# the factor structure explains is the caller's number rather than an accident of the
# draws. The bearish signal enters the *next* observation's shock, so a descriptor built
# from it predicts a future return with no look-ahead.
function synthetic_panel_returns(rng::Random.AbstractRNG, rt::NamedTuple, fac::NamedTuple,
                                 ind_idx::AbstractVector{<:Integer}, svr::Real)
    dy = SYNTHETIC_DYNAMICS
    T, N = size(fac.style, 1), length(rt.beta)
    systematic = fac.market * rt.beta' .+ view(fac.industry, :, ind_idx) .+
                 fac.style * rt.loadings'
    sys_var = Statistics.mean(Statistics.var(systematic; dims = 2, corrected = false))
    dvol = sqrt(sys_var * (1 - svr) / svr) .* rt.idio_shape ./
           sqrt(Statistics.mean(rt.idio_shape .^ 2))
    transitory = synthetic_fat_tailed_normal(rng, dy.tail_dof, T, N)
    momentum = synthetic_ar1_filter(randn(rng, T, N), dy.momentum_autocorr)
    pressure = synthetic_ar1_filter(randn(rng, T, N), dy.reversal_autocorr)
    reversal = zeros(T, N)
    scale = 1 / sqrt(2 * (1 - dy.reversal_autocorr))
    for i in 1:N, t in 2:T
        reversal[t, i] = (pressure[t, i] - pressure[t - 1, i]) * scale
    end
    signal = synthetic_ar1_filter(randn(rng, T, N), dy.signal_autocorr)
    signal .-= Statistics.mean(signal; dims = 2)
    sd = Statistics.std(signal; dims = 2, corrected = false)
    signal ./= ifelse.(iszero.(sd), 1.0, sd)
    shock = sqrt(1 - dy.momentum_weight - dy.reversal_weight - dy.signal_weight) .*
            transitory .+ sqrt(dy.momentum_weight) .* momentum .+
            sqrt(dy.reversal_weight) .* reversal
    lead = sqrt(dy.signal_weight)
    for i in 1:N, t in 2:T
        shock[t, i] -= lead * signal[t - 1, i]
    end
    eps = dvol' .* shock
    return (; X = max.(systematic .+ eps, -0.99), eps = eps, ivar = dvol .^ 2,
            signal = signal)
end

# Draw the listing window of every asset, and the active and estimation masks it defines.
# An asset lists at the first observation unless it is drawn as a late lister, and it
# stays listed to the last observation unless it is drawn as a delisting. A delisting is
# never sooner than one year after the listing. The first asset is forced active
# throughout, so no observation has an empty cross-section.
function synthetic_panel_masks(rng::Random.AbstractRNG, T::Integer, N::Integer, late::Real,
                               delist::Real)
    start, stop = ones(Int, N), fill(T, N)
    late_idx = findall(rand(rng, N) .< late)
    start[late_idx] .= rand(rng, 2:max(2, floor(Int, 0.6 * T)), length(late_idx))
    for i in findall(rand(rng, N) .< delist)
        stop[i] = rand(rng, clamp(start[i] - 1 + SYNTHETIC_PERIODS_PER_YEAR, 1, T):T)
    end
    amsk = falses(T, N)
    for i in 1:N
        amsk[start[i]:stop[i], i] .= true
    end
    amsk[:, 1] .= true
    emsk = copy(amsk)
    emsk[:, StatsBase.sample(rng, 2:N, min(max(1, floor(Int, 0.05 * N)), N - 1); replace = false)] .= false
    return amsk, emsk
end

# Build every numeric field of the panel, in the column order it takes on the feature
# axis. The accounting identities hold: the market capitalisation is the price times the
# share count, the enterprise value is the market capitalisation plus the debt less the
# cash, and the total assets are the sales over the asset turnover. The fundamentals grow
# at each asset's own expected return, so its valuation ratios are stationary in
# expectation.
function synthetic_panel_fields(rng::Random.AbstractRNG, rt::NamedTuple, fac::NamedTuple,
                                amsk::AbstractMatrix{Bool}, X::AbstractMatrix{<:Real},
                                signal::AbstractMatrix{<:Real})
    md, ppy = SYNTHETIC_MEDIANS, SYNTHETIC_PERIODS_PER_YEAR
    T, N = size(X)
    adj_close = rt.base_price' .* exp.(cumsum(log1p.(ifelse.(amsk, X, 0.0)); dims = 1))
    issuance = (exp.(rt.sales_growth) .^ (1 / ppy) .- 1) .* 0.3
    steps = issuance' .+ 5e-5 .* randn(rng, T, N)
    shares = rt.base_shares' .* exp.(cumsum(ifelse.(amsk, steps, 0.0); dims = 1))
    market_cap = adj_close .* shares
    drift = rt.beta .* fac.market_daily_mean .+ rt.loadings * fac.style_daily_mean .+
            issuance .+ (rt.sales_growth .- Statistics.median(rt.sales_growth)) ./ ppy
    growth = exp.((0:(T - 1)) .* drift')
    sales = (rt.sales_to_price .* rt.base_market_cap)' .* growth
    total_assets = sales ./ rt.asset_turnover'
    total_debt = (rt.market_leverage .* rt.base_market_cap)' .* growth
    eps_ntm = rt.forward_earnings_to_price' .* adj_close
    sens = SYNTHETIC_DYNAMICS.signal_sensitivity
    dispersion = exp.(log(md.eps_dispersion_ratio) .+ rt.dispersion_offset' .+
                      sens .* signal)
    short_ratio = clamp.(exp.(log(md.short_interest_ratio) .+ rt.short_interest_offset' .+
                              sens .* signal), 0.0, 0.4)
    return ["adj_close" => adj_close,
            "adj_volume" =>
                shares .* 1e6 .* rt.base_daily_turnover' .* exp.(0.4 .* randn(rng, T, N)),
            "adj_shares_outstanding" => shares, "market_cap" => market_cap,
            "ebitda_ttm" => rt.ebitda_margin' .* sales,
            "enterprise_value" =>
                market_cap .+ total_debt .- rt.cash_to_assets' .* total_assets,
            "net_income_ttm" => (rt.earnings_to_price .* rt.base_market_cap)' .* growth,
            "sales_ttm" => sales,
            "dividends_ttm" => (rt.dividend_yield .* rt.base_market_cap)' .* growth,
            "net_buybacks_ttm" => (rt.buyback_yield .* rt.base_market_cap)' .* growth,
            "book_equity" => (rt.book_to_price .* rt.base_market_cap)' .* growth,
            "operating_cash_flow_ttm" =>
                (rt.cash_flow_to_price .* rt.base_market_cap)' .* growth,
            "total_debt" => total_debt, "total_assets" => total_assets,
            "cost_of_revenue_ttm" => (1 .- rt.gross_margin)' .* sales,
            "capex_ttm" => rt.capex_intensity' .* total_assets,
            "short_interest" => short_ratio .* shares .* 1e6, "eps_ntm" => eps_ntm,
            "dps_ntm" => rt.forward_dividend_yield' .* adj_close,
            "eps_ntm_std" => dispersion .* abs.(eps_ntm)]
end

# Blank the cells of every numeric field that a real panel would not carry. Two rules
# blank a cell, and the second one is unconditional: a field is observed on only part of
# its active cells, and no field is observed at all where the asset is not active. A
# blank is written as a `NaN`, which `asset_panel` resolves into an observed-mask column.
function synthetic_panel_blank!(rng::Random.AbstractRNG,
                                fields::AbstractVector{<:Pair{String, <:AbstractMatrix}},
                                amsk::AbstractMatrix{Bool}, missing_ratio::Real)::Nothing
    T, N = size(amsk)
    for (name, arr) in fields
        keep = get(SYNTHETIC_COVERAGE, name, 1 - missing_ratio)
        if keep < 1
            arr[(rand(rng, T, N) .> keep) .& amsk] .= NaN
        end
        arr[.!amsk] .= NaN
    end
    return nothing
end

# Wrap the panel's numeric fields in the raw inputs `asset_panel` takes. Every numeric
# field blanks, so each one carries a `ForwardPanelFill` and earns an observed-mask
# column. The classification carries no blank: an industry is a time-invariant fact of
# the asset, and the active mask already says when the asset is listed.
function synthetic_panel_inputs(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}},
                                industry::AbstractMatrix{<:AbstractString},
                                levels::AbstractVector{<:AbstractString})
    inputs = PortfolioOptimisers.AbstractPanelFieldInput[NumericPanelInput(; name = name,
                                                                           vals = arr,
                                                                           alg = ForwardPanelFill(;
                                                                                                  val = 0.0))
                                                         for (name, arr) in fields]
    push!(inputs,
          CategoricalPanelInput(; name = "industry", vals = industry, levels = levels))
    return inputs
end

"""
Draw a synthetic point-in-time Asset Panel from a known factor model, and return the
model beside it (issue #656).

The returns follow `X = f * B' + eps`, floored at `-0.99`. The factor exposures `B` are
per-asset traits, and every field is a noisy function of the same traits, so a
cross-sectional regression fit on the panel recovers the factor returns the generator
drew. This is what makes the panel an acceptance test rather than a demonstration: the
answer is known before the estimator runs.

Three factor families span the model. The market family is one fat-tailed factor whose
exposure is the asset's beta. The industry family is one zero-mean factor per industry,
and its exposures are the one-hot classification. The style family is the ten
mean-reverting factors of `SYNTHETIC_STYLES`, and its exposures are the standardised
characteristics.

The panel rides the carrier the same way a real one does: the returns are `rd.X`, the
field values are `rd.Z`, their names are `rd.nz`, and the field index and the two masks
are `rd.pnl`. The default size (`n_assets = 500`, `n_observations = 2520`) allocates
about half a gigabyte, so a test passes smaller dimensions.

Returns `(; rd::ReturnsResult, truth::NamedTuple)`, where `truth` holds the factor names
`nf`, the factor family of each factor `fgrp`, the factor exposures `B`
(assets × factors), the factor returns `f` (observations × factors), the idiosyncratic
returns `eps` (observations × assets) and the idiosyncratic variances `ivar` (assets).
"""
function synthetic_asset_panel(; n_assets::Integer = 500, n_observations::Integer = 2520,
                               n_industries::Integer = 10,
                               start_date::Dates.Date = Dates.Date(2015, 1, 1),
                               systematic_variance_ratio::Real = 0.5,
                               late_listing_proba::Real = 0.15,
                               delisting_proba::Real = 0.15, missing_ratio::Real = 0.01,
                               rng::Random.AbstractRNG = Random.default_rng())
    if !(n_assets >= 2 && n_observations >= 2)
        throw(DomainError((n_assets, n_observations),
                          "a synthetic panel needs at least two assets and two observations: the estimation mask drops an asset that is not the first, and the mean-reverting component reads the previous observation"))
    end
    if !(1 <= n_industries <= length(SYNTHETIC_INDUSTRIES))
        throw(DomainError(n_industries,
                          "n_industries takes the first entries of SYNTHETIC_INDUSTRIES, so it must lie in 1:$(length(SYNTHETIC_INDUSTRIES))"))
    end
    if !(0 < systematic_variance_ratio < 1)
        throw(DomainError(systematic_variance_ratio,
                          "systematic_variance_ratio scales the idiosyncratic variance by (1 - svr) / svr, so it must lie strictly between zero and one"))
    end
    if !(all(x -> 0 <= x <= 1, (late_listing_proba, delisting_proba, missing_ratio)))
        throw(DomainError((late_listing_proba, delisting_proba, missing_ratio),
                          "late_listing_proba, delisting_proba and missing_ratio are probabilities, so each must lie in [0, 1]"))
    end
    T, N = n_observations, n_assets
    levels = [SYNTHETIC_INDUSTRIES[i][1] for i in 1:n_industries]
    cw = cumsum([SYNTHETIC_INDUSTRIES[i][2] for i in 1:n_industries])
    ind_idx = [count(<(u), cw) + 1 for u in rand(rng, N) .* cw[end]]
    rt = synthetic_panel_ratios(rng, N)
    fac = synthetic_panel_factors(rng, T, n_industries)
    ret = synthetic_panel_returns(rng, rt, fac, ind_idx, systematic_variance_ratio)
    amsk, emsk = synthetic_panel_masks(rng, T, N, late_listing_proba, delisting_proba)
    fields = synthetic_panel_fields(rng, rt, fac, amsk, ret.X, ret.signal)
    synthetic_panel_blank!(rng, fields, amsk, missing_ratio)
    industry = repeat(reshape(levels[ind_idx], 1, N), T, 1)
    pnl = asset_panel(synthetic_panel_inputs(fields, industry, levels); amsk = amsk,
                      emsk = emsk)
    days = start_date:Dates.Day(1):(start_date + Dates.Day(2 * T + 7))
    rd = ReturnsResult(; nx = ["A" * lpad(i, 5, '0') for i in 1:N],
                       X = ifelse.(amsk, ret.X, NaN),
                       ts = filter(d -> Dates.dayofweek(d) <= 5, days)[1:T], pnl...)
    onehot = Float64[ind_idx[i] == k for i in 1:N, k in 1:n_industries]
    truth = (;
             nf = vcat("market", ["industry=" * l for l in levels],
                       [s[1] for s in SYNTHETIC_STYLES]),
             fgrp = vcat("market", fill("industry", n_industries),
                         fill("style", length(SYNTHETIC_STYLES))),
             B = hcat(rt.beta, onehot, rt.loadings),
             f = hcat(fac.market, fac.industry, fac.style), eps = ret.eps, ivar = ret.ivar)
    return (; rd = rd, truth = truth)
end
