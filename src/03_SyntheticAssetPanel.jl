"""
    const SYNTHETIC_PERIODS_PER_YEAR = 252

Observations per year the synthetic panel's annualised parameters are divided by.

Every volatility and mean in [`SYNTHETIC_STYLES`](@ref) and [`SYNTHETIC_DYNAMICS`](@ref) is stated per year, and the generator works per observation. The panel's clock is a business-day calendar, which is what makes 252 the right divisor.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`SYNTHETIC_STYLES`](@ref)
  - [`SYNTHETIC_DYNAMICS`](@ref)
"""
const SYNTHETIC_PERIODS_PER_YEAR = 252
"""
    const SYNTHETIC_INDUSTRIES

Industry labels of the synthetic panel, each with the weight its membership is sampled at.

The weights make some industries larger than others, so the one-hot Factor Family the classification produces has uneven column sums, as a real one does. `n_industries` of [`synthetic_asset_panel`](@ref) takes the first entries of this tuple, so it can never exceed its length.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`CategoricalPanelInput`](@ref)
"""
const SYNTHETIC_INDUSTRIES = (("Real Estate", 390.0), ("Software", 280.0), ("Banks", 270.0),
                              ("Energy", 240.0), ("Capital Goods", 230.0),
                              ("Commercial", 210.0), ("Financials", 200.0),
                              ("Tech Hardware", 170.0), ("Pharma & Biotech", 170.0),
                              ("Health Care", 160.0), ("Materials", 160.0),
                              ("Food & Beverage", 150.0), ("Utilities", 150.0),
                              ("Retail", 140.0), ("Insurance", 140.0),
                              ("Semiconductors", 130.0))
"""
    const SYNTHETIC_STYLES

Style factors of the synthetic panel, each with its annualised volatility, its annualised mean and its autocorrelation.

One entry per style factor, in the column order the loadings and the factor returns take. Market beta, momentum and reversal are deliberately absent: beta rides the market factor, and momentum and reversal emerge from the realised return path rather than from a drawn factor.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`synthetic_panel_factors`](@ref)
  - [`synthetic_panel_ratios`](@ref)
"""
const SYNTHETIC_STYLES = (("size", 0.033, 0.005, -0.05), ("value", 0.014, -0.008, 0.01),
                          ("earnings_yield", 0.017, 0.015, 0.06),
                          ("profitability", 0.012, 0.002, 0.02),
                          ("growth", 0.011, -0.001, 0.04),
                          ("investment", 0.010, 0.000, 0.04),
                          ("leverage", 0.013, -0.006, 0.11),
                          ("dividend_yield", 0.012, 0.001, 0.04),
                          ("liquidity", 0.019, 0.007, -0.01),
                          ("volatility", 0.031, 0.003, 0.02))
"""
    const SYNTHETIC_MEDIANS

Central value of each per-asset characteristic the synthetic panel draws around.

Each asset starts from these values, and its latent traits and its own noise then move it. Two entries are levels rather than ratios: `log_market_cap` is a log size and `price` is a share price.

# Related

  - [`synthetic_panel_ratios`](@ref)
  - [`synthetic_panel_fields`](@ref)
"""
const SYNTHETIC_MEDIANS = (log_market_cap = 8.5, price = 42.0, book_to_price = 0.34,
                           earnings_to_price = 0.05, forward_earnings_to_price = 0.06,
                           sales_to_price = 0.45, cash_flow_to_price = 0.07,
                           gross_margin = 0.36, ebitda_margin = 0.18, asset_turnover = 0.57,
                           market_leverage = 0.19, sales_growth = 0.065,
                           capex_intensity = 0.025, daily_turnover = 0.008,
                           dividend_yield = 0.02, forward_dividend_yield = 0.017,
                           short_interest_ratio = 0.02, eps_dispersion_ratio = 0.06,
                           cash_to_assets = 0.10)
"""
    const SYNTHETIC_COVERAGE

Fraction of active cells each named Panel Field of the synthetic panel is observed on.

A Panel Field the dictionary does not name is observed on `1 - missing_ratio` of its active cells. The four entries at `1.0` are the price, the volume, the share count and the market capitalisation, which real data populates whenever the asset trades. The forward-looking and lower-coverage fields sit below it.

# Related

  - [`synthetic_panel_blank!`](@ref)
  - [`synthetic_asset_panel`](@ref)
"""
const SYNTHETIC_COVERAGE = Dict("adj_close" => 1.0, "adj_volume" => 1.0,
                                "adj_shares_outstanding" => 1.0, "market_cap" => 1.0,
                                "ebitda_ttm" => 0.90, "enterprise_value" => 0.90,
                                "cost_of_revenue_ttm" => 0.90, "eps_ntm" => 0.80,
                                "dps_ntm" => 0.80, "eps_ntm_std" => 0.80)
"""
    const SYNTHETIC_DYNAMICS

Volatilities, tail thickness and component weights of the synthetic panel's return process.

The market and the industry entries set the systematic block, `beta_spread` sets the cross-sectional dispersion of the market loading, and `tail_dof` is the Student-t degrees of freedom shared by the market shock and the transitory idiosyncratic shock. The four weights split the idiosyncratic variance between a transitory shock, a persistent momentum component, a mean-reverting pressure component and a predictable bearish component, so the total idiosyncratic variance is preserved.

# Related

  - [`synthetic_panel_factors`](@ref)
  - [`synthetic_panel_returns`](@ref)
  - [`synthetic_panel_fields`](@ref)
"""
const SYNTHETIC_DYNAMICS = (market_ann_vol = 0.18, market_ann_mean = 0.13,
                            industry_ann_vol = 0.11, industry_ann_vol_spread = 0.04,
                            beta_spread = 0.40, idio_log_sigma = 0.5, tail_dof = 6.0,
                            momentum_autocorr = 0.98, momentum_weight = 0.005,
                            reversal_autocorr = 0.95, reversal_weight = 0.3,
                            signal_autocorr = 0.99, signal_weight = 0.000625,
                            signal_sensitivity = 0.8)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Shift a vector to zero mean and scale it to unit standard deviation.

The standard deviation is the uncorrected one, because the vector is the whole cross-section rather than a sample of it. A constant vector is only centred, so a Factor Exposure built from a characteristic every asset shares is zero rather than undefined.

# Arguments

  - `x`: The vector to standardise.

# Returns

  - `y::Vector{Float64}`: The standardised vector.

# Related

  - [`synthetic_panel_ratios`](@ref)
"""
function synthetic_standardise(x::AbstractVector{<:Real})
    m, s = Statistics.mean(x), Statistics.std(x; corrected = false)
    return iszero(s) ? x .- m : (x .- m) ./ s
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Simulate stationary autoregressive factor-return paths of order one.

The parameters are annualised, and the innovation scale is set so the stationary volatility matches `ann_vol` whatever the autocorrelation is.

# Algorithm

 1. Divide `ann_vol` by `sqrt(SYNTHETIC_PERIODS_PER_YEAR)` and `ann_mean` by `SYNTHETIC_PERIODS_PER_YEAR`.
 2. Draw the innovations, and scale each factor's column by its per-observation volatility times `sqrt(1 - rho^2)`.
 3. Draw the first observation from the stationary distribution.
 4. Walk the remaining observations, each one the mean plus `rho` times the previous deviation plus the innovation.

# Arguments

  - `rng`: Random number generator.
  - `T`: Number of observations.
  - `ann_vol`: Annualised volatility, one entry per factor.
  - `ann_mean`: Annualised mean, one entry per factor.
  - `rho`: Autocorrelation, one entry per factor.

# Returns

  - `paths::Matrix{Float64}`: The factor-return paths (observations × factors).

# Related

  - [`synthetic_panel_factors`](@ref)
  - [`synthetic_ar1_filter`](@ref)
  - [`SYNTHETIC_PERIODS_PER_YEAR`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply an autoregressive recursion of order one down the observation axis, keeping the stationary variance at one.

# Algorithm

 1. Take the first observation of `innov` as the first observation of the output.
 2. Walk the remaining observations, each one `rho` times the previous output plus `sqrt(1 - rho^2)` times the innovation.

# Arguments

  - `innov`: Unit-variance innovations (observations × assets).
  - `rho`: The autocorrelation.

# Returns

  - `out::Matrix{Float64}`: The filtered process, in the innovations' own shape.

# Related

  - [`synthetic_panel_returns`](@ref)
  - [`synthetic_ar1_paths`](@ref)
"""
function synthetic_ar1_filter(innov::AbstractMatrix{<:Real}, rho::Real)
    scale = sqrt(1 - rho^2)
    out = Matrix{Float64}(undef, size(innov))
    out[1, :] = view(innov, 1, :)
    for t in 2:size(innov, 1)
        @views out[t, :] = rho .* out[t - 1, :] .+ scale .* innov[t, :]
    end
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Draw unit-variance fat-tailed shocks from a scale mixture of normals.

A standard normal is divided by the square root of a chi-squared variate over its degrees of freedom, which is a Student-t variate, and the result is rescaled to unit variance. `dof` must exceed two, or the Student-t variance does not exist.

# Arguments

  - `rng`: Random number generator.
  - `dof`: Degrees of freedom of the mixing distribution.
  - `dims`: The shape to draw.

# Returns

  - `z::Array{Float64}`: The shocks, of shape `dims`.

# Related

  - [`synthetic_panel_factors`](@ref)
  - [`synthetic_panel_returns`](@ref)
  - [`SYNTHETIC_DYNAMICS`](@ref)
"""
function synthetic_fat_tailed_normal(rng::Random.AbstractRNG, dof::Real, dims::Integer...)
    z = randn(rng, dims...)
    s = sqrt.(dof ./ rand(rng, Distributions.Chisq(dof), dims...))
    return z .* s .* sqrt((dof - 2) / dof)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Draw the latent traits of every asset, and the per-asset characteristics and Factor Exposures they anchor.

Six independent traits — size, value, quality, risk, growth and liquidity — are time-invariant, and every characteristic is a noisy function of them. The Factor Exposures are the standardised characteristics, so each style's exposure is a noisy proxy of the trait that drives that style's factor return, which is what makes the model recoverable from the panel alone.

# Algorithm

 1. Draw the six traits as independent standard normals.
 2. Build each characteristic from its median in [`SYNTHETIC_MEDIANS`](@ref), its trait loadings, and its own noise.
 3. Draw the market loading around one, and the shape of the cross-sectional idiosyncratic volatility.
 4. Standardise ten characteristics with [`synthetic_standardise`](@ref) into the style Factor Exposures, in the column order of [`SYNTHETIC_STYLES`](@ref).

# Arguments

  - `rng`: Random number generator.
  - `N`: Number of assets.

# Returns

  - `rt::NamedTuple`: The per-asset characteristics, the market loading `beta`, the idiosyncratic volatility shape `idio_shape`, and the style Factor Exposures `loadings` (assets × styles).

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`synthetic_panel_fields`](@ref)
  - [`SYNTHETIC_MEDIANS`](@ref)
  - [`SYNTHETIC_STYLES`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Draw the factor-return series of the synthetic panel's three Factor Families.

The market factor is fat-tailed and carries a positive mean, the industry factors are zero-mean and serially independent, and the style factors are mean-reverting.

# Algorithm

 1. Draw the style paths with [`synthetic_ar1_paths`](@ref), from the parameters of [`SYNTHETIC_STYLES`](@ref).
 2. Draw the market path as a per-observation mean plus a fat-tailed shock from [`synthetic_fat_tailed_normal`](@ref).
 3. Draw one annualised volatility per industry, and draw the industry paths with no mean and no autocorrelation.

# Arguments

  - `rng`: Random number generator.
  - `T`: Number of observations.
  - `n_ind`: Number of industries.

# Returns

  - `fac::NamedTuple`: The `market` path (observations), the `industry` paths (observations × industries), the `style` paths (observations × styles), and the per-observation means the fundamentals grow at.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`synthetic_panel_returns`](@ref)
  - [`SYNTHETIC_STYLES`](@ref)
  - [`SYNTHETIC_DYNAMICS`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the total returns of the synthetic panel from its systematic block and its idiosyncratic block.

The idiosyncratic level is set from `svr`, so the share of cross-sectional return variance the factor structure explains is the caller's number rather than an accident of the draws. The bearish signal enters the *next* observation's shock, so a descriptor built from it predicts a future return with no look-ahead.

# Algorithm

 1. Add the market block, the industry block and the style block into the systematic returns.
 2. Scale the idiosyncratic volatility so the systematic share of the cross-sectional variance is `svr`.
 3. Mix a fat-tailed transitory shock, a persistent momentum component and a mean-reverting pressure component, at the weights of [`SYNTHETIC_DYNAMICS`](@ref).
 4. Standardise the bearish signal across the cross-section, and subtract its lagged value from the shock.
 5. Add the two blocks, and floor the total return at `-0.99`.

# Arguments

  - `rng`: Random number generator.
  - `rt`: The per-asset characteristics of [`synthetic_panel_ratios`](@ref).
  - `fac`: The factor returns of [`synthetic_panel_factors`](@ref).
  - `ind_idx`: The industry of each asset, one index per asset.
  - `svr`: Share of the cross-sectional return variance the factor structure explains.

# Returns

  - `ret::NamedTuple`: The total returns `X` (observations × assets), the idiosyncratic returns `eps`, the true idiosyncratic variances `ivar`, and the latent bearish `signal` two descriptors observe.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`synthetic_ar1_filter`](@ref)
  - [`synthetic_fat_tailed_normal`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Draw the listing window of every asset, and the two masks of the [`AssetPanel`](@ref) they define.

An asset lists at the first observation unless it is drawn as a late lister, and it stays listed to the last observation unless it is drawn as a delisting. A delisting is never sooner than one year after the listing. The first asset is forced active throughout, so no observation has an empty cross-section.

# Algorithm

 1. Draw the late listers, and draw each one's first active observation.
 2. Draw the delistings, and draw each one's last active observation, no sooner than [`SYNTHETIC_PERIODS_PER_YEAR`](@ref) observations after its listing.
 3. Fill the active mask from the two windows, and force the first asset active.
 4. Copy the active mask into the estimation mask, and drop a twentieth of the assets from it, never the first.

# Arguments

  - `rng`: Random number generator.
  - `T`: Number of observations.
  - `N`: Number of assets.
  - `late`: Probability that an asset lists after the first observation.
  - `delist`: Probability that an asset delists before the last observation.

# Returns

  - `amsk::BitMatrix`: The active mask (observations × assets).
  - `emsk::BitMatrix`: The estimation mask (observations × assets), a subset of `amsk`.

# Related

  - [`AssetPanel`](@ref)
  - [`synthetic_asset_panel`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build every numeric Panel Field of the synthetic panel, in the column order it takes on the feature axis.

The accounting identities hold: the market capitalisation is the price times the share count, the enterprise value is the market capitalisation plus the debt less the cash, and the total assets are the sales over the asset turnover. The fundamentals grow at each asset's own expected return, so its valuation ratios are stationary in expectation.

# Algorithm

 1. Compound the active returns into the price, drift the share count, and multiply the two into the market capitalisation.
 2. Build the traded volume from the share count and the asset's own turnover.
 3. Grow every fundamental at the asset's expected return plus its issuance drift and its sales growth.
 4. Build the two forward-looking per-share fields from the price, and the two bearish-signal fields from `signal`.

# Arguments

  - `rng`: Random number generator.
  - `rt`: The per-asset characteristics of [`synthetic_panel_ratios`](@ref).
  - `fac`: The factor returns of [`synthetic_panel_factors`](@ref).
  - `amsk`: The active mask (observations × assets).
  - `X`: The total returns (observations × assets).
  - `signal`: The latent bearish signal (observations × assets).

# Returns

  - `fields::Vector{Pair{String, Matrix{Float64}}}`: One Panel Field per pair, its name and its values.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`synthetic_panel_blank!`](@ref)
  - [`SYNTHETIC_MEDIANS`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Blank the cells of every Panel Field that a real panel would not carry.

Two rules blank a cell, and the second one is unconditional: a Panel Field is observed on only part of its active cells, and no Panel Field is observed at all where the asset is not active. A blank is written as a `NaN`, which is what [`is_panel_blank`](@ref) reads, so [`asset_panel`](@ref) resolves it and each blanking Panel Field earns its observed-mask column.

# Algorithm

 1. Read the Panel Field's coverage from [`SYNTHETIC_COVERAGE`](@ref), or take `1 - missing_ratio` when it names none.
 2. Blank a random share of its active cells when that coverage is below one.
 3. Blank every inactive cell.

# Arguments

  - `rng`: Random number generator.
  - `fields`: The Panel Fields of [`synthetic_panel_fields`](@ref), modified in place.
  - `amsk`: The active mask (observations × assets).
  - `missing_ratio`: Share of active cells to blank on a Panel Field [`SYNTHETIC_COVERAGE`](@ref) does not name.

# Returns

  - `nothing`.

# Related

  - [`synthetic_asset_panel`](@ref)
  - [`SYNTHETIC_COVERAGE`](@ref)
  - [`asset_panel`](@ref)
  - [`is_panel_blank`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Wrap the synthetic panel's Panel Fields in the raw inputs [`asset_panel`](@ref) takes.

Every numeric Panel Field blanks, so each one carries a [`ForwardPanelFill`](@ref) and earns an observed-mask column. The classification carries no blank: an industry is a time-invariant fact of the asset, and the active mask already says when the asset is listed.

# Arguments

  - `fields`: The blanked Panel Fields of [`synthetic_panel_fields`](@ref).
  - `industry`: The industry label of each cell (observations × assets).
  - `levels`: The industry labels, in the one-hot column order.

# Returns

  - `inputs::Vector{AbstractPanelFieldInput}`: The raw Panel Fields, in feature-axis order.

# Related

  - [`asset_panel`](@ref)
  - [`NumericPanelInput`](@ref)
  - [`CategoricalPanelInput`](@ref)
  - [`ForwardPanelFill`](@ref)
"""
function synthetic_panel_inputs(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}},
                                industry::AbstractMatrix{<:AbstractString},
                                levels::AbstractVector{<:AbstractString})
    inputs = AbstractPanelFieldInput[NumericPanelInput(; name = name, vals = arr,
                                                       alg = ForwardPanelFill(; val = 0.0))
                                     for (name, arr) in fields]
    push!(inputs,
          CategoricalPanelInput(; name = "industry", vals = industry, levels = levels))
    return inputs
end
"""
    synthetic_asset_panel(;
        n_assets::Integer = 500,
        n_observations::Integer = 2520,
        n_industries::Integer = 10,
        start_date::Dates.Date = Dates.Date(2015, 1, 1),
        systematic_variance_ratio::Real = 0.5,
        late_listing_proba::Real = 0.15,
        delisting_proba::Real = 0.15,
        missing_ratio::Real = 0.01,
        rng::Random.AbstractRNG = Random.default_rng()
    ) -> @NamedTuple{rd::ReturnsResult, truth::NamedTuple}

Draw a synthetic point-in-time Asset Panel from a known factor model, and return the model beside it.

The returns follow `X = f * B' + eps`, floored at `-0.99`. The Factor Exposures `B` are per-asset traits, and every Panel Field is a noisy function of the same traits, so a Cross-Sectional Regression fit on the panel recovers the factor returns the generator drew. This is what makes the panel an acceptance test rather than a demonstration: the answer is known before the estimator runs.

Three Factor Families span the model. The market family is one fat-tailed factor whose exposure is the asset's beta. The industry family is one zero-mean factor per industry, and its exposures are the one-hot classification. The style family is the ten mean-reverting factors of [`SYNTHETIC_STYLES`](@ref), and its exposures are the standardised characteristics.

The panel rides the carrier the same way a real one does: the returns are `rd.X`, the Panel Field values are `rd.Z`, their names are `rd.nz`, and the field index and the two masks are `rd.pnl`. A cell the asset is not active on is a `NaN` in `rd.X` and a blank that [`asset_panel`](@ref) resolved in `rd.Z`, where the Panel Field's observed-mask column records that it was never observed.

!!! warning "The default size allocates about half a gigabyte"

    `rd.Z` is `n_observations × n_assets × features`, and the defaults give 2520 × 500 × 50 `Float64`. Pass smaller dimensions in a test.

# Algorithm

 1. Assign each asset an industry, weighted by the memberships of [`SYNTHETIC_INDUSTRIES`](@ref).
 2. Draw the per-asset characteristics and Factor Exposures with [`synthetic_panel_ratios`](@ref).
 3. Draw the factor returns with [`synthetic_panel_factors`](@ref), and the total returns with [`synthetic_panel_returns`](@ref).
 4. Draw the listing windows and the two masks with [`synthetic_panel_masks`](@ref).
 5. Build the Panel Fields with [`synthetic_panel_fields`](@ref), and blank them with [`synthetic_panel_blank!`](@ref).
 6. Build the carrier through [`asset_panel`](@ref), and assemble the truths.

# Arguments

  - `n_assets::Integer = 500`: Number of assets, the coverage universe.
  - `n_observations::Integer = 2520`: Number of observations.
  - `n_industries::Integer = 10`: Number of industries.
  - `start_date::Dates.Date = Dates.Date(2015, 1, 1)`: First observation. The clock is a business-day calendar.
  - `systematic_variance_ratio::Real = 0.5`: Share of the cross-sectional return variance the factor structure explains.
  - `late_listing_proba::Real = 0.15`: Probability that an asset lists after the first observation.
  - `delisting_proba::Real = 0.15`: Probability that an asset delists before the last observation.
  - `missing_ratio::Real = 0.01`: Share of active cells blanked on a Panel Field [`SYNTHETIC_COVERAGE`](@ref) does not name.
  - `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator. Pass a `StableRNG` for a panel that is reproducible across Julia versions.

# Validation

  - `n_assets >= 2`, because the estimation mask drops an asset that is not the first.
  - `n_observations >= 2`, because the mean-reverting component reads the previous observation.
  - `1 <= n_industries <= length(SYNTHETIC_INDUSTRIES)`.
  - `0 < systematic_variance_ratio < 1`.
  - `0 <= late_listing_proba <= 1`, `0 <= delisting_proba <= 1` and `0 <= missing_ratio <= 1`.

# Returns

  - `rd::ReturnsResult`: The carrier, holding the returns, the panel and the business-day clock.
  - `truth::NamedTuple`: The model the panel was drawn from, holding the factor names `nf`, the Factor Family of each factor `fgrp`, the Factor Exposures `B` (assets × factors), the factor returns `f` (observations × factors), the idiosyncratic returns `eps` (observations × assets) and the idiosyncratic variances `ivar` (assets).

# Examples

```jldoctest
julia> res = synthetic_asset_panel(; n_assets = 6, n_observations = 40, n_industries = 2,
                                   rng = StableRNG(123));

julia> size(res.rd.X), length(res.rd.nz)
((40, 6), 42)

julia> res.truth.nf[1:4]
4-element Vector{String}:
 "market"
 "industry=Real Estate"
 "industry=Software"
 "size"

julia> size(res.truth.B), size(res.truth.f)
((6, 13), (40, 13))
```

# Related

  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`asset_panel`](@ref)
  - [`SYNTHETIC_INDUSTRIES`](@ref)
  - [`SYNTHETIC_STYLES`](@ref)
  - [`SYNTHETIC_DYNAMICS`](@ref)
"""
function synthetic_asset_panel(; n_assets::Integer = 500, n_observations::Integer = 2520,
                               n_industries::Integer = 10,
                               start_date::Dates.Date = Dates.Date(2015, 1, 1),
                               systematic_variance_ratio::Real = 0.5,
                               late_listing_proba::Real = 0.15,
                               delisting_proba::Real = 0.15, missing_ratio::Real = 0.01,
                               rng::Random.AbstractRNG = Random.default_rng())
    @argcheck(n_assets >= 2 && n_observations >= 2,
              DomainError((n_assets, n_observations),
                          "a synthetic panel needs at least two assets and two observations: the estimation mask drops an asset that is not the first, and the mean-reverting component reads the previous observation"))
    @argcheck(1 <= n_industries <= length(SYNTHETIC_INDUSTRIES),
              DomainError(n_industries,
                          "n_industries takes the first entries of SYNTHETIC_INDUSTRIES, so it must lie in 1:$(length(SYNTHETIC_INDUSTRIES))"))
    @argcheck(0 < systematic_variance_ratio < 1,
              DomainError(systematic_variance_ratio,
                          "systematic_variance_ratio scales the idiosyncratic variance by (1 - svr) / svr, so it must lie strictly between zero and one"))
    @argcheck(all(x -> 0 <= x <= 1, (late_listing_proba, delisting_proba, missing_ratio)),
              DomainError((late_listing_proba, delisting_proba, missing_ratio),
                          "late_listing_proba, delisting_proba and missing_ratio are probabilities, so each must lie in [0, 1]"))
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

export synthetic_asset_panel
