#=
```@meta
Description = "Turn prices into returns and returns into a prior with prices_to_returns and prior, which give the expected returns and covariance every optimiser uses."
```

# Data and priors

The first stage turns prices into a prior, which holds the expected returns vector and the
covariance matrix that every optimiser uses. Two calls cover the common path,
[`prices_to_returns`](@ref) and [`prior`](@ref). For the other moment estimators and the priors that
take views, see the
[moments and priors examples](../examples/2_moments_priors/01_Expected_Returns_Estimation.md).
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, LinearAlgebra,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Prices to returns

Market data usually arrives as prices, and an optimiser needs returns.
[`prices_to_returns`](@ref) converts a `TimeArray` of prices into a [`ReturnsResult`](@ref) that
holds the asset names `nx` and the return matrix `X`. Before it converts its inputs, it checks
that their names, dates and shapes match.

A real price table has gaps. An asset that lists or delists inside the sample has no price before
or after it trades, a halted asset has no quotes, and two exchanges close on different holidays.
[`price_ingestion`](@ref) reads the first and the last quote of each asset, its listing span.
The conversion keeps every gap in the returns, and it returns an [`AssetPanel`](@ref) on
`rd.pnl` that states which assets have data on each date. To fill a gap inside the listing span,
use [`PriceGapFill`](@ref). To drop the assets or the dates with too much missing data, use
[`MissingDataFilter`](@ref). The page
[Data preprocessing and the ingestion layer](../examples/1_foundations/02_Data_Preprocessing.md)
covers these steps in depth.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
On a price table, `prices_to_returns(X)` is the same call as
`prices_to_returns(price_ingestion(PriceIngestion(), X))`. Write the two calls when you have more
than one table. [`price_ingestion`](@ref) takes factor prices as `F`, benchmark prices as `B`, and
implied volatilities as `iv`. By default it aligns them on the dates of the asset prices. The
[`PriceIngestion`](@ref) estimator also sets how to collapse the prices to a lower frequency. The
conversion then puts the factor returns in `rd.F` and the benchmark returns in `rd.B`, on the same
[`ReturnsResult`](@ref) as the asset returns.
[The point-in-time universe](08_Point_in_Time_Universe.md) runs these two calls on a price table
with gaps.

## 2. Returns to a prior

[`prior`](@ref) applies a prior estimator to a [`ReturnsResult`](@ref) and returns the moments.
The default is [`EmpiricalPrior`](@ref), which computes the sample mean and the sample
covariance, and then repairs the covariance to the nearest positive definite matrix if it needs
to. We compute it and print the mean and the volatility of each asset.
=#

pr = prior(EmpiricalPrior(), rd)

pretty_table(DataFrame("Asset" => rd.nx, "Expected return" => pr.mu,
                       "Volatility" => sqrt.(diag(pr.sigma))); formatters = [resfmt],
             title = "Empirical prior: per-asset mean and volatility")

#=
## 3. Other priors

You call every prior estimator with `prior(pe, rd)`, and you change the prior by changing its
first argument. The common alternatives are these:

  - [`FactorPrior`](@ref) computes the moments from a factor model. See
    [Factor Priors](../examples/2_moments_priors/04_Factor_Priors.md).
  - [`BlackLittermanPrior`](@ref) combines the moments of a market equilibrium with your views.
    See [Black-Litterman](../examples/2_moments_priors/05_Black_Litterman.md). Three more priors
    of the same family are [`BayesianBlackLittermanPrior`](@ref),
    [`FactorBlackLittermanPrior`](@ref), which takes views on the factors, and
    [`AugmentedBlackLittermanPrior`](@ref), which takes views on the assets and on the factors
    together. See
    [Advanced Black-Litterman](../examples/2_moments_priors/06_Advanced_Black_Litterman.md).
  - [`EntropyPoolingPrior`](@ref) changes the probability of each historical scenario until the
    scenarios satisfy your views, and it takes views on any moment. See
    [Entropy Pooling](../examples/2_moments_priors/07_Entropy_Pooling.md).
  - [`OpinionPoolingPrior`](@ref) combines several entropy pooling priors, each with its own
    views, into one probability for each scenario, and computes the moments under those
    probabilities. See [Opinion Pooling](../examples/2_moments_priors/08_Opinion_Pooling.md).
  - [`CrossSectionalFactorPrior`](@ref) computes the moments from a factor model that it fits
    across the assets on each date, not through time. On each date it regresses the returns of
    the assets on their exposures of the date before, by default. It therefore needs no factor
    returns, and the set of assets can change from date to date.

A cross-sectional prior needs more input than the others. It does not read `F`. It reads an
[`AssetPanel`](@ref) of data for each asset on each date, such as a market capitalisation, a book
equity or an industry label, which the returns hold in `rd.pnl`. You give it a list of pairs. Each
pair names a factor and the estimator that computes the exposures of the assets to that factor from
the panel. An industry label gives one factor for each industry, and one pair then gives many
factors. The page
[Cross-sectional factor model, end to end](../examples/7_putting_it_together/05_Cross_Sectional_Factor_Model.md)
fits one, and
[Cross-sectional factor model through a Pipeline](../examples/7_putting_it_together/06_Cross_Sectional_Factor_Pipeline.md)
gets the same weights through a [`Pipeline`](@ref).

A prior also holds a covariance estimator, and you can change it for another, such as a denoised
covariance or the Gerber covariance. The page
[Covariance Estimation](../examples/2_moments_priors/02_Covariance_Estimation.md) compares them. A
moment estimator can also use only the last part of the sample, or weight the recent returns more
than the old ones. Use this when the recent returns describe the market better than the full
sample does, as the page
[Windowed Estimators](../examples/2_moments_priors/10_Windowed_Estimators.md) shows.
[`RegimeAdjustedExpWeightedVariance`](@ref) and [`RegimeAdjustedExpWeightedCovariance`](@ref)
multiply an exponentially weighted estimate by a factor that measures how far the recent
standardised returns are from their usual size.

## 4. A first look at the data

[`plot_prior`](@ref) shows a prior in one figure: the expected returns, the volatility of each
asset and the correlation matrix. Look at it before you optimise, to find an asset with an
expected return or a volatility that you do not expect.
=#

plot_prior(pr, rd)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow guide page split from the monolith §1 (Computing returns). Keeps the blessed
#src   minimal path (prices_to_returns → prior(EmpiricalPrior())) and defers depth to the
#src   2_moments_priors examples via cross-links. Verified end-to-end on kaimon.
#src - Uses the 1-year SP500 slice (single TimeArray) rather than the monolith's heterogeneous
#src   asset+factor+benchmark load, to keep the front-of-guide example minimal; the factor/
#src   benchmark capability is described in prose in §1.
