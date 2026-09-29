#=
Parity of the Cross-Sectional Factor Prior on a panel with gaps (#1377, map #1375). The prior
passes both universe masks to the idiosyncratic variance and to the refinement of the regression
weights. Without them an asset that delists and lists again kept its variance over the inactive
stretch, and the regime statistic read an active asset outside the estimation universe.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "The prior reads both masks on a panel with gaps, at parity (#1377)" begin
    # Each stored case is the fit of `parity_small_panel()` with a market factor and the two
    # passthrough styles, `minra = 5`, and the default of every other parameter. `MasksOnePass`
    # keeps the market-cap weights, and `MasksTwoPass` blends them half-way towards the inverse
    # idiosyncratic variance. The outputs are the idiosyncratic variance history, the
    # regression weights, `mu` and `sigma`.
    fx = parity_small_panel()
    at = fx.at
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    # The factor covariance clips its regime multiplier to (0.7, 1.6), as the oracle does by
    # default. Our default does not clip, and the two-pass case is the first measured case
    # where the clip binds: without it the factor covariance is 0.991 times the stored one.
    # That default belongs to #1383; this file measures the masks.
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centred = true,
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    load(c, o) = parity_load("CrossSectionalFactorPrior", c, o)
    i4 = at.relist[1]

    @testset "$(c)" for (c, wa) in (("MasksOnePass", MarketCapWeights()),
                                    ("MasksTwoPass", BlendedInverseVarianceWeights(; lambda = 0.5)))
        pr = prior(CrossSectionalFactorPrior(; factors = factors, minra = 5, wa = wa,
                                             pe = pe), fx.rd)
        # The variance resets when the asset delists, and warms up again after it lists, so
        # the non-finite cells of the relisted asset match. The regime statistic reads the
        # estimation universe alone. Measured maxrel 1.5e-15.
        @test parity_compare(pr.rr.vs, load(c, "IdioVariances"); name = "$(c) vs").ok
        # Measured maxrel 4.2e-16.
        @test parity_compare(pr.rr.rw, load(c, "RegressionWeights"); name = "$(c) rw").ok
        # The relisted asset is still in its warm-up at the latest observation, so neither side
        # states its idiosyncratic variance. Both sides state its mean and its covariances
        # with the other assets, which the model determines, and neither states its variance
        # (#1384).
        @test isnan(pr.rr.vs[end, i4]) && isnan(load(c, "IdioVariances")[end, i4])
        @test isfinite(pr.mu[i4]) && isnan(pr.sigma[i4, i4])
        # Measured maxrel 2.9e-14.
        @test parity_compare(pr.mu, vec(load(c, "Mu")); name = "$(c) mu").ok
        # A covariance compares against its largest entry, because its small off-diagonal
        # entries come from a cancellation (#1376). Measured maxscaled 3.3e-13, and maxrel 1.6e-11 cell by cell.
        @test parity_compare(pr.sigma, load(c, "Sigma"); scale = :array,
                             name = "$(c) sigma").ok
    end
end
