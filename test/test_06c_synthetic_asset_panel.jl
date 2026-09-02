#=
The synthetic Asset Panel generator (issue #656, map #643).

The generator draws a panel from a factor model it also returns, so every test here asks one
question: does the panel the caller receives still hold the model the generator drew? The
shapes, the two masks, the reconciliation `X = f B' + eps` and the reproducibility of one
seed are the four the acceptance tests of the prior stand on.
=#
using Statistics, Dates

@testset "Synthetic panel shapes and carrier" begin
    res = synthetic_asset_panel(; n_assets = 8, n_observations = 60, n_industries = 3,
                                rng = StableRNG(987))
    rd, tr = res.rd, res.truth
    T, N, K = 60, 8, 1 + 3 + length(PortfolioOptimisers.SYNTHETIC_STYLES)

    @test size(rd.X) == (T, N)
    @test length(rd.nx) == N
    @test rd.nx[1] == "A00001"
    @test length(rd.ts) == T
    @test allunique(rd.ts)
    @test all(d -> Dates.dayofweek(d) <= 5, rd.ts)
    @test issorted(rd.ts)

    # Twenty numeric Panel Fields, each with its observed-mask column, then the one-hot
    # industry block.
    @test size(rd.Z) == (T, N, 43)
    @test length(rd.nz) == 43
    @test rd.nz[1] == "adj_close"
    @test rd.nz[2] == "adj_close::observed"
    @test rd.nz[(end - 2):end] ==
          ["industry=Real Estate", "industry=Software", "industry=Banks"]
    @test all(isfinite, rd.Z)
    @test length(rd.pnl.pf) == 21
    @test PortfolioOptimisers.panel_field(rd.pnl, "market_cap").kind == NumericPanelField()
    @test PortfolioOptimisers.panel_field(rd.pnl, "industry").kind.levels ==
          ["Real Estate", "Software", "Banks"]

    @test size(tr.B) == (N, K)
    @test size(tr.f) == (T, K)
    @test size(tr.eps) == (T, N)
    @test length(tr.ivar) == N
    @test length(tr.nf) == K
    @test length(tr.fgrp) == K
    @test tr.nf[1] == "market"
    @test tr.fgrp == vcat("market", fill("industry", 3), fill("style", 10))
    @test tr.nf[5:end] == [s[1] for s in PortfolioOptimisers.SYNTHETIC_STYLES]
    @test all(x -> x > 0, tr.ivar)
end

@testset "Synthetic panel masks" begin
    res = synthetic_asset_panel(; n_assets = 40, n_observations = 400, n_industries = 4,
                                rng = StableRNG(4242))
    amsk, emsk = res.rd.pnl.amsk, res.rd.pnl.emsk

    # The subset invariant the carrier checks, and the listing structure it describes.
    @test all(emsk .<= amsk)
    @test !all(amsk)
    @test all(view(amsk, :, 1))
    @test all(view(emsk, :, 1))
    @test count(!, emsk) > count(!, amsk)

    # A cell that is not active carries a `NaN` return and no observed value.
    @test all(isnan, res.rd.X[.!amsk])
    @test all(isfinite, res.rd.X[amsk])
    obs = res.rd.Z[:, :, findfirst(==("market_cap::observed"), res.rd.nz)]
    @test all(iszero, obs[.!amsk])
    @test all(isone, obs[amsk])

    # Every active window is contiguous: an asset lists once and delists once.
    for i in axes(amsk, 2)
        col = findall(view(amsk, :, i))
        @test col == first(col):last(col)
    end

    # Both probabilities at zero leave every asset listed throughout.
    full = synthetic_asset_panel(; n_assets = 10, n_observations = 50, n_industries = 2,
                                 late_listing_proba = 0.0, delisting_proba = 0.0,
                                 rng = StableRNG(11))
    @test all(full.rd.pnl.amsk)
    @test all(isfinite, full.rd.X)

    # Both probabilities at one drive the late-listing and the delisting branches.
    thin = synthetic_asset_panel(; n_assets = 10, n_observations = 400, n_industries = 2,
                                 late_listing_proba = 1.0, delisting_proba = 1.0,
                                 rng = StableRNG(12))
    tmsk = thin.rd.pnl.amsk
    @test !any(view(tmsk, 1, 2:10))
    @test all(view(tmsk, :, 1))
end

@testset "Synthetic panel reconciles with its own model" begin
    svr = 0.4
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 120, n_industries = 3,
                                systematic_variance_ratio = svr, rng = StableRNG(2024))
    rd, tr = res.rd, res.truth
    amsk = rd.pnl.amsk

    # The floor at -0.99 is part of the model, so the reconciliation carries it.
    recon = max.(tr.f * tr.B' .+ tr.eps, -0.99)
    @test isapprox(rd.X[amsk], recon[amsk]; rtol = 1e-12)

    # The market exposure is positive and bounded, and the industry block is one-hot.
    onehot = tr.B[:, 2:4]
    @test all(x -> 0.1 <= x <= 3.0, tr.B[:, 1])
    @test all(x -> x == 0 || x == 1, onehot)
    @test all(isone, sum(onehot; dims = 2))

    # The style exposures are standardised across the cross-section.
    styles = tr.B[:, 5:end]
    @test isapprox(vec(sum(styles; dims = 1)), zeros(10); atol = 1e-10)
    @test isapprox(vec(Statistics.std(styles; dims = 1, corrected = false)), ones(10);
                   rtol = 1e-10)

    # The systematic share of the cross-sectional variance is the caller's number.
    sysvar = Statistics.mean(Statistics.var(tr.f * tr.B'; dims = 2, corrected = false))
    @test isapprox(sysvar / (sysvar + Statistics.mean(tr.ivar)), svr; rtol = 1e-10)
end

@testset "Synthetic panel is reproducible from one seed" begin
    kwargs = (; n_assets = 7, n_observations = 45, n_industries = 2, missing_ratio = 0.05)
    a = synthetic_asset_panel(; kwargs..., rng = StableRNG(7))
    b = synthetic_asset_panel(; kwargs..., rng = StableRNG(7))
    c = synthetic_asset_panel(; kwargs..., rng = StableRNG(8))

    # `X` carries a `NaN` where an asset is not active, so equality is `isequal`.
    @test isequal(a.rd.X, b.rd.X)
    @test a.rd.Z == b.rd.Z
    @test a.rd.nz == b.rd.nz
    @test a.rd.ts == b.rd.ts
    @test a.rd.pnl.amsk == b.rd.pnl.amsk
    @test a.rd.pnl.emsk == b.rd.pnl.emsk
    @test a.truth.B == b.truth.B
    @test a.truth.f == b.truth.f
    @test a.truth.eps == b.truth.eps
    @test a.truth.ivar == b.truth.ivar
    @test !isequal(a.rd.X, c.rd.X)
end

@testset "Synthetic panel coverage and blanking" begin
    res = synthetic_asset_panel(; n_assets = 30, n_observations = 300, n_industries = 3,
                                missing_ratio = 0.2, rng = StableRNG(55))
    rd = res.rd
    amsk = rd.pnl.amsk
    obs_share = name -> begin
        col = rd.Z[:, :, findfirst(==(name * "::observed"), rd.nz)]
        return sum(col[amsk]) / sum(amsk)
    end

    # A protected Panel Field is observed on every active cell; a named one sits at its
    # own coverage; an unnamed one sits at `1 - missing_ratio`.
    @test obs_share("adj_close") == 1
    @test obs_share("market_cap") == 1
    @test isapprox(obs_share("eps_ntm"), 0.8; atol = 0.05)
    @test isapprox(obs_share("ebitda_ttm"), 0.9; atol = 0.05)
    @test isapprox(obs_share("sales_ttm"), 0.8; atol = 0.05)

    # Nothing is blanked when the coverage is complete and no Panel Field is named below
    # one, so the observed columns of the unnamed fields are the active mask itself.
    none = synthetic_asset_panel(; n_assets = 6, n_observations = 40, n_industries = 2,
                                 missing_ratio = 0.0, rng = StableRNG(56))
    ncol = none.rd.Z[:, :, findfirst(==("sales_ttm::observed"), none.rd.nz)]
    @test ncol == none.rd.pnl.amsk
end

@testset "Synthetic panel accounting identities" begin
    res = synthetic_asset_panel(; n_assets = 10, n_observations = 80, n_industries = 2,
                                missing_ratio = 0.0, rng = StableRNG(99))
    rd = res.rd
    amsk = rd.pnl.amsk
    field = name -> rd.Z[:, :, findfirst(==(name), rd.nz)]

    # The market capitalisation is the price times the share count, on every active cell.
    mcap = field("market_cap")[amsk]
    @test isapprox(mcap, (field("adj_close") .* field("adj_shares_outstanding"))[amsk];
                   rtol = 1e-10)

    # The levels every descriptor divides by stay strictly positive, and the gross margin
    # keeps the cost of revenue below the sales.
    @test all(x -> x > 0, mcap)
    @test all(x -> x > 0, field("adj_close")[amsk])
    @test all(x -> x > 0, field("total_assets")[amsk])
    @test all(x -> x >= 0, field("adj_volume")[amsk])
    @test all(x -> x >= 0, field("short_interest")[amsk])
    @test all(x -> x >= 0, field("eps_ntm_std")[amsk])
    @test all((field("cost_of_revenue_ttm") .< field("sales_ttm"))[amsk])
end

@testset "Synthetic panel helpers" begin
    # A constant cross-section standardises to zero rather than to a division by zero.
    @test PortfolioOptimisers.synthetic_standardise(fill(3.0, 5)) == zeros(5)
    @test isapprox(PortfolioOptimisers.synthetic_standardise([1.0, 2.0, 3.0]),
                   [-sqrt(1.5), 0.0, sqrt(1.5)]; rtol = 1e-10)

    # The autoregressive filter keeps the stationary variance at one.
    rng = StableRNG(3)
    filt = PortfolioOptimisers.synthetic_ar1_filter(randn(rng, 20000, 2), 0.9)
    @test isapprox(Statistics.var(filt[:, 1]), 1.0; rtol = 0.1)
    @test isapprox(Statistics.cor(filt[2:end, 1], filt[1:(end - 1), 1]), 0.9; rtol = 0.1)

    # The paths hit their stated annualised volatility and mean.
    paths = PortfolioOptimisers.synthetic_ar1_paths(StableRNG(4), 20000, [0.2], [0.1],
                                                    [0.0])
    @test isapprox(Statistics.std(paths[:, 1]) * sqrt(252), 0.2; rtol = 0.1)
    @test isapprox(Statistics.mean(paths[:, 1]) * 252, 0.1; atol = 0.05)

    # The fat-tailed draw is unit variance and heavier tailed than a normal.
    z = PortfolioOptimisers.synthetic_fat_tailed_normal(StableRNG(5), 6.0, 40000)
    @test isapprox(Statistics.var(z), 1.0; rtol = 0.15)
    @test StatsBase.kurtosis(z) > 1
end

@testset "Synthetic panel statistical truths" begin
    res = synthetic_asset_panel(; n_assets = 25, n_observations = 1500, n_industries = 4,
                                rng = StableRNG(606))
    tr = res.truth

    # The idiosyncratic variances the generator reports are the ones it drew.
    realised = vec(Statistics.var(tr.eps; dims = 1, corrected = false))
    @test isapprox(Statistics.mean(realised ./ tr.ivar), 1.0; rtol = 0.15)

    # The idiosyncratic returns are close to uncorrelated with the factor returns.
    @test maximum(abs.(Statistics.cor(tr.f, tr.eps))) < 0.25
end

@testset "Synthetic panel refusals" begin
    @test_throws DomainError synthetic_asset_panel(; n_assets = 1)
    @test_throws DomainError synthetic_asset_panel(; n_observations = 1)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   n_industries = 0)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   n_industries = 17)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   systematic_variance_ratio = 0.0)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   systematic_variance_ratio = 1.0)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   late_listing_proba = 1.5)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   delisting_proba = -0.1)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   missing_ratio = 2.0)

    # One industry is the smallest classification the generator admits.
    one = synthetic_asset_panel(; n_assets = 4, n_observations = 20, n_industries = 1,
                                rng = StableRNG(1))
    @test one.truth.nf[2] == "industry=Real Estate"
    @test all(isone, one.truth.B[:, 2])
end
