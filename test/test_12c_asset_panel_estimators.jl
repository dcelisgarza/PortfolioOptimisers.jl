#=
The producer family: an Asset Panel built at the point of use.

`#804` moved the derived feature source onto the distance estimator. `FeatureDistance` carries one
slot, `ape`, bound to `Option{<:AbstractAssetPanelEstimator}`: `nothing` reads the panel the data
carrier holds, and a producer builds a **static** `AssetPanel` holding **one** `TensorPanelField`
from the prior result and the returns of the subproblem that runs it.

This file pins the four claims that decision rests on.

 1. The slot admits a producer and nothing else. A literal matrix is refused by the type, which is
    what keeps data off an estimator.
 2. A producer returns a static panel with one tensor field, named for what it holds.
 3. The trailing axis is labelled off the data carrier where a name exists there, and positionally
    otherwise.
 4. A producer is configuration: a view passes it through, and `factory` reaches into it.
=#
# The synthetic point-in-time Asset Panel of the cross-sectional testset.
include(joinpath(@__DIR__, "test06c_setup.jl"))

@testset "The `ape` slot admits a producer and refuses a literal" begin
    @test isnothing(FeatureDistance().ape)
    @test isa(RegressionPanel(), PortfolioOptimisers.AbstractAssetPanelEstimator)
    @test isa(PhylogenyPanel(), PortfolioOptimisers.AbstractAssetPanelEstimator)
    @test FeatureDistance(; ape = RegressionPanel()).ape === RegressionPanel()

    # A literal is refused by the field's type bound, not by a runtime check.
    @test_throws TypeError FeatureDistance(; ape = rand(3, 2))
    @test_throws TypeError FeatureDistance(;
                                           ape = asset_panel([NumericPanelInput(;
                                                                                name = "a",
                                                                                vals = [1.0,
                                                                                        2.0])]))

    # The field order is `metric, alg, sim, ape, sel, strict`, and the printed block says so.
    @test fieldnames(FeatureDistance) == (:metric, :alg, :sim, :ape, :sel, :strict)
end

@testset "RegressionPanel reads the loadings, and labels them off the carrier" begin
    rng = StableRNG(20260906)
    T, N, K = 80, 5, 3
    X = randn(rng, T, N) / 100
    F = randn(rng, T, K) / 100
    nx = ["A", "B", "C", "D", "E"]
    nf = ["mkt", "size", "value"]
    rd = ReturnsResult(; nx = nx, X = X, nf = nf, F = F)
    pr = prior(FactorPrior(), rd)

    pnl = PortfolioOptimisers.asset_panel(RegressionPanel(), pr, rd, X)
    @test isa(pnl, AssetPanel)
    @test PortfolioOptimisers.panel_is_static(pnl)
    @test length(pnl.pf) == 1
    f = PortfolioOptimisers.panel_field(pnl, "loadings")
    @test isa(f, TensorPanelField)
    @test f.axis == "factor"
    @test f.vals == pr.rr.L
    # `L` is `assets × factors`, so the panel's asset axis is the carrier's with no transpose.
    @test size(f.vals, 1) == N

    # A `Regression` whose `L` is unset has the raw `M` as its loadings, so its trailing axis
    # *is* the carrier's factor axis and takes the carrier's names.
    @test isnothing(getfield(pr.rr, :L))
    @test f.labels == nf

    # With no data carrier there is no name, so the labels are positional.
    @test PortfolioOptimisers.panel_field(PortfolioOptimisers.asset_panel(RegressionPanel(),
                                                                          pr, nothing, X),
                                          "loadings").labels ==
          [string(k) for k in 1:size(pr.rr.L, 2)]

    # A wrapped prior forwards `rr`, so nesting order does not matter.
    prw = prior(BlackLittermanPrior(; pe = FactorPrior(),
                                    sets = UniverseSets(;
                                                        dict = Dict("nx" => nx, "nf" => nf)),
                                    views = LinearConstraintEstimator(;
                                                                      val = ["A == 0.01"])),
                rd)
    @test PortfolioOptimisers.panel_field(PortfolioOptimisers.asset_panel(RegressionPanel(),
                                                                          prw, rd, X),
                                          "loadings").vals == prw.rr.L

    # A prior that never computed a regression is refused, and the message names the slot.
    @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.asset_panel(RegressionPanel(),
                                                                                    prior(EmpiricalPrior(),
                                                                                          rd),
                                                                                    rd, X)
    # And a call with no prior at all names the pre-prior site it runs at.
    res = @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.asset_panel(RegressionPanel(),
                                                                                          nothing,
                                                                                          rd,
                                                                                          X)
    @test occursin("RegressionPanel", res.value.msg)
    @test occursin("PhylogenyPanel", res.value.msg)
end

@testset "RegressionPanel on a cross-sectional prior: named off the block, expanded to the universe" begin
    #=
    ADR 0045's eighth amendment reads a name off the data carrier "where a name exists", and
    its first text labelled a `CrossSectionalFactorModel` positionally as a block whose
    factors "no data names". The block has carried `nf` since #724 landed beside that
    decision, so the producer reads it, through the same verb the diagnostics label their
    factor axis with, which maps the raw names onto the re-based axis under a family
    re-basis. And a prior fitted on a point-in-time Asset Panel writes `NaN` on every asset
    outside its Investable Mask (ADR 0117), which a Panel Field refuses; the producer reads
    the loadings on the mask and answers the full universe with a zero row and a false
    observed mask outside it, as #1062 ruled for every uncertainty set fitted standalone on
    such a prior, so a view of the field at the mask recovers the reduced loadings. The
    reduced prior is what an optimiser hands the producer, and the routed path solves with no
    weight outside the mask.
    =#
    PO = PortfolioOptimisers
    rdp = synthetic_asset_panel(; n_assets = 40, n_observations = 200, n_industries = 3,
                                rng = StableRNG(725_001)).rd
    factors = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" =>
                   CompositeExposure(; descriptors = [LogMarketCap()], family = "style"),
               "value" =>
                   CompositeExposure(; descriptors = [BookToPrice()], family = "style")]
    prp = prior(CrossSectionalFactorPrior(; factors = factors), rdp)
    msk = PO.investable_mask(prp)
    @test count(msk) < length(msk)
    nnf = count(i -> !all(isfinite, view(prp.rr.L, i, :)), axes(prp.rr.L, 1))
    @test nnf == length(msk) - count(msk)

    # Standalone on the unreduced prior: the full universe, zero rows and a false observed
    # mask outside the Investable Mask, the reduced loadings on it, and no `NaN` anywhere.
    idx = findall(msk)
    prr = PO.port_opt_view(prp, idx)
    rdr = PO.port_opt_view(rdp, idx)
    pf = PO.asset_panel(RegressionPanel(), prp, rdp, rdp.X)
    ff = PO.panel_field(pf, "loadings")
    @test size(ff.vals) == (length(msk), length(prp.rr.nf))
    @test all(isfinite, ff.vals)
    @test ff.vals[msk, :] == prr.rr.L
    @test all(iszero, ff.vals[.!msk, :])
    @test size(ff.omsk) == size(ff.vals)
    @test all(ff.omsk[msk, :]) && !any(ff.omsk[.!msk, :])
    @test ff.labels == prp.rr.nf
    # The view at the mask recovers the reduced panel, and `:observed` is the mask as a column.
    @test PO.panel_field(PO.port_opt_view(pf, idx), "loadings").vals == prr.rr.L
    @test vec(feature_matrix(pf, ["loadings" => :observed])) == msk
    # A zero row is a zero feature vector, so an excluded asset sits at distance one from
    # every asset with loadings under the default metric.
    D = distance(FeatureDistance(; ape = RegressionPanel()), nothing, rdp.X; pr = prp,
                 rd = rdp)
    @test all(isone, D[.!msk, msk])

    # Reduced to the mask, the panel builds over the reduced universe with no mask, and the
    # labels are the block's own raw factor names: no family re-basis, so `L` is `M` and the
    # axis is `nf`.
    f = PO.panel_field(PO.asset_panel(RegressionPanel(), prr, rdr, rdr.X), "loadings")
    @test size(f.vals) == (count(msk), length(prr.rr.nf))
    @test isnothing(f.omsk)
    @test f.labels == prr.rr.nf
    @test f.labels == PO.cs_diagnostic_factor_names(prr.rr)
    @test f.labels[1] == "market"
    @test any(startswith("industry="), f.labels)
    # The carrier is not read for the names: the block carries them.
    @test PO.panel_field(PO.asset_panel(RegressionPanel(), prr, nothing, rdr.X),
                         "loadings").labels == f.labels

    # Under a family re-basis `L` is narrower than `M`, and the labels follow the reduced
    # axis: the dropped member of the constrained family is absent, and the rest keep their
    # names.
    prf = PO.port_opt_view(prior(CrossSectionalFactorPrior(; factors = factors,
                                                           families = ["industry" =>
                                                                           nothing]), rdp),
                           idx)
    @test !isnothing(prf.rr.fcb)
    fb = PO.panel_field(PO.asset_panel(RegressionPanel(), prf, rdr, rdr.X), "loadings")
    @test size(fb.vals, 2) == size(prf.rr.L, 2) < size(prf.rr.M, 2)
    @test fb.labels == PO.cs_diagnostic_factor_names(prf.rr)
    @test length(fb.labels) == size(prf.rr.L, 2)
    @test fb.labels ⊆ prf.rr.nf

    # The routed path: the optimiser reduces the prior before the kernel runs, so the
    # producer never meets the unreduced block, and no weight lands outside the mask.
    opt = HierarchicalOptimiser(; pe = CrossSectionalFactorPrior(; factors = factors),
                                cle = ClustersEstimator(;
                                                        de = FeatureDistance(;
                                                                             ape = RegressionPanel())))
    res = optimise(HierarchicalRiskParity(; opt = opt), rdp)
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(sum(res.w), 1)
    @test all(iszero, res.w[.!msk])
    @test feature_labels(opt.cle.de, res.pr, rdr, rdr.X) ==
          ["loadings" => n for n in prr.rr.nf]
end
@testset "PhylogenyPanel grades a graph, and its labels are the assets" begin
    rng = StableRNG(20260907)
    T, N = 90, 6
    X = randn(rng, T, N) / 100
    nx = ["A", "B", "C", "D", "E", "F"]
    rd = ReturnsResult(; nx = nx, X = X)

    for alg in (Proximity(), Proximity(; decay = NoDecay())),
        pl in (NetworkEstimator(), ClustersEstimator())

        ape = PhylogenyPanel(; pl = pl, alg = alg)
        pnl = PortfolioOptimisers.asset_panel(ape, nothing, rd, X)
        @test PortfolioOptimisers.panel_is_static(pnl)
        @test length(pnl.pf) == 1
        f = PortfolioOptimisers.panel_field(pnl, "proximity")
        @test f.axis == "asset"
        @test f.labels == nx
        @test size(f.vals) == (N, N)
        @test f.vals == phylogeny_features(alg, pl, X)
        # The one producer whose trailing axis *is* the asset axis.
        @test PortfolioOptimisers.features_are_assets(f, nx)
    end

    # It reads no prior, so it runs at a pre-prior site.
    @test isa(PortfolioOptimisers.asset_panel(PhylogenyPanel(), nothing, rd, X), AssetPanel)
    # With no data carrier the labels are positional.
    @test PortfolioOptimisers.panel_field(PortfolioOptimisers.asset_panel(PhylogenyPanel(),
                                                                          nothing, nothing,
                                                                          X),
                                          "proximity").labels == [string(k) for k in 1:N]

    # `pl` is bound by `NwE_ClE`: both source kinds, both estimators. A precomputed result of
    # either kind is refused by the type -- an Estimator does not hold a Result.
    @test isa(PhylogenyPanel(; pl = ClustersEstimator()), PhylogenyPanel)
    @test_throws TypeError PhylogenyPanel(; pl = clusterise(ClustersEstimator(), X))
    @test_throws TypeError PhylogenyPanel(; pl = PhylogenyResult(; X = zeros(N, N)))
end

@testset "A producer is configuration: the view passes it through and `factory` reaches it" begin
    rng = StableRNG(20260908)
    X = randn(rng, 60, 5) / 100
    de = FeatureDistance(; ape = PhylogenyPanel())

    # Nothing views the distance's producer: it refits on the subproblem's own data.
    v = PortfolioOptimisers.port_opt_view(de, [1, 3])
    @test v.ape === de.ape

    # `factory` reaches into it, because the producer holds estimators of its own.
    @test isa(factory(de, prior(EmpiricalPrior(), X)).ape, PhylogenyPanel)
end

@testset "The kernel resolves the panel through the producer" begin
    rng = StableRNG(20260909)
    T, N, K = 80, 5, 3
    X = randn(rng, T, N) / 100
    F = randn(rng, T, K) / 100
    nx = ["A", "B", "C", "D", "E"]
    rd = ReturnsResult(; nx = nx, X = X, nf = ["mkt", "size", "value"], F = F)
    pr = prior(FactorPrior(), rd)

    de = FeatureDistance(; ape = RegressionPanel())
    Z = feature_matrix(de, pr, rd, X)
    @test Z == pr.rr.L
    @test feature_labels(de, pr, rd, X) ==
          ["loadings" => "mkt", "loadings" => "size", "loadings" => "value"]

    # The label vector is itself a selector that rebuilds the same matrix.
    lbl = feature_labels(de, pr, rd, X)
    @test feature_matrix(FeatureDistance(; ape = RegressionPanel(), sel = lbl), pr, rd,
                         X) == Z

    # And the distance is the metric applied to that matrix.
    S, D = PortfolioOptimisers.cor_and_dist(de, nothing, X; pr = pr, rd = rd)
    @test D == distance(FeatureDistance(), Z)
    @test size(S) == (N, N)

    # A selector cuts the produced panel by label.
    des = FeatureDistance(; ape = RegressionPanel(), sel = ["loadings" => "mkt"])
    @test size(feature_matrix(des, pr, rd, X)) == (N, 1)
    @test feature_matrix(des, pr, rd, X) == reshape(pr.rr.L[:, 1], N, 1)
end
