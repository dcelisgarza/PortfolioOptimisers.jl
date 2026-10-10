#=
The row key of a walk-forward attribution through a Pipeline, over a FactorPrior, and after a
capped refit (#1495, map #1375, ADR 0195).

`factor_attribution(pred::MultiPeriodPredictionResult, pr)` matches each fold to the block by a
row key (#1493). Three routes recorded no key on one side, or a wrong one:

  - A Pipeline fold called the whole-sample `predict`, so it recorded no positions. It now records
    `test_idx` when every fitted step keeps the observations, which a step states with
    `keeps_observations`. A step that makes no promise records nothing, and a step that promises
    and drops a row is refused.
  - A `Regression` recorded no key, so a walk-forward over a `FactorPrior` kept the tail rule and
    refused a series shorter than the block. It now records the positions and the timestamps of the
    observations of its prior result. A Scenario Cap of the factor prior refused to fit before,
    because the asset side kept every row and the factor side the last ones.
  - A refit over a capped buffer recorded the positions inside the buffer, `2:120` for data rows
    `132:250`, so an attribution matched the wrong rows with no error. It now records no positions.
=#
using Dates, Statistics
include(joinpath(@__DIR__, "parity_harness.jl"))

# A returns-level step that drops the first observation of each window. It makes no promise.
struct RowKeyDropFirst <: PortfolioOptimisers.AbstractReturnsPreprocessingEstimator end
struct RowKeyDropFirstResult <: PortfolioOptimisers.AbstractReturnsPreprocessingResult end
PortfolioOptimisers.fit_preprocessing(::RowKeyDropFirst, ::Any) = RowKeyDropFirstResult()
function PortfolioOptimisers.apply_preprocessing(::RowKeyDropFirstResult,
                                                 rd::PortfolioOptimisers.AbstractReturnsResult)
    return PortfolioOptimisers.port_opt_view(rd, 2:size(rd.X, 1), :)
end
# The same step with a wrong promise.
struct RowKeyLyingDrop <: PortfolioOptimisers.AbstractReturnsPreprocessingEstimator end
struct RowKeyLyingDropResult <: PortfolioOptimisers.AbstractReturnsPreprocessingResult end
PortfolioOptimisers.fit_preprocessing(::RowKeyLyingDrop, ::Any) = RowKeyLyingDropResult()
function PortfolioOptimisers.apply_preprocessing(::RowKeyLyingDropResult,
                                                 rd::PortfolioOptimisers.AbstractReturnsResult)
    return PortfolioOptimisers.port_opt_view(rd, 2:size(rd.X, 1), :)
end
PortfolioOptimisers.keeps_observations(::RowKeyLyingDropResult) = true

@testset "The row key of a walk-forward attribution, by route (#1495)" begin
    PO = PortfolioOptimisers
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    est() = CrossSectionalFactorPrior(; lambda = 1, factors = factors)
    rd = parity_large_panel().rd
    wf = IndexWalkForward(120, 40)
    tests = [(121 + 40 * (k - 1)):(160 + 40 * (k - 1)) for k in 1:3]
    pcs = prior(est(), rd)
    function same_attribution(a, b)
        return all(f -> isequal(getfield(a.sys, f), getfield(b.sys, f)) &&
                        isequal(getfield(a.idio, f), getfield(b.idio, f)) &&
                        isequal(getfield(a.total, f), getfield(b.total, f)),
                   (:vol, :vol_contrib, :mu_contrib)) &&
               isequal(a.fbd.vol_contrib, b.fbd.vol_contrib)
    end

    @testset "A Pipeline fold records its rows when every step keeps them" begin
        pipe = Pipeline(; steps = ("prior" => est(), "opt" => InverseVolatility()))
        pcv = cross_val_predict(pipe, rd, wf)
        @test [p.idx for p in pcv.pred] == tests
        # The data carries no timestamps, so the fold matches by position, and the attribution
        # equals the one of the direct route exactly.
        @test isnothing(rd.ts)
        dcv = cross_val_predict(InverseVolatility(; pe = est()), rd, wf)
        @test same_attribution(factor_attribution(pcv, pcs), factor_attribution(dcv, pcs))
        # A step that makes no promise records no rows, so the tail rule refuses the short series.
        drop = Pipeline(;
                        steps = ("drop" => RowKeyDropFirst(), "prior" => est(),
                                 "opt" => InverseVolatility()))
        dpcv = cross_val_predict(drop, rd, wf)
        @test all(p -> isnothing(p.idx), dpcv.pred)
        @test_throws DimensionMismatch factor_attribution(dpcv, pcs)
        # A step that promises and drops a row is refused, not matched to the wrong rows.
        lie = Pipeline(;
                       steps = ("drop" => RowKeyLyingDrop(), "prior" => est(),
                                "opt" => InverseVolatility()))
        @test_throws DimensionMismatch cross_val_predict(lie, rd, wf)
    end

    @testset "A fitted step states whether it keeps the observations, at its own level" begin
        @test !PO.keeps_observations(RowKeyDropFirstResult())
        @test !PO.keeps_observations(PricesToReturns())
        @test PO.keeps_observations(PO.AssetSelectorResult(rd.nx))
        # A step of the other level is an identity on the window, so it keeps the observations.
        @test PO.fitted_step_keeps_observations(PricesToReturns(), rd)
        @test PO.fitted_step_keeps_observations(pcs, rd)
        @test !PO.fitted_step_keeps_observations(RowKeyDropFirstResult(), rd)
        # A whole-sample prediction records no rows.
        pred = predict(optimise(InverseVolatility(; pe = est()), rd), rd)
        @test PO.pipeline_fold_prediction(pred, :, true) === pred
    end

    # A FactorPrior on returns data with factor returns, with and without timestamps.
    rng = StableRNG(1495)
    T, N, K = 250, 10, 3
    F = 0.01 .* randn(rng, T, K)
    X = F * (0.5 .+ rand(rng, K, N)) .+ 0.005 .* randn(rng, T, N)
    ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
    nx = ["a$i" for i in 1:N]
    nf = ["f$i" for i in 1:K]
    rdf = ReturnsResult(; nx = nx, X = X, nf = nf, F = F)
    rdt = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, ts = ts)

    @testset "A Regression records the observations of its prior result" begin
        pf = prior(FactorPrior(), rdf)
        @test pf.rr.idx == 1:T && isnothing(pf.rr.ts)
        pft = prior(FactorPrior(), rdt)
        @test pft.rr.idx == 1:T && pft.rr.ts == ts
        @test PO.attribution_row_key(pft.rr) == (; idx = 1:T, ts = ts)
        # The view by asset, the expansion to the full universe and the residual block keep it.
        @test PO.port_opt_view(pft.rr, [1, 3]).ts == ts
        @test PO.set_idiosyncratic_covariance(pft.rr, nothing, nothing, nothing).idx == 1:T
        # The walk-forward attribution answers, and its systematic part is w' M f of each test
        # row. Measured: a difference of 2.0e-18 on a volatility of 0.0116.
        cv = cross_val_predict(InverseVolatility(; pe = FactorPrior()), rdf, wf)
        fa = factor_attribution(cv, pf)
        W, r = PO.attribution_prediction_history(cv)
        M = pf.rr.M
        sys = [dot(W[t, :], M * F[120 + t, :]) for t in axes(W, 1)]
        @test fa.sys.vol ≈ std(sys) rtol = 1e-14
        @test fa.total.vol ≈ std(r) rtol = 1e-14
        # The match by timestamp equals the match by position.
        cvt = cross_val_predict(InverseVolatility(; pe = FactorPrior()), rdt, wf)
        @test same_attribution(factor_attribution(cvt, pft), fa)
        # A key whose positions do not increase, or whose two parts differ in length, is refused.
        @test_throws ArgumentError Regression(; M = M, idx = [2, 1])
        @test_throws DimensionMismatch Regression(; M = M, idx = 1:3, ts = ts[1:2])
        @test_throws DimensionMismatch prior(FactorPrior(), X, F; ts = ts[1:10])
    end

    @testset "A Scenario Cap of the factor prior keeps the last rows on both sides" begin
        pft = prior(FactorPrior(), rdt)
        pc = prior(FactorPrior(; pe = EmpiricalPrior(; max_scenarios = 50)), rdt)
        @test size(pc.X) == (50, N) && size(pc.o_X) == (50, N) && size(pc.fpr.X) == (50, K)
        @test pc.rr.idx == 201:T && pc.rr.ts == ts[201:T]
        # The cut changes no moment, and the rows are the last rows of the uncapped fit.
        @test pc.mu == pft.mu && pc.sigma == pft.sigma
        @test pc.X == pft.X[201:T, :] && pc.o_X == X[201:T, :]
    end

    # The systematic volatility of each path against w' M f over the data rows its folds name.
    function fp_path_gap(mp, pr)
        fa = factor_attribution(mp, pr)
        W, _ = PO.attribution_prediction_history(mp)
        rows = reduce(vcat, p.idx for p in mp.pred)
        sys = [dot(W[k, :], pr.rr.M * F[rows[k], :]) for k in eachindex(rows)]
        return abs(fa.sys.vol - std(sys)) / std(sys)
    end
    paths(x::MultiPeriodPredictionResult) = [x]
    paths(x::PopulationPredictionResult) = x.pred
    hs = HindsightSplit(; start = 121)
    ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)

    @testset "Every scheme through a Pipeline equals the direct route, $(nm)" for (nm,
                                                                                   cv) in
                                                                                  ("KFold" =>
                                                                                       KFold(;
                                                                                             n = 3),
                                                                                   "Hindsight" =>
                                                                                       hs,
                                                                                   "Combinatorial" =>
                                                                                       ccv,
                                                                                   "Combinatorial split" =>
                                                                                       split(ccv,
                                                                                             rd))
        pipe = Pipeline(; steps = ("prior" => est(), "opt" => InverseVolatility()))
        pp = cross_val_predict(pipe, rd, cv)
        dp = cross_val_predict(InverseVolatility(; pe = est()), rd, cv)
        for (a, b) in zip(paths(pp), paths(dp))
            @test [p.idx for p in a.pred] == [p.idx for p in b.pred]
            # Measured: a difference of exactly zero.
            @test same_attribution(factor_attribution(a, pcs), factor_attribution(b, pcs))
        end
    end

    @testset "A randomised Pipeline path against a prior on its own assets" begin
        pe5() = CrossSectionalFactorPrior(; lambda = 1, factors = factors, minra = 5)
        mr = MultipleRandomised(IndexWalkForward(60, 30); n_subsets = 2, subset_size = 8,
                                window_size = 150, seed = 1)
        pipe = Pipeline(; steps = ("prior" => pe5(), "opt" => InverseVolatility()))
        # The split of the scheme takes the same route (#1496). It fell to the single-path
        # method, which joined the paths into one series over every asset.
        for cv in (mr, split(mr, rd))
            pp = cross_val_predict(pipe, rd, cv)
            dp = cross_val_predict(InverseVolatility(; pe = pe5()), rd, cv)
            @test pp isa PopulationPredictionResult && length(pp.pred) == 2
            for (a, b) in zip(pp.pred, dp.pred)
                c = [findfirst(==(n), rd.nx) for n in a.pred[1].rd.nx]
                @test length(c) == 8 && all(p -> p.rd.nx == a.pred[1].rd.nx, a.pred)
                prs = prior(pe5(), PO.port_opt_view(rd, :, c))
                @test [p.idx for p in a.pred] == [p.idx for p in b.pred]
                @test same_attribution(factor_attribution(a, prs),
                                       factor_attribution(b, prs))
            end
        end
    end

    @testset "An online Pipeline equals the expanding walk-forward" begin
        pipe = Pipeline(; steps = ("prior" => est(), "opt" => InverseVolatility()))
        pp = cross_val_predict(Online(pipe), rd, OnlineIndexWalkForward(120, 40))
        dp = cross_val_predict(InverseVolatility(; pe = est()), rd,
                               IndexWalkForward(120, 40; expand_train = true))
        @test [p.idx for p in pp.pred] == tests
        @test same_attribution(factor_attribution(pp, pcs), factor_attribution(dp, pcs))
    end

    @testset "A FactorPrior block under every scheme, $(nm)" for (nm, cv, pe) in
                                                                 (("KFold", KFold(; n = 3),
                                                                   FactorPrior()),
                                                                  ("Hindsight", hs,
                                                                   FactorPrior()),
                                                                  ("Combinatorial", ccv,
                                                                   FactorPrior()),
                                                                  ("OnlineWalkForward",
                                                                   OnlineIndexWalkForward(120,
                                                                                          40),
                                                                   Online(FactorPrior())))
        pf = prior(FactorPrior(), rdf)
        mp = cross_val_predict(InverseVolatility(; pe = pe), rdf, cv)
        # Measured: a relative difference below 1e-15 on each path.
        @test all(p -> fp_path_gap(p, pf) <= 1e-14, paths(mp))
    end

    @testset "A Pipeline over prices matches each fold by timestamp" begin
        P = 100 .* cumprod(vcat(ones(1, N), 1 .+ X); dims = 1)
        FP = 100 .* cumprod(vcat(ones(1, K), 1 .+ F); dims = 1)
        tsp = Date(2020, 1, 1) .+ Day.(0:T)
        px = PricesResult(; X = TimeArray(tsp, P, Symbol.(nx)),
                          F = TimeArray(tsp, FP, Symbol.(nf)))
        rdp = prices_to_returns(px)
        prp = prior(FactorPrior(), rdp)
        @test prp.rr.ts == tsp[2:end]
        pipe = Pipeline(;
                        steps = ("p2r" => PricesToReturns(),
                                 "opt" => InverseVolatility(; pe = FactorPrior())))
        pcv = cross_val_predict(pipe, px, wf)
        # The price folds record no positions, and the first price row of each test window
        # anchors its returns, so a fold holds 39 returns of its 40 price rows.
        @test all(p -> isnothing(p.idx) && length(p.rd.ts) == 39, pcv.pred)
        fa = factor_attribution(pcv, prp)
        W, r = PO.attribution_prediction_history(pcv)
        rows = [findfirst(==(t), prp.rr.ts) for t in pcv.mrd.ts]
        sys = [dot(W[k, :], prp.rr.M * rdp.F[rows[k], :]) for k in eachindex(rows)]
        @test fa.sys.vol ≈ std(sys) rtol = 1e-14
        @test fa.total.vol ≈ std(r) rtol = 1e-14
    end

    @testset "A refit over a capped buffer records no positions" begin
        rows(r, i) = PO.port_opt_view(r, i, :)
        online(e; kwargs...) = PO.update_online_estimator(Online(e; kwargs...))
        function stepped(e, data)
            for (a, b) in ((1, 90), (91, 170), (171, 250))
                e = partial_fit!(e, rows(data, a:b))
            end
            return prior(e)
        end
        # With no cap the buffer holds every folded row, so the positions are right.
        @test stepped(online(est()), rd).rr.idx == pcs.rr.idx
        # A capped buffer kept rows 131 to 250 and counted them from 1.
        @test isnothing(stepped(online(est(); max_history = 120), rd).rr.idx)
        @test isnothing(stepped(online(FactorPrior(); max_history = 120), rdf).rr.idx)
        @test stepped(online(FactorPrior()), rdf).rr.idx == 1:T
    end
end
