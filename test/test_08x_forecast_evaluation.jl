#=
Check `src/08_Moments/45_ReturnForecasts/07_ForecastEvaluation.jl` and
`src/08_Moments/45_ReturnForecasts/08_ForecastHistory.jl` against the contract their
docstrings state, and against the reference implementation the map of issue #931 ports.
Issues #934 and #935.

FOUR CONVENTIONS SHAPE THE PROBES.

1. THE TWO OBSERVATION AXES ARE RECONCILED BY THE TARGET, NOT BY THE EVALUATION. A Return
   Forecast history lives on the factor-model block's rows, and the block is a suffix of the
   carrier. `forecast_target_history` cuts the asset returns and the Panel Fields down to
   that suffix, and the idiosyncratic returns already live there, so the probes build a
   carrier that is strictly longer than the block and assert the cut.

2. THE EVALUATION DATES ARE ROW INDICES, NOT TIMESTAMPS. They index `alpha` and `y`, they
   start at the first observation some asset carries a finite pair at, they end at the last
   one, and they stride by `step`. An unscorable observation between the two bounds is kept,
   so the stride means the same thing everywhere in the sample.

3. THESE TICKETS SHIP THE PAIRING, NOT THE STATISTICS. The level-2 verbs are later tickets,
   so the perfect-forecast probe scores the pairing with `cs_spearman_correlation`, which map
   #643 already ships, rather than with a verb this file does not yet carry.

4. THE REFIT LOOP IS PROVED BY AN IDENTITY, AND ITS SIGNAL BY A PLANTED ONE. Refitting a
   member at each observation of the block must reproduce, exactly, the history a member that
   computes one publishes; `evaluation_fixture` carries no relation between the Descriptors
   and the idiosyncratic returns, so it proves the loop and nothing else. Under
   `planted = true` the same fixture drives the idiosyncratic return off the composite
   score, so a refitted `TargetReturnForecast` has something real to find and its
   information coefficient is a measurement rather than noise. Both are asserted, because a
   positive coefficient means nothing without the fixture that reports none.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# The synthetic panel of issue #656, with a factor-model block fitted on a strict suffix of
# the carrier so every probe of the cut has something to cut.
#
# `planted` drives the idiosyncratic return off the composite score instead of off a
# sinusoid, so a member refitted at each date has a relation to find. Convention 4.
function evaluation_fixture(; n_observations::Integer = 60, drop::Integer = 8,
                            planted::Bool = false)
    sp = synthetic_asset_panel(; n_assets = 20, n_observations = n_observations,
                               n_industries = 4, late_listing_proba = 0.3,
                               delisting_proba = 0.3, missing_ratio = 0.08,
                               rng = StableRNG(987654321))
    rd = sp.rd
    pnl = rd.pnl
    T, N = size(pnl.amsk)
    rows = (drop + 1):T
    Tb = length(rows)
    ct_out = CrossSectionalWinsoriser()
    ct_sco = CrossSectionalStandardiser(; min_group_size = 2)
    xc = CompositeExposure(;
                           descriptors = [Passthrough(; field = "book_equity"),
                                          Passthrough(; field = "market_cap")],
                           weights = [0.4, 0.6], min_coverage = 0.5, outlier = ct_out,
                           scoring = ct_sco, group = "industry", bw = "market_cap")
    Lo = factor_exposure(OneHotExposure(; field = "industry", family = "industry"), rd)
    K = 1 + size(Lo, 3)
    Ms = Array{Float64, 3}(undef, T, N, K)
    Z = factor_exposure(xc, rd)
    Ms[:, :, 1] = Z
    for k in 1:size(Lo, 3)
        Ms[:, :, k + 1] = Lo[:, :, k]
    end
    nf = ["style"; ["ind$k" for k in 1:size(Lo, 3)]]
    fam = ["style"; fill("industry", size(Lo, 3))]
    vs = [pnl.amsk[t, i] ? 0.0004 * (1.5 + sin(0.3 * t + 0.7 * i)) : NaN
          for t in rows, i in 1:N]
    rng = StableRNG(24680)
    eps = if planted
        [if pnl.amsk[t, i] && isfinite(Z[t, i])
             0.01 * Z[t, i] + 0.003 * randn(rng)
         else
             NaN
         end
         for t in rows, i in 1:N]
    else
        [if pnl.amsk[t, i]
             0.01 * sin(0.7 * t + 0.29 * i) + 0.004 * cos(0.11 * t * i)
         else
             NaN
         end
         for t in rows, i in 1:N]
    end
    csr = CrossSectionalRegression(; f = zeros(Tb, K), eps = eps, n = fill(N, Tb))
    csfm = CrossSectionalFactorModel(; M = Ms[end, :, :], b = zeros(N), csr = csr,
                                     Ms = Ms[rows, :, :], vs = vs, nf = nf, fam = fam)
    scores = DescriptorScores(;
                              descriptors = [Passthrough(; field = "book_equity"),
                                             Passthrough(; field = "market_cap")],
                              outlier = ct_out, scoring = ct_sco, group = "industry")
    return (; rd = rd, csfm = csfm, scores = scores, rows = rows, T = T, N = N, Tb = Tb)
end

@testset "The forward target family reads the history each member names" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm, rows = fx.rd, fx.csfm, fx.rows

    @testset "The idiosyncratic member reads the block's residuals" begin
        X = PO.forecast_target_history(IdiosyncraticTarget(), rd, csfm)
        @test X === csfm.csr.eps
        @test size(X) == (fx.Tb, fx.N)
    end

    @testset "The asset member reads the carrier's returns, cut to the block" begin
        X = PO.forecast_target_history(AssetReturnTarget(), rd, csfm)
        @test size(X) == (fx.Tb, fx.N)
        @test isequal(X, rd.X[rows, :])
    end

    @testset "The Panel Field member reads the named field, cut to the block" begin
        X = PO.forecast_target_history(PanelFieldTarget(; name = "market_cap"), rd, csfm)
        @test size(X) == (fx.Tb, fx.N)
        @test isequal(X, PO.panel_field_values(rd, "market_cap")[rows, :])
    end

    @testset "A Panel Field the panel does not hold is refused" begin
        @test_throws KeyError PO.forecast_target_history(PanelFieldTarget(;
                                                                          name = "not_a_field"),
                                                         rd, csfm)
    end

    @testset "A Panel Field target with no name is refused at construction" begin
        @test_throws PO.IsEmptyError PanelFieldTarget(; name = "")
    end

    @testset "A block with no cross-sectional fit refuses the idiosyncratic target" begin
        bare = CrossSectionalFactorModel(; M = reshape(fill(1.0, fx.N), fx.N, 1),
                                         b = zeros(fx.N))
        @test_throws PO.IsNothingError PO.forecast_target_history(IdiosyncraticTarget(), rd,
                                                                  bare)
    end
end

@testset "The evaluation dates bound the sample and stride by step" begin
    PO = PortfolioOptimisers
    alpha = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0; 11.0 12.0]
    y = [NaN NaN; 1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; NaN NaN]

    @testset "A stride of one takes every observation between the bounds" begin
        @test PO.forecast_evaluation_dates(alpha, y, 1) == [2, 3, 4, 5]
    end

    @testset "A stride of the horizon takes non-overlapping windows" begin
        @test PO.forecast_evaluation_dates(alpha, y, 2) == [2, 4]
        @test PO.forecast_evaluation_dates(alpha, y, 3) == [2, 5]
    end

    @testset "An unscorable observation inside the bounds is kept" begin
        gap = [NaN NaN; 1.0 2.0; NaN NaN; 5.0 6.0; 7.0 8.0; NaN NaN]
        @test PO.forecast_evaluation_dates(alpha, gap, 1) == [2, 3, 4, 5]
    end

    @testset "An asset enters only when both of its values are finite" begin
        one = [NaN 2.0; 3.0 NaN; NaN NaN]
        two = [1.0 NaN; NaN 4.0; NaN NaN]
        @test_throws PO.IsEmptyError PO.forecast_evaluation_dates(one, two, 1)
    end

    @testset "A sample with no scorable observation is refused" begin
        @test_throws PO.IsEmptyError PO.forecast_evaluation_dates(alpha, fill(NaN, 6, 2), 1)
    end
end

@testset "The bare layer pairs two matrices and carries its parameters" begin
    PO = PortfolioOptimisers
    alpha = [1.0 2.0; 2.0 1.0; 3.0 4.0; 4.0 3.0]
    y = PO.forward_mean_returns(alpha, 1, 1)

    @testset "The pairing is carried unchanged, and step defaults to the horizon" begin
        fe = forecast_evaluation(alpha, y; horizon = 2, lag = 1)
        @test fe.alpha === alpha
        @test fe.y === y
        @test fe.step == 2
        @test fe.horizon == 2
        @test fe.lag == 1
        @test fe.min_count == 3
        @test fe.ppy == 1
        @test isa(fe.target, IdiosyncraticTarget)
        @test isa(fe, ForecastEvaluationResult)
    end

    @testset "Every parameter is carried as given" begin
        tgt = PanelFieldTarget(; name = "market_cap")
        fe = forecast_evaluation(alpha, y; target = tgt, horizon = 3, lag = 2, step = 1,
                                 min_count = 5, ppy = 252)
        @test fe.target === tgt
        @test fe.horizon == 3
        @test fe.lag == 2
        @test fe.step == 1
        @test fe.min_count == 5
        @test fe.ppy == 252
        @test fe.dates == [1, 2, 3]
    end

    @testset "An empty forecast and a mismatched target are refused" begin
        @test_throws PO.IsEmptyError forecast_evaluation(Matrix{Float64}(undef, 0, 0),
                                                         Matrix{Float64}(undef, 0, 0))
        @test_throws DimensionMismatch forecast_evaluation(alpha, y[:, 1:1])
        @test_throws DimensionMismatch forecast_evaluation(alpha, y[1:3, :])
    end

    @testset "Every parameter outside its domain is refused" begin
        @test_throws DomainError forecast_evaluation(alpha, y; horizon = 0)
        @test_throws DomainError forecast_evaluation(alpha, y; lag = -1)
        @test_throws DomainError forecast_evaluation(alpha, y; step = 0)
        @test_throws DomainError forecast_evaluation(alpha, y; min_count = 0)
        @test_throws DomainError forecast_evaluation(alpha, y; ppy = 0)
    end
end

@testset "The Result layer scores a fitted member against its target" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm, scores, rows = fx.rd, fx.csfm, fx.scores, fx.rows
    rf = return_forecast(ExpWeightedReturnForecast(; scores = scores, half_life = 10.0,
                                                   min_obs = 1, horizon = 2, lag = 1), rd,
                         csfm)

    @testset "The forecast history and the forward target are the pairing" begin
        fe = forecast_evaluation(rf, rd, csfm; horizon = 2, lag = 1)
        @test fe.alpha === rf.hist
        @test isequal(fe.y, PO.forward_mean_returns(csfm.csr.eps, 2, 1))
        @test fe.step == 2
        @test size(fe.alpha) == (fx.Tb, fx.N)
    end

    @testset "The two layers agree on the same pair" begin
        fe = forecast_evaluation(rf, rd, csfm; horizon = 2, lag = 1, step = 1, ppy = 252)
        bare = forecast_evaluation(rf.hist, PO.forward_mean_returns(csfm.csr.eps, 2, 1);
                                   horizon = 2, lag = 1, step = 1, ppy = 252)
        @test fe.dates == bare.dates
        @test isequal(fe.alpha, bare.alpha)
        @test isequal(fe.y, bare.y)
    end

    @testset "The dates match under a stride of the horizon and a stride of one" begin
        wide = forecast_evaluation(rf, rd, csfm; horizon = 3, lag = 1)
        thin = forecast_evaluation(rf, rd, csfm; horizon = 3, lag = 1, step = 1)
        @test wide.dates == thin.dates[1:3:end]
        @test thin.dates == first(thin.dates):last(thin.dates)
        @test length(wide.dates) == length(1:3:length(thin.dates))
    end

    @testset "The target member changes the pairing and nothing else" begin
        idio = forecast_evaluation(rf, rd, csfm; horizon = 2, lag = 1)
        asset = forecast_evaluation(rf, rd, csfm; target = AssetReturnTarget(), horizon = 2,
                                    lag = 1)
        @test isequal(idio.alpha, asset.alpha)
        @test !isequal(idio.y, asset.y)
        @test isequal(asset.y, PO.forward_mean_returns(rd.X[rows, :], 2, 1))
    end

    @testset "A perfect forecast scores a rank correlation of one at every date" begin
        fe = forecast_evaluation(rf, rd, csfm; horizon = 2, lag = 1, step = 1)
        perfect = forecast_evaluation(fe.y, fe.y; horizon = 2, lag = 1, step = 1)
        ic = [PO.cs_spearman_correlation(view(perfect.alpha, t, :), view(perfect.y, t, :))
              for t in perfect.dates]
        @test count(isfinite, ic) > 0
        @test all(x -> x ≈ 1, filter(isfinite, ic))
    end

    @testset "A member that carries no history is refused, naming the member" begin
        stated = return_forecast(CustomValueReturnForecast(; mu = fill(0.01, fx.N)), rd,
                                 csfm)
        @test isnothing(stated.hist)
        @test_throws PO.IsNothingError forecast_evaluation(stated, rd, csfm)
        tgt = return_forecast(TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                                   calibrate = false), rd, csfm)
        @test isnothing(tgt.hist)
        @test_throws PO.IsNothingError forecast_evaluation(tgt, rd, csfm)
    end

    @testset "A block the forecast was not fitted on is refused" begin
        short = evaluation_fixture(; n_observations = 40, drop = 4)
        @test_throws DimensionMismatch forecast_evaluation(rf, short.rd, short.csfm)
    end
end

@testset "A member's history is published, or refit along the evaluation grid" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm, scores, rows = fx.rd, fx.csfm, fx.scores, fx.rows
    fw = FixedWeightedReturnForecast(; scores = scores, scale = 1.0, weights = [0.4, 0.6])
    ew = ExpWeightedReturnForecast(; scores = scores, horizon = 2, lag = 1, scale = 1.0,
                                   min_obs = 3)

    @testset "The refit loop reproduces the history a member publishes" begin
        for rfe in (fw, ew)
            rf = return_forecast(rfe, rd, csfm)
            refit = PO.forecast_history_refit(rfe, rd, csfm, rf.mu, 1)
            @test size(refit) == size(rf.hist)
            @test isequal(refit, rf.hist)
        end
    end

    @testset "A member that publishes a history is handed it, not refitted" begin
        for rfe in (fw, ew)
            @test isequal(forecast_history(rfe, rd, csfm),
                          return_forecast(rfe, rd, csfm).hist)
            @test isequal(forecast_history(rfe, rd, csfm; step = 4),
                          forecast_history(rfe, rd, csfm))
        end
    end

    @testset "The block prefix cuts the observation axis and keeps the asset axis" begin
        tb = 20
        cut = PO.forecast_history_block(csfm, tb)
        @test size(cut.Ms) == (tb, fx.N, size(csfm.Ms, 3))
        @test isequal(cut.Ms, csfm.Ms[1:tb, :, :])
        @test isequal(cut.M, csfm.Ms[tb, :, :])
        @test isequal(cut.csr.eps, csfm.csr.eps[1:tb, :])
        @test size(cut.csr.f, 1) == tb
        @test length(cut.csr.n) == tb
        @test isequal(cut.vs, csfm.vs[1:tb, :])
        @test cut.b == csfm.b
        @test cut.nf == csfm.nf
        @test cut.fam == csfm.fam
        @test isnothing(getfield(cut, :L))
        @test isnothing(cut.fcb)
        @test isnothing(cut.rf)
    end

    @testset "A block that carries no history is cut to itself" begin
        bare = CrossSectionalFactorModel(; M = reshape(fill(1.0, fx.N), fx.N, 1),
                                         b = zeros(fx.N))
        cut = PO.forecast_history_block(bare, 3)
        @test cut.M == bare.M
        @test cut.b == bare.b
        @test isnothing(cut.csr)
        @test isnothing(cut.Ms)
        @test isnothing(cut.vs)
        @test isnothing(cut.rw)
        @test isnothing(cut.bw)
    end

    @testset "The per-observation intercept of a fit is cut with its residuals" begin
        csr = CrossSectionalRegression(; f = reshape(collect(1.0:6.0), 6, 1),
                                       eps = reshape(collect(1.0:12.0), 6, 2),
                                       n = fill(2, 6), b = collect(1.0:6.0))
        with = CrossSectionalFactorModel(; M = reshape([1.0, 1.0], 2, 1), b = [0.0, 0.0],
                                         csr = csr)
        cut = PO.forecast_history_block(with, 4)
        @test cut.csr.b == collect(1.0:4.0)
        @test size(cut.csr.eps) == (4, 2)
    end

    @testset "The prefix at the last observation is the whole block" begin
        whole = PO.forecast_history_block(csfm, fx.Tb)
        @test isequal(return_forecast(fw, rd, whole).mu, return_forecast(fw, rd, csfm).mu)
    end

    @testset "A member that publishes none is refitted, and its last row is its fit" begin
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        @test isnothing(return_forecast(tgt, rd, csfm).hist)
        hist = forecast_history(tgt, rd, csfm)
        @test size(hist) == (fx.Tb, fx.N)
        @test count(t -> any(isfinite, view(hist, t, :)), 1:fx.Tb) == fx.Tb
        @test isequal(view(hist, fx.Tb, :), return_forecast(tgt, rd, csfm).mu)
    end

    @testset "The refit grid is anchored at the first observation and strides by step" begin
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        hist = forecast_history(tgt, rd, csfm; step = 5)
        grid = 1:5:fx.Tb
        @test all(t -> any(isfinite, view(hist, t, :)), grid)
        @test all(t -> all(isnan, view(hist, t, :)), setdiff(1:fx.Tb, grid))
    end

    @testset "A refit reads nothing after the observation it answers for" begin
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        hist = forecast_history(tgt, rd, csfm)
        tb = 30
        short = return_forecast(tgt, PO.port_opt_view(rd, 1:rows[tb], :),
                                PO.forecast_history_block(csfm, tb))
        @test isequal(view(hist, tb, :), short.mu)
    end

    @testset "A stated forecast has no history at any observation but its own" begin
        cv = CustomValueReturnForecast(; mu = fill(0.01, fx.N))
        @test_throws PO.ConflictingArgumentError forecast_history(cv, rd, csfm)
        err = try
            forecast_history(cv, rd, csfm)
        catch e
            e
        end
        @test occursin("CustomValueReturnForecast", err.msg)
    end

    @testset "A stride below one is refused" begin
        @test_throws DomainError forecast_history(fw, rd, csfm; step = 0)
        @test_throws DomainError forecast_history(fw, rd, csfm; step = -2)
    end
end

@testset "The Estimator layer evaluates every member the family ships" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm, scores = fx.rd, fx.csfm, fx.scores
    fw = FixedWeightedReturnForecast(; scores = scores, scale = 1.0, weights = [0.4, 0.6])

    @testset "The Estimator and the Result layers agree on a member that publishes one" begin
        rf = return_forecast(fw, rd, csfm)
        est = forecast_evaluation(fw, rd, csfm; horizon = 2, lag = 1, ppy = 252)
        res = forecast_evaluation(rf, rd, csfm; horizon = 2, lag = 1, ppy = 252)
        @test est.dates == res.dates
        @test isequal(est.alpha, res.alpha)
        @test isequal(est.y, res.y)
        @test est.horizon == res.horizon
        @test est.lag == res.lag
        @test est.step == res.step
        @test est.ppy == res.ppy
    end

    @testset "A member that publishes no history is evaluable through the Estimator" begin
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        @test_throws PO.IsNothingError forecast_evaluation(return_forecast(tgt, rd, csfm),
                                                           rd, csfm)
        fe = forecast_evaluation(tgt, rd, csfm; horizon = 2, lag = 1, step = 1)
        @test size(fe.alpha) == (fx.Tb, fx.N)
        @test !isempty(fe.dates)
        @test all(t -> any(isfinite, view(fe.alpha, t, :)), fe.dates)
    end

    @testset "Every evaluation date is an observation the member was refit at" begin
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        fe = forecast_evaluation(tgt, rd, csfm; horizon = 3, lag = 1)
        @test fe.step == 3
        @test issubset(fe.dates, 1:3:fx.Tb)
        @test all(t -> any(isfinite, view(fe.alpha, t, :)), fe.dates)
    end

    @testset "A planted relation is recovered out of sample" begin
        px = evaluation_fixture(; planted = true)
        tgt = TargetReturnForecast(; scores = px.scores, horizon = 2, lag = 1,
                                   calibrate = false)
        fe = forecast_evaluation(tgt, px.rd, px.csfm; horizon = 2, lag = 1, step = 1)
        ic = [PO.cs_spearman_correlation(view(fe.alpha, t, :), view(fe.y, t, :))
              for t in fe.dates]
        ic = filter(isfinite, ic)
        @test length(ic) == length(fe.dates)
        @test sum(ic) / length(ic) > 0.5

        wide = forecast_evaluation(tgt, px.rd, px.csfm; horizon = 2, lag = 1, step = 3)
        icw = [PO.cs_spearman_correlation(view(wide.alpha, t, :), view(wide.y, t, :))
               for t in wide.dates]
        icw = filter(isfinite, icw)
        @test length(icw) == length(wide.dates)
        @test sum(icw) / length(icw) > 0.5
    end

    @testset "The unplanted fixture finds nothing, which is what proves the planted one" begin
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        fe = forecast_evaluation(tgt, rd, csfm; horizon = 2, lag = 1, step = 1)
        ic = [PO.cs_spearman_correlation(view(fe.alpha, t, :), view(fe.y, t, :))
              for t in fe.dates]
        ic = filter(isfinite, ic)
        @test abs(sum(ic) / length(ic)) < 0.2
    end
end
