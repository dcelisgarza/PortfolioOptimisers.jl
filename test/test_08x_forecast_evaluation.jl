#=
Check `src/08_Moments/45_ReturnForecasts/07_ForecastEvaluation.jl`,
`src/08_Moments/45_ReturnForecasts/08_ForecastHistory.jl` and
`src/08_Moments/45_ReturnForecasts/09_ForecastInformationCoefficient.jl`,
`src/08_Moments/45_ReturnForecasts/10_ForecastPortfolios.jl`,
`src/08_Moments/45_ReturnForecasts/11_ForecastFactorCorrelation.jl`,
`src/08_Moments/45_ReturnForecasts/12_ForecastForwardWindows.jl` and
`src/08_Moments/45_ReturnForecasts/13_ForecastCalibration.jl` against the contract their
docstrings state, and against the reference implementation the map of issue #931 ports.
Issues #934, #935, #936, #937, #938, #939 and #940.

TEN CONVENTIONS SHAPE THE PROBES.

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

5. THE STATISTICS ARE ORACLED BY RUNNING THE REFERENCE, NOT BY READING IT. `IC_ALPHA` and
   its gapped variant were put through the reference's own diagnostic and its correlation
   summary, and the literals below are what it answered. The one place the port diverges is
   the hit rate: the library reads it against every date, so a date with no coefficient is a
   miss, where the reference reads it against the dates that carried one. That is
   `exposure_ic_factor_summary`'s convention, which both summaries now share, and it is
   asserted as a divergence rather than papered over.

6. THE TWO ALPHA PORTFOLIOS ARE PINNED BY THEIR INVARIANTS, NOT BY A STORED NUMBER. Both
   are centred and scaled to 200 % gross, so every date holds one unit long and one unit
   short and nets to zero whatever the forecast is. A perfect forecast -- `alpha` set to
   `y` itself -- must earn a positive mean, and its negation must earn exactly the
   opposite, which pins the sign convention without a fixture that carries signal.

7. THE PORTFOLIO SUMMARY IS OF THE COMPRESSED PATH, AND THAT IS ASSERTED RATHER THAN
   AVOIDED. A date below `min_count` has no portfolio return, and `performance_summary`
   states that its series must be finite. The gaps are dropped before the call, so
   `max_drawdown` and `calmar` read a path that joins the date before a gap to the date
   after it. The probes assert the compressed answer and, beside it, what the uncompressed
   series would have answered, which is what makes the caveat legible. The hit rate beside
   that summary therefore counts against the FINITE dates, which is the opposite of
   convention 5's denominator: the two summaries are computed on different series, and each
   docstring states which.

8. THE FACTOR CORRELATION IS PINNED BY A DESIGN, NOT BY A LITERAL. It is contemporaneous,
   so there is nothing forward-looking to oracle: the probes build a cross-section whose
   correlation is exactly 1 and exactly 0 by construction, under BOTH the weighted form and
   the rank form, so the two are separable without a stored number. `FC_ORTHO` is the
   permutation whose centred values and whose ordinal ranks are both orthogonal to
   `[1, 2, 3, 4]`, which an arbitrary orthogonal vector would not be -- a tied vector such
   as `[1, -1, -1, 1]` is Pearson-orthogonal and reads 0.4 under ordinal ranks. The
   `ExposureNeutralisation` probe is the one that reads a real fixture, and it is
   DIRECTIONAL rather than near-zero: issue #950 records that a Neutralisation fits a
   cross-sectional regression with no intercept, so its residual is orthogonal to the
   target in the UNCENTRED sense and keeps a large Pearson correlation with it. The probe
   asserts the un-neutralised correlation above 0.9, the neutralised one below it and
   still above 0.5, and it is tightened when #950 is settled.

9. A FORWARD-WINDOW TABLE IS PINNED ROW BY ROW AGAINST THE REFERENCE, AND ITS DATE RULE IS
   PINNED SEPARATELY. `WINDOW_ALPHA` was put through the reference's own holding-period and
   decay diagnostics and the twenty-two literals below are what it answered, to every digit
   it printed. The date rule is the one thing those literals cannot pin, because the
   reference and the port agree on it: every row of a table is read on the dates every
   window of the grid can be scored at, so `n` sets the sample as well as the depth. The
   ticket asked for the opposite -- that shortening `n` leave the rows that remain -- and
   that is FALSE in general and asserted as false. It holds only when the base evaluation
   already stops before the deepest window matures, and that case is asserted beside it.

10. THE CALIBRATION IS POOLED OVER PAIRS, SO IT READS NO THRESHOLD. Every other statistic
    here is a statistic of a cross-section and refuses one that carries fewer than
    `min_count` assets. The slope, the curve and the pooled moments read the pairs of every
    evaluation date as one sample, so a thin cross-section contributes few pairs rather
    than an unreliable number: there is nothing to threshold, and the probes assert that
    raising `min_count` past the universe moves none of the four answers. The slope is also
    the one reading of a forecast that a rescaling moves, and that is asserted beside the
    two readings it does not move.
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

# The oracle of the information coefficients, measured by running the reference
# implementation on the same two matrices. `IC_ALPHA` is a forecast whose ordering of the
# four assets is good at the first date, mixed at the second and wrong at the third, and
# whose *spacing* is uneven, so the rank column and the level column disagree — which is the
# whole reason both are answered. Issue #936.
const IC_ALPHA = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 2.0 3.0; 3.0 1.0 2.0 6.0]
const IC_W = [1.0 1.0 1.0 1.0; 1.0 2.0 3.0 4.0; 4.0 3.0 2.0 1.0; 1.0 1.0 1.0 1.0]
const IC_REF_SPEARMAN = [1.0, 0.4, -0.4]
const IC_REF_PEARSON = [0.9404839524466372, 0.10090550403524001, -0.2710523708715754]
const IC_REF_PEARSON_W = [0.9404839524466372, 0.0521748308303865, -0.49733055872774606]
const IC_REF_GAP_SPEARMAN = [1.0, 1.0, 0.5]
const IC_REF_GAP_PEARSON = [1.0, 1.0, 0.720576692122892]
const IC_REF_GAP_COVERAGE = [0.75, 0.5, 0.75]

function ic_gap_fixture()
    alpha = copy(IC_ALPHA)
    alpha[2, 4] = NaN
    alpha[3, 2] = NaN
    return alpha
end

@testset "The information coefficients reproduce the reference implementation" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)

    @testset "Both columns are answered, and both match the reference" begin
        ic = forecast_ic(fe)
        @test size(ic) == (length(fe.dates), 2)
        @test ic[:, 1] ≈ IC_REF_SPEARMAN
        @test ic[:, 2] ≈ IC_REF_PEARSON
        # The two columns disagree, which is why the verb answers both rather than one.
        @test !isapprox(ic[:, 1], ic[:, 2])
    end

    @testset "Each column is the cross-sectional helper applied to the pairing" begin
        ic = forecast_ic(fe)
        for (j, t) in enumerate(fe.dates)
            a = view(fe.alpha, t, :)
            b = view(fe.y, t, :)
            @test ic[j, 1] == PO.cs_spearman_correlation(a, b; min_count = fe.min_count)
            @test ic[j, 2] == PO.cs_weighted_correlation(a, b, ones(size(IC_ALPHA, 2));
                                                         min_count = fe.min_count)
        end
    end

    @testset "A weighting moves the Pearson column and leaves the Spearman one" begin
        ic = forecast_ic(fe, IC_W)
        @test ic[:, 1] ≈ IC_REF_SPEARMAN
        @test ic[:, 2] ≈ IC_REF_PEARSON_W
        @test !isapprox(ic[:, 2], IC_REF_PEARSON)
    end

    @testset "The block method reads the weight history the metric names" begin
        # `IdentityMetric` resolves to no history, so the block method and the bare method
        # with no weights are the same call.
        fx = evaluation_fixture()
        rd, csfm = fx.rd, fx.csfm
        fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                         weights = [0.4, 0.6])
        fb = forecast_evaluation(fw, rd, csfm; horizon = 2, lag = 1)
        @test isequal(forecast_ic(fb, csfm), forecast_ic(fb))
        # A block whose regression weights vary over the assets moves the Pearson column.
        icr = forecast_ic(fb, csfm; weighting = InverseIdiosyncraticVarianceMetric())
        @test isequal(icr[:, 1], forecast_ic(fb)[:, 1])
        @test !isequal(icr[:, 2], forecast_ic(fb)[:, 2])
        # A metric naming a history the block does not carry refuses by name.
        @test_throws PO.IsNothingError forecast_ic(fb, csfm;
                                                   weighting = BenchmarkWeightMetric())
    end

    @testset "A gap in the panel moves both columns, as it does in the reference" begin
        gap = ic_gap_fixture()
        fg = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 2)
        ic = forecast_ic(fg)
        @test ic[:, 1] ≈ IC_REF_GAP_SPEARMAN
        @test ic[:, 2] ≈ IC_REF_GAP_PEARSON
    end
end

@testset "The summary names its two series and reports a t-statistic" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    ic = forecast_ic(forecast_evaluation(IC_ALPHA, y))
    s = forecast_ic_summary(ic)

    @testset "The five figures of each series match the reference" begin
        @test keys(s) == (:spearman, :pearson)
        @test keys(s.spearman) == (:mean_ic, :std_ic, :ic_ir, :t_stat, :hit_rate)
        @test s.spearman.mean_ic ≈ 0.3333333333333333
        @test s.spearman.std_ic ≈ 0.7023769168568493
        @test s.spearman.ic_ir ≈ 0.4745789978762494
        @test s.spearman.t_stat ≈ 0.8219949365267862
        @test s.spearman.hit_rate ≈ 0.6666666666666666
        @test s.pearson.mean_ic ≈ 0.2567790285367673
        @test s.pearson.std_ic ≈ 0.6206266852224849
        @test s.pearson.ic_ir ≈ 0.4137415207093715
        @test s.pearson.t_stat ≈ 0.7166213350694421
        @test s.pearson.hit_rate ≈ 0.6666666666666666
    end

    @testset "The t-statistic is the mean over the standard error of the mean" begin
        for k in 1:2
            v = filter(isfinite, ic[:, k])
            n = length(v)
            mu = sum(v) / n
            sd = sqrt(sum(x -> (x - mu)^2, v) / (n - 1))
            sk = getfield(s, k == 1 ? :spearman : :pearson)
            @test sk.t_stat ≈ mu / (sd / sqrt(n))
            @test sk.t_stat ≈ sk.ic_ir * sqrt(n)
        end
    end

    @testset "The two series are named, and the naming is what the columns lack" begin
        @test s.spearman == PO.exposure_ic_factor_summary(ic, 1)
        @test s.pearson == PO.exposure_ic_factor_summary(ic, 2)
    end

    @testset "The summary refuses a series it cannot name" begin
        @test_throws PO.IsEmptyError forecast_ic_summary(Matrix{Float64}(undef, 0, 0))
        @test_throws DimensionMismatch forecast_ic_summary(ones(3, 1))
        @test_throws DimensionMismatch forecast_ic_summary(ones(3, 3))
    end
end

@testset "A date under the threshold carries no coefficient, and counts as a miss" begin
    PO = PortfolioOptimisers
    # A forecast and its target are paired one observation apart, so a row of gaps thins two
    # dates: the one it is the forecast of, and the one it is the target of. Here that is the
    # middle pair, which carries two finite pairs against four at the ends, so a threshold of
    # three silences the middle two dates and leaves the ends.
    alpha = [1.0 2.0 4.0 8.0; 2.0 3.0 5.0 40.0; 1.0 5.0 NaN NaN; 3.0 1.0 2.0 6.0;
             2.0 4.0 1.0 3.0]
    y = PO.forward_mean_returns(alpha, 1, 1)
    fe = forecast_evaluation(alpha, y; min_count = 3)

    @testset "The silenced dates are NaN in both columns" begin
        ic = forecast_ic(fe)
        @test fe.dates == [1, 2, 3, 4]
        @test all(isnan, ic[2, :])
        @test all(isnan, ic[3, :])
        @test all(isfinite, ic[1, :])
        @test all(isfinite, ic[4, :])
    end

    @testset "The threshold is re-parameterisable without re-pairing" begin
        loose = forecast_ic(fe; min_count = 2)
        @test all(isfinite, loose[2, :])
        @test all(isfinite, loose[3, :])
        @test isequal(forecast_ic(fe; min_count = 3), forecast_ic(fe))
        @test_throws DomainError forecast_ic(fe; min_count = 0)
    end

    @testset "The hit rate counts a NaN as a miss, and the t-statistic drops it" begin
        # The library's convention, which `exposure_ic_factor_summary` already holds: the
        # hit rate is read against every date and the other four against the dates that
        # carried a score. The reference divides its hit rate by the finite count instead, so
        # it would report 1/2 where this reports 1/4.
        ic = forecast_ic(fe)
        s = forecast_ic_summary(ic)
        v = filter(isfinite, ic[:, 1])
        @test length(v) == 2
        @test count(>(0), v) == 1
        @test s.spearman.hit_rate == 1 / 4
        @test s.spearman.hit_rate == count(>(0), v) / length(fe.dates)
        @test s.spearman.t_stat ≈ s.spearman.ic_ir * sqrt(2)
    end

    @testset "A threshold no date reaches gives no score and a hit rate of zero" begin
        none = forecast_ic(fe; min_count = 5)
        @test all(isnan, none)
        s = forecast_ic_summary(none)
        @test isnan(s.spearman.mean_ic)
        @test isnan(s.spearman.std_ic)
        @test isnan(s.spearman.ic_ir)
        @test isnan(s.spearman.t_stat)
        @test s.spearman.hit_rate == 0
    end
end

@testset "The coverage says what share of the universe was scored" begin
    PO = PortfolioOptimisers
    gap = ic_gap_fixture()
    fg = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 2)

    @testset "It matches the reference, one entry per evaluation date" begin
        c = forecast_coverage(fg)
        @test length(c) == length(fg.dates)
        @test c ≈ IC_REF_GAP_COVERAGE
    end

    @testset "It counts the assets carrying a finite pair over the universe" begin
        c = forecast_coverage(fg)
        for (j, t) in enumerate(fg.dates)
            n = count(i -> isfinite(fg.alpha[t, i]) && isfinite(fg.y[t, i]),
                      axes(fg.alpha, 2))
            @test c[j] == n / size(fg.alpha, 2)
        end
    end

    @testset "The threshold is deliberately not applied to the coverage" begin
        # The coverage is what says why a date carries no coefficient, so it answers where
        # the coefficient does not.
        tight = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 4)
        @test all(isnan, forecast_ic(tight)[2, :])
        @test isequal(forecast_coverage(tight), forecast_coverage(fg))
        @test all(isfinite, forecast_coverage(tight))
    end

    @testset "A weight history is the universe, and an empty one has nothing to cover" begin
        # The first date's universe is empty, so it carries `NaN` rather than a share. The
        # other two narrow the denominator to the assets of positive weight, and the numerator
        # to the ones of those that carry a finite pair.
        u = [0.0 0.0 0.0 0.0; 1.0 1.0 0.0 0.0; 1.0 1.0 1.0 0.0; 1.0 1.0 1.0 1.0]
        c = forecast_coverage(fg, u)
        @test isnan(c[1])
        @test c[2] == 1 / 2
        @test c[3] == 2 / 3
        for (j, t) in enumerate(fg.dates)
            n = count(i -> u[t, i] > 0, axes(fg.alpha, 2))
            k = count(i -> u[t, i] > 0 && isfinite(fg.alpha[t, i]) && isfinite(fg.y[t, i]),
                      axes(fg.alpha, 2))
            @test isequal(c[j], n > 0 ? k / n : NaN)
        end
    end

    @testset "The block method reads the weight history the metric names" begin
        fx = evaluation_fixture()
        fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                         weights = [0.4, 0.6])
        fb = forecast_evaluation(fw, fx.rd, fx.csfm; horizon = 2, lag = 1)
        @test isequal(forecast_coverage(fb, fx.csfm), forecast_coverage(fb))
        cv = forecast_coverage(fb, fx.csfm;
                               weighting = InverseIdiosyncraticVarianceMetric())
        @test all(x -> isnan(x) || 0 <= x <= 1, cv)
        @test length(cv) == length(fb.dates)
    end

    @testset "A weight history that does not fit the forecast is refused" begin
        @test_throws DimensionMismatch forecast_coverage(fg, ones(2, 4))
        @test_throws DimensionMismatch forecast_ic(fg, ones(4, 3))
        @test_throws DomainError forecast_ic(fg, fill(-1.0, 4, 4))
        @test_throws DomainError forecast_coverage(fg, fill(-1.0, 4, 4))
        @test PO.forecast_ic_weights(gap, nothing) == ones(4, 4)
        @test PO.forecast_ic_weights(gap, IC_W) === IC_W
    end
end

@testset "The exposure summary gained the same t-statistic" begin
    PO = PortfolioOptimisers
    # One statistic, one kernel: `exposure_ic_factor_summary` is what both summaries read,
    # so the exposure diagnostics report a t-statistic on the same terms rather than
    # diverging from the evaluation.
    ic = [0.1 0.4; -0.2 NaN; 0.3 0.5; NaN 0.1]
    s = PO.exposure_ic_summary(ic)
    @test keys(s) == (:mean_ic, :std_ic, :ic_ir, :t_stat, :hit_rate)
    for k in 1:2
        m = PO.exposure_ic_factor_summary(ic, k)
        v = filter(isfinite, ic[:, k])
        @test s.t_stat[k] ≈ m.ic_ir * sqrt(length(v))
        @test m.t_stat ≈ m.ic_ir * sqrt(length(v))
    end
    @test length(s.t_stat) == 2
end

@testset "A planted forecast scores, and an unplanted one does not" begin
    PO = PortfolioOptimisers
    tgt_of(fx) = TargetReturnForecast(; scores = fx.scores, horizon = 2, lag = 1,
                                      calibrate = false)

    @testset "The planted fixture reports a coefficient the summary believes" begin
        px = evaluation_fixture(; planted = true)
        fe = forecast_evaluation(tgt_of(px), px.rd, px.csfm; horizon = 2, lag = 1, step = 1)
        s = forecast_ic_summary(forecast_ic(fe))
        @test s.spearman.mean_ic > 0.5
        @test s.spearman.t_stat > 20
        @test s.spearman.hit_rate > 0.9
        @test s.pearson.mean_ic > 0.5
        # The panel lists and delists assets and drops 8% of its cells, so the universe is
        # never wholly scored and the coverage is what says so.
        cv = forecast_coverage(fe)
        @test all(x -> 0.4 < x <= 1, cv)
        @test sum(cv) / length(cv) > 0.75
        @test any(x -> x < 1, cv)
    end

    @testset "The unplanted fixture reports none, which is what proves the planted one" begin
        fx = evaluation_fixture()
        fe = forecast_evaluation(tgt_of(fx), fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1)
        s = forecast_ic_summary(forecast_ic(fe))
        @test abs(s.spearman.mean_ic) < 0.2
        @test abs(s.spearman.t_stat) < 5
        @test s.spearman.hit_rate < 0.5
    end
end

@testset "The alpha portfolios hold the forecast and nothing else" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm, scores = fx.rd, fx.csfm, fx.scores
    fw = FixedWeightedReturnForecast(; scores = scores, scale = 1.0, weights = [0.4, 0.6])
    fe = forecast_evaluation(fw, rd, csfm; horizon = 2, lag = 1, step = 1)

    @testset "Both kinds are dollar neutral at 200 % gross" begin
        for kind in (:rank, :zscore)
            p = forecast_portfolio(fe; kind = kind)
            @test size(p.w) == (length(fe.dates), fx.N)
            gross = [sum(abs, view(p.w, k, :)) for k in axes(p.w, 1)]
            net = [sum(view(p.w, k, :)) for k in axes(p.w, 1)]
            @test all(g -> isapprox(g, 2) || iszero(g), gross)
            @test any(g -> isapprox(g, 2), gross)
            @test maximum(abs, net) < 1e-12
        end
    end

    @testset "A `:zscore` weight is the centred forecast, rescaled" begin
        p = forecast_portfolio(fe; kind = :zscore)
        k = 1
        t = fe.dates[k]
        v = [isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]) for i in 1:(fx.N)]
        a = [v[i] ? fe.alpha[t, i] : 0.0 for i in 1:(fx.N)]
        c = [v[i] ? a[i] - sum(a) / count(v) : 0.0 for i in 1:(fx.N)]
        @test view(p.w, k, :) ≈ 2 * c / sum(abs, c)
    end

    @testset "A `:rank` weight reads the order and not the level" begin
        # Pushing the date's largest forecast far further out leaves it largest, so every
        # ordinal rank is unchanged and the rank book is too. The z-score book is not,
        # because it reads the level. That is the whole difference between the two kinds.
        alpha = copy(Matrix(fe.alpha))
        t = fe.dates[1]
        v = [isfinite(alpha[t, j]) && isfinite(fe.y[t, j]) for j in 1:(fx.N)]
        i = argmax([v[j] ? alpha[t, j] : -Inf for j in 1:(fx.N)])
        alpha[t, i] += 1e3 * (abs(alpha[t, i]) + 1)
        blown = forecast_evaluation(alpha, fe.y; step = fe.step, min_count = fe.min_count)
        @test view(forecast_portfolio(blown; kind = :rank).w, 1, :) ≈
              view(forecast_portfolio(fe; kind = :rank).w, 1, :)
        @test !isapprox(view(forecast_portfolio(blown; kind = :zscore).w, 1, :),
                        view(forecast_portfolio(fe; kind = :zscore).w, 1, :))
    end

    @testset "The portfolio return is the contraction of the weights with the target" begin
        p = forecast_portfolio(fe; kind = :rank)
        ref = map(enumerate(fe.dates)) do (k, t)
            yr = [isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]) ? fe.y[t, i] : 0.0
                  for i in 1:(fx.N)]
            return only(calc_net_returns(view(p.w, k, :), reshape(yr, 1, :)))
        end
        fin = isfinite.(p.ret)
        @test all(fin)
        @test p.ret[fin] ≈ ref[fin]
    end

    @testset "`min_count` gates the return and the turnover together" begin
        p = forecast_portfolio(fe; kind = :rank)
        @test isnan(p.turnover[1])
        @test all(k -> isfinite(p.turnover[k]) == isfinite(p.ret[k]), 2:length(p.ret))
        @test p.turnover[2:end] ≈ PO.calc_turnover(p.w)[2:end]
        @test p.hit_rate ≈
              count(x -> x > 0, filter(isfinite, p.ret)) / count(isfinite, p.ret)
        @test p.mean_turnover ≈
              sum(filter(isfinite, p.turnover)) / count(isfinite, p.turnover)
    end

    @testset "A cross-section no date can fill leaves nothing to summarise" begin
        # `performance_summary` needs one finite return, and the refusal is its own rather
        # than a guard this verb adds. The spread answers `NaN` instead, because it
        # summarises each column and never builds a path.
        hi = forecast_evaluation(fe.alpha, fe.y; min_count = fx.N + 1)
        @test_throws ArgumentError forecast_portfolio(hi)
        @test all(isnan, forecast_quantile_spread(hi).spread)
        @test all(isnan, forecast_quantile_spread(hi).ann_mean)
    end

    @testset "`kind` takes two values and refuses every other" begin
        @test_throws PO.ConflictingArgumentError forecast_portfolio(fe; kind = :equal)
        @test_throws PO.ConflictingArgumentError PO.forecast_portfolio_weights(fe.alpha,
                                                                               fe.y,
                                                                               fe.dates,
                                                                               :inverse_vol)
    end
end

@testset "A perfect forecast earns, and its negation loses exactly as much" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    fe = forecast_evaluation(fw, fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1)
    perfect = forecast_evaluation(fe.y, fe.y; min_count = 3)
    inverted = forecast_evaluation(-fe.y, fe.y; min_count = 3)
    mean_finite(x) = sum(filter(isfinite, x)) / count(isfinite, x)

    @testset "The rank portfolio of a perfect forecast is positive" begin
        pp = forecast_portfolio(perfect; kind = :rank)
        pi_ = forecast_portfolio(inverted; kind = :rank)
        @test mean_finite(pp.ret) > 0
        @test mean_finite(pp.ret) ≈ -mean_finite(pi_.ret)
        @test pp.summary.ann_return > 0
        @test pp.hit_rate == 1
    end

    @testset "The z-score portfolio of a perfect forecast is positive" begin
        pz = forecast_portfolio(perfect; kind = :zscore)
        @test mean_finite(pz.ret) > 0
        @test pz.hit_rate == 1
    end

    @testset "The quantile spread of a perfect forecast is positive at every quantile" begin
        q = forecast_quantile_spread(perfect; quantiles = (0.1, 0.25, 0.5))
        @test all(>(0), q.ann_mean)
        @test all(q.hit_rate .== 1)
        qi = forecast_quantile_spread(inverted; quantiles = (0.1, 0.25, 0.5))
        @test all(<(0), qi.ann_mean)
        @test q.ann_mean ≈ -qi.ann_mean
    end
end

@testset "`ppy` annualises the summaries and leaves the turnover alone" begin
    fx = evaluation_fixture()
    fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    fe1 = forecast_evaluation(fw, fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1, ppy = 1)
    fe4 = forecast_evaluation(fw, fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1, ppy = 4)

    @testset "The portfolio summary scales and the turnover does not" begin
        p1 = forecast_portfolio(fe1)
        p4 = forecast_portfolio(fe4)
        @test p4.summary.periods_per_year == 4
        @test p4.summary.ann_return ≈ 4 * p1.summary.ann_return
        @test p4.summary.ann_volatility ≈ 2 * p1.summary.ann_volatility
        @test p4.summary.sharpe ≈ 2 * p1.summary.sharpe
        @test p4.mean_turnover ≈ p1.mean_turnover
        @test p4.hit_rate == p1.hit_rate
        @test p4.w == p1.w
        @test isequal(p4.ret, p1.ret)
    end

    @testset "The quantile summary scales the same way" begin
        q1 = forecast_quantile_spread(fe1; quantiles = (0.1, 0.3))
        q4 = forecast_quantile_spread(fe4; quantiles = (0.1, 0.3))
        @test size(q1.spread) == (length(fe1.dates), 2)
        @test isequal(q4.spread, q1.spread)
        @test q4.ann_mean ≈ 4 * q1.ann_mean
        @test q4.ann_vol ≈ 2 * q1.ann_vol
        @test q4.ann_ir ≈ 2 * q1.ann_ir
        @test q4.hit_rate == q1.hit_rate
    end
end

@testset "The quantile spread cuts two tails and refuses an impossible cut" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    fe = forecast_evaluation(fw, fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1)

    @testset "The spread is the top mean less the bottom mean" begin
        q = forecast_quantile_spread(fe; quantiles = (0.2,))
        k = 1
        t = fe.dates[k]
        v = [isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]) for i in 1:(fx.N)]
        av = [fe.alpha[t, i] for i in 1:(fx.N) if v[i]]
        yv = [fe.y[t, i] for i in 1:(fx.N) if v[i]]
        lo = quantile(av, 0.2)
        hi = quantile(av, 0.8)
        top = [yv[i] for i in eachindex(av) if av[i] >= hi]
        bot = [yv[i] for i in eachindex(av) if av[i] <= lo]
        @test q.spread[k, 1] ≈ sum(top) / length(top) - sum(bot) / length(bot)
    end

    @testset "A half-quantile splits the cross-section in two" begin
        q = forecast_quantile_spread(fe; quantiles = (0.5,))
        k = 1
        t = fe.dates[k]
        v = [isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]) for i in 1:(fx.N)]
        av = [fe.alpha[t, i] for i in 1:(fx.N) if v[i]]
        yv = [fe.y[t, i] for i in 1:(fx.N) if v[i]]
        m = quantile(av, 0.5)
        top = [yv[i] for i in eachindex(av) if av[i] >= m]
        bot = [yv[i] for i in eachindex(av) if av[i] <= m]
        @test q.spread[k, 1] ≈ sum(top) / length(top) - sum(bot) / length(bot)
    end

    @testset "One column per quantile, in the order they were asked for" begin
        q = forecast_quantile_spread(fe; quantiles = (0.1, 0.3, 0.5))
        @test size(q.spread) == (length(fe.dates), 3)
        @test length(q.ann_mean) == length(q.ann_vol) == length(q.ann_ir) == 3
        for (j, p) in enumerate((0.1, 0.3, 0.5))
            @test q.spread[:, j] ≈
                  forecast_quantile_spread(fe; quantiles = (p,)).spread[:, 1]
        end
    end

    @testset "A quantile outside `(0, 0.5]` is refused" begin
        @test_throws PO.IsEmptyError forecast_quantile_spread(fe; quantiles = ())
        @test_throws DomainError forecast_quantile_spread(fe; quantiles = (0.0,))
        @test_throws DomainError forecast_quantile_spread(fe; quantiles = (0.6,))
        @test_throws DomainError forecast_quantile_spread(fe; quantiles = (0.1, NaN))
    end
end

@testset "The portfolio summary is of the compressed path" begin
    PO = PortfolioOptimisers
    # Row 2 carries one asset, so it is below `min_count` and has no portfolio return. The
    # summary is therefore taken over rows 1, 3, 4 and 5 joined end to end.
    alpha = [3.0 1.0 2.0 4.0
             1.0 NaN NaN NaN
             2.0 4.0 1.0 3.0
             4.0 3.0 2.0 1.0
             1.0 2.0 4.0 3.0]
    y = [0.02 -0.01 0.00 0.03
         0.01 0.02 0.03 0.04
         0.05 -0.04 0.06 -0.02
         -0.03 -0.01 0.02 0.04
         -0.01 0.00 0.02 0.01]
    fe = forecast_evaluation(alpha, y; min_count = 3, ppy = 1)
    p = forecast_portfolio(fe; kind = :rank)

    @testset "The gapped date drops out of the series and out of the summary" begin
        @test fe.dates == 1:5
        @test isnan(p.ret[2])
        @test count(isfinite, p.ret) == 4
        @test p.summary.n_periods == 4
        @test p.summary ==
              performance_summary(p.ret[isfinite.(p.ret)]; periods_per_year = 1)
    end

    @testset "The uncompressed series is what the compression avoids" begin
        # `performance_summary`'s Precomputed-returns contract: a `NaN` makes the mean and
        # the drawdown non-finite, and the tail figure answers a number rather than a `NaN`.
        raw = performance_summary(p.ret; periods_per_year = 1)
        @test isnan(raw.ann_return)
        @test isnan(raw.max_drawdown)
        @test isfinite(raw.cvar)
        @test isfinite(p.summary.max_drawdown)
        @test p.summary.max_drawdown < 0
    end

    @testset "The turnover into and out of the gap is dropped with it" begin
        @test isnan(p.turnover[1])
        @test isnan(p.turnover[2])
        @test isfinite(p.turnover[3])
        @test p.mean_turnover ≈
              sum(filter(isfinite, p.turnover)) / count(isfinite, p.turnover)
    end
end

# The forecast-against-exposure design of convention 8. Each row of `FC_ALPHA` is an
# increasing affine function of `[1, 2, 3, 4]`, so it correlates exactly `1` with the first
# exposure under both the weighted and the rank form. The second exposure is the
# permutation `[3, 1, 4, 2]`, whose centred values AND whose ordinal ranks are both
# orthogonal to `[1, 2, 3, 4]`, so it correlates exactly `0` under both forms and the two
# answers are separable. Issue #940.
const FC_ALPHA = [1.0 2.0 3.0 4.0
                  2.0 4.0 6.0 8.0
                  0.5 1.0 1.5 2.0
                  3.0 4.0 5.0 6.0]
const FC_TIED = repeat([1.0 2.0 3.0 4.0], 4, 1)
const FC_ORTHO = repeat([3.0 1.0 4.0 2.0], 4, 1)
const FC_B = cat(FC_TIED, FC_ORTHO; dims = 3)
# A weight that breaks the orthogonality of `FC_ORTHO`. `IC_W` does not: its rows happen to
# leave the weighted covariance of `[1, 2, 3, 4]` against `[3, 1, 4, 2]` at exactly zero.
const FC_W = repeat([1.0 1.0 1.0 5.0], 4, 1)
# A per-date exposure design. Every row of `FC_TIED` and `FC_ORTHO` is the same, so a
# forecast whose rows are affine in `[1, 2, 3, 4]` correlates identically at every date and
# a summary of it has no dispersion to report. `FC_G` varies by row, so the series it
# produces against `IC_ALPHA` does.
const FC_G = [1.0 0.0 1.0 0.0
              0.0 1.0 0.0 1.0
              1.0 1.0 0.0 0.0
              0.0 0.0 1.0 1.0]
const FC_BG = cat(FC_G, reverse(FC_G; dims = 2); dims = 3)

@testset "The exposure axis is checked against the forecast, once" begin
    PO = PortfolioOptimisers
    fe = forecast_evaluation(FC_ALPHA, PO.forward_mean_returns(FC_ALPHA, 1, 1))

    @testset "The checked history is handed back unchanged" begin
        @test PO.forecast_factor_exposures(fe.alpha, FC_B) === FC_B
    end

    @testset "An empty tensor and a mismatched axis are refused" begin
        @test_throws PO.IsEmptyError PO.forecast_factor_exposures(fe.alpha,
                                                                  Array{Float64, 3}(undef,
                                                                                    0, 0,
                                                                                    0))
        # One observation short, and one asset short.
        @test_throws DimensionMismatch PO.forecast_factor_exposures(fe.alpha,
                                                                    FC_B[1:3, :, :])
        @test_throws DimensionMismatch PO.forecast_factor_exposures(fe.alpha,
                                                                    FC_B[:, 1:3, :])
        @test_throws DimensionMismatch forecast_factor_correlation(fe, FC_B[1:3, :, :])
    end

    @testset "A threshold below one is refused" begin
        @test_throws DomainError forecast_factor_correlation(fe, FC_B; min_count = 0)
    end
end

@testset "A forecast is correlated against the exposures it is meant to add alpha over" begin
    PO = PortfolioOptimisers
    fe = forecast_evaluation(FC_ALPHA, PO.forward_mean_returns(FC_ALPHA, 1, 1))
    c = forecast_factor_correlation(fe, FC_B)

    @testset "A pure multiple of an exposure correlates one with it and zero with an orthogonal one" begin
        @test size(c) == (length(fe.dates), 2)
        @test c[:, 1] ≈ ones(length(fe.dates))
        @test c[:, 2] ≈ zeros(length(fe.dates)) atol = 1e-12
    end

    @testset "The rank form agrees where the design is rank-orthogonal too" begin
        cr = forecast_factor_correlation(fe, FC_B; rank = true)
        @test cr[:, 1] ≈ ones(length(fe.dates))
        @test cr[:, 2] ≈ zeros(length(fe.dates)) atol = 1e-12
    end

    @testset "Each entry is the cross-sectional helper applied to the pairing" begin
        cr = forecast_factor_correlation(fe, FC_B; rank = true)
        u = ones(size(FC_ALPHA, 2))
        for k in 1:2, (j, t) in enumerate(fe.dates)
            a = view(fe.alpha, t, :)
            b = view(FC_B, t, :, k)
            @test c[j, k] == PO.cs_weighted_correlation(a, b, u; min_count = fe.min_count)
            @test cr[j, k] == PO.cs_spearman_correlation(a, b; min_count = fe.min_count)
        end
    end

    @testset "A weighting moves the weighted form and leaves the rank one" begin
        cw = forecast_factor_correlation(fe, FC_B, FC_W)
        # A perfect linear relation is weight invariant, and an orthogonal one is not.
        @test cw[:, 1] ≈ ones(length(fe.dates))
        @test !isapprox(cw[:, 2], zeros(length(fe.dates)); atol = 1e-12)
        @test isequal(forecast_factor_correlation(fe, FC_B, FC_W; rank = true),
                      forecast_factor_correlation(fe, FC_B; rank = true))
    end

    @testset "A cross-section under the threshold carries no correlation" begin
        gap = copy(FC_ALPHA)
        gap[2, 3] = NaN
        gap[2, 4] = NaN
        fg = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 2)
        cg = forecast_factor_correlation(fg, FC_B; min_count = 3)
        j = findfirst(==(2), fg.dates)
        @test all(isnan, cg[j, :])
        @test all(isfinite, forecast_factor_correlation(fg, FC_B; min_count = 2)[j, :])
    end
end

@testset "The block method reads both histories off the block" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm = fx.rd, fx.csfm
    fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    fb = forecast_evaluation(fw, rd, csfm; horizon = 2, lag = 1)

    @testset "The exposure history is the block's own, unlagged" begin
        @test isequal(forecast_factor_correlation(fb, csfm),
                      forecast_factor_correlation(fb, PO.cs_diagnostic_exposures(csfm)))
        @test size(forecast_factor_correlation(fb, csfm), 2) == length(csfm.nf)
    end

    @testset "A weighting moves the weighted form and leaves the rank one" begin
        cw = forecast_factor_correlation(fb, csfm;
                                         weighting = InverseIdiosyncraticVarianceMetric())
        @test !isequal(cw, forecast_factor_correlation(fb, csfm))
        @test isequal(forecast_factor_correlation(fb, csfm; rank = true,
                                                  weighting = InverseIdiosyncraticVarianceMetric()),
                      forecast_factor_correlation(fb, csfm; rank = true))
        # A metric naming a history the block does not carry refuses by name.
        @test_throws PO.IsNothingError forecast_factor_correlation(fb, csfm;
                                                                   weighting = BenchmarkWeightMetric())
    end

    @testset "A block that carries no exposure history refuses by name" begin
        bare = CrossSectionalFactorModel(; M = csfm.M, b = csfm.b, csr = csfm.csr,
                                         nf = csfm.nf, fam = csfm.fam)
        @test_throws PO.IsNothingError forecast_factor_correlation(fb, bare)
    end
end

@testset "A neutralised forecast is less correlated with what it was neutralised against" begin
    fx = evaluation_fixture()
    rd, csfm = fx.rd, fx.csfm
    ds = fx.scores
    # The Descriptors and the style factor are built from the same two Panel Fields, so an
    # un-neutralised forecast restates the style factor rather than adding alpha over it.
    # `neutralise = "style"` is the `ExposureNeutralisation` path, and it is what the
    # correlation must see.
    dn = DescriptorScores(; descriptors = ds.descriptors, neutralise = "style",
                          outlier = ds.outlier, scoring = ds.scoring, group = ds.group)
    raw = FixedWeightedReturnForecast(; scores = ds, scale = 1.0, weights = [0.4, 0.6])
    neu = FixedWeightedReturnForecast(; scores = dn, scale = 1.0, weights = [0.4, 0.6])
    cr = forecast_factor_correlation(forecast_evaluation(raw, rd, csfm; horizon = 2), csfm)
    cn = forecast_factor_correlation(forecast_evaluation(neu, rd, csfm; horizon = 2), csfm)
    mraw = abs(sum(filter(isfinite, view(cr, :, 1))) / count(isfinite, view(cr, :, 1)))
    mneu = abs(sum(filter(isfinite, view(cn, :, 1))) / count(isfinite, view(cn, :, 1)))

    @testset "The un-neutralised forecast restates the style factor" begin
        @test mraw > 0.9
    end

    @testset "The neutralised one is less correlated with it, and not near zero" begin
        # THE ASSERTION IS DIRECTIONAL, AND ISSUE #950 IS WHY. A Neutralisation fits
        # `CrossSectionalLinearRegression()`, whose `intercept` defaults to `false`, so the
        # residual is exactly orthogonal to the target in the UNCENTRED sense and keeps a
        # large Pearson correlation with it. Tighten this to a near-zero assertion when
        # #950 is settled.
        @test mneu < mraw
        @test mneu > 0.5
    end
end

@testset "The correlation series is summarised by the shared kernel" begin
    PO = PortfolioOptimisers
    # `FC_BG` varies by date, so the series has dispersion and the ratio and the
    # t-statistic are numbers rather than the `NaN` a constant series earns.
    fe = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 1, 1))
    c = forecast_factor_correlation(fe, FC_BG)
    s = exposure_ic_summary(c)

    @testset "No third summary ships, and the kernel answers one entry per factor" begin
        @test keys(s) == (:mean_ic, :std_ic, :ic_ir, :t_stat, :hit_rate)
        @test all(v -> length(v) == 2, values(s))
        @test all(isfinite, s.mean_ic)
        @test all(isfinite, s.std_ic)
        @test all(isfinite, s.ic_ir)
        @test all(isfinite, s.t_stat)
        # The two factors are one another's mirror, so their correlations are the negatives
        # of each other and every figure of the summary follows.
        @test s.mean_ic[1] ≈ -s.mean_ic[2]
        @test s.std_ic[1] ≈ s.std_ic[2]
        @test s.ic_ir[1] ≈ -s.ic_ir[2]
        @test all(h -> 0 <= h <= 1, s.hit_rate)
    end

    @testset "It is the same helper the information coefficient summary reads" begin
        for k in 1:2
            m = PO.exposure_ic_factor_summary(c, k)
            @test isequal(m.mean_ic, s.mean_ic[k])
            @test isequal(m.std_ic, s.std_ic[k])
            @test isequal(m.ic_ir, s.ic_ir[k])
            @test isequal(m.t_stat, s.t_stat[k])
            @test isequal(m.hit_rate, s.hit_rate[k])
        end
        # The two-column case is exactly what `forecast_ic_summary` names, so the two
        # summaries agree figure for figure on the same input.
        fs = forecast_ic_summary(c)
        @test isequal(fs.spearman, PO.exposure_ic_factor_summary(c, 1))
        @test isequal(fs.pearson, PO.exposure_ic_factor_summary(c, 2))
    end

    @testset "A constant series is summarised, and its ratio has no answer" begin
        # The `FC_B` design is constant across dates by construction, which is what pins
        # the exact `1` and `0`; a series with no dispersion has no ratio.
        cc = forecast_factor_correlation(forecast_evaluation(FC_ALPHA,
                                                             PO.forward_mean_returns(FC_ALPHA,
                                                                                     1, 1)),
                                         FC_B)
        sc = exposure_ic_summary(cc)
        @test sc.mean_ic ≈ [1.0, 0.0] atol = 1e-12
        @test sc.std_ic ≈ [0.0, 0.0] atol = 1e-12
        @test sc.hit_rate ≈ [1.0, 0.0]
        @test all(isnan, sc.ic_ir)
        @test all(isnan, sc.t_stat)
    end
end

# The 8 x 4 forecast the forward-window tables are oracled on. Every entry is a permutation
# of `1:4` down to a constant column, so the ranks move at every observation and the two
# books differ from each other only where the levels do. Convention 9.
const WINDOW_ALPHA = [1.0 2.0 3.0 4.0; 3.0 1.0 2.0 4.0; 2.0 3.0 1.0 4.0; 1.0 3.0 2.0 4.0;
                      2.0 1.0 3.0 4.0; 4.0 2.0 1.0 3.0; 1.0 4.0 3.0 2.0; 3.0 2.0 4.0 1.0]

@testset "The forward-window grid places the two tables' windows" begin
    PO = PortfolioOptimisers

    @testset "A cumulative grid lengthens the window and holds its start" begin
        @test PO.forecast_window_grid(2, 1, 3, :cumulative) == [(2, 1), (4, 1), (6, 1)]
        @test PO.forecast_window_grid(1, 0, 4, :cumulative) ==
              [(1, 0), (2, 0), (3, 0), (4, 0)]
    end

    @testset "A disjoint grid holds the window and pushes its start out" begin
        @test PO.forecast_window_grid(2, 1, 3, :disjoint) == [(2, 1), (2, 3), (2, 5)]
        @test PO.forecast_window_grid(1, 0, 4, :disjoint) ==
              [(1, 0), (1, 1), (1, 2), (1, 3)]
    end

    @testset "The two grids agree at the first period and nowhere else" begin
        c = PO.forecast_window_grid(3, 2, 5, :cumulative)
        d = PO.forecast_window_grid(3, 2, 5, :disjoint)
        @test c[1] == d[1] == (3, 2)
        @test all(c[p] != d[p] for p in 2:5)
    end

    @testset "A depth below one and an unknown kind are refused" begin
        @test_throws DomainError PO.forecast_window_grid(1, 1, 0, :cumulative)
        @test_throws PO.ConflictingArgumentError PO.forecast_window_grid(1, 1, 2, :rolling)
    end
end

@testset "The common dates are intersected across the whole grid" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    ys = [PO.forward_mean_returns(alpha, h, 1) for h in 1:3]

    @testset "A deeper window shortens the set the whole table is read on" begin
        @test PO.forecast_common_dates(alpha, ys[1:1], 1:7, 3) == collect(1:7)
        @test PO.forecast_common_dates(alpha, ys[1:2], 1:7, 3) == collect(1:6)
        @test PO.forecast_common_dates(alpha, ys, 1:7, 3) == collect(1:5)
    end

    @testset "A grid deeper than the sample leaves no date at all" begin
        deep = [PO.forward_mean_returns(alpha, h, 1) for h in 1:8]
        @test isempty(PO.forecast_common_dates(alpha, deep, 1:7, 3))
    end

    @testset "`min_count` gates a date under every window" begin
        gapped = copy(alpha)
        gapped[3, 2:4] .= NaN
        gys = [PO.forward_mean_returns(gapped, h, 1) for h in 1:2]
        @test 3 ∉ PO.forecast_common_dates(gapped, gys, 1:6, 3)
        @test 3 ∈ PO.forecast_common_dates(gapped, gys, 1:6, 1)
    end

    @testset "A threshold below one is refused" begin
        @test_throws DomainError PO.forecast_common_dates(alpha, ys, 1:7, 0)
    end
end

@testset "The two tables reproduce the reference implementation" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    fe = forecast_evaluation(alpha, PO.forward_mean_returns(alpha, 1, 1); step = 1)
    h = forecast_holding_period(fe, alpha; n = 3)
    d = forecast_decay(fe, alpha; n = 3)

    @testset "Both tables carry fifteen columns and three rows" begin
        @test keys(h) ==
              keys(d) ==
              (:period, :horizon, :lag, :dates, :spearman_mean_ic, :spearman_ic_ir,
               :spearman_t_stat, :pearson_mean_ic, :pearson_ic_ir, :pearson_t_stat,
               :rank_ann_return, :rank_sharpe, :zscore_ann_return, :zscore_sharpe,
               :mean_coverage)
        @test h.period == d.period == [1, 2, 3]
        @test h.horizon == [1, 2, 3]
        @test h.lag == [1, 1, 1]
        @test d.horizon == [1, 1, 1]
        @test d.lag == [1, 2, 3]
        @test h.dates == d.dates == [1, 2, 3, 4, 5]
    end

    @testset "The holding-period table is what the reference answered" begin
        @test isapprox(h.spearman_mean_ic, [0.4, 0.12, 0.28])
        @test isapprox(h.spearman_ic_ir, [1.414214, 0.395628, 0.639010]; rtol = 1e-6)
        @test isapprox(h.spearman_t_stat, [3.162278, 0.884652, 1.428869]; rtol = 1e-6)
        @test isapprox(h.pearson_mean_ic, [0.4, 0.180180, 0.400988]; rtol = 1e-5)
        @test isapprox(h.pearson_ic_ir, [1.414214, 0.371014, 0.605921]; rtol = 1e-6)
        @test isapprox(h.pearson_t_stat, [3.162278, 0.829613, 1.354880]; rtol = 1e-6)
        @test isapprox(h.rank_ann_return, [1.0, 0.55, 0.733333]; rtol = 1e-6)
        @test isapprox(h.rank_sharpe, [1.414214, 0.792825, 0.964764]; rtol = 1e-6)
        @test isapprox(h.zscore_ann_return, h.rank_ann_return)
        @test isapprox(h.zscore_sharpe, h.rank_sharpe)
        @test h.mean_coverage == [1.0, 1.0, 1.0]
    end

    @testset "The decay table is what the reference answered" begin
        @test isapprox(d.spearman_mean_ic, [0.4, 0.04, 0.44])
        @test isapprox(d.spearman_ic_ir, [1.414214, 0.121716, 1.073490]; rtol = 1e-6)
        @test isapprox(d.spearman_t_stat, [3.162278, 0.272166, 2.400397]; rtol = 1e-6)
        @test isapprox(d.pearson_mean_ic, d.spearman_mean_ic)
        @test isapprox(d.pearson_ic_ir, d.spearman_ic_ir)
        @test isapprox(d.pearson_t_stat, d.spearman_t_stat)
        @test isapprox(d.rank_ann_return, [1.0, 0.1, 1.1]; rtol = 1e-6)
        @test isapprox(d.rank_sharpe, [1.414214, 0.121716, 1.073490]; rtol = 1e-6)
        @test isapprox(d.zscore_ann_return, d.rank_ann_return)
        @test isapprox(d.zscore_sharpe, d.rank_sharpe)
        @test d.mean_coverage == [1.0, 1.0, 1.0]
    end
end

@testset "The first row of both tables is the base evaluation, on the common dates" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    y = PO.forward_mean_returns(alpha, 1, 1)
    fe = forecast_evaluation(alpha, y; step = 1)
    h = forecast_holding_period(fe, alpha; n = 3)
    d = forecast_decay(fe, alpha; n = 3)
    fb = PO.ForecastEvaluationResult(alpha, y, h.dates, fe.target, fe.horizon, fe.lag,
                                     fe.step, fe.min_count, fe.ppy)
    ic = forecast_ic_summary(forecast_ic(fb))
    rk = forecast_portfolio(fb; kind = :rank)
    zs = forecast_portfolio(fb; kind = :zscore)

    @testset "The two tables agree at the first period" begin
        for k in (:spearman_mean_ic, :spearman_ic_ir, :spearman_t_stat, :pearson_mean_ic,
                  :pearson_ic_ir, :pearson_t_stat, :rank_ann_return, :rank_sharpe,
                  :zscore_ann_return, :zscore_sharpe, :mean_coverage)
            @test h[k][1] == d[k][1]
        end
    end

    @testset "The first period is the base evaluation restricted to those dates" begin
        @test h.spearman_mean_ic[1] == ic.spearman.mean_ic
        @test h.spearman_ic_ir[1] == ic.spearman.ic_ir
        @test h.spearman_t_stat[1] == ic.spearman.t_stat
        @test h.pearson_mean_ic[1] == ic.pearson.mean_ic
        @test h.rank_ann_return[1] == rk.summary.ann_return
        @test h.rank_sharpe[1] == rk.summary.sharpe
        @test h.zscore_ann_return[1] == zs.summary.ann_return
        @test h.mean_coverage[1] ==
              sum(forecast_coverage(fb)) / length(forecast_coverage(fb))
    end
end

@testset "The depth sets the common dates, so it moves every row it keeps" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA

    @testset "A shallower table is read on more dates, so its rows differ" begin
        fe = forecast_evaluation(alpha, PO.forward_mean_returns(alpha, 1, 1); step = 1)
        t1 = forecast_holding_period(fe, alpha; n = 1)
        t3 = forecast_holding_period(fe, alpha; n = 3)
        @test t1.dates == collect(1:7)
        @test t3.dates == collect(1:5)
        @test t1.spearman_mean_ic[1] != t3.spearman_mean_ic[1]
    end

    @testset "A base evaluation that already stops early is depth invariant" begin
        y = PO.forward_mean_returns(alpha, 1, 1)
        y[6:8, :] .= NaN
        fe = forecast_evaluation(alpha, y; step = 1)
        @test fe.dates == collect(1:5)
        t1 = forecast_holding_period(fe, alpha; n = 1)
        t3 = forecast_holding_period(fe, alpha; n = 3)
        @test t1.dates == t3.dates == collect(1:5)
        @test t1.spearman_mean_ic[1] == t3.spearman_mean_ic[1]
        @test t1.rank_sharpe[1] == t3.rank_sharpe[1]
    end
end

@testset "A perfect forecast scores one at the first decay period" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    X = PO.forecast_target_history(IdiosyncraticTarget(), fx.rd, fx.csfm)
    y = PO.forward_mean_returns(X, 1, 1)
    fe = forecast_evaluation(y, y; step = 1)

    @testset "The first period of both tables is a perfect coefficient" begin
        d = forecast_decay(fe, X; n = 3)
        h = forecast_holding_period(fe, X; n = 3)
        @test isapprox(d.spearman_mean_ic[1], 1)
        @test isapprox(d.pearson_mean_ic[1], 1)
        @test isapprox(h.spearman_mean_ic[1], 1)
    end

    @testset "The later periods score a target the forecast is not aimed at" begin
        d = forecast_decay(fe, X; n = 3)
        @test all(abs(v) < 1 for v in d.spearman_mean_ic[2:3])
        @test all(isfinite, d.rank_ann_return)
    end
end

@testset "The block method builds the target and the weights the bare one takes" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture(; planted = true)
    rd, csfm = fx.rd, fx.csfm
    rfe = TargetReturnForecast(; scores = fx.scores, horizon = 1, lag = 1)
    fe = forecast_evaluation(rfe, rd, csfm; step = 3)
    X = PO.forecast_target_history(fe.target, rd, csfm)

    @testset "The two methods answer the same table" begin
        a = forecast_holding_period(fe, rd, csfm; n = 3)
        b = forecast_holding_period(fe, X, PO.cs_diagnostic_weights(IdentityMetric(), csfm);
                                    n = 3)
        @test a.dates == b.dates
        @test isapprox(a.spearman_mean_ic, b.spearman_mean_ic)
        @test isapprox(a.pearson_mean_ic, b.pearson_mean_ic)
    end

    @testset "A planted fixture finds signal at the first period of both tables" begin
        h = forecast_holding_period(fe, rd, csfm; n = 3)
        d = forecast_decay(fe, rd, csfm; n = 3)
        @test h.spearman_mean_ic[1] > 0.5
        @test d.spearman_mean_ic[1] == h.spearman_mean_ic[1]
        @test all(0 .<= filter(isfinite, h.mean_coverage) .<= 1)
    end
end

@testset "A table refuses what it cannot score and prints what it cannot fill" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    fe = forecast_evaluation(alpha, PO.forward_mean_returns(alpha, 1, 1); step = 1)

    @testset "A target history of the wrong shape is refused" begin
        @test_throws DimensionMismatch forecast_holding_period(fe, alpha[:, 1:3]; n = 2)
        @test_throws DimensionMismatch forecast_decay(fe, alpha[1:4, :]; n = 2)
    end

    @testset "An empty grid is refused" begin
        @test_throws PO.IsEmptyError PO.forecast_window_table(fe, alpha, nothing,
                                                              Tuple{Int, Int}[], 3)
    end

    @testset "A grid deeper than the sample prints a table of NaN" begin
        t = forecast_holding_period(fe, alpha; n = 8)
        @test isempty(t.dates)
        @test t.period == collect(1:8)
        @test all(isnan, t.spearman_mean_ic)
        @test all(isnan, t.rank_sharpe)
        @test all(isnan, t.mean_coverage)
    end

    @testset "`min_count` is overridden without a second pairing" begin
        gapped = copy(alpha)
        gapped[3, 3:4] .= NaN
        fg = forecast_evaluation(gapped, PO.forward_mean_returns(gapped, 1, 1); step = 1)
        @test 3 ∉ forecast_holding_period(fg, gapped; n = 2).dates
        @test 3 ∈ forecast_holding_period(fg, gapped; n = 2, min_count = 2).dates
    end
end

@testset "`ppy` annualises the table's portfolio columns" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    y = PO.forward_mean_returns(alpha, 1, 1)
    t1 = forecast_holding_period(forecast_evaluation(alpha, y; step = 1), alpha; n = 3)
    t4 = forecast_holding_period(forecast_evaluation(alpha, y; step = 1, ppy = 4), alpha;
                                 n = 3)

    @testset "The return scales by `ppy` and the ratio by its square root" begin
        @test isapprox(t4.rank_ann_return, 4 * t1.rank_ann_return)
        @test isapprox(t4.rank_sharpe, sqrt(4) * t1.rank_sharpe)
        @test isapprox(t4.zscore_ann_return, 4 * t1.zscore_ann_return)
    end

    @testset "The coefficients and the coverage do not scale" begin
        @test isapprox(t4.spearman_mean_ic, t1.spearman_mean_ic)
        @test isapprox(t4.pearson_t_stat, t1.pearson_t_stat)
        @test t4.mean_coverage == t1.mean_coverage
    end
end

# The oracle of the calibration, measured by running the reference implementation on
# `IC_ALPHA` and its forward target -- the same two matrices the information coefficients
# are oracled on, so the two sets of literals describe one forecast. `IC_ALPHA` spaces its
# assets very unevenly, which is what makes a scale statistic worth taking on it: the
# forecast of `40` at the second date drags the slope far below `1` while the ordering it
# states is good. Issue #938.
const CAL_REF_SLOPE = 0.29795686719636777
const CAL_REF_SLOPE_W = 0.14120994309673554
const CAL_REF_MEAN_ALPHA = 6.333333333333333
const CAL_REF_STD_ALPHA = 10.790006599823858
const CAL_REF_MEAN_Y = 6.083333333333333
const CAL_REF_STD_Y = 10.799480907839406
# The reference numbers its buckets from zero and the library from one, so the indices below
# are its `[0, 2, 4, 5, 7, 8]` shifted by one. The four bins it never fills are dropped by
# both.
const CAL_REF_BIN = [1, 3, 5, 6, 8, 9]
const CAL_REF_BIN_ALPHA = [1.0, 2.0, 3.0, 4.0, 5.0, 24.0]
const CAL_REF_BIN_Y = [2.5, 2.0, 5.5, 5.0, 1.5, 21.5]
const CAL_REF_BIN_COUNT = [2, 3, 2, 1, 2, 2]

@testset "The calibration reproduces the reference implementation" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)
    c = forecast_calibration(fe)

    @testset "The slope and the pooled moments match, weighted and unweighted" begin
        @test c.slope ≈ CAL_REF_SLOPE
        @test forecast_calibration(fe, IC_W).slope ≈ CAL_REF_SLOPE_W
        @test c.mean_alpha ≈ CAL_REF_MEAN_ALPHA
        @test c.std_alpha ≈ CAL_REF_STD_ALPHA
        @test c.mean_y ≈ CAL_REF_MEAN_Y
        @test c.std_y ≈ CAL_REF_STD_Y
        # A good ordering and a bad scale: the coefficients of #936 are positive at the
        # first two dates and the slope is a third of one all the same.
        @test c.slope < 0.5
    end

    @testset "The curve matches bin for bin, and the empty bins are dropped" begin
        @test c.n_bins == length(CAL_REF_BIN)
        @test c.curve.bin == CAL_REF_BIN
        @test c.curve.mean_alpha ≈ CAL_REF_BIN_ALPHA
        @test c.curve.mean_y ≈ CAL_REF_BIN_Y
        @test c.curve.count == CAL_REF_BIN_COUNT
        @test c.n_bins < 10
    end

    @testset "The curve reads every pair, and only the pairs" begin
        n = sum(t -> count(i -> isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]),
                           axes(fe.alpha, 2)), fe.dates)
        @test sum(c.curve.count) == n
        @test issorted(c.curve.mean_alpha)
    end
end

@testset "The calibration slope states a scale and nothing else" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)

    @testset "A perfect forecast is already in target units" begin
        perfect = forecast_evaluation(fe.y, fe.y)
        @test forecast_calibration(perfect).slope == 1
        @test forecast_calibration(perfect, ones(size(fe.y))).slope == 1
    end

    @testset "Scaling the forecast by `c` scales the slope by `1/c`" begin
        for k in (0.5, 2.0, 10.0)
            @test forecast_calibration(forecast_evaluation(k * IC_ALPHA, y)).slope ≈
                  forecast_calibration(fe).slope / k
        end
        @test forecast_calibration(forecast_evaluation(-IC_ALPHA, y)).slope ≈ -CAL_REF_SLOPE
    end

    @testset "The line is pinned through the origin" begin
        # A fitted intercept would absorb a shift of the forecast and leave the slope where
        # it was. This one does not, which is what makes it a statement about scale.
        shifted = forecast_calibration(forecast_evaluation(IC_ALPHA .+ 100.0, y)).slope
        @test !isapprox(shifted, CAL_REF_SLOPE)
    end

    @testset "A forecast that is identically zero states no scale" begin
        @test isnan(forecast_calibration(forecast_evaluation(zeros(size(IC_ALPHA)), y)).slope)
        @test isnan(PO.forecast_calibration_slope(Float64[], Float64[], Float64[]))
    end
end

@testset "The calibration curve cuts quantile bins" begin
    PO = PortfolioOptimisers

    @testset "The edges are the quantiles, and they are answered once each" begin
        # The cut writes out the linear interpolation `Statistics.quantile` applies by
        # default, so it must answer exactly what that verb answers -- which is what keeps
        # the curve at parity with the reference implementation.
        for x in ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0, 1.0, 2.0, 40.0],
                  collect(range(-3.0, 5.0, 17)))
            for bins in (1, 2, 3, 10)
                @test PO.forecast_calibration_edges(x, bins) ≈
                      unique([quantile(x, p) for p in range(0, 1, bins + 1)])
            end
        end
        @test PO.forecast_calibration_edges([2.0, 2.0, 2.0], 4) == [2.0]
        @test length(PO.forecast_calibration_edges(collect(1.0:100.0), 10)) == 11
    end

    @testset "An empty pooling answers four empty vectors" begin
        e = PO.forecast_calibration_curve(Float64[], Float64[], 3)
        @test isempty(e.bin)
        @test isempty(e.mean_alpha)
        @test isempty(e.mean_y)
        @test isempty(e.count)
    end

    @testset "A forecast with one distinct value leaves one bin" begin
        f = PO.forecast_calibration_curve(fill(2.0, 5), collect(1.0:5.0), 4)
        @test f.bin == [1]
        @test f.mean_alpha == [2.0]
        @test f.mean_y == [3.0]
        @test f.count == [5]
    end

    @testset "A tie that spans an edge collapses the bins it spans" begin
        t = PO.forecast_calibration_curve([1.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 4)
        @test t.bin == [1, 2]
        @test t.count == [3, 1]
        @test t.mean_alpha == [1.0, 2.0]
        @test t.mean_y == [2.0, 4.0]
    end

    @testset "`bins = 1` pools everything and `bins = 0` is refused" begin
        o = PO.forecast_calibration_curve([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1)
        @test o.bin == [1]
        @test o.count == [3]
        @test_throws DomainError PO.forecast_calibration_curve([1.0], [1.0], 0)
        @test_throws DomainError forecast_calibration(forecast_evaluation(IC_ALPHA,
                                                                          PO.forward_mean_returns(IC_ALPHA,
                                                                                                  1,
                                                                                                  1));
                                                      bins = 0)
    end

    @testset "The pooled moments answer `NaN` where they cannot be taken" begin
        @test PO.forecast_pooled_moments([1.0]).mean == 1
        @test isnan(PO.forecast_pooled_moments([1.0]).std)
        @test isnan(PO.forecast_pooled_moments(Float64[]).mean)
        @test isnan(PO.forecast_pooled_moments(Float64[]).std)
    end
end

@testset "A weight shapes the slope alone, and the threshold reaches nothing" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)
    plain = forecast_calibration(fe)

    @testset "A zero weight drops a pair from the slope and keeps it everywhere else" begin
        w = ones(size(IC_ALPHA))
        w[1, 1] = 0.0
        c = forecast_calibration(fe, w)
        @test !isapprox(c.slope, plain.slope)
        @test c.curve == plain.curve
        @test c.mean_alpha == plain.mean_alpha
        @test c.mean_y == plain.mean_y
        @test c.n_bins == plain.n_bins
    end

    @testset "A weight that is not finite is read as a zero one" begin
        w = ones(size(IC_ALPHA))
        w[1, 1] = 0.0
        wn = ones(size(IC_ALPHA))
        wn[1, 1] = NaN
        @test forecast_calibration(fe, wn).slope == forecast_calibration(fe, w).slope
        @test forecast_calibration(fe, wn).curve == plain.curve
    end

    @testset "A history of ones is the equal-weight case" begin
        @test forecast_calibration(fe, ones(size(IC_ALPHA))).slope == plain.slope
    end

    @testset "The weight history is checked against the forecast" begin
        @test_throws DimensionMismatch forecast_calibration(fe, ones(2, 2))
        @test_throws DomainError forecast_calibration(fe, -ones(size(IC_ALPHA)))
    end

    @testset "`min_count` is not a parameter of this statistic" begin
        # Every other statistic of an evaluation refuses a thin cross-section. This one
        # pools the pairs of every date and reads them as one sample, so there is no
        # cross-sectional count to threshold and the answer does not move with it.
        hi = forecast_evaluation(IC_ALPHA, y; min_count = size(IC_ALPHA, 2) + 1)
        c = forecast_calibration(hi)
        @test c.slope == plain.slope
        @test c.curve == plain.curve
        @test c.mean_alpha == plain.mean_alpha
        @test c.n_bins == plain.n_bins
    end
end

@testset "The calibration of a fitted member is measured on the panel" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture(; planted = true)
    fw = FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    fe = forecast_evaluation(fw, fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1)

    @testset "A member that never calibrated states whatever scale it carries" begin
        # `FixedWeightedReturnForecast` blends standardised Descriptor scores, so its
        # magnitude is that of a score and not that of a return. The slope is therefore
        # very small and finite, and the curve still has ten bins to report it over.
        c = forecast_calibration(fe)
        @test isfinite(c.slope)
        @test c.slope > 0
        @test c.slope < 0.01
        @test c.n_bins == 10
        @test sum(c.curve.count) ==
              sum(t -> count(i -> isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]),
                             axes(fe.alpha, 2)), fe.dates)
        @test issorted(c.curve.mean_alpha)
        @test abs(c.mean_alpha) < abs(c.std_alpha)
    end

    @testset "The block method resolves the weights the metric names" begin
        c = forecast_calibration(fe, fx.csfm)
        @test c.slope == forecast_calibration(fe,
                               PO.cs_diagnostic_weights(PO.IdentityMetric(), fx.csfm)).slope
        @test c.curve == forecast_calibration(fe).curve
        cb = forecast_calibration(fe, fx.csfm;
                                  weighting = PO.InverseIdiosyncraticVarianceMetric(),
                                  bins = 4)
        @test cb.n_bins == 4
        @test !isapprox(cb.slope, c.slope)
        @test cb.curve != c.curve
        # A metric the block cannot serve is refused by name, by the resolver rather than
        # by this verb: the fixture's block carries no benchmark weight history.
        @test_throws PO.IsNothingError forecast_calibration(fe, fx.csfm;
                                                            weighting = PO.BenchmarkWeightMetric())
    end

    @testset "The scale is the one reading a rescaling moves" begin
        # The coefficients and the portfolio are invariant to a rescaling of the forecast,
        # and the calibration is not. That is the whole reason this verb exists.
        scaled = forecast_evaluation(100 .* fe.alpha, fe.y; horizon = 2, lag = 1, step = 1)
        @test forecast_ic(scaled)[:, 1] ≈ forecast_ic(fe)[:, 1]
        @test forecast_portfolio(scaled).ret ≈ forecast_portfolio(fe).ret
        @test forecast_calibration(scaled).slope ≈ forecast_calibration(fe).slope / 100
    end
end
