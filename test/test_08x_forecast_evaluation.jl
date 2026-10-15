#=
Check `src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/07_ForecastEvaluation.jl`,
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/08_ForecastHistory.jl` and
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/09_ForecastInformationCoefficient.jl`,
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/10_ForecastPortfolios.jl`,
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/11_ForecastFactorCorrelation.jl`,
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/12_ForecastForwardWindows.jl`,
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/13_ForecastCalibration.jl` and
`src/05_Moments/32_CrossSectionalFactorModel/07_ReturnForecasts/14_ForecastSummary.jl` against the contract their
docstrings state, and against the stored oracle of map #931.
Issues #934, #935, #936, #937, #938, #939, #940 and #941.

ELEVEN CONVENTIONS SHAPE THE PROBES.

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

5. THE STATISTICS ARE MEASURED BY RUNNING THE ORACLE, NOT BY READING IT. `IC_ALPHA` and
   its gapped variant were put through the oracle's own diagnostic and its correlation
   summary, and the literals below are what it answered. The one place the port diverges is
   the hit rate: the library reads it against the dates that carried a coefficient, where
   the oracle's exposure summary counts a date with no coefficient as a miss. That is
   `exposure_ic_factor_summary`'s convention, which both summaries share, and it is
   asserted as a divergence rather than papered over. The second divergence is the
   t-statistic: the oracle scales the ratio by the root of the date count wherever it
   is read, and the port's standard error reads the overlap of the forward windows through
   `forecast_ic_lags`, so the two agree only where the windows are disjoint. The
   holding-period table at a stride of one is where they part, and the oracle's number
   is asserted there as the plain ratio beside the corrected column.

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
   that summary counts against the FINITE dates, which is convention 5's denominator.

8. THE FACTOR CORRELATION IS PINNED BY A DESIGN, NOT BY A LITERAL. It is contemporaneous,
   so there is nothing forward-looking to oracle: the probes build a cross-section whose
   correlation is exactly 1 and exactly 0 by construction, under BOTH the weighted form and
   the rank form, so the two are separable without a stored number. `FC_ORTHO` is the
   permutation whose centred values and whose ordinal ranks are both orthogonal to
   `[1, 2, 3, 4]`, which an arbitrary orthogonal vector would not be -- a tied vector such
   as `[1, -1, -1, 1]` is Pearson-orthogonal and reads 0.4 under ordinal ranks. The
   `ExposureNeutralisation` probe is the one that reads a real fixture. Since #1521 the
   Neutralisation of `DescriptorScores` fits an intercept by default, so its residual is
   orthogonal to the constant and to the target, hence uncorrelated with the target. The
   probe pins that to round-off on the scores themselves, under the weights of the
   Neutralisation. Through the evaluation it is DIRECTIONAL: the grouped scoring step
   rescales each industry on its own, and the evaluation weighs the active assets that the
   estimation mask leaves out, so the correlation falls far but not to zero. The
   oracle's rule, `intercept = false`, keeps a correlation above 0.5.

9. A FORWARD-WINDOW TABLE IS PINNED ROW BY ROW AGAINST THE ORACLE, AND ITS DATE RULE IS
   PINNED SEPARATELY. `WINDOW_ALPHA` was put through the oracle's own holding-period and
   decay diagnostics and the literals below are what it answered, to every digit (#1558;
   before it, to the six digits it printed). The date rule is the one thing those literals cannot pin, because the
   oracle and the port agree on it: every row of a table is read on the dates every
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

11. THE SUMMARY IS ORACLED WHOLE, AND ITS PLUMBING IS PINNED SEPARATELY. Every column of
    `ForecastSummaryResult` was put through the oracle's own five
    summary methods over `IC_ALPHA` and over the gapped variant, and `FS_REF` and
    `FS_REF_GAP` are what it answered, to every digit -- the coefficients, both books,
    the calibration, the coverage and the quantile spread alike. Beside those literals
    sits a second set of probes asserting that each column IS the level-2 verb it reads,
    which is what a summary that computes nothing of its own owes: a literal that matches
    proves the number, and the identity proves the number came from the verb rather than
    from a second copy of it. The gapped fixture is what separates `mean_coverage` from
    `min_coverage`, because a full panel reads 1 for both. The hit rates of conventions 5
    and 7 meet here with one denominator: a date with no coefficient and a date with no
    trade are both unmeasured, and the coverage columns count them.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "test06c_setup.jl"))
include(joinpath(@__DIR__, "forecast_evaluation_fixture.jl"))

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
        @test_throws PO.ConflictingArgumentError forecast_evaluation(alpha, y;
                                                                     ties = :dense)
        @test forecast_evaluation(alpha, y).ties === :average
        @test forecast_evaluation(alpha, y; ties = :ordinal).ties === :ordinal
    end

    @testset "The pair is scored in its own element type, which must hold a fraction and NaN" begin
        # The evaluation neither converts nor refuses a pair by its type, so an integer pair
        # fails where a statistic first writes a fraction or a NaN, and a caller converts it.
        ai = rand(StableRNG(1), -5:5, 30, 8)
        yi = rand(StableRNG(2), -5:5, 30, 8)
        umsk = rand(StableRNG(3), Bool, 30, 8)
        fe = forecast_evaluation(ai, yi)
        @test eltype(fe.alpha) == Int
        @test_throws InexactError forecast_ic(fe)
        @test_throws InexactError forecast_portfolio(fe)
        @test_throws InexactError forecast_calibration(fe)
        @test_throws InexactError forecast_quantile_spread(fe)
        @test_throws InexactError forecast_evaluation(ai, yi; umsk = umsk)
        @test eltype(forecast_ic(forecast_evaluation(float.(ai), float.(yi)))) == Float64

        fe32 = forecast_evaluation(Float32.(ai), Float32.(yi); umsk = umsk)
        @test eltype(fe32.alpha) == Float32
        @test eltype(forecast_ic(fe32)) == Float32
        @test eltype(forecast_portfolio(fe32).w) == Float32

        fer = forecast_evaluation(Rational.(ai), Rational.(yi))
        @test eltype(forecast_ic(fer)) == Rational{Int}
        @test_throws InexactError forecast_portfolio(fer)
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
        # The history is written onto the estimation universe at the pairing (#1074).
        @test isequal(fe.alpha, PO.forecast_evaluation_mask(rf.hist, rd.pnl.emsk[rows, :]))
        @test fe.umsk == rd.pnl.emsk[rows, :]
        @test isequal(fe.y, PO.forward_mean_returns(csfm.csr.eps, 2, 1))
        @test fe.step == 2
        @test size(fe.alpha) == (fx.Tb, fx.N)
    end

    @testset "The two layers agree on the same pair" begin
        fe = forecast_evaluation(rf, rd, csfm; horizon = 2, lag = 1, step = 1, ppy = 252)
        bare = forecast_evaluation(rf.hist, PO.forward_mean_returns(csfm.csr.eps, 2, 1);
                                   umsk = rd.pnl.emsk[rows, :], horizon = 2, lag = 1,
                                   step = 1, ppy = 252)
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
        # Causality is proved by perturbation, not by recomputing the loop's own call: every
        # row after the block's row `tb` -- the carrier's returns, every numeric Panel Field,
        # and the block's residual, variance and exposure histories -- is overwritten with
        # noise, and the rows through `tb` must come back bit for bit. A row after `tb` must
        # move, or the perturbation reached nothing.
        tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1,
                                   calibrate = false)
        hist = forecast_history(tgt, rd, csfm)
        tb = 30
        rng = StableRNG(1)
        rdp = deepcopy(rd)
        csfmp = deepcopy(csfm)
        r_after = (rows[tb] + 1):fx.T
        rdp.X[r_after, :] .= 10 .* randn(rng, length(r_after), fx.N)
        for f in rdp.pnl.pf
            if f isa PO.NumericPanelField
                f.vals[r_after, :] .= 100 .* rand(rng, length(r_after), fx.N)
            end
        end
        b_after = (tb + 1):fx.Tb
        csfmp.csr.eps[b_after, :] .= 5 .* randn(rng, length(b_after), fx.N)
        csfmp.vs[b_after, :] .= 9.0
        csfmp.Ms[b_after, :, :] .= randn(rng, length(b_after), fx.N, size(csfm.Ms, 3))
        histp = forecast_history(tgt, rdp, csfmp)
        @test isequal(view(histp, 1:tb, :), view(hist, 1:tb, :))
        @test !isequal(view(histp, tb + 1, :), view(hist, tb + 1, :))
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

# The stored oracle of the information coefficients, measured by running the oracle
# on the same two matrices. `IC_ALPHA` is a forecast whose ordering of the
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

@testset "The information coefficients reproduce the stored oracle" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)

    @testset "Both columns are answered, and both match the oracle" begin
        ic = forecast_ic(fe)
        @test size(ic) == (length(fe.dates), 2)
        # Every IC, IC summary and coverage literal of the oracle in this file measures
        # bit-equal, cell by cell (#1558). `rtol = 1e-14` only leaves room for a `sqrt` or a
        # sum that rounds one ulp apart on another host.
        @test parity_compare(ic[:, 1], IC_REF_SPEARMAN; rtol = 1e-14, name = "ic spearman").ok
        @test parity_compare(ic[:, 2], IC_REF_PEARSON; rtol = 1e-14, name = "ic pearson").ok
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
        @test parity_compare(ic[:, 1], IC_REF_SPEARMAN; rtol = 1e-14,
                             name = "ic w spearman").ok
        @test parity_compare(ic[:, 2], IC_REF_PEARSON_W; rtol = 1e-14,
                             name = "ic w pearson").ok
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

    @testset "A gap in the panel moves both columns, as it does in the oracle" begin
        gap = ic_gap_fixture()
        fg = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 2)
        ic = forecast_ic(fg)
        @test parity_compare(ic[:, 1], IC_REF_GAP_SPEARMAN; rtol = 1e-14,
                             name = "ic gap spearman").ok
        @test parity_compare(ic[:, 2], IC_REF_GAP_PEARSON; rtol = 1e-14,
                             name = "ic gap pearson").ok
    end
end

@testset "The summary names its two series and reports a t-statistic" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    ic = forecast_ic(forecast_evaluation(IC_ALPHA, y))
    s = forecast_ic_summary(ic; mt = OverlapMeanTest())

    @testset "The five figures of each series match the oracle" begin
        @test keys(s) == (:spearman, :pearson)
        @test keys(s.spearman) ==
              (:mean_ic, :std_ic, :ic_ir, :t_stat, :p_value, :mean_ci_lower, :mean_ci_upper,
               :hit_rate)
        @test parity_compare(collect(values(s.spearman))[[1, 2, 3, 4, 8]],
                             [0.3333333333333333, 0.7023769168568493, 0.4745789978762494,
                              0.8219949365267862, 0.6666666666666666]; rtol = 1e-14,
                             name = "ic summary spearman").ok
        @test parity_compare(collect(values(s.pearson))[[1, 2, 3, 4, 8]],
                             [0.2567790285367673, 0.6206266852224849, 0.4137415207093715,
                              0.7166213350694421, 0.6666666666666666]; rtol = 1e-14,
                             name = "ic summary pearson").ok
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
        @test s.spearman == PO.exposure_ic_factor_summary(ic, 1; mt = OverlapMeanTest())
        @test s.pearson == PO.exposure_ic_factor_summary(ic, 2; mt = OverlapMeanTest())
    end

    @testset "The summary refuses a series it cannot name" begin
        @test_throws PO.IsEmptyError forecast_ic_summary(Matrix{Float64}(undef, 0, 0);
                                                         mt = OverlapMeanTest())
        @test_throws DimensionMismatch forecast_ic_summary(ones(3, 1);
                                                           mt = OverlapMeanTest())
        @test_throws DimensionMismatch forecast_ic_summary(ones(3, 3);
                                                           mt = OverlapMeanTest())
        @test_throws DomainError forecast_ic_summary(ic; lags = -1, mt = OverlapMeanTest())
    end
end

@testset "The t-statistic reads the overlap of the forward windows" begin
    PO = PortfolioOptimisers
    # Hand computation of the kernel's long-run standard error: the variance plus twice the
    # first `L` autocovariances, every one over `n - 2L - 1`, which removes the bias of the
    # centring under a moving average of order `L`, pairs read at their positions.
    function hand_t(v, L)
        f = isfinite.(v)
        n = count(f)
        m = sum(v[f]) / n
        g = zeros(L + 1)
        for j in 0:L, t in 1:(length(v) - j)
            if f[t] && f[t + j]
                g[j + 1] += (v[t] - m) * (v[t + j] - m)
            end
        end
        nu = n - 2L - 1
        lrv = (g[1] + 2 * sum(g[2:end])) / nu
        return nu > 0 && lrv > 0 ? m / sqrt(lrv) * sqrt(n) : NaN
    end

    @testset "The lag is one less than the number of strides a window spans" begin
        @test PO.forecast_ic_lags(1, 1) == 0
        @test PO.forecast_ic_lags(5, 5) == 0
        @test PO.forecast_ic_lags(5, 1) == 4
        @test PO.forecast_ic_lags(5, 2) == 2
        @test PO.forecast_ic_lags(4, 2) == 1
        @test PO.forecast_ic_lags(2, 5) == 0
        @test_throws DomainError PO.forecast_ic_lags(0, 1)
        @test_throws DomainError PO.forecast_ic_lags(1, 0)
        # The Result method reads the evaluation's own window and stride.
        y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
        @test PO.forecast_ic_lags(forecast_evaluation(IC_ALPHA, y)) == 0
        @test PO.forecast_ic_lags(forecast_evaluation(IC_ALPHA, y; horizon = 3, step = 1)) ==
              2
        @test PO.forecast_ic_lags(forecast_evaluation(IC_ALPHA, y; horizon = 3, step = 2)) ==
              1
    end

    @testset "At no lag the statistic is the one the oracle states, bit for bit" begin
        y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
        ic = forecast_ic(forecast_evaluation(IC_ALPHA, y))
        for k in 1:2
            a = PO.exposure_ic_factor_summary(ic, k; mt = OverlapMeanTest())
            b = PO.exposure_ic_factor_summary(ic, k; lags = 0, mt = OverlapMeanTest())
            @test a == b
            @test a.t_stat == a.ic_ir * sqrt(count(isfinite, ic[:, k]))
        end
    end

    @testset "A positive lag adds the autocovariances to the standard error" begin
        # A smooth series, so every autocovariance up to the third lag is positive and the
        # long-run variance is positive at every lag probed.
        v = [0.10, 0.18, 0.26, 0.30, 0.24, 0.16, 0.08, 0.04, 0.12, 0.22, 0.28, 0.20]
        ic = hcat(v, -v)
        for L in 0:3
            s = PO.exposure_ic_factor_summary(ic, 1; lags = L, mt = OverlapMeanTest())
            @test s.t_stat ≈ hand_t(v, L)
            # The mean, the deviation, the ratio and the hit rate do not read the lag.
            s0 = PO.exposure_ic_factor_summary(ic, 1; mt = OverlapMeanTest())
            @test (s.mean_ic, s.std_ic, s.ic_ir, s.hit_rate) ==
                  (s0.mean_ic, s0.std_ic, s0.ic_ir, s0.hit_rate)
            # The forecast summary hands the same lag to both of its series.
            fs = forecast_ic_summary(ic; lags = L, mt = OverlapMeanTest())
            @test isequal(fs.spearman, s)
            @test isequal(fs.pearson,
                          PO.exposure_ic_factor_summary(ic, 2; lags = L,
                                                        mt = OverlapMeanTest()))
            @test fs.pearson.t_stat ≈ -s.t_stat
        end
        # A positively autocorrelated series has a smaller statistic at a positive lag.
        w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        t0 = PO.exposure_ic_factor_summary(hcat(w), 1; mt = OverlapMeanTest()).t_stat
        t1 = PO.exposure_ic_factor_summary(hcat(w), 1; lags = 1, mt = OverlapMeanTest()).t_stat
        @test 0 < t1 < t0
        @test_throws DomainError PO.exposure_ic_factor_summary(ic, 1; lags = -1,
                                                               mt = OverlapMeanTest())
        @test_throws DomainError PO.exposure_ic_summary(ic; lags = -1,
                                                        mt = OverlapMeanTest())
    end

    @testset "A NaN row is in no pair, and the lag is a distance in the series" begin
        v = [0.3, NaN, 0.4, 0.2, NaN, 0.5, 0.1, 0.0, 0.3, -0.1]
        for L in 1:2
            @test PO.exposure_ic_factor_summary(hcat(v), 1; lags = L,
                                                mt = OverlapMeanTest()).t_stat ≈
                  hand_t(v, L)
        end
        # Dropping the gaps first would pair rows that are two apart as neighbours, and
        # that is not what the kernel does.
        u = filter(isfinite, v)
        @test !(PO.exposure_ic_factor_summary(hcat(v), 1; lags = 1,
                                              mt = OverlapMeanTest()).t_stat ≈ hand_t(u, 1))
    end

    @testset "A non-positive long-run variance has no statistic" begin
        # An alternating series has a lag-one autocovariance more negative than half its
        # variance, so the sum is negative and the statistic is NaN rather than clamped.
        v = [0.5, -0.4, 0.5, -0.4, 0.5, -0.4, 0.5, -0.4]
        s = PO.exposure_ic_factor_summary(hcat(v), 1; lags = 1, mt = OverlapMeanTest())
        @test isnan(s.t_stat)
        @test isfinite(s.ic_ir)
        @test isfinite(PO.exposure_ic_factor_summary(hcat(v), 1; mt = OverlapMeanTest()).t_stat)
    end

    @testset "The evaluation summaries derive the lag from the window and the stride" begin
        rng = StableRNG(7)
        alpha = randn(rng, 40, 6)
        X = randn(rng, 40, 6)
        y = PO.forward_mean_returns(X, 3, 1)
        # A stride of one under a window of three: two neighbours on either side overlap.
        fe = forecast_evaluation(alpha, y; horizon = 3, step = 1)
        ic = forecast_ic(fe)
        s = forecast_evaluation_summary([fe]; mt = OverlapMeanTest())
        @test s.spearman_t_stat[1] ≈
              forecast_ic_summary(ic; lags = 2, mt = OverlapMeanTest()).spearman.t_stat
        @test s.pearson_t_stat[1] ≈
              forecast_ic_summary(ic; lags = 2, mt = OverlapMeanTest()).pearson.t_stat
        @test !(s.spearman_t_stat[1] ≈
                forecast_ic_summary(ic; mt = OverlapMeanTest()).spearman.t_stat)
        # A stride of the window: no overlap, the plain statistic.
        fe3 = forecast_evaluation(alpha, y; horizon = 3)
        s3 = forecast_evaluation_summary([fe3]; mt = OverlapMeanTest())
        @test s3.spearman_t_stat[1] ≈
              forecast_ic_summary(forecast_ic(fe3); mt = OverlapMeanTest()).spearman.t_stat
        # The holding-period table's row `p` spans `p` strides at the default stride, so it
        # reads `p - 1` lags; the decay table's windows stay one stride long.
        fe1 = forecast_evaluation(alpha, PO.forward_mean_returns(X, 1, 1))
        h = forecast_holding_period(fe1, X; n = 4, mt = OverlapMeanTest())
        d = forecast_decay(fe1, X; n = 4, mt = OverlapMeanTest())
        for p in 1:4
            fp = PO.ForecastEvaluationResult(alpha,
                                             PO.forward_mean_returns(X, h.horizon[p],
                                                                     h.lag[p]), fe1.umsk,
                                             h.dates, fe1.target, h.horizon[p], h.lag[p],
                                             fe1.step, fe1.min_count, fe1.ties, fe1.ppy)
            @test PO.forecast_ic_lags(fp) == p - 1
            @test isequal(h.spearman_t_stat[p],
                          forecast_ic_summary(forecast_ic(fp); lags = p - 1,
                                              mt = OverlapMeanTest()).spearman.t_stat)
            fq = PO.ForecastEvaluationResult(alpha,
                                             PO.forward_mean_returns(X, d.horizon[p],
                                                                     d.lag[p]), fe1.umsk,
                                             d.dates, fe1.target, d.horizon[p], d.lag[p],
                                             fe1.step, fe1.min_count, fe1.ties, fe1.ppy)
            @test PO.forecast_ic_lags(fq) == 0
            @test d.spearman_t_stat[p] ≈
                  forecast_ic_summary(forecast_ic(fq); mt = OverlapMeanTest()).spearman.t_stat
        end
    end

    @testset "A forecast with no skill is not made significant by a longer window" begin
        # A persistent forecast against an independent target has no skill by construction.
        # Its coefficient at a window of `p` observations, scored every observation, is a
        # moving average of order `p - 1`, and the plain statistic's dispersion across
        # samples grows with `p` where the corrected one's does not.
        S = 120
        T = 160
        N = 12
        n = 5
        plain = zeros(S, n)
        fixed = zeros(S, n)
        for s in 1:S
            rng = StableRNG(1000 + s)
            alpha = Matrix{Float64}(undef, T, N)
            alpha[1, :] = randn(rng, N)
            for t in 2:T
                alpha[t, :] = 0.99 .* alpha[t - 1, :] .+ sqrt(1 - 0.99^2) .* randn(rng, N)
            end
            X = randn(rng, T, N)
            fe = forecast_evaluation(alpha, PO.forward_mean_returns(X, 1, 1))
            h = forecast_holding_period(fe, X; n = n, mt = OverlapMeanTest())
            m = length(h.dates)
            fixed[s, :] = h.spearman_t_stat
            plain[s, :] = h.spearman_ic_ir .* sqrt(m)
        end
        sd(x) = sqrt(sum(abs2, x .- sum(x) / length(x)) / (length(x) - 1))
        # Under the null the corrected statistic is dispersed like a unit normal at every
        # depth; the plain one is at the first row only and by the last is far wider.
        for p in 1:n
            @test 0.75 < sd(fixed[:, p]) < 1.3
        end
        @test 0.75 < sd(plain[:, 1]) < 1.3
        @test sd(plain[:, n]) > 1.6
        @test sd(plain[:, n]) > 1.5 * sd(fixed[:, n])
        @test plain[:, 1] == fixed[:, 1]
    end
end

# A kernel that implements only the weight of the interface, so the Mean Test reaches it
# through the generic long-run variance and the generic bootstrap.
struct GenericBartlett <: PortfolioOptimisers.AbstractLongRunKernel end
function PortfolioOptimisers.long_run_kernel_weight(::GenericBartlett, x::Real)
    return max(1 - abs(x), 0)
end
# The truncated kernel: its matrix of weights has negative eigenvalues, so an eigen factor of
# it has a part whose squares subtract.
struct TruncatedKernel <: PortfolioOptimisers.AbstractLongRunKernel end
function PortfolioOptimisers.long_run_kernel_weight(::TruncatedKernel, x::Real)
    return abs(x) <= 1 ? one(x) : zero(x)
end

@testset "A Mean Test reads the dependence of the series past its overlap" begin
    PO = PortfolioOptimisers
    # Hand computation of a fixed-bandwidth statistic: every pair of finite entries weighted
    # by the kernel at `|i - j| / B`, over `n` as the source divides, with `B = b * T`.
    function hand_t(v, b, k = BartlettKernel(); grid = false)
        f = isfinite.(v)
        n = count(f)
        m = sum(v[f]) / n
        i1 = findfirst(f)
        i2 = findlast(f)
        B = b * (grid ? length(v) : i2 - i1 + 1)
        Q = 0.0
        for i in i1:i2, j in i1:i2
            if f[i] && f[j]
                w = i == j ? 1.0 : PO.long_run_kernel_weight(k, abs(i - j) / B)
                Q += w * (v[i] - m) * (v[j] - m)
            end
        end
        return Q > 0 ? m / sqrt(Q / n) * sqrt(n) : NaN
    end
    # Hand computation of the bootstrap, draw by draw, with the draws in the order the
    # library takes them: one per finite entry, draw after draw.
    function hand_bootstrap(v, b, k, M, q, rng)
        f = isfinite.(v)
        n = count(f)
        m = sum(v[f]) / n
        t = hand_t(v, b, k)
        i1 = findfirst(f)
        i2 = findlast(f)
        B = b * (i2 - i1 + 1)
        d = [f[i] ? v[i] - m : 0.0 for i in i1:i2]
        ff = f[i1:i2]
        P = length(d)
        ats = zeros(M)
        for s in 1:M
            u = [ff[i] ? randn(rng) * d[i] : 0.0 for i in 1:P]
            ms = sum(u) / n
            u = [ff[i] ? u[i] - ms : 0.0 for i in 1:P]
            Q = 0.0
            for i in 1:P, j in 1:P
                w = i == j ? 1.0 : PO.long_run_kernel_weight(k, abs(i - j) / B)
                Q += w * u[i] * u[j]
            end
            ts = Q > 0 ? ms / sqrt(Q / n) * sqrt(n) : NaN
            ats[s] = isfinite(ts) ? abs(ts) : Inf
        end
        K = count(>=(abs(t)), ats)
        c = maximum(x for x in 0:M if x / (M + 1) <= q)
        h = sort(ats)[M + 1 - c] * m / t
        return (; t_stat = t, p_value = (1 + K) / (M + 1), mean_ci_lower = m - h,
                mean_ci_upper = m + h)
    end
    ar1(rng, T, phi) = begin
        e = randn(rng, T + 50)
        x = zeros(T + 50)
        for t in 2:(T + 50)
            x[t] = phi * x[t - 1] + e[t]
        end
        x[51:end]
    end

    @testset "The constructors hold the defaults of the source and refuse a bad value" begin
        mt = WildBootstrapMeanTest()
        @test mt.kernel isa BartlettKernel
        @test mt.b == 0.4
        @test mt.span isa FiniteEntrySpan
        @test mt.ddof == 0
        @test_throws DomainError WildBootstrapMeanTest(; ddof = -1)
        @test mt.n_sim == 1_999
        @test mt.q == 0.05
        @test isnothing(mt.min_obs) && isnothing(mt.min_periods)
        @test isnothing(mt.seed)
        @test mt.rng === PO.Random.default_rng()
        @test_throws DomainError WildBootstrapMeanTest(; b = 0)
        @test_throws DomainError WildBootstrapMeanTest(; b = 1.5)
        @test WildBootstrapMeanTest(; b = 1).b == 1
        @test_throws DomainError WildBootstrapMeanTest(; n_sim = 0)
        @test_throws DomainError WildBootstrapMeanTest(;
                                                       n_sim = PO.RESOURCE_LIMITS[].max_n_sim +
                                                               1)
        @test_throws DomainError WildBootstrapMeanTest(; q = 0)
        @test_throws DomainError WildBootstrapMeanTest(; q = 1)
        # Nine draws cannot place a 5% interval, and nineteen can.
        @test_throws DomainError WildBootstrapMeanTest(; n_sim = 9)
        @test WildBootstrapMeanTest(; n_sim = 19).n_sim == 19
        @test_throws DomainError WildBootstrapMeanTest(; min_obs = 1)
        @test_throws DomainError WildBootstrapMeanTest(; min_periods = 0)
        o = OverlapMeanTest()
        @test o.q == 0.05 && isnothing(o.min_obs) && isnothing(o.min_periods)
        @test_throws DomainError OverlapMeanTest(; q = 1.5)
        @test_throws DomainError OverlapMeanTest(; min_obs = 0)
        @test isnothing(PO.mean_test_rng(OverlapMeanTest()))
        # The draws follow the kernel by default, and a kernel with no quadrature features
        # refuses the quadrature draws.
        @test mt.alg isa ObservationDraws
        qs = WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel())
        @test qs.alg isa QuadratureSumDraws
        @test isnothing(qs.alg.tol) && isnothing(qs.alg.nodes)
        @test WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel(),
                                    alg = EigenSumDraws()).alg isa EigenSumDraws
        @test_throws ConflictingArgumentError WildBootstrapMeanTest(;
                                                                    alg = QuadratureSumDraws())
        @test_throws DomainError QuadratureSumDraws(; tol = 0)
        @test_throws DomainError QuadratureSumDraws(; tol = 1)
        @test_throws DomainError QuadratureSumDraws(; nodes = 0)
        @test QuadratureSumDraws(; tol = 1e-10, nodes = 7).nodes == 7
        @test QuadratureSumDraws().form isa FeatureLongRunForm
        @test QuadratureSumDraws(; form = ExactLongRunForm()).form isa ExactLongRunForm
        # A seed gives a private copy, so the caller's generator does not move.
        rng = StableRNG(3)
        r = PO.mean_test_rng(WildBootstrapMeanTest(; rng = rng, seed = 9))
        @test r !== rng
        @test rand(r) ==
              rand(PO.mean_test_rng(WildBootstrapMeanTest(; rng = rng, seed = 9)))
        @test rand(rng) == rand(StableRNG(3))
    end

    @testset "The count of the interval is read off the division of the p-value" begin
        # `0.29 * 100` rounds below 29, so a count read off the product would be one short.
        @test 0.29 * 100 < 29
        @test PO.bootstrap_exceedance_cap(0.29, 99) == 29
        # A statistic of a draw that is not finite sorts above every finite one.
        @test PO.finite_abs(NaN) == Inf
        @test PO.finite_abs(-2.0) == 2.0
        for q in 0.01:0.01:0.49, M in (19, 24, 49, 99, 999)
            if q * (M + 1) >= 1
                @test PO.bootstrap_exceedance_cap(q, M) ==
                      maximum(x for x in 0:M if x / (M + 1) <= q)
            end
        end
    end

    @testset "The kernels weigh the lags as their sources state" begin
        @test PO.long_run_kernel_weight(BartlettKernel(), 0.25) == 0.75
        @test PO.long_run_kernel_weight(BartlettKernel(), 1.5) == 0
        k = QuadraticSpectralKernel()
        @test PO.long_run_kernel_weight(k, 0.0) == 1
        @test PO.long_run_kernel_weight(k, 1e-4) ≈ 1 atol = 1e-6
        for x in (0.3, 1.0, 2.5)
            z = 6 * pi * x / 5
            @test PO.long_run_kernel_weight(k, x) ≈
                  25 / (12 * pi^2 * x^2) * (sin(z) / z - cos(z))
        end
        @test PO.long_run_kernel_weight(k, 2.0) < 0
        # The closed form cancels near zero, and the Taylor series does not: against the
        # integral form in 256-bit arithmetic, on both sides of `|z| = 1`.
        qsb(x) = (z = 6 * big(pi) * x / 5; 3 * (sin(z) - z * cos(z)) / z^3)
        for x in (1e-6, 1e-4, 1e-3, 0.05, 0.1, 0.13, 0.1326, 0.1327, 0.2, 0.5)
            @test abs(PO.long_run_kernel_weight(k, x) - qsb(big(x))) / qsb(big(x)) < 4e-16
        end
        @test PO.long_run_kernel_weight(k, 1.0f-3) isa Float32
    end

    @testset "The quadrature features give the quadratic spectral weights" begin
        k = QuadraticSpectralKernel()
        for b in (1.0, 0.4, 0.1, 0.02, 0.01), P in (2, 37, 300)
            B = b * P
            Phi = PO.quadrature_kernel_features(k, QuadratureSumDraws(), P, B)
            @test maximum(abs, Phi * Phi' - PO.kernel_weight_matrix(k, P, B)) < 1e-13
        end
        # The count is the least one whose bound holds, and it depends on `b` alone.
        for b in (1.0, 0.4, 0.1, 0.02), tol in (1e-8, 1e-14)
            c = 6 * pi / (5 * b)
            N = PO.quadrature_node_count(c, tol)
            @test PO.quadrature_error_bound(N, c) <= log(tol)
            @test N == 1 || PO.quadrature_error_bound(N - 1, c) > log(tol)
        end
        # A count overrides the derivation, and a count too small gives a wrong kernel.
        Phi = PO.quadrature_kernel_features(k, QuadratureSumDraws(; nodes = 4), 300, 30.0)
        @test size(Phi, 2) == 8
        @test maximum(abs, Phi * Phi' - PO.kernel_weight_matrix(k, 300, 30.0)) > 1e-3
        # The feature form of the observed sum agrees with the exact sum to the error of
        # the quadrature, with a gap and at a long span.
        for P in (37, 3_000)
            d = randn(StableRNG(P), P)
            d[[2, P - 3]] .= 0
            d .-= sum(d) / P
            B = 0.1 * P
            Phi = PO.quadrature_kernel_features(k, QuadratureSumDraws(), P, B)
            buf = zeros(P + 1)
            ex = PO.observed_long_run_form(ExactLongRunForm(), k, d, B, buf, Phi)
            @test ex == PO.long_run_quadratic_form(k, d, B, buf)
            @test isapprox(PO.observed_long_run_form(FeatureLongRunForm(), k, d, B, buf,
                                                     Phi), ex; rtol = 1e-12)
        end
        # A sum that is not positive takes no draw.
        Q, ats = PO.wild_bootstrap_statistics(k, QuadratureSumDraws(), 99, zeros(10),
                                              trues(10), 10, 10, 1.0, zeros(11),
                                              StableRNG(1))
        @test iszero(Q) && isempty(ats)
        u, w = PO.gauss_legendre_positive_nodes(5, Float64)
        @test sum(w) ≈ 1
        @test sum(w .* u .^ 4) ≈ 1 / 5
    end

    @testset "Every draw algorithm draws the same law of the statistic" begin
        # A no-skill series of overlapping windows with gaps, and a series with a mean. The
        # p-values of two algorithms are two Monte Carlo estimates of one probability, so
        # they agree within the error of their difference.
        e = randn(StableRNG(21), 184)
        v = [sum(view(e, t:(t + 4))) / 5 for t in 1:180]
        v[[1, 40, 41, 42, 133]] .= NaN
        @test minimum(LinearAlgebra.eigvals(PO.kernel_weight_matrix(TruncatedKernel(), 40,
                                                                    6.0))) < 0
        M = 9_999
        for x in (v, v .+ 0.08)
            for (k, algs) in ((QuadraticSpectralKernel(),
                               (ObservationDraws(), QuadratureSumDraws(), EigenSumDraws(),
                                QuadratureSumDraws(; form = ExactLongRunForm()))),
                              (BartlettKernel(), (ObservationDraws(), EigenSumDraws())),
                              (GenericBartlett(), (ObservationDraws(), EigenSumDraws())),
                              (TruncatedKernel(), (ObservationDraws(), EigenSumDraws())))
                r = [PO.mean_test(WildBootstrapMeanTest(; kernel = k, alg = a, b = 0.15,
                                                        n_sim = M), x, 5, StableRNG(j))
                     for (j, a) in enumerate(algs)]
                for s in r[2:end]
                    # The feature form reads the kernel of the quadrature, to its tolerance.
                    @test isapprox(s.t_stat, r[1].t_stat; rtol = 1e-12)
                    p = r[1].p_value
                    @test abs(s.p_value - p) <= 4 * sqrt(2 * p * (1 - p) / M) + 2 / M
                    @test isapprox(s.mean_ci_upper - s.mean_ci_lower,
                                   r[1].mean_ci_upper - r[1].mean_ci_lower; rtol = 0.06)
                end
            end
        end
        # The exact form reads the observed series at the exact weights, as the draws per
        # observation do.
        mq = WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel(), b = 0.15,
                                   n_sim = 19)
        @test PO.mean_test(WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel(),
                                                 alg = QuadratureSumDraws(;
                                                                          form = ExactLongRunForm()),
                                                 b = 0.15, n_sim = 19), v, 5,
                           StableRNG(1)).t_stat ==
              PO.mean_test(WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel(),
                                                 alg = ObservationDraws(), b = 0.15,
                                                 n_sim = 19), v, 5, StableRNG(1)).t_stat
        @test mq.alg.form isa FeatureLongRunForm
        # A constant series has no statistic under any algorithm.
        for a in (QuadratureSumDraws(), EigenSumDraws(),
                  QuadratureSumDraws(; form = ExactLongRunForm()))
            mt = WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel(), alg = a,
                                       n_sim = 19)
            @test all(isnan, values(PO.mean_test(mt, fill(0.03, 40), 1, StableRNG(1))))
        end
    end

    @testset "The Bartlett form reads the window sums at every real bandwidth" begin
        # `GenericBartlett` implements the weight alone, so it takes the generic path, and
        # the two paths must agree on the Bartlett weights, at an integer bandwidth and
        # between two integers alike.
        for P in (1, 5, 17, 40), B in (0.3, 1.0, 1.5, 2.7, 4.0, 13.2, 40.0)
            d = randn(StableRNG(P + round(Int, 10B)), P)
            d[2:3:end] .= 0
            buf = zeros(P + 1)
            direct = sum(max(1 - abs(i - j) / B, 0) * d[i] * d[j] for i in 1:P, j in 1:P)
            @test PO.long_run_quadratic_form(BartlettKernel(), d, B, buf) ≈ direct
            @test PO.long_run_quadratic_form(GenericBartlett(), d, B, buf) ≈ direct
        end
    end

    @testset "The statistic is the hand computation, gaps and ends included" begin
        for v in ([0.10, 0.18, 0.26, 0.30, 0.24, 0.16, 0.08, 0.04, 0.12, 0.22, 0.28, 0.20],
                  [0.3, NaN, 0.4, 0.2, NaN, 0.5, 0.1, 0.0, 0.3, -0.1],
                  [NaN, NaN, 0.2, -0.1, 0.4, NaN, 0.3, 0.1, NaN, NaN, NaN]),
            b in (0.1, 0.37, 1.0), k in (BartlettKernel(), QuadraticSpectralKernel())

            mt = WildBootstrapMeanTest(; kernel = k, b = b, n_sim = 99, seed = 2)
            @test PO.mean_test(mt, v, 1, PO.mean_test_rng(mt)).t_stat ≈ hand_t(v, b, k)
            mg = WildBootstrapMeanTest(; kernel = k, b = b, n_sim = 99, seed = 2,
                                       span = FullGridSpan())
            @test PO.mean_test(mg, v, 1, PO.mean_test_rng(mg)).t_stat ≈
                  hand_t(v, b, k; grid = true)
        end
        # A bandwidth below one entry reads no autocovariance. The variance then divides by
        # `n`, so the statistic is the plain one times `sqrt(n / (n - 1))`.
        v = randn(StableRNG(5), 40)
        mt = WildBootstrapMeanTest(; b = 0.01, n_sim = 19, seed = 1)
        s = PO.exposure_ic_factor_summary(hcat(v), 1; mt = mt)
        @test s.t_stat ≈ s.ic_ir * sqrt(40) * sqrt(40 / 39)
        # The test reads the window of the grid only in the history rule.
        @test PO.mean_test(mt, v, 6, PO.mean_test_rng(mt)) ==
              PO.mean_test(mt, v, 1, PO.mean_test_rng(mt))
        @test_throws DomainError PO.mean_test(mt, v, 0, PO.mean_test_rng(mt))
        @test_throws DomainError PO.mean_test(OverlapMeanTest(), v, 0, nothing)
    end

    @testset "The bootstrap is the draw-by-draw computation, on both paths" begin
        v = [0.3, NaN, 0.4, 0.2, NaN, 0.5, 0.1, 0.0, 0.3, -0.1, 0.2, 0.25, NaN, 0.15]
        for k in (BartlettKernel(), QuadraticSpectralKernel(), GenericBartlett()),
            q in (0.05, 0.29)

            mt = WildBootstrapMeanTest(; kernel = k, alg = ObservationDraws(), b = 0.37,
                                       n_sim = 199, q = q)
            a = PO.mean_test(mt, v, 1, StableRNG(8))
            h = hand_bootstrap(v, 0.37, k, 199, q, StableRNG(8))
            @test a.t_stat ≈ h.t_stat
            @test a.p_value == h.p_value
            @test a.mean_ci_lower ≈ h.mean_ci_lower
            @test a.mean_ci_upper ≈ h.mean_ci_upper
        end
        # The generic path and the Bartlett path take the same draws.
        a = PO.mean_test(WildBootstrapMeanTest(; kernel = GenericBartlett(), n_sim = 499),
                         v, 1, StableRNG(1))
        b = PO.mean_test(WildBootstrapMeanTest(; n_sim = 499), v, 1, StableRNG(1))
        @test a.p_value == b.p_value
        @test a.mean_ci_lower ≈ b.mean_ci_lower && a.mean_ci_upper ≈ b.mean_ci_upper
    end

    @testset "The p-value is a count of draws, and a seed repeats it" begin
        v = randn(StableRNG(11), 50) .+ 0.2
        mt = WildBootstrapMeanTest(; n_sim = 499, seed = 4)
        p = PO.mean_test(mt, v, 1, PO.mean_test_rng(mt)).p_value
        @test 1 / 500 <= p <= 1
        @test isinteger(p * 500)
        @test p == PO.mean_test(mt, v, 1, PO.mean_test_rng(mt)).p_value
        # A strong mean is far in the tail of every draw.
        @test PO.mean_test(mt, v .+ 5, 1, PO.mean_test_rng(mt)).p_value == 1 / 500
        # The divisor moves the t-statistic alone: the same draws give the same p-value and
        # the same interval.
        m1 = WildBootstrapMeanTest(; n_sim = 499, seed = 4, ddof = 1)
        r0 = PO.mean_test(mt, v, 1, PO.mean_test_rng(mt))
        r1 = PO.mean_test(m1, v, 1, PO.mean_test_rng(m1))
        # A larger divisor of the variance gives a smaller statistic: `n - 1` over `n`.
        @test r1.t_stat ≈ r0.t_stat * sqrt(49 / 50)
        @test r1.p_value == r0.p_value
        @test r1.mean_ci_lower ≈ r0.mean_ci_lower && r1.mean_ci_upper ≈ r0.mean_ci_upper
        # A divisor of zero or less has no statistic.
        m9 = WildBootstrapMeanTest(; n_sim = 19, ddof = 50)
        @test all(isnan, values(PO.mean_test(m9, v, 1, PO.mean_test_rng(m9))))
        # One resolved generator per summary: two equal series take different draws.
        s = exposure_ic_summary(hcat(v, v); mt = mt)
        @test s.t_stat[1] == s.t_stat[2]
        @test s.p_value[1] != s.p_value[2]
        @test isequal(s, exposure_ic_summary(hcat(v, v); mt = mt))
    end

    @testset "The interval excludes zero exactly when the p-value is at most q" begin
        mism = 0
        for r in 1:400
            rng = StableRNG(9000 + r)
            x = randn(rng, rand(rng, 8:40)) .+ 0.4 * randn(rng)
            mt = WildBootstrapMeanTest(; n_sim = rand(rng, (19, 99, 199)),
                                       q = rand(rng, (0.05, 0.1, 0.29, 0.44)))
            res = PO.mean_test(mt, x, 1, rng)
            excl = !(res.mean_ci_lower <= 0 <= res.mean_ci_upper)
            mism += excl != (res.p_value <= mt.q)
        end
        @test mism == 0
        # The overlap test centres its interval on the mean, at the quantile of the Student
        # law with `n - 2L - 1` degrees of freedom: thirty entries at a window of three.
        v = randn(StableRNG(12), 30) .+ 0.3
        o = PO.mean_test(OverlapMeanTest(; q = 0.1), v, 3, nothing)
        D = PO.Distributions.TDist(30 - 2 * 2 - 1)
        se = (sum(v) / 30) / o.t_stat
        @test o.p_value ≈ 2 * (1 - PO.Distributions.cdf(D, abs(o.t_stat)))
        @test o.mean_ci_upper - o.mean_ci_lower ≈
              2 * PO.Distributions.quantile(D, 0.95) * se
        @test (o.mean_ci_upper + o.mean_ci_lower) / 2 ≈ sum(v) / 30
    end

    @testset "A series with too little evidence has no test" begin
        mt = WildBootstrapMeanTest(; n_sim = 19, seed = 1)
        for v in ([NaN, NaN], [0.2, NaN], fill(0.3, 6), Float64[])
            for t in (mt, OverlapMeanTest())
                r = PO.mean_test(t, v, 1, PO.mean_test_rng(t))
                @test all(isnan, values(r))
            end
        end
        # A constant series whose sum over its count misses its value by round-off: the
        # centring residue must not make a long-run variance, with or without a gap.
        @test sum(fill(0.03, 100)) / 100 != 0.03
        g = fill(0.03, 100)
        g[[1, 40, 41, 100]] .= NaN
        for v in (fill(0.03, 100), g), t in (mt, OverlapMeanTest())
            @test all(isnan, values(PO.mean_test(t, v, 1, PO.mean_test_rng(t))))
        end
        # An alternating series sums the untapered long-run variance below zero, and the
        # Bartlett one stays positive, so only the overlap test answers NaN.
        v = [0.5, -0.4, 0.5, -0.4, 0.5, -0.4, 0.5, -0.4]
        @test isnan(PO.mean_test(OverlapMeanTest(), v, 2, nothing).t_stat)
        @test isfinite(PO.mean_test(mt, v, 2, PO.mean_test_rng(mt)).t_stat)
        # The minimum-data rules: twenty finite entries, and ten windows in the span.
        w = randn(StableRNG(4), 30)
        w[[3, 9, 17]] .= NaN
        for t in (WildBootstrapMeanTest(; n_sim = 19, min_obs = 27),
                  OverlapMeanTest(; min_obs = 27))
            @test isfinite(PO.mean_test(t, w, 1, PO.mean_test_rng(t)).t_stat)
        end
        for t in (WildBootstrapMeanTest(; n_sim = 19, min_obs = 28),
                  OverlapMeanTest(; min_obs = 28))
            @test all(isnan, values(PO.mean_test(t, w, 1, PO.mean_test_rng(t))))
        end
        # Thirty entries hold ten windows of three and seven and a half windows of four. A
        # window of `5 // 2` entries is a horizon of five at a stride of two: twelve periods.
        t = WildBootstrapMeanTest(; n_sim = 19, min_periods = 10)
        @test isfinite(PO.mean_test(t, w, 3, PO.mean_test_rng(t)).p_value)
        @test isnan(PO.mean_test(t, w, 4, PO.mean_test_rng(t)).p_value)
        t12 = WildBootstrapMeanTest(; n_sim = 19, min_periods = 12)
        @test isfinite(PO.mean_test(t12, w, 5 // 2, PO.mean_test_rng(t12)).p_value)
        @test isnan(PO.mean_test(t12, w, 3, PO.mean_test_rng(t12)).p_value)
        # The span rule decides what the history rule reads: two missing ends.
        z = vcat(NaN, randn(StableRNG(6), 28), NaN)
        tf = WildBootstrapMeanTest(; n_sim = 19, min_periods = 30)
        tg = WildBootstrapMeanTest(; n_sim = 19, min_periods = 30, span = FullGridSpan())
        @test isnan(PO.mean_test(tf, z, 1, PO.mean_test_rng(tf)).t_stat)
        @test isfinite(PO.mean_test(tg, z, 1, PO.mean_test_rng(tg)).t_stat)
        @test PO.mean_test_span_length(FiniteEntrySpan(), 30, 2, 29) == 28
        @test PO.mean_test_span_length(FullGridSpan(), 30, 2, 29) == 30
    end

    @testset "The summaries take the wild bootstrap by default" begin
        y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
        ic = forecast_ic(forecast_evaluation(IC_ALPHA, y))
        s = forecast_ic_summary(ic)
        mt = WildBootstrapMeanTest()
        for (k, sk) in ((1, s.spearman), (2, s.pearson))
            @test isequal(sk.t_stat,
                          PO.mean_test(mt, ic[:, k], 1, PO.mean_test_rng(mt)).t_stat)
            @test isnan(sk.p_value) || 0 < sk.p_value <= 1
        end
        # Every figure but the test is the one the overlap test reports beside it.
        o = forecast_ic_summary(ic; mt = OverlapMeanTest())
        for f in (:mean_ic, :std_ic, :ic_ir, :hit_rate)
            @test isequal(getfield(s.spearman, f), getfield(o.spearman, f))
        end
        rng = StableRNG(7)
        alpha = randn(rng, 40, 6)
        X = randn(rng, 40, 6)
        fe = forecast_evaluation(alpha, PO.forward_mean_returns(X, 1, 1))
        mt = WildBootstrapMeanTest(; n_sim = 199, seed = 3)
        fs = forecast_evaluation_summary(fe; mt = mt)
        f1 = forecast_ic_summary(forecast_ic(fe); mt = mt)
        @test fs.spearman_t_stat[1] == f1.spearman.t_stat
        @test fs.spearman_p_value[1] == f1.spearman.p_value
        @test fs.pearson_p_value[1] == f1.pearson.p_value
        @test fs.spearman_mean_ci_lower[1] == f1.spearman.mean_ci_lower
        @test fs.pearson_mean_ci_upper[1] == f1.pearson.mean_ci_upper
        h = forecast_holding_period(fe, X; n = 3, mt = mt)
        d = forecast_decay(fe, X; n = 3, mt = mt)
        @test all(p -> 0 < p <= 1, h.spearman_p_value)
        @test all(h.spearman_mean_ci_lower .<
                  h.spearman_mean_ic .<
                  h.spearman_mean_ci_upper)
        @test h.spearman_p_value[1] == d.spearman_p_value[1]
        @test h.pearson_mean_ci_lower[1] == d.pearson_mean_ci_lower[1]
        e = exposure_ic_summary(ic; mt = mt)
        @test e.mean_ci_lower[1] < e.mean_ic[1] < e.mean_ci_upper[1]
    end

    @testset "The window states the holding period exactly, and agrees with the lag" begin
        @test PO.forecast_ic_window(5, 2) == 5 // 2
        @test PO.forecast_ic_window(5, 5) == 1
        @test PO.forecast_ic_window(3, 5) == 3 // 5
        @test_throws DomainError PO.forecast_ic_window(0, 1)
        @test_throws DomainError PO.forecast_ic_window(1, 0)
        for (h, st) in ((5, 2), (4, 2), (3, 1), (2, 5), (7, 3))
            @test ceil(Int, PO.forecast_ic_window(h, st)) - 1 == PO.forecast_ic_lags(h, st)
        end
        y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
        fe = forecast_evaluation(IC_ALPHA, y; horizon = 3, step = 2)
        @test PO.forecast_ic_window(fe) == 3 // 2
        ic = randn(StableRNG(2), 30, 2)
        o = OverlapMeanTest()
        # A window of five halves overlaps two entries on each side.
        @test isequal(PO.exposure_ic_factor_summary(ic, 1; lags = 2, window = 5 // 2,
                                                    mt = o),
                      PO.exposure_ic_factor_summary(ic, 1; window = 5 // 2, mt = o))
        @test isequal(PO.exposure_ic_factor_summary(ic, 1; window = 5 // 2, mt = o).t_stat,
                      PO.exposure_ic_factor_summary(ic, 1; lags = 2, mt = o).t_stat)
        @test_throws PO.ConflictingArgumentError PO.exposure_ic_factor_summary(ic, 1;
                                                                               lags = 1,
                                                                               window = 5 //
                                                                                        2,
                                                                               mt = o)
        @test_throws PO.ConflictingArgumentError forecast_ic_summary(ic; lags = 3,
                                                                     window = 2, mt = o)
        @test_throws DomainError exposure_ic_summary(ic; window = 0, mt = o)
        # The history rule reads the exact window: thirty dates at a horizon of five and a
        # stride of two cover twelve holding periods, and a window rounded up to three covers
        # ten.
        t = OverlapMeanTest(; min_periods = 11)
        @test isfinite(forecast_ic_summary(ic; window = 5 // 2, mt = t).spearman.t_stat)
        @test isnan(forecast_ic_summary(ic; lags = 2, mt = t).spearman.t_stat)
    end

    @testset "A no-skill series persistent past the overlap keeps its size" begin
        # An autoregressive series of mean zero has no overlap, so the overlap test reads it
        # as independent and rejects far too often. The wild bootstrap reads the persistence.
        R = 300
        T = 60
        over = 0
        wild = 0
        mt = WildBootstrapMeanTest(; n_sim = 199)
        for r in 1:R
            rng = StableRNG(5000 + r)
            v = ar1(rng, T, 0.5)
            over += PO.mean_test(OverlapMeanTest(), v, 1, nothing).p_value <= 0.05
            wild += PO.mean_test(mt, v, 1, rng).p_value <= 0.05
        end
        @test over / R > 0.18
        @test wild / R < 0.11
    end

    @testset "The test keeps the element type of the series" begin
        v = Float32[0.3, NaN, 0.4, 0.2, NaN, 0.5, 0.1, 0.0, 0.3, -0.1]
        # The bandwidth is `b * T`, so the statistic promotes the series with `b`: a
        # `Float32` fraction keeps the type of the series, and the default `Float64` one
        # promotes it.
        mt = WildBootstrapMeanTest(; b = 0.4f0, n_sim = 19, seed = 1)
        @test PO.mean_test(mt, v, 1, PO.mean_test_rng(mt)).t_stat isa Float32
        md = WildBootstrapMeanTest(; n_sim = 19, seed = 1)
        @test PO.mean_test(md, v, 1, PO.mean_test_rng(md)).t_stat isa Float64
        @test PO.mean_test(OverlapMeanTest(), v, 2, nothing).t_stat isa Float32
        s = exposure_ic_summary(hcat(v, v); mt = mt)
        @test eltype(s.p_value) == Float32
        @test eltype(s.mean_ci_lower) == Float32
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

    @testset "Every figure of the summary, the hit rate included, drops a NaN" begin
        # A `NaN` coefficient is a date at which nothing was measured, not a miss, so the hit
        # rate is read against the dates that carried a score, as the mean, the ratio and the
        # t-statistic beside it are, and as the oracle reads it. Until 2026-09-14 the hit
        # rate alone counted every date and reported 1/4 here.
        ic = forecast_ic(fe)
        s = forecast_ic_summary(ic; mt = OverlapMeanTest())
        v = filter(isfinite, ic[:, 1])
        @test length(v) == 2
        @test count(>(0), v) == 1
        @test s.spearman.hit_rate == 1 / 2
        @test s.spearman.hit_rate == count(>(0), v) / length(v)
        @test s.spearman.t_stat ≈ s.spearman.ic_ir * sqrt(2)
    end

    @testset "A threshold no date reaches gives no score, and no hit rate" begin
        none = forecast_ic(fe; min_count = 5)
        @test all(isnan, none)
        s = forecast_ic_summary(none; mt = OverlapMeanTest())
        @test isnan(s.spearman.mean_ic)
        @test isnan(s.spearman.std_ic)
        @test isnan(s.spearman.ic_ir)
        @test isnan(s.spearman.t_stat)
        @test isnan(s.spearman.hit_rate)
    end
end

@testset "The coverage says what share of the universe was scored" begin
    PO = PortfolioOptimisers
    gap = ic_gap_fixture()
    fg = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 2)

    @testset "It matches the oracle, one entry per evaluation date" begin
        c = forecast_coverage(fg)
        @test length(c) == length(fg.dates)
        @test parity_compare(c, IC_REF_GAP_COVERAGE; rtol = 1e-14, name = "coverage gap").ok
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

    @testset "The universe mask is the denominator, and the bare method defaults it to every asset" begin
        # #1070. The gapped fixture's two `NaN` cells are outside this mask, so the share is
        # one at both dates: a cell the panel does not admit is not a missed one. The
        # numerator the summary reports is counted inside the same mask.
        @test fg.umsk == trues(size(gap))
        umsk = [true true true true; true false true false; true false true true;
                true true true true]
        fm = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); umsk = umsk,
                                 min_count = 2)
        @test fm.umsk === umsk
        @test forecast_coverage(fm) == [0.75, 1.0, 1.0]
        @test PO.forecast_summary_scored(fm, PO.forecast_ic_weights(gap, nothing)) ==
              [3.0, 2.0, 3.0]
        # A weight history narrows the universe further, never widens it.
        u = [1.0 1.0 1.0 1.0; 1.0 1.0 0.0 1.0; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
        @test forecast_coverage(fm, u) == [0.75, 1.0, 1.0]
        @test PO.forecast_summary_scored(fm, u) == [3.0, 1.0, 3.0]
        @test_throws DimensionMismatch forecast_evaluation(gap,
                                                           PO.forward_mean_returns(gap, 1,
                                                                                   1);
                                                           umsk = trues(4, 3))
    end

    @testset "On a point-in-time panel the universe is the estimation mask, so a late lister is not a lost Descriptor" begin
        # #1070. The planted fixture lists assets late, and its block's listed count runs
        # from 10 to 19 over the evaluation dates. Every listed asset is scored at every date,
        # so the share is one throughout and the count is what moves. Divided by every asset
        # the panel ever held, the same dates read as low as one half, which is the port's
        # defect the oracle does not share.
        px = evaluation_fixture(; planted = true)
        fw = FixedWeightedReturnForecast(; scores = px.scores, scale = 1.0,
                                         weights = [0.4, 0.6])
        fe = forecast_evaluation(fw, px.rd, px.csfm; horizon = 2, lag = 1)
        @test fe.umsk == px.rd.pnl.emsk[px.rows, :]
        listed = [count(view(fe.umsk, t, :)) for t in fe.dates]
        @test first(listed) < last(listed)
        @test all(==(1), forecast_coverage(fe))
        n = PO.forecast_summary_scored(fe, PO.forecast_ic_weights(fe.alpha, nothing))
        @test n == listed
        @test minimum(n ./ px.N) < 0.6
        cv = PO.forecast_summary_coverage(fe, nothing)
        @test cv.mean_coverage == 1
        @test cv.min_coverage == 1
        @test cv.min_n_scored == first(listed)
        # The window tables inherit the same universe.
        h = forecast_holding_period(fe, px.rd, px.csfm; n = 2, mt = OverlapMeanTest())
        @test h.mean_coverage[1] == 1
    end
end

@testset "The exposure summary gained the same t-statistic" begin
    PO = PortfolioOptimisers
    # One statistic, one kernel: `exposure_ic_factor_summary` is what both summaries read,
    # so the exposure diagnostics report a t-statistic on the same terms rather than
    # diverging from the evaluation.
    ic = [0.1 0.4; -0.2 NaN; 0.3 0.5; NaN 0.1]
    s = PO.exposure_ic_summary(ic; mt = OverlapMeanTest())
    @test keys(s) ==
          (:mean_ic, :std_ic, :ic_ir, :t_stat, :p_value, :mean_ci_lower, :mean_ci_upper,
           :hit_rate)
    for k in 1:2
        m = PO.exposure_ic_factor_summary(ic, k; mt = OverlapMeanTest())
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
        s = forecast_ic_summary(forecast_ic(fe); mt = OverlapMeanTest())
        @test s.spearman.mean_ic > 0.5
        @test s.spearman.t_stat > 20
        @test s.spearman.hit_rate > 0.9
        @test s.pearson.mean_ic > 0.5
        # The panel drops 8% of its cells, so a refit member that needs both Descriptors
        # scores fewer assets than the estimation mask admits on some dates, and the coverage
        # is what says so. Its listings move the universe itself, not the coverage (#1070).
        cv = forecast_coverage(fe)
        @test all(x -> 0.4 < x <= 1, cv)
        @test sum(cv) / length(cv) > 0.75
        @test any(x -> x < 1, cv)
    end

    @testset "The unplanted fixture reports none, which is what proves the planted one" begin
        fx = evaluation_fixture()
        fe = forecast_evaluation(tgt_of(fx), fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1)
        s = forecast_ic_summary(forecast_ic(fe); mt = OverlapMeanTest())
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

    @testset "Equal forecasts take equal `:rank` weights under the default tie rule" begin
        at = [1.0 1.0 2.0 3.0; 2.0 1.0 1.0 3.0; 1.0 2.0 3.0 4.0]
        yt = [1.0 2.0 3.0 4.0; 4.0 3.0 2.0 1.0; 1.0 3.0 2.0 4.0]
        wa = forecast_portfolio(forecast_evaluation(at, yt)).w
        wo = forecast_portfolio(forecast_evaluation(at, yt; ties = :ordinal)).w
        @test wa[1, 1] == wa[1, 2]
        @test wa[2, 2] == wa[2, 3]
        @test wo[1, 1] < wo[1, 2]
        @test wo[2, 2] < wo[2, 3]
        # A row with no tie is the same under both rules, and both books stay dollar
        # neutral at 200 % gross.
        @test wa[3, :] == wo[3, :]
        @test all(k -> isapprox(sum(wa[k, :]), 0; atol = 1e-12), 1:3)
        @test all(k -> sum(abs, wa[k, :]) ≈ 2, 1:3)
        @test wa == PO.forecast_portfolio_weights(at, yt, 1:3, :rank, :average)
        @test wo == PO.forecast_portfolio_weights(at, yt, 1:3, :rank, :ordinal)
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
                                                                               :inverse_vol,
                                                                               :average)
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
        @test p4.summary.ppy == 4
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
        @test p.summary == performance_summary(p.ret[isfinite.(p.ret)]; ppy = 1)
    end

    @testset "The uncompressed series is what the compression avoids" begin
        # `performance_summary`'s Precomputed-returns contract: a `NaN` makes the mean and
        # the drawdown non-finite, and the tail figure answers a number rather than a `NaN`.
        raw = performance_summary(p.ret; ppy = 1)
        @test isnan(raw.ann_return)
        @test isnan(raw.max_drawdown)
        @test isfinite(raw.cvar)
        @test isfinite(p.summary.max_drawdown)
        @test p.summary.max_drawdown < 0
    end

    @testset "The turnover into the gap is dropped with it, and the one out of it is kept" begin
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
    # The statistic is contemporaneous, so the verb reads every observation of the
    # forecast, not the evaluation grid: the last row has no forward target and is off
    # `fe.dates`, and it is scored here all the same. Issue #1071.
    T = size(FC_ALPHA, 1)

    @testset "A pure multiple of an exposure correlates one with it and zero with an orthogonal one" begin
        @test length(fe.dates) < T
        @test size(c) == (T, 2)
        @test c[:, 1] ≈ ones(T)
        @test c[:, 2] ≈ zeros(T) atol = 1e-12
    end

    @testset "The rank form agrees where the design is rank-orthogonal too" begin
        cr = forecast_factor_correlation(fe, FC_B; rank = true)
        @test cr[:, 1] ≈ ones(T)
        @test cr[:, 2] ≈ zeros(T) atol = 1e-12
    end

    @testset "Each entry is the cross-sectional helper applied to the observation" begin
        cr = forecast_factor_correlation(fe, FC_B; rank = true)
        u = ones(size(FC_ALPHA, 2))
        for k in 1:2, t in 1:T
            a = view(fe.alpha, t, :)
            b = view(FC_B, t, :, k)
            @test c[t, k] == PO.cs_weighted_correlation(a, b, u; min_count = fe.min_count)
            @test cr[t, k] == PO.cs_spearman_correlation(a, b; min_count = fe.min_count)
        end
    end

    @testset "A weighting moves the weighted form and leaves the rank one" begin
        cw = forecast_factor_correlation(fe, FC_B, FC_W)
        # A perfect linear relation is weight invariant, and an orthogonal one is not.
        @test cw[:, 1] ≈ ones(T)
        @test !isapprox(cw[:, 2], zeros(T); atol = 1e-12)
        @test isequal(forecast_factor_correlation(fe, FC_B, FC_W; rank = true),
                      forecast_factor_correlation(fe, FC_B; rank = true))
    end

    @testset "A cross-section under the threshold carries no correlation" begin
        gap = copy(FC_ALPHA)
        gap[2, 3] = NaN
        gap[2, 4] = NaN
        fg = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 2)
        cg = forecast_factor_correlation(fg, FC_B; min_count = 3)
        @test all(isnan, cg[2, :])
        @test all(isfinite, forecast_factor_correlation(fg, FC_B; min_count = 2)[2, :])
    end

    @testset "A weight removes an asset, and a constant cross-section is NaN in the weighted form only" begin
        # Issue #951. Every asset carries a finite pair, so each NaN below comes from the
        # weight or from the constant exposure, and not from a missing value.
        w0 = ones(size(FC_ALPHA))
        w0[1, 2] = 0.0
        w0[2, 3] = NaN
        cz = forecast_factor_correlation(fe, FC_B, w0; min_count = 4)
        @test all(isnan, cz[1:2, :])
        @test all(isfinite, cz[3:T, :])
        # A constant exposure has no weighted correlation. Under the default tie rule its
        # ranks are equal, so it has no rank correlation either. Under `:ordinal`, equal
        # values take their position on the asset axis, so the rank form answers the rank
        # correlation of the forecast with that order.
        bc = cat(FC_TIED, fill(1.0, size(FC_ALPHA)); dims = 3)
        @test all(isnan, forecast_factor_correlation(fe, bc)[:, 2])
        @test all(isnan, forecast_factor_correlation(fe, bc; rank = true)[:, 2])
        fo = forecast_evaluation(FC_ALPHA, PO.forward_mean_returns(FC_ALPHA, 1, 1);
                                 ties = :ordinal)
        crc = forecast_factor_correlation(fo, bc; rank = true)
        @test crc[:, 2] == crc[:, 1]
        @test crc[:, 2] ≈ ones(T)
    end

    @testset "The evaluation grid is opt-in, and any row set is read the same way" begin
        # `dates = fe.dates` is the row subset of the whole-axis answer, entry for entry,
        # so a caller who wants the correlations beside the coefficients of the same dates
        # has them; and a row set of their own is read on the same terms.
        @test isequal(forecast_factor_correlation(fe, FC_B; dates = fe.dates),
                      c[fe.dates, :])
        @test isequal(forecast_factor_correlation(fe, FC_B; dates = 1:T), c)
        @test isequal(forecast_factor_correlation(fe, FC_B; dates = [T, 1]), c[[T, 1], :])
        @test isequal(forecast_factor_correlation(fe, FC_B, FC_W; dates = fe.dates,
                                                  rank = true),
                      forecast_factor_correlation(fe, FC_B; rank = true)[fe.dates, :])
    end

    @testset "The rank form ranks a tie by `fe.ties`, and `:average` reads no asset order" begin
        # Every row of `FC_G` is one-hot, two blocks of two, so every asset is in a tie. A
        # reversal of the asset axis moves the `:ordinal` answer and not the default one.
        yt = PO.forward_mean_returns(FC_ALPHA, 1, 1)
        rv = size(FC_ALPHA, 2):-1:1
        fo = forecast_evaluation(FC_ALPHA, yt; ties = :ordinal)
        far = forecast_evaluation(FC_ALPHA[:, rv], yt[:, rv])
        fro = forecast_evaluation(FC_ALPHA[:, rv], yt[:, rv]; ties = :ordinal)
        ca = forecast_factor_correlation(fe, FC_BG; rank = true)
        co = forecast_factor_correlation(fo, FC_BG; rank = true)
        @test ca ≈ forecast_factor_correlation(far, FC_BG[:, rv, :]; rank = true)
        @test !isapprox(co, forecast_factor_correlation(fro, FC_BG[:, rv, :]; rank = true))
        # Row 1: the ranks of `[1, 2, 3, 4]` against the midranks of `[1, 0, 1, 0]`.
        @test ca[1, 1] ≈ -2 / sqrt(20)
        @test co[1, 1] ≈ 0.0 atol = 1e-12
        for k in 1:2, t in 1:T
            @test co[t, k] ==
                  PO.cs_spearman_correlation(view(fe.alpha, t, :), view(FC_BG, t, :, k);
                                             min_count = fe.min_count, ties = :ordinal)
        end
    end

    @testset "An empty row set and a row off the axis are refused" begin
        @test_throws PO.IsEmptyError forecast_factor_correlation(fe, FC_B; dates = Int[])
        @test_throws DomainError forecast_factor_correlation(fe, FC_B; dates = [1, T + 1])
        @test_throws DomainError forecast_factor_correlation(fe, FC_B; dates = 0:T)
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
        @test isequal(forecast_factor_correlation(fb, csfm).X,
                      forecast_factor_correlation(fb, PO.cs_diagnostic_exposures(csfm)))
        @test size(forecast_factor_correlation(fb, csfm).X) ==
              (size(fb.alpha, 1), length(csfm.nf))
    end

    @testset "The answer carries the names and the families of the raw axis" begin
        c = forecast_factor_correlation(fb, csfm)
        @test c.nf == csfm.nf && isequal(c.fam, csfm.fam)
        # The summary of the answer keeps the labels, and a view by name selects one
        # factor of both.
        s = exposure_ic_summary(c; mt = OverlapMeanTest())
        @test s.nf == csfm.nf
        k = csfm.nf[end]
        cv = PO.port_opt_view(c, [k])
        @test isequal(cv.X, c.X[:, end:end])
        @test isequal(exposure_ic_summary(cv; mt = OverlapMeanTest()).mean_ic,
                      PO.port_opt_view(s, [k]).mean_ic)
    end

    @testset "The row set passes through to the bare method" begin
        @test isequal(forecast_factor_correlation(fb, csfm; dates = fb.dates).X,
                      forecast_factor_correlation(fb, PO.cs_diagnostic_exposures(csfm);
                                                  dates = fb.dates))
        @test isequal(forecast_factor_correlation(fb, csfm; dates = fb.dates).X,
                      forecast_factor_correlation(fb, csfm).X[fb.dates, :])
    end

    @testset "A weighting moves the weighted form and leaves the rank one" begin
        cw = forecast_factor_correlation(fb, csfm;
                                         weighting = InverseIdiosyncraticVarianceMetric()).X
        @test !isequal(cw, forecast_factor_correlation(fb, csfm).X)
        @test isequal(forecast_factor_correlation(fb, csfm; rank = true,
                                                  weighting = InverseIdiosyncraticVarianceMetric()).X,
                      forecast_factor_correlation(fb, csfm; rank = true).X)
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
    cr = forecast_factor_correlation(forecast_evaluation(raw, rd, csfm; horizon = 2), csfm).X
    cn = forecast_factor_correlation(forecast_evaluation(neu, rd, csfm; horizon = 2), csfm).X
    mraw = abs(sum(filter(isfinite, view(cr, :, 1))) / count(isfinite, view(cr, :, 1)))
    mneu = abs(sum(filter(isfinite, view(cn, :, 1))) / count(isfinite, view(cn, :, 1)))

    @testset "The un-neutralised forecast restates the style factor" begin
        @test mraw > 0.9
    end

    @testset "The neutralised one is far less correlated with it (#1521)" begin
        # THE ASSERTION IS DIRECTIONAL. The default `cre` fits an intercept, so each score
        # leaves the regression uncorrelated with the style exposure, as the next testset
        # pins. The grouped scoring step after the fit rescales each industry on its own,
        # and the evaluation weighs the active assets that the estimation mask leaves out,
        # so the forecast keeps a small correlation.
        @test dn.cre.intercept
        @test mneu < 0.4
    end

    @testset "`intercept = false`, the oracle's rule, keeps a large correlation (#950)" begin
        # The residual is orthogonal to the target in the UNCENTRED sense alone, so it keeps
        # a large Pearson correlation with it.
        dn0 = DescriptorScores(; descriptors = ds.descriptors, neutralise = "style",
                               cre = CrossSectionalLinearRegression(; intercept = false),
                               outlier = ds.outlier, scoring = ds.scoring, group = ds.group)
        ne0 = FixedWeightedReturnForecast(; scores = dn0, scale = 1.0, weights = [0.4, 0.6])
        cn0 = forecast_factor_correlation(forecast_evaluation(ne0, rd, csfm; horizon = 2),
                                          csfm).X
        mne0 = abs(sum(filter(isfinite, view(cn0, :, 1))) /
                   count(isfinite, view(cn0, :, 1)))
        @test mneu < mne0 < mraw
        @test mne0 > 0.5
    end

    @testset "Each default score is uncorrelated with the target to round-off (#1521)" begin
        # The scores themselves, under the weights of the Neutralisation, with an ungrouped
        # standardiser after the fit: it maps each cross-section by one affine map, so it
        # keeps the zero weighted mean and the orthogonality that the intercept gives.
        PO = PortfolioOptimisers
        X = csfm.Ms[:, :, 1]
        function worst(cre)
            d = DescriptorScores(; descriptors = ds.descriptors, neutralise = "style",
                                 cre = cre, outlier = ds.outlier, scoring = ds.scoring)
            r = descriptor_scores(d, rd, csfm)
            w = PO.return_forecast_weights(rd)[r.rows, :]
            wc = 0.0
            wo = 0.0
            for k in axes(r.S, 3), t in axes(X, 1)
                s = r.S[r.rows[t], :, k]
                m = isfinite.(s) .& isfinite.(view(X, t, :)) .& (view(w, t, :) .> 0)
                u, s, x = w[t, m], s[m], X[t, m]
                sc = s .- sum(u .* s) / sum(u)
                xc = x .- sum(u .* x) / sum(u)
                wc = max(wc,
                         abs(sum(u .* sc .* xc)) /
                         sqrt(sum(u .* sc .^ 2) * sum(u .* xc .^ 2)))
                wo = max(wo,
                         abs(sum(u .* s .* x)) / sqrt(sum(u .* s .^ 2) * sum(u .* x .^ 2)))
            end
            return (; corr = wc, orth = wo)
        end
        d1 = worst(CrossSectionalLinearRegression(; intercept = true))
        d0 = worst(CrossSectionalLinearRegression(; intercept = false))
        @test d1.corr < 1e-12
        @test d1.orth < 1e-12
        # Without the intercept the standardiser re-centres a residual whose weighted mean
        # is not zero, so neither property survives it.
        @test d0.corr > 0.5
        @test d0.orth > 0.3
    end
end

@testset "The correlation series is summarised by the shared kernel" begin
    PO = PortfolioOptimisers
    # `FC_BG` varies by date, so the series has dispersion and the ratio and the
    # t-statistic are numbers rather than the `NaN` a constant series earns.
    fe = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 1, 1))
    c = forecast_factor_correlation(fe, FC_BG)
    s = exposure_ic_summary(c; mt = OverlapMeanTest())

    @testset "No third summary ships, and the kernel answers one entry per factor" begin
        @test keys(s) ==
              (:mean_ic, :std_ic, :ic_ir, :t_stat, :p_value, :mean_ci_lower, :mean_ci_upper,
               :hit_rate)
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
            m = PO.exposure_ic_factor_summary(c, k; mt = OverlapMeanTest())
            @test isequal(m.mean_ic, s.mean_ic[k])
            @test isequal(m.std_ic, s.std_ic[k])
            @test isequal(m.ic_ir, s.ic_ir[k])
            @test isequal(m.t_stat, s.t_stat[k])
            @test isequal(m.hit_rate, s.hit_rate[k])
        end
        # The two-column case is exactly what `forecast_ic_summary` names, so the two
        # summaries agree figure for figure on the same input.
        fs = forecast_ic_summary(c; mt = OverlapMeanTest())
        @test isequal(fs.spearman,
                      PO.exposure_ic_factor_summary(c, 1; mt = OverlapMeanTest()))
        @test isequal(fs.pearson,
                      PO.exposure_ic_factor_summary(c, 2; mt = OverlapMeanTest()))
    end

    @testset "A constant series is summarised, and its ratio has no answer" begin
        # The `FC_B` design is constant across dates by construction, which is what pins
        # the exact `1` and `0`; a series with no dispersion has no ratio.
        cc = forecast_factor_correlation(forecast_evaluation(FC_ALPHA,
                                                             PO.forward_mean_returns(FC_ALPHA,
                                                                                     1, 1)),
                                         FC_B)
        sc = exposure_ic_summary(cc; mt = OverlapMeanTest())
        @test sc.mean_ic ≈ [1.0, 0.0] atol = 1e-12
        @test sc.std_ic ≈ [0.0, 0.0] atol = 1e-12
        @test sc.hit_rate ≈ [1.0, 0.0]
        @test all(isnan, sc.ic_ir)
        @test all(isnan, sc.t_stat)
    end
end

# The oracle's factor diagnostics on the planted fixture of the testset
# below, run on 2026-09-14 through its own `_compute_factor_correlation_diagnostics` and
# `_correlation_stats` over the fixture's exported forecast history, exposure history and
# estimation mask, with equal weights, Pearson and `min_count = 3`. The style factor's
# column, and its five-figure summary. Issues #1071 and #1074.
const FC_REF_COUNT = 52
const FC_REF_STYLE = [0.9874127453938115, 0.9870981609818839, 0.9865119263442621,
                      0.9921459947686977, 0.9915404370359904, 0.9912131795260033,
                      0.9917956248307396, 0.9885403249092338, 0.9959730507424697,
                      0.9882258345887717, 0.9884379212434494, 0.9884548481476073,
                      0.9884646748783673, 0.9852190219604461, 0.9901001320352284,
                      0.9800736981924713, 0.9813372828078473, 0.9812620555718751,
                      0.9810661512449548, 0.9806037177644529, 0.9774016331244408,
                      0.9804536126708862, 0.9806921820687978, 0.9780612297890581,
                      0.9766512225743983, 0.981351648031914, 0.9796165970862136,
                      0.9810342619366895, 0.9807490363096569, 0.9811884221947792,
                      0.9810726555341553, 0.9808378426058264, 0.9811471882269012,
                      0.9812827692837984, 0.9816927433790472, 0.9822828091727407,
                      0.9817288190852552, 0.9806598237162722, 0.9818546753077076,
                      0.9818547174170653, 0.9819585143062083, 0.9820684711165474,
                      0.9818545709323333, 0.982065434202447, 0.981810669952924,
                      0.982400091394535, 0.9822964949272698, 0.982548482602381,
                      0.9826176358134895, 0.982818893768118, 0.982554103550104,
                      0.983025494100696]
const FC_REF_SUMMARY = (; mean = 0.9835597986375235, std = 0.004192196953212379,
                        ir = 234.61679153309947, t_stat = 1691.8457439148713,
                        hit_rate = 1.0)

@testset "A contemporaneous statistic is read on every observation, as the oracle reads it" begin
    PO = PortfolioOptimisers
    # Issue #1071. The oracle correlates the forecast against every
    # exposure over the whole forecast history and summarises it over that count; it has no
    # evaluation grid for a statistic that looks nowhere forward. The port read `fe.dates`
    # only, so under `step = horizon` it saw `1 / horizon` of the observations and its
    # t-statistic was smaller by the root of that ratio. The planted fixture carries a
    # signal, `FixedWeightedReturnForecast` writes a forecast on every row, and
    # `horizon = 5` makes the two grids far apart.
    px = evaluation_fixture(; planted = true)
    fw = FixedWeightedReturnForecast(; scores = px.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    fe = forecast_evaluation(fw, px.rd, px.csfm; horizon = 5, lag = 1)
    c = forecast_factor_correlation(fe, px.csfm).X
    cg = forecast_factor_correlation(fe, px.csfm; dates = fe.dates).X
    s = exposure_ic_summary(c; mt = OverlapMeanTest())
    sg = exposure_ic_summary(cg; mt = OverlapMeanTest())

    @testset "The whole axis carries every finite pairing, and the grid a fraction of it" begin
        @test fe.step == 5
        @test size(c, 1) == size(fe.alpha, 1)
        @test size(cg, 1) == length(fe.dates)
        @test count(isfinite, view(c, :, 1)) == FC_REF_COUNT
        @test count(isfinite, view(cg, :, 1)) == length(fe.dates)
        @test count(isfinite, view(c, :, 1)) > 4 * count(isfinite, view(cg, :, 1))
    end

    @testset "The mean barely moves and the evidence for it does" begin
        @test isapprox(s.mean_ic[1], sg.mean_ic[1]; atol = 0.01)
        @test s.t_stat[1] > 2 * sg.t_stat[1]
    end

    @testset "The oracle's factor diagnostics are reproduced" begin
        # The default of the verb is parity: the same kernel over the same axis answers
        # the oracle's column to the last bit, and the shared summary kernel its five
        # figures. The hit rate leaves a `NaN` coefficient out of its denominator on both
        # sides (ADR 0149); this fixture scores every row, so the last assertion states
        # that no row is left out.
        # The oracle masks the forecast by the estimation mask before it correlates,
        # and the pairing writes the forecast onto that mask once (#1074); the fixture
        # carries one active asset off the mask, so the pins would miss by up to `0.098`
        # in a cell were either side to read it.
        # Measured cell by cell, with and without the flags of `Pkg.test` (#1558): the column
        # 5.6e-16 at most, the summary 7.9e-15 at most. The column
        # sits near 0.98 with a spread of 0.004, so its standard deviation cancels: the ratio
        # is 235 and the ratio and the t-statistic carry about 235 times the round-off.
        @test parity_compare(c[:, 1], FC_REF_STYLE; rtol = 1e-14, name = "factor column").ok
        @test parity_compare([s.mean_ic[1], s.std_ic[1], s.ic_ir[1], s.t_stat[1]],
                             [FC_REF_SUMMARY.mean, FC_REF_SUMMARY.std, FC_REF_SUMMARY.ir,
                              FC_REF_SUMMARY.t_stat]; rtol = 5e-14, name = "factor summary").ok
        @test s.hit_rate[1] == FC_REF_SUMMARY.hit_rate
        @test count(isfinite, view(c, :, 1)) == size(c, 1)
    end

    @testset "A member refit along the grid carries no correlation off it" begin
        # `forecast_history` writes such a member's forecast on its refit grid only, which
        # is anchored at the block's first row and strides by `step` through its last, so
        # the whole axis answers `NaN` on every row the member was never asked for and a
        # correlation on every row it was. `fe.dates` is the scorable subset of that grid:
        # it stops where the forward target does, and the last written row lies past it,
        # so the whole axis scores one row the grid never sees.
        tgt = TargetReturnForecast(; scores = px.scores, horizon = 2, lag = 1,
                                   calibrate = false)
        ft = forecast_evaluation(tgt, px.rd, px.csfm; horizon = 2, lag = 1)
        ct = forecast_factor_correlation(ft, px.csfm).X
        T = size(ft.alpha, 1)
        written = [any(isfinite, view(ft.alpha, t, :)) for t in 1:T]
        @test findall(written) == collect(1:ft.step:T)
        @test ft.dates ⊆ findall(written)
        @test last(findall(written)) > last(ft.dates)
        @test all(isnan, view(ct, .!written, :))
        @test findall(isfinite, view(ct, :, 1)) == findall(written)
        # The grid reads the style column finite on every date, and the whole axis reads
        # the same entries there.
        cg = forecast_factor_correlation(ft, px.csfm; dates = ft.dates).X
        @test all(isfinite, view(cg, :, 1))
        @test isequal(cg, ct[ft.dates, :])
    end
end

@testset "The forecast is written onto the estimation universe at the pairing" begin
    PO = PortfolioOptimisers
    # Issue #1074. The oracle masks the forecast by the estimation mask
    # before every statistic; the pairing does it once, so every verb above it inherits
    # the universe from `fe.alpha` and none carries a mask of its own.
    y = PO.forward_mean_returns(FC_ALPHA, 1, 1)
    m = trues(size(FC_ALPHA))
    m[2, 3] = false
    m[4, 1] = false

    @testset "The bare method writes the forecast onto the universe it is given" begin
        # A mask that admits every asset, the default, hands the history back as it is.
        fe = forecast_evaluation(FC_ALPHA, y)
        @test fe.alpha === FC_ALPHA
        @test PO.forecast_evaluation_mask(FC_ALPHA, trues(size(FC_ALPHA))) === FC_ALPHA
        fm = forecast_evaluation(FC_ALPHA, y; umsk = m)
        @test isnan(fm.alpha[2, 3])
        @test isnan(fm.alpha[4, 1])
        @test count(isnan, fm.alpha) == 2
        @test all(t -> all(i -> !m[t, i] || fm.alpha[t, i] == FC_ALPHA[t, i],
                           axes(FC_ALPHA, 2)), axes(FC_ALPHA, 1))
        @test fm.y === y
        # A mask that empties a row leaves the target alone and the row unscorable.
        me = copy(m)
        me[1, :] .= false
        @test first(forecast_evaluation(FC_ALPHA, y; umsk = me).dates) == 2
        @test isequal(forecast_evaluation(FC_ALPHA, y; umsk = me).y, y)
        # The caller's history is not written to.
        @test !any(isnan, FC_ALPHA)
    end

    @testset "A mask off the forecast's axis is refused" begin
        @test_throws DimensionMismatch PO.forecast_evaluation_mask(FC_ALPHA, m[1:3, :])
        @test_throws DimensionMismatch PO.forecast_evaluation_mask(FC_ALPHA, m[:, 1:3])
    end

    @testset "The carrier-and-block methods read the mask off the Asset Panel" begin
        px = evaluation_fixture(; planted = true)
        rd, csfm = px.rd, px.csfm
        emsk = rd.pnl.emsk[px.rows, :]
        amsk = rd.pnl.amsk[px.rows, :]
        off = [i for i in axes(emsk, 2) if all(view(amsk, :, i)) && !any(view(emsk, :, i))]
        # The synthetic panel clears the estimation mask for a few assets over the whole
        # panel, so the fixture carries at least one active asset the forecast is written
        # for and the estimate never reads.
        @test !isempty(off)
        fw = FixedWeightedReturnForecast(; scores = px.scores, scale = 1.0,
                                         weights = [0.4, 0.6])
        fe = forecast_evaluation(fw, rd, csfm; horizon = 5, lag = 1)
        @test all(t -> all(i -> emsk[t, i] || isnan(fe.alpha[t, i]), axes(fe.alpha, 2)),
                  axes(fe.alpha, 1))
        @test any(isfinite, view(PO.forecast_history(fw, rd, csfm; step = 5), :, off[1]))
        @test all(isnan, view(fe.alpha, :, off[1]))
        # The Estimator method is the bare method handed the cut mask, entry for entry,
        # and the Result method the same.
        hist = PO.forecast_history(fw, rd, csfm; step = 5)
        fb = forecast_evaluation(hist, fe.y; umsk = emsk, horizon = 5, lag = 1)
        @test isequal(fb.alpha, fe.alpha)
        @test fb.dates == fe.dates
        fr = forecast_evaluation(return_forecast(fw, rd, csfm), rd, csfm; horizon = 5,
                                 lag = 1)
        @test isequal(fr.alpha, fe.alpha)
        # Every statistic inherits the universe: the coefficient of the masked pairing is
        # the coefficient of the bare pairing with the mask, and differs from the one
        # without.
        @test isequal(forecast_ic(fe), forecast_ic(fb))
        @test !isequal(forecast_ic(fe),
                       forecast_ic(forecast_evaluation(hist, fe.y; horizon = 5, lag = 1)))
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

@testset "The two tables reproduce the stored oracle" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    # The forward means of a longer window tie on this fixture, and the oracle breaks a
    # tie by the asset order here, so `ties = :ordinal` reproduces its numbers.
    fe = forecast_evaluation(alpha, PO.forward_mean_returns(alpha, 1, 1); step = 1,
                             ties = :ordinal)
    h = forecast_holding_period(fe, alpha; n = 3, mt = OverlapMeanTest())
    d = forecast_decay(fe, alpha; n = 3, mt = OverlapMeanTest())

    @testset "Under the default tie rule, the rows whose targets tie move and the first does not" begin
        fa = forecast_evaluation(alpha, PO.forward_mean_returns(alpha, 1, 1); step = 1)
        ha = forecast_holding_period(fa, alpha; n = 3, mt = OverlapMeanTest())
        @test parity_compare([ha.spearman_mean_ic[1]; ha.pearson_mean_ic],
                             [h.spearman_mean_ic[1]; h.pearson_mean_ic]; rtol = 1e-14,
                             name = "untied rows").ok
        @test !isapprox(ha.spearman_mean_ic[2:3], h.spearman_mean_ic[2:3])
    end

    @testset "Both tables carry twenty-one columns and three rows" begin
        @test keys(h) ==
              keys(d) ==
              (:period, :horizon, :lag, :dates, :spearman_mean_ic, :spearman_ic_ir,
               :spearman_t_stat, :spearman_p_value, :spearman_mean_ci_lower,
               :spearman_mean_ci_upper, :pearson_mean_ic, :pearson_ic_ir, :pearson_t_stat,
               :pearson_p_value, :pearson_mean_ci_lower, :pearson_mean_ci_upper,
               :rank_ann_return, :rank_sharpe, :zscore_ann_return, :zscore_sharpe,
               :mean_coverage)
        @test h.period == d.period == [1, 2, 3]
        @test h.horizon == [1, 2, 3]
        @test h.lag == [1, 1, 1]
        @test d.horizon == [1, 1, 1]
        @test d.lag == [1, 2, 3]
        @test h.dates == d.dates == [1, 2, 3, 4, 5]
    end

    @testset "The holding-period table is what the oracle answered" begin
        # The literals of both tables are the oracle's full digits (#1558; the six-digit
        # printouts before it held the tables to `rtol = 1e-5`). Every cell measures
        # bit-equal, and so do the hand literals of the overlap rows.
        @test parity_compare(hcat(h.spearman_mean_ic, h.spearman_ic_ir, h.pearson_mean_ic,
                                  h.pearson_ic_ir),
                             [0.4 1.414213562373095 0.4 1.414213562373095;
                              0.12 0.39562828403747213 0.18017986772428515 0.37101420854458156;
                              0.27999999999999997 0.6390096504226938 0.40098776003427916 0.6059206450877959];
                             rtol = 1e-14, name = "holding ic").ok
        # The oracle's t-statistic is the ratio times the root of the five dates at every
        # row. The port answers that at the first row alone: at a stride of one, row `p`'s
        # windows overlap their `p - 1` neighbours, and the port's standard error reads the
        # overlap where the oracle's does not. That is a deliberate divergence, asserted
        # here beside the oracle's number rather than papered over.
        @test parity_compare(hcat(h.spearman_ic_ir, h.pearson_ic_ir) .* sqrt(5),
                             [3.162277660168379 3.162277660168379;
                              0.8846517369293827 0.8296129909239677;
                              1.4288690166235205 1.3548797513868356]; rtol = 1e-14,
                             name = "holding oracle t").ok
        @test parity_compare([h.spearman_t_stat[1], h.pearson_t_stat[1]],
                             [3.162277660168379, 3.162277660168379]; rtol = 1e-14,
                             name = "holding t first row").ok
        for p in 2:3
            fp = PO.ForecastEvaluationResult(alpha,
                                             PO.forward_mean_returns(alpha, h.horizon[p],
                                                                     h.lag[p]), fe.umsk,
                                             h.dates, fe.target, h.horizon[p], h.lag[p],
                                             fe.step, fe.min_count, fe.ties, fe.ppy)
            s = forecast_ic_summary(forecast_ic(fp); lags = p - 1, mt = OverlapMeanTest())
            @test isequal(h.spearman_t_stat[p], s.spearman.t_stat)
            @test isequal(h.pearson_t_stat[p], s.pearson.t_stat)
        end
        # The overlap rows divide by `n - 2L - 1`, five dates less `2L + 1`, where the
        # released rule divided by `n - 1`: row 2 is the released value times `sqrt(2 / 4)`.
        @test parity_compare([h.spearman_t_stat[2], h.pearson_t_stat[2]],
                             [0.8624393618641034, 0.7453765425294355] .* sqrt(2 / 4);
                             rtol = 1e-14, name = "holding t overlap").ok
        # Five dates at two lags leave no degree of freedom, so row 3 has no statistic.
        @test isnan(h.spearman_t_stat[3])
        @test isnan(h.pearson_t_stat[3])
        @test parity_compare(hcat(h.rank_ann_return, h.rank_sharpe),
                             [1.0 1.414213562373095; 0.55 0.7928249671720919;
                              0.7333333333333332 0.964763821237732]; rtol = 1e-14,
                             name = "holding rank book").ok
        @test parity_compare(hcat(h.zscore_ann_return, h.zscore_sharpe),
                             hcat(h.rank_ann_return, h.rank_sharpe); rtol = 1e-14,
                             name = "holding zscore book").ok
        @test h.mean_coverage == [1.0, 1.0, 1.0]
    end

    @testset "The decay table is what the oracle answered" begin
        @test parity_compare(hcat(d.spearman_mean_ic, d.spearman_ic_ir, d.spearman_t_stat),
                             [0.4 1.414213562373095 3.162277660168379;
                              0.040000000000000015 0.12171612389003696 0.2721655269759088;
                              0.43999999999999995 1.0734900802433864 2.400396792595916];
                             rtol = 1e-14, name = "decay spearman").ok
        @test parity_compare(hcat(d.pearson_mean_ic, d.pearson_ic_ir, d.pearson_t_stat),
                             hcat(d.spearman_mean_ic, d.spearman_ic_ir, d.spearman_t_stat);
                             rtol = 1e-14, name = "decay pearson").ok
        @test parity_compare(hcat(d.rank_ann_return, d.rank_sharpe),
                             [1.0 1.414213562373095; 0.1 0.12171612389003691;
                              1.1 1.0734900802433864]; rtol = 1e-14,
                             name = "decay rank book").ok
        @test parity_compare(hcat(d.zscore_ann_return, d.zscore_sharpe),
                             hcat(d.rank_ann_return, d.rank_sharpe); rtol = 1e-14,
                             name = "decay zscore book").ok
        @test d.mean_coverage == [1.0, 1.0, 1.0]
    end
end

@testset "The first row of both tables is the base evaluation, on the common dates" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    y = PO.forward_mean_returns(alpha, 1, 1)
    fe = forecast_evaluation(alpha, y; step = 1)
    h = forecast_holding_period(fe, alpha; n = 3, mt = OverlapMeanTest())
    d = forecast_decay(fe, alpha; n = 3, mt = OverlapMeanTest())
    fb = PO.ForecastEvaluationResult(alpha, y, fe.umsk, h.dates, fe.target, fe.horizon,
                                     fe.lag, fe.step, fe.min_count, fe.ties, fe.ppy)
    ic = forecast_ic_summary(forecast_ic(fb); mt = OverlapMeanTest())
    rk = forecast_portfolio(fb; kind = :rank)
    zs = forecast_portfolio(fb; kind = :zscore)

    @testset "The two tables agree at the first period" begin
        for k in (:spearman_mean_ic, :spearman_ic_ir, :spearman_t_stat, :spearman_p_value,
                  :spearman_mean_ci_lower, :spearman_mean_ci_upper, :pearson_mean_ic,
                  :pearson_ic_ir, :pearson_t_stat, :pearson_p_value, :pearson_mean_ci_lower,
                  :pearson_mean_ci_upper, :rank_ann_return, :rank_sharpe, :zscore_ann_return,
                  :zscore_sharpe, :mean_coverage)
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
        t1 = forecast_holding_period(fe, alpha; n = 1, mt = OverlapMeanTest())
        t3 = forecast_holding_period(fe, alpha; n = 3, mt = OverlapMeanTest())
        @test t1.dates == collect(1:7)
        @test t3.dates == collect(1:5)
        @test t1.spearman_mean_ic[1] != t3.spearman_mean_ic[1]
    end

    @testset "A base evaluation that already stops early is depth invariant" begin
        y = PO.forward_mean_returns(alpha, 1, 1)
        y[6:8, :] .= NaN
        fe = forecast_evaluation(alpha, y; step = 1)
        @test fe.dates == collect(1:5)
        t1 = forecast_holding_period(fe, alpha; n = 1, mt = OverlapMeanTest())
        t3 = forecast_holding_period(fe, alpha; n = 3, mt = OverlapMeanTest())
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
        d = forecast_decay(fe, X; n = 3, mt = OverlapMeanTest())
        h = forecast_holding_period(fe, X; n = 3, mt = OverlapMeanTest())
        @test isapprox(d.spearman_mean_ic[1], 1)
        @test isapprox(d.pearson_mean_ic[1], 1)
        @test isapprox(h.spearman_mean_ic[1], 1)
    end

    @testset "The later periods score a target the forecast is not aimed at" begin
        d = forecast_decay(fe, X; n = 3, mt = OverlapMeanTest())
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
        a = forecast_holding_period(fe, rd, csfm; n = 3, mt = OverlapMeanTest())
        b = forecast_holding_period(fe, X, PO.cs_diagnostic_weights(IdentityMetric(), csfm);
                                    n = 3, mt = OverlapMeanTest())
        @test a.dates == b.dates
        @test isapprox(a.spearman_mean_ic, b.spearman_mean_ic)
        @test isapprox(a.pearson_mean_ic, b.pearson_mean_ic)
    end

    @testset "A planted fixture finds signal at the first period of both tables" begin
        h = forecast_holding_period(fe, rd, csfm; n = 3, mt = OverlapMeanTest())
        d = forecast_decay(fe, rd, csfm; n = 3, mt = OverlapMeanTest())
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
        @test_throws DimensionMismatch forecast_holding_period(fe, alpha[:, 1:3]; n = 2,
                                                               mt = OverlapMeanTest())
        @test_throws DimensionMismatch forecast_decay(fe, alpha[1:4, :]; n = 2,
                                                      mt = OverlapMeanTest())
    end

    @testset "An empty grid is refused" begin
        @test_throws PO.IsEmptyError PO.forecast_window_table(fe, alpha, nothing,
                                                              Tuple{Int, Int}[], 3, 1,
                                                              OverlapMeanTest())
    end

    @testset "A grid deeper than the sample prints a table of NaN" begin
        t = forecast_holding_period(fe, alpha; n = 8, mt = OverlapMeanTest())
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
        @test 3 ∉ forecast_holding_period(fg, gapped; n = 2, mt = OverlapMeanTest()).dates
        @test 3 ∈ forecast_holding_period(fg, gapped; n = 2, min_count = 2,
                                  mt = OverlapMeanTest()).dates
    end
end

@testset "`ppy` annualises the table's portfolio columns" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    y = PO.forward_mean_returns(alpha, 1, 1)
    t1 = forecast_holding_period(forecast_evaluation(alpha, y; step = 1), alpha; n = 3,
                                 mt = OverlapMeanTest())
    t4 = forecast_holding_period(forecast_evaluation(alpha, y; step = 1, ppy = 4), alpha;
                                 n = 3, mt = OverlapMeanTest())

    @testset "The return scales by `ppy` and the ratio by its square root" begin
        @test isapprox(t4.rank_ann_return, 4 * t1.rank_ann_return)
        @test isapprox(t4.rank_sharpe, sqrt(4) * t1.rank_sharpe)
        @test isapprox(t4.zscore_ann_return, 4 * t1.zscore_ann_return)
    end

    @testset "The coefficients and the coverage do not scale" begin
        @test isapprox(t4.spearman_mean_ic, t1.spearman_mean_ic)
        @test isapprox(t4.pearson_t_stat, t1.pearson_t_stat; nans = true)
        @test t4.mean_coverage == t1.mean_coverage
    end
end

# The stored oracle of the calibration, measured by running the oracle on
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
# The oracle numbers its buckets from zero and the library from one, so the indices below
# are its `[0, 2, 4, 5, 7, 8]` shifted by one. The four bins it never fills are dropped by
# both.
const CAL_REF_BIN = [1, 3, 5, 6, 8, 9]
const CAL_REF_BIN_ALPHA = [1.0, 2.0, 3.0, 4.0, 5.0, 24.0]
const CAL_REF_BIN_Y = [2.5, 2.0, 5.5, 5.0, 1.5, 21.5]
const CAL_REF_BIN_COUNT = [2, 3, 2, 1, 2, 2]

@testset "The calibration reproduces the stored oracle" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)
    c = forecast_calibration(fe)

    @testset "The slope and the pooled moments match, weighted and unweighted" begin
        # Measured bit-equal, cell by cell, here and on the curve (#1558).
        @test parity_compare([c.slope, forecast_calibration(fe, IC_W).slope, c.mean_alpha,
                              c.std_alpha, c.mean_y, c.std_y],
                             [CAL_REF_SLOPE, CAL_REF_SLOPE_W, CAL_REF_MEAN_ALPHA,
                              CAL_REF_STD_ALPHA, CAL_REF_MEAN_Y, CAL_REF_STD_Y];
                             rtol = 1e-14, name = "calibration").ok
        # A good ordering and a bad scale: the coefficients of #936 are positive at the
        # first two dates and the slope is a third of one all the same.
        @test c.slope < 0.5
    end

    @testset "The curve matches bin for bin, and the empty bins are dropped" begin
        @test c.n_bins == length(CAL_REF_BIN)
        @test c.curve.bin == CAL_REF_BIN
        @test parity_compare(hcat(c.curve.mean_alpha, c.curve.mean_y),
                             hcat(CAL_REF_BIN_ALPHA, CAL_REF_BIN_Y); rtol = 1e-14,
                             name = "calibration curve").ok
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
        # Measured maxrel 0.0: the negation is exact.
        @test parity_compare([forecast_calibration(forecast_evaluation(-IC_ALPHA, y)).slope],
                             [-CAL_REF_SLOPE]; rtol = 1e-14, name = "negated slope").ok
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
        # default, so it answers what that verb answers to rounding. That keeps the curve at
        # parity with the oracle.
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

@testset "The docstrings of 13_ForecastCalibration.jl against numbers" begin
    PO = PortfolioOptimisers

    @testset "The pairs follow the order of `dates`, and a weight that is not finite is zero" begin
        A = [1.0 NaN 3.0; 4.0 5.0 6.0; 7.0 8.0 NaN]
        Y = [1.0 2.0 3.0; NaN 5.0 6.0; 7.0 8.0 9.0]
        U = [1.0 1.0 NaN; 2.0 3.0 4.0; 5.0 Inf 6.0]
        a, b, q = PO.forecast_calibration_pairs(A, Y, U, [3, 1])
        @test a == [7.0, 8.0, 1.0, 3.0]
        @test b == [7.0, 8.0, 1.0, 3.0]
        @test q == [5.0, 0.0, 1.0, 0.0]
    end

    @testset "The slope is the weighted least-squares fit through the origin" begin
        rng = StableRNG(953)
        for _ in 1:50
            n = rand(rng, 1:40)
            a = randn(rng, n)
            b = randn(rng, n)
            q = rand(rng, n)
            @test PO.forecast_calibration_slope(a, b, q) ≈ (sqrt.(q) .* a) \ (sqrt.(q) .* b) rtol = 1e-10
        end
        # A weight of zero at every pair leaves no positive weighted square.
        @test isnan(PO.forecast_calibration_slope([1.0, 2.0], [1.0, 2.0], [0.0, 0.0]))
    end

    @testset "An edge that falls on an order statistic equals it exactly" begin
        # At `n = 43` and `B = 14`, edge 9 sits at the order statistic `0.28`. A floating
        # `h = (n - 1) i / B + 1` lands one ulp above 28, and the edge came out above
        # `0.28`, so the pair at `0.28` fell into the bin below.
        x = collect(1:43) ./ 100
        e = PO.forecast_calibration_edges(x, 14)
        @test e[10] == 0.28
        @test all(i -> x[div(42 * i, 14) + 1] == e[i + 1], 0:14)
        @test PO.forecast_calibration_curve(x, x, 14).count == [fill(3, 13); 4]
        # Elsewhere the edges agree with `Statistics.quantile` to rounding.
        rng = StableRNG(9532)
        for _ in 1:200
            n = rand(rng, 1:60)
            y = round.(randn(rng, n); digits = 2)
            B = rand(rng, 1:12)
            @test PO.forecast_calibration_edges(y, B) ≈
                  unique([quantile(y, p) for p in range(0, 1, B + 1)]) atol = 1e-12
        end
    end

    @testset "Each bin holds the pairs between its edges, and the means rise" begin
        rng = StableRNG(9533)
        for _ in 1:100
            n = rand(rng, 1:50)
            a = round.(randn(rng, n); digits = 1)
            b = randn(rng, n)
            B = rand(rng, 1:10)
            c = PO.forecast_calibration_curve(a, b, B)
            e = PO.forecast_calibration_edges(a, B)
            m = length(e)
            @test sum(c.count) == n
            @test all(>(0), diff(c.mean_alpha))
            for (j, bj) in enumerate(c.bin)
                lo = m == 1 ? -Inf : e[bj]
                hi = (m == 1 || bj == m - 1) ? Inf : e[bj + 1]
                k = findall(v -> lo <= v < hi, a)
                @test length(k) == c.count[j]
                @test c.mean_alpha[j] ≈ sum(a[k]) / length(k)
                @test c.mean_y[j] ≈ sum(b[k]) / length(k)
            end
        end
        # A bin with no pair between its edges is dropped, so `bin` skips its index.
        @test PO.forecast_calibration_curve([0.0, 10.0], [1.0, 2.0], 4).bin == [1, 4]
    end

    @testset "The pooled moments are the mean and the corrected deviation" begin
        v = [1.0, 2.0, 4.0, -3.0]
        m = PO.forecast_pooled_moments(v)
        @test m.mean == sum(v) / 4
        @test m.std ≈ sqrt(sum(abs2, v .- sum(v) / 4) / 3)
        @test isnan(PO.forecast_pooled_moments([1.0]).std)
    end

    @testset "Only a positive rescaling leaves the coefficients and the books unmoved" begin
        rng = StableRNG(1)
        A = randn(rng, 12, 8)
        Y = PO.forward_mean_returns(A .+ 0.3 .* randn(rng, 12, 8), 1, 1)
        fe = forecast_evaluation(A, Y)
        fn = forecast_evaluation(-2 .* A, Y)
        @test forecast_ic(forecast_evaluation(2 .* A, Y)) ≈ forecast_ic(fe)
        @test forecast_ic(fn) ≈ -forecast_ic(fe)
        @test filter(isfinite, forecast_portfolio(fn).ret) ≈
              -filter(isfinite, forecast_portfolio(fe).ret)
        @test forecast_calibration(fn).slope ≈ -forecast_calibration(fe).slope / 2
    end

    @testset "A Float32 evaluation stays Float32" begin
        A = Float32[1 2 3 4; 2 4 6 8; 4 3 2 1]
        c = forecast_calibration(forecast_evaluation(A, PO.forward_mean_returns(A, 1, 1)))
        @test c.slope isa Float32
        @test c.std_y isa Float32
        @test eltype(c.curve.mean_alpha) == Float32
        @test eltype(PO.forecast_calibration_edges(Float32[3, 1, 2, 5], 3)) == Float32
    end
end

# The stored oracle of the summary, measured by running the oracle's own
# `ic_summary`, `portfolio_summary`, `quantile_summary`, `calibration_summary` and
# `coverage_summary` over `IC_ALPHA` and over the gapped variant, on the evaluation dates
# both agree on. Convention 11. Issue #941.
const FS_REF = (; spearman_mean_ic = 0.3333333333333333,
                spearman_std_ic = 0.7023769168568493, spearman_ic_ir = 0.4745789978762494,
                spearman_t_stat = 0.8219949365267862,
                spearman_hit_rate = 0.6666666666666666,
                pearson_mean_ic = 0.2567790285367673, pearson_std_ic = 0.6206266852224849,
                pearson_ic_ir = 0.4137415207093715, pearson_t_stat = 0.7166213350694421,
                pearson_hit_rate = 0.6666666666666666, rank_ann_return = 9.75,
                rank_ann_volatility = 16.68270062070287, rank_sharpe = 0.5844377491196157,
                rank_hit_rate = 0.6666666666666666, rank_mean_turnover = 1.0,
                zscore_ann_return = 11.604040404040402,
                zscore_ann_volatility = 20.853167491514082,
                zscore_sharpe = 0.5564641634784026, zscore_hit_rate = 0.6666666666666666,
                zscore_mean_turnover = 1.5737373737373739,
                calibration_slope = 0.29795686719636777, mean_alpha = 6.333333333333333,
                std_alpha = 10.790006599823858, mean_y = 6.083333333333333,
                std_y = 10.799480907839406, n_bins = 6, mean_coverage = 1.0,
                min_coverage = 1.0, mean_n_scored = 4.0, min_n_scored = 4.0,
                spread_ann_return = 12.666666666666666,
                spread_ann_volatility = 22.03028218914441,
                spread_sharpe = 0.5749661560353623, spread_hit_rate = 0.6666666666666666)
const FS_REF_GAP = (; spearman_mean_ic = 0.8333333333333334,
                    spearman_std_ic = 0.28867513459481287,
                    spearman_ic_ir = 2.886751345948129, spearman_t_stat = 5.0,
                    spearman_hit_rate = 1.0, pearson_mean_ic = 0.9068588973742973,
                    pearson_std_ic = 0.16132512202070398, pearson_ic_ir = 5.621312328887709,
                    pearson_t_stat = 9.736398558846842, pearson_hit_rate = 1.0,
                    rank_ann_return = 2.3333333333333335,
                    rank_ann_volatility = 1.1547005383792515,
                    rank_sharpe = 2.0207259421636903, rank_hit_rate = 1.0,
                    rank_mean_turnover = 1.0, zscore_ann_return = 2.266666666666666,
                    zscore_ann_volatility = 1.10151410945722,
                    zscore_sharpe = 2.0577736110739284, zscore_hit_rate = 1.0,
                    zscore_mean_turnover = 1.2000000000000002, calibration_slope = 1.015625,
                    mean_alpha = 2.5, std_alpha = 1.4142135623730951, mean_y = 3.0,
                    std_y = 1.6903085094570331, n_bins = 5,
                    mean_coverage = 0.6666666666666666, min_coverage = 0.5,
                    mean_n_scored = 2.6666666666666665, min_n_scored = 2.0,
                    spread_ann_return = 2.3333333333333335,
                    spread_ann_volatility = 1.1547005383792515,
                    spread_sharpe = 2.0207259421636903, spread_hit_rate = 1.0)

# The thirty columns of the oracle's summary whose axis is the forecast, in the order the
# Result declares them. The three scored-coverage columns of #1512 follow them.
const FS_CORE = (:spearman_mean_ic, :spearman_std_ic, :spearman_ic_ir, :spearman_t_stat,
                 :spearman_hit_rate, :pearson_mean_ic, :pearson_std_ic, :pearson_ic_ir,
                 :pearson_t_stat, :pearson_hit_rate, :rank_ann_return, :rank_ann_volatility,
                 :rank_sharpe, :rank_hit_rate, :rank_mean_turnover, :zscore_ann_return,
                 :zscore_ann_volatility, :zscore_sharpe, :zscore_hit_rate,
                 :zscore_mean_turnover, :calibration_slope, :mean_alpha, :std_alpha,
                 :mean_y, :std_y, :n_bins, :mean_coverage, :min_coverage, :mean_n_scored,
                 :min_n_scored)
const FS_SPREAD = (:spread_ann_return, :spread_ann_volatility, :spread_sharpe,
                   :spread_hit_rate)

@testset "The summary reproduces the stored oracle, column for column" begin
    PO = PortfolioOptimisers

    @testset "Every core column matches, bit for bit" begin
        fs = forecast_evaluation_summary(forecast_evaluation(IC_ALPHA,
                                                             PO.forward_mean_returns(IC_ALPHA,
                                                                                     1, 1));
                                         quantiles = (0.1,), mt = OverlapMeanTest())
        # Every column measures bit-equal, on this panel and on the gapped one (#1558).
        @test parity_compare([[getfield(fs, f)[1] for f in FS_CORE];
                              [getfield(fs, f)[1, 1] for f in FS_SPREAD]],
                             [getfield(FS_REF, f) for f in (FS_CORE..., FS_SPREAD...)];
                             rtol = 1e-14, name = "summary").ok
    end

    @testset "A gapped panel moves the coverage block, and the oracle agrees" begin
        # Two gaps thin two of the three cross-sections, so the coverage columns stop
        # reading the whole universe and the count columns say how far they fell. This is
        # the fixture that separates `mean_coverage` from `min_coverage`, which a full
        # panel cannot.
        gap = ic_gap_fixture()
        fs = forecast_evaluation_summary(forecast_evaluation(gap,
                                                             PO.forward_mean_returns(gap, 1,
                                                                                     1);
                                                             min_count = 2);
                                         quantiles = (0.25,), mt = OverlapMeanTest())
        @test parity_compare([[getfield(fs, f)[1] for f in FS_CORE];
                              [getfield(fs, f)[1, 1] for f in FS_SPREAD]],
                             [getfield(FS_REF_GAP, f) for f in (FS_CORE..., FS_SPREAD...)];
                             rtol = 1e-14, name = "summary gap").ok
        @test parity_compare([fs.mean_coverage[1], fs.min_coverage[1], fs.mean_n_scored[1]],
                             [sum(IC_REF_GAP_COVERAGE) / 3, minimum(IC_REF_GAP_COVERAGE),
                              sum(IC_REF_GAP_COVERAGE .* 4) / 3]; rtol = 1e-14,
                             name = "summary gap coverage").ok
        @test fs.min_n_scored[1] == 2
    end
end

@testset "Every column is the level-2 verb it reads, and nothing else" begin
    PO = PortfolioOptimisers
    fe = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 1, 1); ppy = 12)
    fs = forecast_evaluation_summary(fe; quantiles = (0.1, 0.25), bins = 4,
                                     mt = OverlapMeanTest())
    ic = forecast_ic_summary(forecast_ic(fe); mt = OverlapMeanTest())
    rk = forecast_portfolio(fe; kind = :rank)
    zs = forecast_portfolio(fe; kind = :zscore)
    cb = forecast_calibration(fe; bins = 4)
    qs = forecast_quantile_spread(fe; quantiles = (0.1, 0.25))
    c = forecast_coverage(fe)

    @testset "The coefficient block is the shared summary kernel" begin
        for (p, s) in ((:spearman, ic.spearman), (:pearson, ic.pearson))
            @test getfield(fs, Symbol(p, :_mean_ic))[1] == s.mean_ic
            @test getfield(fs, Symbol(p, :_std_ic))[1] == s.std_ic
            @test getfield(fs, Symbol(p, :_ic_ir))[1] == s.ic_ir
            @test getfield(fs, Symbol(p, :_t_stat))[1] == s.t_stat
            @test getfield(fs, Symbol(p, :_hit_rate))[1] == s.hit_rate
        end
    end

    @testset "The two books are the two kinds of the portfolio verb" begin
        for (p, b) in ((:rank, rk), (:zscore, zs))
            @test getfield(fs, Symbol(p, :_ann_return))[1] == b.summary.ann_return
            @test getfield(fs, Symbol(p, :_ann_volatility))[1] == b.summary.ann_volatility
            @test getfield(fs, Symbol(p, :_sharpe))[1] == b.summary.sharpe
            @test getfield(fs, Symbol(p, :_hit_rate))[1] == b.hit_rate
            @test getfield(fs, Symbol(p, :_mean_turnover))[1] == b.mean_turnover
        end
    end

    @testset "The calibration, the spreads and the coverage are their own verbs" begin
        @test fs.calibration_slope[1] == cb.slope
        @test fs.mean_alpha[1] == cb.mean_alpha
        @test fs.std_alpha[1] == cb.std_alpha
        @test fs.mean_y[1] == cb.mean_y
        @test fs.std_y[1] == cb.std_y
        @test fs.n_bins[1] == cb.n_bins
        @test fs.spread_ann_return[1, :] == qs.ann_mean
        @test fs.spread_ann_volatility[1, :] == qs.ann_vol
        @test fs.spread_sharpe[1, :] == qs.ann_ir
        @test fs.spread_hit_rate[1, :] == qs.hit_rate
        @test fs.mean_coverage[1] ≈ sum(c) / length(c)
        @test fs.min_coverage[1] ≈ minimum(c)
    end

    @testset "The annualisation is the evaluation's own, and it is carried" begin
        # `ppy` reaches the summary off the Result rather than as a keyword of its own, so
        # a caller cannot annualise a table under a factor the evaluation did not use.
        @test fs.ppy == 12
        @test fs.rank_ann_return[1] ≈
              12 *
              forecast_portfolio(forecast_evaluation(fe.alpha, fe.y)).summary.ann_return
    end

    @testset "The drawdown family is one call down, not a column" begin
        @test !any(f -> occursin("calmar", String(f)) ||
                        occursin("drawdown", String(f)) ||
                        occursin("sortino", String(f)) ||
                        occursin("cvar", String(f)), fieldnames(typeof(fs)))
        @test isfinite(rk.summary.calmar) || isnan(rk.summary.calmar)
    end
end

@testset "One evaluation is the length-1 case of the comparison" begin
    PO = PortfolioOptimisers
    fe = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 1, 1))
    one = forecast_evaluation_summary(fe; mt = OverlapMeanTest())
    two = forecast_evaluation_summary([fe, fe]; names = ["a", "b"], mt = OverlapMeanTest())

    @testset "The two carry the same type and the same columns" begin
        @test typeof(one) === typeof(two)
        @test fieldnames(typeof(one)) == fieldnames(typeof(two))
        @test isa(one, PO.AbstractResult)
    end

    @testset "Two evaluations of one forecast compare as identical rows" begin
        @test two.names == ["a", "b"]
        for f in FS_CORE
            v = getfield(two, f)
            @test length(v) == 2
            @test v[1] == v[2]
            @test v[1] == getfield(one, f)[1]
        end
    end

    @testset "An unnamed summary numbers its rows in the order they were given" begin
        @test one.names == ["Forecast 1"]
        @test forecast_evaluation_summary([fe, fe, fe]; mt = OverlapMeanTest()).names ==
              ["Forecast 1", "Forecast 2", "Forecast 3"]
        @test_throws DimensionMismatch forecast_evaluation_summary([fe, fe]; names = ["a"],
                                                                   mt = OverlapMeanTest())
    end

    @testset "A vector method with no evaluation has nothing to summarise" begin
        @test_throws PO.IsEmptyError forecast_evaluation_summary(ForecastEvaluationResult[];
                                                                 mt = OverlapMeanTest())
    end
end

@testset "A listing stacks the summary of each evaluation, and a comparison refuses it" begin
    # R90 of #1416, built by #1512. A comparison of skill is paired: one estimand, one sample
    # and one unit. `paired = false` is a listing, the oracle's table of evaluations.
    PO = PortfolioOptimisers
    fe1 = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 1, 1))
    fe2 = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 2, 1);
                              horizon = 2, ppy = 12)
    @test_throws PO.ConflictingArgumentError forecast_evaluation_summary([fe1, fe2];
                                                                         mt = OverlapMeanTest())
    ls = forecast_evaluation_summary([fe1, fe2]; paired = false, names = ["h1", "h2"],
                                     quantiles = (0.25,), mt = OverlapMeanTest())
    s1 = forecast_evaluation_summary(fe1; quantiles = (0.25,), mt = OverlapMeanTest())
    s2 = forecast_evaluation_summary(fe2; quantiles = (0.25,), mt = OverlapMeanTest())
    @test ls.names == ["h1", "h2"]
    for f in (FS_CORE..., :n_silenced, :mean_coverage_scored, :min_coverage_scored,
              FS_SPREAD...)
        @test isequal(getfield(ls, f), vcat(getfield(s1, f), getfield(s2, f)))
    end
    @test ls.ppy == [1, 12]
    err = try
        forecast_evaluation_summary([fe1, fe2]; paired = false, align = true,
                                    mt = OverlapMeanTest())
        nothing
    catch e
        e
    end
    @test isa(err, PO.ConflictingArgumentError)
    @test occursin("`paired = false`", err.msg) && occursin("`align = true`", err.msg)
    # On a comparable pair the listing has the rows of the comparison.
    cp = forecast_evaluation_summary([fe1, fe1]; mt = OverlapMeanTest())
    lp = forecast_evaluation_summary([fe1, fe1]; paired = false, mt = OverlapMeanTest())
    for f in FS_CORE
        @test isequal(getfield(cp, f), getfield(lp, f))
    end
    @test cp.ppy == 1 && lp.ppy == [1, 1]
    # The spreads take the type of every evaluation, not of the first (R10 of #1630). A
    # `Float32` evaluation listed first narrowed the spreads of a `Float64` one.
    a32 = Float32.(IC_ALPHA)
    fe32 = forecast_evaluation(a32, PO.forward_mean_returns(a32, 1, 1))
    l32 = forecast_evaluation_summary([fe32, fe1]; paired = false, quantiles = (0.25,),
                                      mt = OverlapMeanTest())
    for f in FS_SPREAD
        @test eltype(getfield(l32, f)) == Float64
        @test isequal(getfield(l32, f)[2:2, :], getfield(s1, f))
    end
    @test_throws PO.IsEmptyError forecast_evaluation_summary(ForecastEvaluationResult[];
                                                             paired = false,
                                                             mt = OverlapMeanTest())
end

@testset "A silenced date counts in the coverage, and the scored coverage leaves it out" begin
    # R87 of #1416, built by #1512. The middle date scores no asset of a full universe.
    PO = PortfolioOptimisers
    alpha = copy(IC_ALPHA)
    alpha[2, :] .= NaN
    fe = forecast_evaluation(alpha, PO.forward_mean_returns(IC_ALPHA, 1, 1))
    @test fe.dates == [1, 2, 3]
    c = forecast_coverage(fe)
    @test c == [1, 0, 1]
    cv = PO.forecast_summary_coverage(fe, nothing)
    @test cv.n_silenced == 1
    @test cv.mean_coverage ≈ 2 / 3 && cv.min_coverage == 0
    @test cv.mean_coverage_scored == 1 && cv.min_coverage_scored == 1
    @test cv.mean_coverage_scored ≈ cv.mean_coverage * 3 / (3 - cv.n_silenced)
    # A date whose universe is empty has no coverage, so it is neither scored nor silenced.
    umsk = trues(size(alpha))
    umsk[2, :] .= false
    cu = PO.forecast_summary_coverage(forecast_evaluation(alpha,
                                                          PO.forward_mean_returns(IC_ALPHA,
                                                                                  1, 1);
                                                          umsk = umsk), nothing)
    @test cu.n_silenced == 0 && cu.mean_coverage == 1 && cu.mean_coverage_scored == 1
    # Every date silenced (#1740): only the first asset carries a forecast, and the weights
    # leave it out, so each date has a universe and scores none of it.
    a1 = fill(NaN, size(alpha))
    a1[:, 1] = IC_ALPHA[:, 1]
    f1 = forecast_evaluation(a1, PO.forward_mean_returns(IC_ALPHA, 1, 1))
    w1 = ones(size(a1))
    w1[:, 1] .= 0
    @test forecast_coverage(f1, w1) == [0, 0, 0]
    ca = PO.forecast_summary_coverage(f1, w1)
    @test ca.n_silenced == length(f1.dates) == 3
    @test ca.mean_coverage == 0 && ca.min_coverage == 0
    @test ca.mean_n_scored == 0 && ca.min_n_scored == 0
    @test isnan(ca.mean_coverage_scored) && isnan(ca.min_coverage_scored)
end

@testset "An incomparable pair is refused, and the message names the field" begin
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y)

    @testset "Each of the seven parameters is checked, and the asset axis with them" begin
        for (nm, other) in
            (("target", forecast_evaluation(IC_ALPHA, y; target = PO.AssetReturnTarget())),
             ("horizon", forecast_evaluation(IC_ALPHA, y; horizon = 2)),
             ("lag", forecast_evaluation(IC_ALPHA, y; lag = 0)),
             ("step", forecast_evaluation(IC_ALPHA, y; step = 2)),
             ("min_count", forecast_evaluation(IC_ALPHA, y; min_count = 2)),
             ("ppy", forecast_evaluation(IC_ALPHA, y; ppy = 252)))
            err = try
                forecast_evaluation_summary([fe, other]; mt = OverlapMeanTest())
                nothing
            catch e
                e
            end
            @test isa(err, PO.ConflictingArgumentError)
            @test occursin("`$(nm)`", err.msg)
        end
    end

    @testset "A different date set and a different universe are refused too" begin
        # `step = 2` moves both `step` and `dates`, so the date rule is checked on a pair
        # that agrees on every parameter: a forecast whose last row carries no finite value
        # scores one date fewer.
        short = copy(IC_ALPHA)
        short[3, :] .= NaN
        e1 = try
            forecast_evaluation_summary([fe,
                                         forecast_evaluation(short,
                                                             PO.forward_mean_returns(short,
                                                                                     1, 1))];
                                        mt = OverlapMeanTest())
            nothing
        catch e
            e
        end
        @test isa(e1, PO.ConflictingArgumentError)
        @test occursin("`dates`", e1.msg)

        wide = hcat(IC_ALPHA, IC_ALPHA[:, 1])
        e2 = try
            forecast_evaluation_summary([fe,
                                         forecast_evaluation(wide,
                                                             PO.forward_mean_returns(wide,
                                                                                     1, 1))];
                                        mt = OverlapMeanTest())
            nothing
        catch e
            e
        end
        @test isa(e2, PO.ConflictingArgumentError)
        @test occursin("asset axis", e2.msg)

        # Two evaluations on the same asset axis over two universe masks are two different
        # denominators for every coverage column, so the mask is compared cell for cell.
        umsk = trues(size(IC_ALPHA))
        umsk[2, 4] = false
        e3 = try
            forecast_evaluation_summary([fe, forecast_evaluation(IC_ALPHA, y; umsk = umsk)],
                                        mt = OverlapMeanTest())
            nothing
        catch e
            e
        end
        @test isa(e3, PO.ConflictingArgumentError)
        @test occursin("`umsk`", e3.msg)
        @test occursin("1 cell(s)", e3.msg)
    end

    @testset "A Panel Field target is the same target only where it names the same field" begin
        @test PO.forecast_summary_same_target(PO.IdiosyncraticTarget(),
                                              PO.IdiosyncraticTarget())
        @test !PO.forecast_summary_same_target(PO.IdiosyncraticTarget(),
                                               PO.AssetReturnTarget())
        @test PO.forecast_summary_same_target(PanelFieldTarget(; name = "volume"),
                                              PanelFieldTarget(; name = "volume"))
        @test !PO.forecast_summary_same_target(PanelFieldTarget(; name = "volume"),
                                               PanelFieldTarget(; name = "market_cap"))
    end
end

@testset "Two evaluations on two grids are aligned onto the grid they share" begin
    # Issue #1073. A member that publishes its history is scorable from the block's first
    # row and a member that is refit warms up, so the two rarely share a grid and the
    # summary refuses them on `dates`. The verb rebuilds each Result on
    # `max(lo):step:min(hi)` and touches nothing else, which is what the example's
    # hand-written blank through the bare method did, and the summary's `align = true`
    # calls it. The pair is asserted UNTOUCHED by identity, not by equality.
    PO = PortfolioOptimisers
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    early = forecast_evaluation(IC_ALPHA, y)
    late_alpha = copy(IC_ALPHA)
    late_alpha[1, :] .= NaN
    late = forecast_evaluation(late_alpha, y)

    @testset "The common grid starts at the later start, and the pair is untouched" begin
        @test early.dates == [1, 2, 3]
        @test late.dates == [2, 3]
        a, b = forecast_evaluation_align([early, late])
        @test a.dates == [2, 3]
        @test b.dates == [2, 3]
        @test a.alpha === early.alpha
        @test a.y === early.y
        @test a.umsk === early.umsk
        @test b.alpha === late.alpha
        for f in (:target, :horizon, :lag, :step, :min_count, :ties, :ppy)
            @test getfield(a, f) === getfield(early, f)
            @test getfield(b, f) === getfield(late, f)
        end
        @test isa(forecast_evaluation_align([early, late]),
                  Vector{<:ForecastEvaluationResult})
    end

    @testset "The common grid ends at the earlier finish, which the blank did not do" begin
        short_alpha = copy(IC_ALPHA)
        short_alpha[3, :] .= NaN
        short = forecast_evaluation(short_alpha, y)
        @test short.dates == [1, 2]
        a, b = forecast_evaluation_align([late, short])
        @test a.dates == [2]
        @test b.dates == [2]
    end

    @testset "The grid is anchored at the later start, so an off-phase member is re-anchored" begin
        # Under `step = 2` the early member's own grid is [1, 3] and the late one's is [2];
        # the intersection as sets is empty, and the common grid is [2]. That is what the
        # blank did too: it blanked the early member before row 2 and found its dates
        # again, at 2:2:3.
        e2 = forecast_evaluation(IC_ALPHA, y; step = 2)
        l2 = forecast_evaluation(late_alpha, y; step = 2)
        @test e2.dates == [1, 3]
        @test l2.dates == [2]
        a, b = forecast_evaluation_align([e2, l2])
        @test a.dates == [2]
        @test b.dates == [2]
    end

    @testset "A set already on one grid is handed back on it, and a singleton is itself" begin
        a, b = forecast_evaluation_align([early, early])
        @test a.dates == early.dates
        @test b.dates == early.dates
        @test only(forecast_evaluation_align([late])).dates == late.dates
    end

    @testset "Two grids that do not overlap are refused" begin
        tail_alpha = copy(IC_ALPHA)
        tail_alpha[1:2, :] .= NaN
        head_alpha = copy(IC_ALPHA)
        head_alpha[3, :] .= NaN
        tail = forecast_evaluation(tail_alpha, y)
        head = forecast_evaluation(head_alpha, y)
        @test tail.dates == [3]
        @test head.dates == [1, 2]
        @test_throws PO.IsEmptyError forecast_evaluation_align([tail, head])
        @test_throws PO.IsEmptyError forecast_evaluation_align(ForecastEvaluationResult[])
    end

    @testset "A set that does not answer one question is refused before it is aligned" begin
        for (nm, other) in (("horizon", forecast_evaluation(late_alpha, y; horizon = 2)),
                            ("step", forecast_evaluation(late_alpha, y; step = 2)),
                            ("ties", forecast_evaluation(late_alpha, y; ties = :ordinal)),
                            ("ppy", forecast_evaluation(late_alpha, y; ppy = 252)))
            err = try
                forecast_evaluation_align([early, other])
                nothing
            catch e
                e
            end
            @test isa(err, PO.ConflictingArgumentError)
            @test occursin("`$(nm)`", err.msg)
        end
        umsk = trues(size(IC_ALPHA))
        umsk[2, 4] = false
        err = try
            forecast_evaluation_align([early,
                                       forecast_evaluation(late_alpha, y; umsk = umsk)])
            nothing
        catch e
            e
        end
        @test isa(err, PO.ConflictingArgumentError)
        @test occursin("`umsk`", err.msg)
    end

    @testset "The summary aligns under the keyword and refuses without it" begin
        err = try
            forecast_evaluation_summary([early, late]; mt = OverlapMeanTest())
            nothing
        catch e
            e
        end
        @test isa(err, PO.ConflictingArgumentError)
        @test occursin("`dates`", err.msg)
        fs = forecast_evaluation_summary([early, late]; align = true, names = ["e", "l"],
                                         quantiles = (0.5,), mt = OverlapMeanTest())
        by_hand = forecast_evaluation_summary(forecast_evaluation_align([early, late]);
                                              names = ["e", "l"], quantiles = (0.5,),
                                              mt = OverlapMeanTest())
        @test fs.names == ["e", "l"]
        for f in FS_CORE
            @test isequal(getfield(fs, f), getfield(by_hand, f))
        end
        for f in FS_SPREAD
            @test isequal(getfield(fs, f), getfield(by_hand, f))
        end
        # The aligned early member is the early member scored on rows 2 and 3, which is
        # what the bare blank answered, so the two roads agree to the digit.
        blank = copy(IC_ALPHA)
        blank[1, :] .= NaN
        blanked = forecast_evaluation_summary(forecast_evaluation(blank, y);
                                              quantiles = (0.5,), mt = OverlapMeanTest())
        for f in FS_CORE
            @test isequal(getfield(fs, f)[1], getfield(blanked, f)[1])
        end
        # The single-evaluation method carries the keyword too, and it changes nothing.
        @test isequal(forecast_evaluation_summary(late; align = true,
                                                  mt = OverlapMeanTest()).spearman_mean_ic,
                      forecast_evaluation_summary(late; mt = OverlapMeanTest()).spearman_mean_ic)
    end
end

@testset "The quantile block is absent unless a quantile is asked for" begin
    PO = PortfolioOptimisers
    fe = forecast_evaluation(IC_ALPHA, PO.forward_mean_returns(IC_ALPHA, 1, 1))

    @testset "No quantile leaves the five entries `nothing`, read by dispatch" begin
        fs = forecast_evaluation_summary(fe; mt = OverlapMeanTest())
        @test isnothing(fs.quantiles)
        for f in FS_SPREAD
            @test isnothing(getfield(fs, f))
        end
        @test PO.forecast_summary_spreads([fe], nothing) ==
              (nothing, nothing, nothing, nothing, nothing)
    end

    @testset "One column per quantile, in the order they were asked for" begin
        fs = forecast_evaluation_summary([fe, fe]; quantiles = (0.5, 0.1),
                                         mt = OverlapMeanTest())
        @test fs.quantiles == [0.5, 0.1]
        for f in FS_SPREAD
            @test size(getfield(fs, f)) == (2, 2)
        end
        @test fs.spread_ann_return[1, :] ==
              forecast_quantile_spread(fe; quantiles = (0.5, 0.1)).ann_mean
        @test fs.spread_ann_return[1, :] == fs.spread_ann_return[2, :]
    end

    @testset "An impossible cut is refused by the verb that owns it" begin
        @test_throws DomainError forecast_evaluation_summary(fe; quantiles = (0.9,),
                                                             mt = OverlapMeanTest())
        @test_throws PO.IsEmptyError forecast_evaluation_summary(fe; quantiles = (),
                                                                 mt = OverlapMeanTest())
    end
end

@testset "Every hit rate of a summary counts against the dates that scored" begin
    PO = PortfolioOptimisers
    # A date under the threshold carries no coefficient and no book, and both hit rates read
    # it as a date at which nothing was measured, so the five hit rates of one Result sit over
    # the same sample as the means beside them. Until 2026-09-14 the coefficient hit rate
    # alone counted every date, so it read 2/3 here against the book's 1.
    y = PO.forward_mean_returns(IC_ALPHA, 1, 1)
    fe = forecast_evaluation(IC_ALPHA, y; min_count = size(IC_ALPHA, 2) + 1)
    gap = ic_gap_fixture()
    thin = forecast_evaluation(gap, PO.forward_mean_returns(gap, 1, 1); min_count = 3)
    fs = forecast_evaluation_summary(thin; mt = OverlapMeanTest())

    @testset "The coefficient hit rate counts against the dates that scored" begin
        # The middle date carries two assets, under the threshold of three, so it takes no
        # coefficient. Two of three dates score, and both score positive.
        @test count(isfinite, forecast_ic(thin)[:, 1]) == 2
        @test fs.spearman_hit_rate[1] ≈ 1.0
        @test fs.pearson_hit_rate[1] ≈ 1.0
        # The silenced date is the coverage's to report, once.
        @test fs.min_coverage[1] < 1
        @test fs.min_n_scored[1] == 2
    end

    @testset "The book hit rate counts against the dates that traded" begin
        rk = forecast_portfolio(thin; kind = :rank)
        @test count(isfinite, rk.ret) == 2
        @test fs.rank_hit_rate[1] ≈ 1.0
        @test fs.zscore_hit_rate[1] ≈ 1.0
    end

    @testset "The one denominator is the two kernels' shared convention" begin
        @test fs.spearman_hit_rate[1] == fs.rank_hit_rate[1]
        @test fs.spearman_hit_rate[1] ==
              PO.exposure_ic_factor_summary(forecast_ic(thin), 1; mt = OverlapMeanTest()).hit_rate
        @test fs.rank_hit_rate[1] ==
              PO.forecast_hit_rate(forecast_portfolio(thin; kind = :rank).ret)
        # A series with no finite entry has no hit rate under either kernel.
        @test isnan(PO.exposure_ic_factor_summary(forecast_ic(fe), 1;
                                                  mt = OverlapMeanTest()).hit_rate)
        @test isnan(PO.forecast_hit_rate(forecast_ic(fe)[:, 1]))
    end

    @testset "A sample no date can score is refused by the portfolio verb" begin
        @test_throws ArgumentError forecast_evaluation_summary(fe; mt = OverlapMeanTest())
    end
end

@testset "Two members of the family are compared on the synthetic panel" begin
    # The closing condition of map #931: two of the four shipped members are scored on the
    # panel of #656 and set side by side in one Result. The fixture is `planted`, so the
    # member that fits a relation has one to find and the member that only weights its
    # Descriptors does not -- which is what makes the comparison a measurement rather than
    # a pair of numbers. Convention 4.
    PO = PortfolioOptimisers
    px = evaluation_fixture(; planted = true)
    fw = FixedWeightedReturnForecast(; scores = px.scores, scale = 1.0,
                                     weights = [0.4, 0.6])
    tgt = TargetReturnForecast(; scores = px.scores, horizon = 2, lag = 1,
                               calibrate = false)
    fes = [forecast_evaluation(m, px.rd, px.csfm; horizon = 2, lag = 1, step = 1,
                               ppy = 252) for m in (fw, tgt)]
    fs = forecast_evaluation_summary(fes; names = ["fixed weighted", "target"],
                                     quantiles = (0.2,), mt = OverlapMeanTest())

    @testset "Both members carry a row, and the rows are not the same row" begin
        @test fs.names == ["fixed weighted", "target"]
        @test length(fs.spearman_mean_ic) == 2
        @test fs.spearman_mean_ic[1] != fs.spearman_mean_ic[2]
        @test size(fs.spread_ann_return) == (2, 1)
        @test fs.ppy == 252
    end

    @testset "Every column of both rows is a number a caller can read" begin
        for f in FS_CORE
            v = getfield(fs, f)
            @test length(v) == 2
            @test all(x -> isa(x, Real), v)
        end
    end

    @testset "The planted relation is what the summary reports as skill" begin
        # The member that refits against the idiosyncratic return finds the planted
        # relation; the one that only weights its Descriptors scores nothing like it.
        @test fs.spearman_mean_ic[2] > 0.5
        @test fs.spearman_t_stat[2] > 2
        @test fs.rank_ann_return[2] > 0
        @test fs.mean_coverage[2] > 0.5
        @test fs.min_n_scored[2] >= fes[2].min_count
    end

    @testset "The comparison is the same call as the single evaluation" begin
        alone = forecast_evaluation_summary(fes[2]; names = ["target"], quantiles = (0.2,),
                                            mt = OverlapMeanTest())
        @test alone.names == ["target"]
        for f in FS_CORE
            @test getfield(alone, f)[1] == getfield(fs, f)[2]
        end
        @test alone.spread_ann_return[1, :] == fs.spread_ann_return[2, :]
    end

    @testset "The block method resolves the weights the metric names" begin
        wf = forecast_evaluation_summary(fes, px.csfm;
                                         weighting = PO.InverseIdiosyncraticVarianceMetric(),
                                         mt = OverlapMeanTest())
        @test wf.spearman_mean_ic == fs.spearman_mean_ic
        @test wf.pearson_mean_ic != fs.pearson_mean_ic
        @test forecast_evaluation_summary(fes[1], px.csfm; mt = OverlapMeanTest()).pearson_mean_ic[1] ==
              fs.pearson_mean_ic[1]
    end

    @testset "Two members on two grids are compared in one call at the Estimator layer" begin
        # Issue #1073. The two members above happen to share a grid; a member whose
        # published history starts late does not, and the comparison is then one call
        # with `align = true` rather than a blank through the bare method. The late
        # member is the fixed-weighted Result with the first five rows of its published
        # history withheld, scored through the Result layer.
        rfr = return_forecast(fw, px.rd, px.csfm)
        hist = copy(rfr.hist)
        hist[1:5, :] .= NaN
        late = forecast_evaluation(FixedWeightedReturnForecastResult(; mu = rfr.mu,
                                                                     hist = hist,
                                                                     weights = rfr.weights),
                                   px.rd, px.csfm; horizon = 2, lag = 1, step = 1,
                                   ppy = 252)
        @test first(late.dates) == 6
        @test first(fes[2].dates) == 1
        @test_throws PO.ConflictingArgumentError forecast_evaluation_summary([late, fes[2]];
                                                                             mt = OverlapMeanTest())
        fa = forecast_evaluation_summary([late, fes[2]], px.csfm; align = true,
                                         names = ["late fixed weighted", "target"],
                                         mt = OverlapMeanTest())
        @test fa.names == ["late fixed weighted", "target"]
        al = forecast_evaluation_align([late, fes[2]])
        @test al[1].dates == al[2].dates == late.dates
        @test fa.spearman_mean_ic[1] ==
              forecast_evaluation_summary(late, px.csfm; mt = OverlapMeanTest()).spearman_mean_ic[1]
        @test fa.spearman_mean_ic[2] !=
              forecast_evaluation_summary(fes[2], px.csfm; mt = OverlapMeanTest()).spearman_mean_ic[1]
        @test fa.spearman_mean_ic[2] ==
              forecast_evaluation_summary(al[2], px.csfm; mt = OverlapMeanTest()).spearman_mean_ic[1]
    end
end

#=
The docstrings of `14_ForecastSummary.jl` against numbers. Each probe computes, by hand, the
number a `# Mathematical definition` or a sentence of the file states, and compares it with
the number the code returns. The fixture holds a date with no forecast, a date under the
threshold and two late listers, so the hit rates, the shares and the counts each read a
sample that differs from the evaluation dates.
=#
function summary_docstring_fixture()
    T, N = 40, 12
    rng = StableRNG(962)
    alpha = randn(rng, T, N)
    X = randn(rng, T, N)
    y = PortfolioOptimisers.forward_mean_returns(X, 1, 1)
    alpha[5, :] .= NaN
    alpha[9, 1:10] .= NaN
    umsk = trues(T, N)
    umsk[1:6, 11:12] .= false
    fe = forecast_evaluation(alpha, y; umsk = umsk, min_count = 3)
    return (; T = T, N = N, y = y, umsk = umsk, fe = fe)
end

@testset "The docstrings of 14_ForecastSummary.jl against numbers" begin
    PO = PortfolioOptimisers
    sx = summary_docstring_fixture()
    fe = sx.fe

    @testset "The scored count is the universe assets that carry a finite pair" begin
        u = PO.forecast_ic_weights(fe.alpha, nothing)
        n = PO.forecast_summary_scored(fe, u)
        nh = [count(i -> fe.umsk[t, i] &&
                         u[t, i] > 0 &&
                         isfinite(fe.alpha[t, i]) &&
                         isfinite(fe.y[t, i]), 1:(sx.N)) for t in fe.dates]
        @test n == nh
        # It is the numerator of the coverage, over the universe of the date.
        U = [count(i -> fe.umsk[t, i] && u[t, i] > 0, 1:(sx.N)) for t in fe.dates]
        @test isequal(forecast_coverage(fe), n ./ U)
        @test n[findfirst(==(5), fe.dates)] == 0
    end

    @testset "The four coverage figures are the means and the minima of the definition" begin
        u = PO.forecast_ic_weights(fe.alpha, nothing)
        c = forecast_coverage(fe)
        n = PO.forecast_summary_scored(fe, u)
        Jc = findall(isfinite, c)
        cv = PO.forecast_summary_coverage(fe, nothing)
        @test cv.mean_coverage ≈ sum(c[Jc]) / length(Jc)
        @test cv.min_coverage == minimum(c[Jc])
        @test cv.mean_n_scored ≈ sum(n) / length(n)
        @test cv.min_n_scored == minimum(n) == 0
        # A date whose universe is empty enters no share figure and counts zero assets.
        w = ones(sx.T, sx.N)
        w[3, :] .= 0
        cw = forecast_coverage(fe, w)
        nw = PO.forecast_summary_scored(fe, w)
        j3 = findfirst(==(3), fe.dates)
        @test isnan(cw[j3]) && nw[j3] == 0
        cvw = PO.forecast_summary_coverage(fe, w)
        @test cvw.mean_coverage ≈ sum(filter(isfinite, cw)) / count(isfinite, cw)
        @test cvw.mean_n_scored ≈ sum(nw) / length(nw)
        # Every universe empty: the two shares are NaN, and the two counts are zero.
        cvz = PO.forecast_summary_coverage(fe, zeros(sx.T, sx.N))
        @test isnan(cvz.mean_coverage) && isnan(cvz.min_coverage)
        @test cvz.mean_n_scored == 0 && cvz.min_n_scored == 0
    end

    @testset "Every hit rate counts the scored dates and not every date" begin
        fs = forecast_evaluation_summary(fe; mt = OverlapMeanTest())
        ic = forecast_ic(fe)
        rk = forecast_portfolio(fe; kind = :rank)
        zs = forecast_portfolio(fe; kind = :zscore)
        scored(x) = count(>(0), filter(isfinite, x)) / count(isfinite, x)
        every(x) = count(>(0), filter(isfinite, x)) / length(x)
        @test fs.spearman_hit_rate[1] == scored(ic[:, 1]) != every(ic[:, 1])
        @test fs.pearson_hit_rate[1] == scored(ic[:, 2]) != every(ic[:, 2])
        @test fs.rank_hit_rate[1] == scored(rk.ret) != every(rk.ret)
        @test fs.zscore_hit_rate[1] == scored(zs.ret) != every(zs.ret)
    end

    @testset "Each book reports three figures that read no order of the returns" begin
        fs = forecast_evaluation_summary(fe; mt = OverlapMeanTest())
        rk = forecast_portfolio(fe; kind = :rank)
        @test fs.rank_ann_return[1] == rk.summary.ann_return
        @test fs.rank_ann_volatility[1] == rk.summary.ann_volatility
        @test fs.rank_sharpe[1] == rk.summary.sharpe
        r = filter(isfinite, rk.ret)
        p = r[randperm(StableRNG(3), length(r))]
        a = performance_summary(r)
        b = performance_summary(p)
        @test a.ann_return ≈ b.ann_return &&
              a.ann_volatility ≈ b.ann_volatility &&
              a.sharpe ≈ b.sharpe
        @test a.sortino ≈ b.sortino && a.cvar ≈ b.cvar && a.sharpe_stderr ≈ b.sharpe_stderr
        @test !(a.max_drawdown ≈ b.max_drawdown)
        @test length(PO.forecast_summary_row(fe, nothing, 10, OverlapMeanTest())) == 39
        @test collect(keys(PO.forecast_summary_row(fe, nothing, 10, OverlapMeanTest()))) ==
              collect(fieldnames(ForecastSummaryResult)[2:40])
    end

    @testset "The aligned grid is the set the definition states, on the later phase" begin
        f1 = forecast_evaluation(fe.alpha, sx.y; umsk = sx.umsk, step = 2, min_count = 3)
        late = copy(fe.alpha)
        late[1:9, :] .= NaN
        f2 = forecast_evaluation(late, sx.y; umsk = sx.umsk, step = 2, min_count = 3)
        @test isodd(first(f1.dates)) && iseven(first(f2.dates))
        lo = max(first(f1.dates), first(f2.dates))
        hi = min(last(f1.dates), last(f2.dates))
        D = [lo + 2 * q for q in 0:hi if lo + 2 * q <= hi]
        al = forecast_evaluation_align([f1, f2])
        @test al[1].dates == al[2].dates == D
        @test al[1].alpha === f1.alpha && al[1].y === f1.y && al[1].umsk === f1.umsk
        # Two grids that share no date are refused.
        g1 = PO.ForecastEvaluationResult(fe.alpha, fe.y, fe.umsk, [1, 2, 3], fe.target,
                                         fe.horizon, fe.lag, fe.step, fe.min_count, fe.ties,
                                         fe.ppy)
        g2 = PO.ForecastEvaluationResult(fe.alpha, fe.y, fe.umsk, [10, 11], fe.target,
                                         fe.horizon, fe.lag, fe.step, fe.min_count, fe.ties,
                                         fe.ppy)
        @test_throws PO.IsEmptyError forecast_evaluation_align([g1, g2])
    end

    @testset "A single tail fraction is read as a vector of one" begin
        a = forecast_evaluation_summary(fe; quantiles = 0.2, mt = OverlapMeanTest())
        b = forecast_evaluation_summary(fe; quantiles = [0.2], mt = OverlapMeanTest())
        @test a.quantiles == [0.2]
        @test a.spread_sharpe == b.spread_sharpe && size(a.spread_sharpe) == (1, 1)
    end

    @testset "Two Panel Field targets are one target only when they name one field" begin
        @test PO.forecast_summary_same_target(PanelFieldTarget(; name = "a"),
                                              PanelFieldTarget(; name = "a"))
        @test !PO.forecast_summary_same_target(PanelFieldTarget(; name = "a"),
                                               PanelFieldTarget(; name = "b"))
        @test !PO.forecast_summary_same_target(IdiosyncraticTarget(), AssetReturnTarget())
    end
end

@testset "The docstrings of 12_ForecastForwardWindows.jl against numbers" begin
    PO = PortfolioOptimisers
    alpha = WINDOW_ALPHA
    fe = forecast_evaluation(alpha, PO.forward_mean_returns(alpha, 1, 1); step = 1)

    @testset "Each grid pair is the half-open window of the definition" begin
        Xw = randn(StableRNG(952), 30, 3)
        kinds = (:cumulative, :disjoint)
        for (h, l, n) in ((1, 0, 4), (2, 1, 3), (3, 2, 3))
            for kind in kinds
                g = PO.forecast_window_grid(h, l, n, kind)
                for p in 1:n
                    lo = kind === :cumulative ? l : l + (p - 1) * h
                    hi = l + p * h
                    Yw = PO.forward_mean_returns(Xw, g[p]...)
                    @test all(Yw[t, i] ≈ mean(Xw[(t + lo):(t + hi - 1), i])
                              for t in 1:(30 - hi + 1), i in 1:3)
                end
            end
        end
    end

    @testset "The common dates are the set of the definition" begin
        rng = StableRNG(9520)
        for _ in 1:100
            nt, na = rand(rng, 5:12), rand(rng, 2:6)
            a = randn(rng, nt, na)
            a[rand(rng, nt, na) .< 0.25] .= NaN
            xr = randn(rng, nt, na)
            xr[rand(rng, nt, na) .< 0.2] .= NaN
            g = PO.forecast_window_grid(rand(rng, 1:2), rand(rng, 0:2), rand(rng, 1:3),
                                        rand(rng, (:cumulative, :disjoint)))
            ys = [PO.forward_mean_returns(xr, h, l) for (h, l) in g]
            mc = rand(rng, 1:na)
            set = [t
                   for t in 1:nt
                   if all(count(i -> isfinite(a[t, i]) && isfinite(y[t, i]), 1:na) >= mc
                          for y in ys)]
            @test PO.forecast_common_dates(a, ys, 1:nt, mc) == set
        end
    end

    @testset "A zero weight leaves the Pearson coefficient NaN on a kept date" begin
        wz = ones(8, 4)
        wz[2, 1:2] .= 0.0
        t = forecast_holding_period(fe, alpha, wz; n = 2, mt = OverlapMeanTest())
        @test 2 ∈ t.dates
        f1 = PO.ForecastEvaluationResult(fe.alpha, PO.forward_mean_returns(alpha, 1, 1),
                                         fe.umsk, t.dates, fe.target, 1, 1, 1, 3, fe.ties,
                                         fe.ppy)
        ic = forecast_ic(f1, wz)
        j = findfirst(==(2), t.dates)
        @test isnan(ic[j, 2])
        @test isfinite(ic[j, 1])
        @test all(isfinite, forecast_portfolio(f1; kind = :rank).ret)
    end

    @testset "Every figure of a row takes the element type of the pairing" begin
        a32 = Float32.(alpha)
        f32 = forecast_evaluation(a32, PO.forward_mean_returns(a32, 1, 1); step = 1)
        for n in (2, 8)
            t = forecast_holding_period(f32, a32, ones(Float64, 8, 4); n = n,
                                        mt = OverlapMeanTest())
            @test all(eltype(t[k]) === Float32 for k in keys(t)[5:end])
        end
    end

    @testset "A wrong weight history raises at every depth" begin
        @test_throws DimensionMismatch forecast_holding_period(fe, alpha, ones(3, 3); n = 2,
                                                               mt = OverlapMeanTest())
        @test_throws DimensionMismatch forecast_holding_period(fe, alpha, ones(3, 3); n = 8,
                                                               mt = OverlapMeanTest())
        @test_throws DimensionMismatch forecast_decay(fe, alpha, ones(3, 3); n = 8,
                                                      mt = OverlapMeanTest())
        @test_throws DomainError forecast_holding_period(fe, alpha, -ones(8, 4); n = 8,
                                                         mt = OverlapMeanTest())
    end

    @testset "The plain t-ratio spreads with the depth only for a persistent forecast" begin
        rng = StableRNG(9522)
        nr, nt, na, depth = 100, 240, 20, 5
        for (rho, grows) in ((1.0, true), (0.0, false))
            plain = zeros(nr, depth)
            corrected = zeros(nr, depth)
            for m in 1:nr
                xm = randn(rng, nt, na)
                am = zeros(nt, na)
                am[1, :] = randn(rng, na)
                for t in 2:nt
                    am[t, :] = rho .* am[t - 1, :] .+ sqrt(1 - rho^2) .* randn(rng, na)
                end
                fm = forecast_evaluation(am, PO.forward_mean_returns(xm, 1, 1))
                tb = forecast_holding_period(fm, xm; n = depth, mt = OverlapMeanTest())
                plain[m, :] = tb.spearman_ic_ir .* sqrt(length(tb.dates))
                corrected[m, :] = tb.spearman_t_stat
            end
            sp = vec(std(plain; dims = 1))
            sc = vec(std(corrected; dims = 1))
            @test (sp[depth] / sp[1] > 1.6) == grows
            @test sc[depth] / sc[1] < 1.3
        end
    end
end

@testset "The docstrings of 09_ForecastInformationCoefficient.jl against numbers" begin
    PO = PortfolioOptimisers
    # The definitions, written out by hand. Every assignment is local, so no helper writes
    # a variable of the testset.
    function ic_hand_wcor(a, b, u, mc)
        local k = [i
                   for i in eachindex(a)
                   if isfinite(a[i]) && isfinite(b[i]) && isfinite(u[i]) && u[i] > 0]
        length(k) < mc && return NaN
        local wk, xk, yk = u[k], a[k], b[k]
        local mx, my = sum(wk .* xk) / sum(wk), sum(wk .* yk) / sum(wk)
        local num = sum(wk .* (xk .- mx) .* (yk .- my))
        local den = sqrt(sum(wk .* (xk .- mx) .^ 2) * sum(wk .* (yk .- my) .^ 2))
        return den > 1e-12 ? num / den : NaN
    end
    function ic_hand_ranks(x)
        local r = similar(x)
        r[sortperm(x)] = 1:length(x)
        return r
    end
    function ic_hand_spearman(a, b, mc)
        local k = [i for i in eachindex(a) if isfinite(a[i]) && isfinite(b[i])]
        length(k) < mc && return NaN
        return ic_hand_wcor(ic_hand_ranks(a[k]), ic_hand_ranks(b[k]), ones(length(k)), mc)
    end

    @testset "Each column is its correlation over the assets the definition admits" begin
        rng = StableRNG(948)
        for _ in 1:150
            nt, na = rand(rng, 5:10), rand(rng, 3:8)
            a = randn(rng, nt, na)
            a[rand(rng, nt, na) .< 0.1] .= NaN
            wr = rand(rng, nt, na)
            wr[rand(rng, nt, na) .< 0.15] .= rand(rng, [0.0, NaN, Inf])
            mc = rand(rng, 1:4)
            fr = forecast_evaluation(a, PO.forward_mean_returns(randn(rng, nt, na), 2, 1);
                                     horizon = 2, step = rand(rng, 1:2), min_count = mc)
            ic = forecast_ic(fr, wr)
            for (j, t) in enumerate(fr.dates)
                s = ic_hand_spearman(fr.alpha[t, :], fr.y[t, :], mc)
                p = ic_hand_wcor(fr.alpha[t, :], fr.y[t, :], wr[t, :], mc)
                @test isequal(s, ic[j, 1]) || isapprox(s, ic[j, 1]; atol = 1e-12)
                @test isequal(p, ic[j, 2]) || isapprox(p, ic[j, 2]; atol = 1e-12)
            end
        end
    end

    @testset "Equal forecasts share a rank under `:average` and take the asset order under `:ordinal`" begin
        flat = [1.0 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0 0.0]
        up = [1.0 2.0 3.0 4.0 5.0; 0.0 0.0 0.0 0.0 0.0]
        # Under the default, a constant forecast has constant ranks, so the Spearman column
        # is `NaN`, as the Pearson column is.
        for yc in (up, reverse(up; dims = 2))
            icc = forecast_ic(forecast_evaluation(flat, yc))
            @test isnan(icc[1, 1])
            @test isnan(icc[1, 2])
        end
        # Under `:ordinal`, the order of the assets sets the Spearman column.
        icu = forecast_ic(forecast_evaluation(flat, up; ties = :ordinal))
        icd = forecast_ic(forecast_evaluation(flat, reverse(up; dims = 2); ties = :ordinal))
        @test icu[1, 1] == 1.0
        @test icd[1, 1] == -1.0
        @test isnan(icu[1, 2])
        @test isnan(icd[1, 2])
        # Two targets that differ only inside the ties of the forecast get one coefficient
        # under the default, and two under `:ordinal`.
        a2 = [1.0 1.0 2.0 2.0; 0.0 0.0 0.0 0.0]
        y1 = [1.0 2.0 3.0 4.0; 0.0 0.0 0.0 0.0]
        y2 = [2.0 1.0 4.0 3.0; 0.0 0.0 0.0 0.0]
        @test forecast_ic(forecast_evaluation(a2, y1))[1, 1] ≈ 4 / sqrt(20)
        @test forecast_ic(forecast_evaluation(a2, y2))[1, 1] ≈ 4 / sqrt(20)
        @test forecast_ic(forecast_evaluation(a2, y1; ties = :ordinal))[1, 1] ≈ 1.0
        @test forecast_ic(forecast_evaluation(a2, y2; ties = :ordinal))[1, 1] ≈ 0.6
    end

    @testset "A weight that is not finite is outside the universe of every verb" begin
        # A zero idiosyncratic variance gives an infinite weight under the inverse-variance
        # metric, which reads 1 ./ vs. Asset 1 carries it at the first date, and no forecast
        # there, so the coverage read it as a missed asset and reported 3/4.
        vs = [0.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
        wi = 1 ./ vs
        a = [NaN 2.0 3.0 4.0; 2.0 1.0 4.0 3.0]
        yi = [1.0 2.0 3.0 4.0; 0.0 0.0 0.0 0.0]
        fi = forecast_evaluation(a, yi; min_count = 2)
        @test isinf(wi[1, 1])
        @test fi.dates == [1, 2]
        @test forecast_coverage(fi, wi)[1] == 1.0
        # With a finite forecast, asset 1 was counted as scored, 4 against the 3 the
        # Pearson column reads.
        a2 = [1.0 2.0 3.0 4.0; 2.0 1.0 4.0 3.0]
        f2 = forecast_evaluation(a2, yi; min_count = 2)
        @test PO.forecast_summary_scored(f2, PO.forecast_ic_weights(f2.alpha, wi)) ==
              [3.0, 4.0]
        @test forecast_coverage(f2, wi) == [1.0, 1.0]
        @test forecast_ic(f2, wi)[1, 2] ≈ ic_hand_wcor(a2[1, 2:4], yi[1, 2:4], ones(3), 2)
        wn = copy(wi)
        wn[1, 1] = NaN
        @test isequal(forecast_coverage(f2, wn), forecast_coverage(f2, wi))
        @test isequal(forecast_ic(f2, wn), forecast_ic(f2, wi))
    end

    @testset "The lag counts the later dates whose windows overlap a given one" begin
        for h in 1:12, s in 1:12
            @test PO.forecast_ic_lags(h, s) == count(j -> j * s < h, 1:12)
        end
    end

    @testset "The t-statistic is the mean over the long-run standard error" begin
        rng = StableRNG(9480)
        for _ in 1:100
            nr = rand(rng, 4:30)
            ic = randn(rng, nr, 2)
            ic[rand(rng, nr, 2) .< 0.1] .= NaN
            nl = rand(rng, 0:4)
            sm = forecast_ic_summary(ic; lags = nl, mt = OverlapMeanTest())
            for (k, v) in ((1, sm.spearman), (2, sm.pearson))
                c = ic[:, k]
                f = filter(isfinite, c)
                n = length(f)
                n < 2 && continue
                m = sum(f) / n
                g = sum(abs2, f .- m)
                for l in 1:nl, t in 1:(nr - l)
                    if isfinite(c[t]) && isfinite(c[t + l])
                        g += 2 * (c[t] - m) * (c[t + l] - m)
                    end
                end
                # The divisor `n - 2L - 1` removes the bias of the centring at order `L`.
                nu = n - 2 * nl - 1
                tt = g > 0 && nu > 0 ? m / sqrt(g / nu) * sqrt(n) : NaN
                @test isequal(tt, v.t_stat) || isapprox(tt, v.t_stat; rtol = 1e-12)
                @test v.mean_ic ≈ m
                @test v.std_ic ≈ sqrt(sum(abs2, f .- m) / (n - 1))
                @test v.hit_rate == count(>(0), f) / n
            end
        end
    end

    @testset "The plain ratio overstates a persistent forecast and not a fresh one" begin
        rng = StableRNG(9481)
        nr, nt, na, h = 100, 200, 16, 4
        for (rho, overstated) in ((0.99, true), (0.0, false))
            plain = zeros(nr)
            corrected = zeros(nr)
            for m in 1:nr
                am = zeros(nt, na)
                am[1, :] = randn(rng, na)
                for t in 2:nt
                    am[t, :] = rho .* am[t - 1, :] .+ sqrt(1 - rho^2) .* randn(rng, na)
                end
                xm = randn(rng, nt, na)
                fm = forecast_evaluation(am, PO.forward_mean_returns(xm, h, 1); horizon = h,
                                         step = 1)
                icm = forecast_ic(fm)
                plain[m] = forecast_ic_summary(icm; mt = OverlapMeanTest()).spearman.ic_ir *
                           sqrt(count(isfinite, icm[:, 1]))
                corrected[m] = forecast_ic_summary(icm; lags = PO.forecast_ic_lags(fm),
                                                   mt = OverlapMeanTest()).spearman.t_stat
            end
            @test (std(plain) > 1.5) == overstated
            @test 0.75 < std(corrected) < 1.3
        end
    end

    @testset "The columns and the coverage take the promotion of the pair and the weights" begin
        a32 = Float32.(IC_ALPHA)
        f32 = forecast_evaluation(a32, PO.forward_mean_returns(a32, 1, 1))
        @test eltype(forecast_ic(f32)) === Float32
        @test eltype(forecast_coverage(f32)) === Float32
        @test eltype(forecast_ic(f32, ones(Float64, 4, 4))) === Float64
        @test eltype(forecast_coverage(f32, ones(Float64, 4, 4))) === Float64
    end
end

@testset "The docstrings of 10_ForecastPortfolios.jl against numbers" begin
    PO = PortfolioOptimisers

    @testset "A flat cross-section holds no book under either kind" begin
        # The mean of equal values rounds off them at 3, 5, 6 and 7 assets, so a centring
        # that trusts it writes a short book of 200 % gross instead of a zero row. Under the
        # default tie rule, the ranks of a flat cross-section are equal too. Under
        # `:ordinal`, they follow the asset axis, so the rank kind holds a book.
        for na in 1:7, v in (0.1, 0.3, 0.7, 1 / 3, 0.123456789, -0.2, 2.2),
            kind in (:rank, :zscore), tr in (:average, :ordinal)

            a = fill(v, 1, na)
            wf = PO.forecast_portfolio_weights(a, a, [1], kind, tr)
            @test if kind === :rank && tr === :ordinal && na > 1
                isapprox(sum(abs, wf), 2)
            else
                all(iszero, wf)
            end
        end
        @test all(iszero,
                  PO.forecast_portfolio_weights(fill(0.1, 1, 3), fill(0.0, 1, 3), [1],
                                                :zscore, :average))
    end

    @testset "A constant series has a zero volatility and no ratio" begin
        for ns in 2:9, v in (0.1, 0.3, 0.7, 1 / 3, 0.123456789, -0.2, 2.2)
            sm = PO.forecast_series_summary(fill(v, ns), 12)
            @test sm.ann_mean == 12 * v
            @test iszero(sm.ann_vol)
            @test isnan(sm.ann_ir)
        end
        @test all(isnan, values(PO.forecast_series_summary(Float64[], 12)))
        sm = PO.forecast_series_summary([NaN, 0.2], 12)
        @test sm.ann_mean ≈ 2.4
        @test isnan(sm.ann_vol) && isnan(sm.ann_ir)
        @test PO.forecast_hit_rate([1.0, -1.0, NaN, 0.0, Inf]) == 1 / 3
        @test isnan(PO.forecast_hit_rate([NaN]))
    end

    # Date 2 carries two assets under `min_count = 3`. Its book is built, and its return
    # is not scored.
    alpha = [1.0 2.0 3.0 4.0
             1.0 2.0 NaN NaN
             1.0 2.0 3.0 4.0]
    gap_y(g) = [0.0 0.0 0.0 -0.1
                -g g 0.0 0.0
                0.0 0.0 0.0 -0.1]

    @testset "A date below the threshold keeps its book, and the next turnover reads it" begin
        fe = forecast_evaluation(alpha, gap_y(0.1); min_count = 3, ppy = 1, step = 1)
        p = forecast_portfolio(fe; kind = :rank)
        @test p.w[2, :] == [-1.0, 1.0, 0.0, 0.0]
        @test isnan(p.ret[2])
        @test isnan(p.turnover[2])
        @test p.turnover[3] ≈ sum(abs, p.w[3, :] - p.w[2, :])
        @test p.turnover[3] ≈ 2.5
    end

    @testset "A gap hides its own return, so the compressed drawdown moves either way" begin
        for (g, deeper) in ((0.1, true), (-0.1, false))
            fe = forecast_evaluation(alpha, gap_y(g); min_count = 3, ppy = 1, step = 1)
            p = forecast_portfolio(fe; kind = :rank)
            held = copy(p.ret)
            held[2] = sum(p.w[2, i] * gap_y(g)[2, i] for i in 1:4)
            full = performance_summary(held; ppy = 1)
            @test p.summary.max_drawdown ≈ -0.15
            @test (p.summary.max_drawdown < full.max_drawdown) == deeper
        end
    end

    @testset "The quantile spread reports the four figures of the series summary" begin
        fe = forecast_evaluation(alpha[[1, 3], :], gap_y(0.1)[[1, 3], :]; min_count = 3,
                                 ppy = 4, step = 1)
        q = forecast_quantile_spread(fe; quantiles = [0.25, 0.5])
        @test keys(q) == (:spread, :ann_mean, :ann_vol, :ann_ir, :hit_rate)
        for j in 1:2
            sm = PO.forecast_series_summary(q.spread[:, j], 4)
            @test isequal((q.ann_mean[j], q.ann_vol[j], q.ann_ir[j], q.hit_rate[j]),
                          values(sm))
        end
        @test isequal(forecast_quantile_spread(fe; quantiles = 0.25).spread,
                      q.spread[:, 1:1])
    end

    @testset "Both tails are inclusive, so a small cross-section puts one asset in both" begin
        @test PO.forecast_tail_spread([1.0], [2.0], 0.1) == 0
        # At q = 0.5 the median asset is in both tails.
        @test PO.forecast_tail_spread([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 0.5) ≈
              (20 + 30) / 2 - (10 + 20) / 2
    end
end

# A member that counts its fits and forwards each one to a shipped member. It carries no
# method of `forecast_history`, so it also proves that a new member needs none.
struct HistoryFitCounter{R} <: PortfolioOptimisers.AbstractReturnForecastEstimator
    inner::R
    n::Base.RefValue{Int}
end
function PortfolioOptimisers.return_forecast(c::HistoryFitCounter,
                                             rd::PortfolioOptimisers.ReturnsResult,
                                             csfm::CrossSectionalFactorModel)
    c.n[] += 1
    return PortfolioOptimisers.return_forecast(c.inner, rd, csfm)
end

@testset "The docstrings of 08_ForecastHistory.jl against numbers" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    rd, csfm, scores, rows = fx.rd, fx.csfm, fx.scores, fx.rows
    Tb, N = fx.Tb, fx.N
    fw = FixedWeightedReturnForecast(; scores = scores, scale = 1.0, weights = [0.4, 0.6])
    ew = ExpWeightedReturnForecast(; scores = scores, horizon = 2, lag = 1, scale = 1.0,
                                   min_obs = 3)
    tgt = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1, calibrate = false)
    tgtc = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1, calibrate = true)
    members = (fw, ew, tgt, tgtc)

    @testset "No member reads b, esigma, L, fcb, rf, or the values of M" begin
        K = size(csfm.Ms, 3)
        fcb = PO.FactorFamilyBasis(; fnm = ["industry"], fi = [collect(2:K)], di = [K - 1],
                                   ratios = ones(Tb, K - 2), K = K)
        full = CrossSectionalFactorModel(; M = 3 .* csfm.M .+ 1,
                                         L = randn(StableRNG(3), N, K - 1),
                                         b = randn(StableRNG(4), N), csr = csfm.csr,
                                         Ms = csfm.Ms, vs = csfm.vs, esigma = fill(0.3, N),
                                         nf = csfm.nf, fam = csfm.fam, fcb = fcb,
                                         rf = return_forecast(fw, rd, csfm))
        for m in members
            mu = return_forecast(m, rd, full).mu
            @test isequal(mu, return_forecast(m, rd, csfm).mu)
            # The fit on the cut at the last row is the fit on the whole sample.
            @test isequal(mu,
                          return_forecast(m, PO.port_opt_view(rd, 1:rows[Tb], :),
                                          PO.forecast_history_block(full, Tb)).mu)
            @test isequal(forecast_history(m, rd, full; step = 3),
                          forecast_history(m, rd, csfm; step = 3))
        end
        cut = PO.forecast_history_block(full, 10)
        @test isequal(cut.M, full.Ms[10, :, :])
        @test cut.b == full.b
        @test cut.esigma == full.esigma
        @test isnothing(getfield(cut, :L))
        @test cut.L === cut.M
        @test isnothing(cut.fcb)
        @test isnothing(cut.rf)
    end

    @testset "A grid row is the fit at that row, and the last row is mu only on the grid" begin
        mu = return_forecast(tgtc, rd, csfm).mu
        for s in (1, 3, 5, 7)
            h = forecast_history(tgtc, rd, csfm; step = s)
            grid = 1:s:Tb
            for tb in grid
                tb == Tb && continue
                @test isequal(h[tb, :],
                              return_forecast(tgtc, PO.port_opt_view(rd, 1:rows[tb], :),
                                              PO.forecast_history_block(csfm, tb)).mu)
            end
            @test all(t -> all(isnan, view(h, t, :)), setdiff(1:Tb, grid))
            if Tb in grid
                @test isequal(h[Tb, :], mu)
            else
                @test all(isnan, view(h, Tb, :))
            end
        end
    end

    @testset "Every evaluation date is a fitted row of the grid" begin
        for s in 1:7, hz in (1, 2, 4)
            fe = forecast_evaluation(tgt, rd, csfm; horizon = hz, lag = 1, step = s)
            grid = 1:s:Tb
            fin = [t
                   for t in grid
                   if any(i -> isfinite(fe.alpha[t, i]) && isfinite(fe.y[t, i]), 1:N)]
            @test issubset(fe.dates, grid)
            @test first(fe.dates) == first(fin)
            @test last(fe.dates) == last(fin)
        end
    end

    @testset "A member with a history is fitted once, and one without is refitted per row" begin
        for (m, s) in ((fw, 1), (fw, 5), (ew, 3), (tgt, 1), (tgt, 3), (tgt, 5))
            c = HistoryFitCounter(m, Ref(0))
            h = forecast_history(c, rd, csfm; step = s)
            grid = 1:s:Tb
            @test c.n[] ==
                  (m isa TargetReturnForecast ? 1 + length(grid) - (Tb in grid) : 1)
            @test size(h) == (length(PO.return_forecast_rows(rd, csfm)), N)
        end
    end

    @testset "A whole-history member reads nothing after the row it forecasts" begin
        tw = TargetReturnForecast(; scores = scores, horizon = 2, lag = 1, calibrate = true,
                                  whole_history = true)
        h = forecast_history(tw, rd, csfm; step = 3)
        for tb in 1:3:(Tb - 1)
            @test isequal(h[tb, :],
                          return_forecast(tw, PO.port_opt_view(rd, 1:rows[tb], :),
                                          PO.forecast_history_block(csfm, tb)).mu)
        end
        @test isequal(h[Tb, :], return_forecast(tw, rd, csfm).mu)
        tb = 31
        rng = StableRNG(7)
        rdp = deepcopy(rd)
        csfmp = deepcopy(csfm)
        ra = (rows[tb] + 1):(fx.T)
        rdp.X[ra, :] .= 10 .* randn(rng, length(ra), N)
        csfmp.csr.eps[(tb + 1):Tb, :] .= 5 .* randn(rng, Tb - tb, N)
        csfmp.vs[(tb + 1):Tb, :] .= 9.0
        hp = forecast_history(tw, rdp, csfmp; step = 3)
        @test isequal(view(hp, 1:tb, :), view(h, 1:tb, :))
        @test !isequal(view(hp, tb + 3, :), view(h, tb + 3, :))
    end

    @testset "A stated forecast is refused whatever the step" begin
        cv = CustomValueReturnForecast(; mu = fill(0.01, N))
        for s in (0, -1, 1, 3)
            @test_throws PO.ConflictingArgumentError forecast_history(cv, rd, csfm;
                                                                      step = s)
        end
    end

    @testset "The refit history takes the element type of mu" begin
        mu = Float32.(return_forecast(tgt, rd, csfm).mu)
        h = PO.forecast_history_refit(tgt, rd, csfm, mu, 5)
        @test eltype(h) === Float32
        @test isequal(h[1, :], Float32.(forecast_history(tgt, rd, csfm; step = 5)[1, :]))
    end
end

@testset "The docstrings of 07_ForecastEvaluation.jl against numbers" begin
    PO = PortfolioOptimisers
    fx = evaluation_fixture()
    ew = ExpWeightedReturnForecast(; scores = fx.scores, half_life = 10.0, min_obs = 1,
                                   horizon = 2, lag = 1)
    rf = return_forecast(ew, fx.rd, fx.csfm)

    @testset "Every method refuses a parameter outside its domain before it reads a history" begin
        # The carrier methods read the history at `t + lag` to build the target, so a
        # negative lag must be refused before that read, not by an index out of bounds.
        for (k, v) in
            ((:horizon, 0), (:lag, -1), (:lag, -3), (:step, 0), (:min_count, 0), (:ppy, 0))
            kw = (k => v,)
            @test_throws DomainError forecast_evaluation(rf, fx.rd, fx.csfm; kw...)
            @test_throws DomainError forecast_evaluation(ew, fx.rd, fx.csfm; kw...)
            @test_throws DomainError forecast_evaluation(rf.hist, rf.hist; kw...)
            @test_throws DomainError PO.forecast_evaluation_pairing(rf.hist, fx.rd, fx.csfm;
                                                                    kw...)
        end
        for fn in (() -> forecast_evaluation(rf, fx.rd, fx.csfm; ties = :dense),
                   () -> forecast_evaluation(ew, fx.rd, fx.csfm; ties = :dense),
                   () -> forecast_evaluation(rf.hist, rf.hist; ties = :dense))
            @test_throws PO.ConflictingArgumentError fn()
        end
        @test isnothing(PO.forecast_evaluation_assert_parameters(; horizon = 1, lag = 0,
                                                                 step = 1, min_count = 1,
                                                                 ties = :ordinal, ppy = 1))
    end

    @testset "The dates are the stride from the first scorable observation" begin
        for trial in 1:500
            rng = StableRNG(trial)
            nt, na = rand(rng, 1:9), rand(rng, 1:5)
            ad = [rand(rng) < 0.3 ? rand(rng, (NaN, Inf, -Inf)) : randn(rng)
                  for _ in 1:nt, _ in 1:na]
            yd = [rand(rng) < 0.3 ? rand(rng, (NaN, Inf)) : randn(rng)
                  for _ in 1:nt, _ in 1:na]
            sd = rand(rng, 1:4)
            sc = [t
                  for t in 1:nt if any(i -> isfinite(ad[t, i]) && isfinite(yd[t, i]), 1:na)]
            if isempty(sc)
                @test_throws PO.IsEmptyError PO.forecast_evaluation_dates(ad, yd, sd)
            else
                dd = PO.forecast_evaluation_dates(ad, yd, sd)
                @test dd == [first(sc) + (j - 1) * sd
                             for j in 1:(fld(last(sc) - first(sc), sd) + 1)]
            end
        end
        # The last date can come before the last scorable observation.
        @test PO.forecast_evaluation_dates(ones(4, 1), ones(4, 1), 2) == [1, 3]
    end

    @testset "The mask writes NaN off the universe, and no more" begin
        for trial in 1:200
            rng = StableRNG(10_000 + trial)
            am = randn(rng, rand(rng, 1:6), rand(rng, 1:5))
            mm = rand(rng, Bool, size(am)...)
            ac = copy(am)
            wm = PO.forecast_evaluation_mask(am, mm)
            @test all(k -> mm[k] ? wm[k] == am[k] : isnan(wm[k]), eachindex(mm))
            @test (wm === am) == all(mm)
            @test am == ac
        end
        @test eltype(PO.forecast_evaluation_mask(Float32[1 2; 3 4],
                                                 [true false; true true])) === Float32
    end

    @testset "A perfect idiosyncratic ranking scores less than one on the asset return" begin
        fi = forecast_evaluation(rf, fx.rd, fx.csfm; horizon = 2, lag = 1, step = 1)
        fa = forecast_evaluation(rf, fx.rd, fx.csfm; target = AssetReturnTarget(),
                                 horizon = 2, lag = 1, step = 1)
        rho_i = [PO.cs_spearman_correlation(view(fi.y, t, :), view(fi.y, t, :))
                 for t in axes(fi.y, 1)]
        rho_a = [PO.cs_spearman_correlation(view(fi.y, t, :), view(fa.y, t, :))
                 for t in axes(fi.y, 1)]
        both = [t for t in axes(fi.y, 1) if isfinite(rho_i[t]) && isfinite(rho_a[t])]
        @test !isempty(both)
        @test all(t -> rho_i[t] ≈ 1, both)
        @test all(t -> rho_a[t] <= 1 + 1e-12, both)
        @test count(t -> rho_a[t] < 1 - 1e-12, both) > 0
    end
end

#=
Parity of the Return Forecast evaluation, series by series and date by date (#1389, map #1375).

`test_08x_forecast_evaluation.jl` pins the summaries of the evaluation against literals. Here every
per-date series, every forward-window table and every summary is pinned against a stored oracle
output, on the fixture of that file. The oracle scores the same forecast history, the same target
history and the same estimation mask through its own public evaluation, so the forecast itself is
out of the comparison: a member's history is at parity since #1386.

Every stored case under `test/assets/Parity_ForecastEvaluation_<Case>_<Output>.csv.gz` is the
oracle output of one row of `FE_CASES`:

  - `Series`: one row per oracle date, the date, the Spearman and the Pearson coefficient, the
    returns and the turnovers of the rank and the z-score book, the quantile spreads, the count of
    scored assets and the coverage.
  - `Factor`: the contemporaneous correlation of the forecast with every exposure, on every row.
  - `Tables`: the holding-period table over the decay table, eleven columns each.
  - `Summary`: the coefficient, book, quantile, coverage and calibration summaries, in the layout
    of `fe_summary`.
  - `Curve`: the calibration curve, one row per bin.

WHAT THE ORACLE RANKS DIFFERENTLY. The oracle ranks a cross-section by an unstable sort, so it
ranks a tie in an arbitrary order. The forecasts of the fixture hold ties, because the winsoriser
clips both tails to one value. A rank statistic of a tie is its midrank by definition, and that is
our default `ties = :average` (#1332, Better since #1381). The stored oracle ran with its rank
replaced by the midrank, so it isolates everything else. `TiesOrdinal` ran with a stable ordinal
rank, which `ties = :ordinal` reproduces: the oracle's rule stays one keyword away.

THE FORECAST IS STORED. `Alpha` holds the forecast that the oracle scored, once for each member
(`Base` the fixed member, `EW` and `Refit`), and a case edits it as it edits its own. The test
compares the forecast of this process with it at `rtol = 1e-12` of its largest entry (measured
8.3e-16 of it under the flags below), and then scores the stored one. A
process under `--check-bounds=yes`, as `Pkg.test` runs, sums in another order: its forecast moves by
an ulp, a tie of the winsoriser breaks, and a rank or a calibration bin moves with it (#1475).

THE TWO DIFFERENCES THAT ADR 0149 RULES.

  1. The t-statistic. The oracle scales the ratio by the root of the date count wherever it reads
     it. Our standard error reads the overlap of the forward windows (`forecast_ic_lags`), so the
     two agree where the windows are disjoint. Every oracle t-statistic is our ratio times the
     root of a whole count of dates, and that identity is asserted where the two differ.
  2. A date with no scorable asset. The oracle drops it from the grid. We keep it with `NaN`
     statistics, so `step` means the same thing everywhere. No statistic reads the kept date but
     the coverage, which counts it as a silenced date by design. The summary also reports the
     count of silenced dates and the coverage over the scored dates, which is the oracle's
     coverage (#1512). The book carries its weights
     over such a date, because the forecast states nothing there: before #1389 the turnover out
     of it read a trade out of an empty book (Defect found).

The coefficients, the books, the spreads, the coverage, the calibration and the curve compare cell
by cell at `rtol = 1e-12`. Measured 3.9e-14 at most on a series and 5.5e-15 on a table. The factor
correlation compares against its largest entry (measured 5.6e-16 of it), because the one-hot columns give cells of round-off size, and the pooled mean of a
standardised forecast is round-off zero, so it compares against the pooled standard deviation
(measured 1.3e-15 of it, #1558). The t-statistic that ADR 0149 rules is the oracle's at
`rtol = 1e-14` (measured 1.8e-15) and the square of its ratio to ours is a whole count to
`rtol = 1e-13` (measured 2.5e-14).
=#

# The layout of a stored `Summary`: the coefficient summary (Spearman then Pearson, each mean,
# std, ratio, t-statistic, hit rate), the book summary (rank then z-score, each annualised mean,
# annualised volatility, ratio, hit rate, mean turnover), the quantile summary (per quantile the
# annualised mean, volatility, ratio and hit rate), the coverage summary (mean and least coverage,
# mean and least count of scored assets) and the calibration summary (slope, mean and std of the
# forecast, mean and std of the target, bins).
function fe_summary(s::ForecastSummaryResult)
    q = vec(permutedims(hcat(vec(s.spread_ann_return), vec(s.spread_ann_volatility),
                             vec(s.spread_sharpe), vec(s.spread_hit_rate))))
    return vcat(s.spearman_mean_ic, s.spearman_std_ic, s.spearman_ic_ir, s.spearman_t_stat,
                s.spearman_hit_rate, s.pearson_mean_ic, s.pearson_std_ic, s.pearson_ic_ir,
                s.pearson_t_stat, s.pearson_hit_rate, s.rank_ann_return,
                s.rank_ann_volatility, s.rank_sharpe, s.rank_hit_rate, s.rank_mean_turnover,
                s.zscore_ann_return, s.zscore_ann_volatility, s.zscore_sharpe,
                s.zscore_hit_rate, s.zscore_mean_turnover, q, s.mean_coverage,
                s.min_coverage, s.mean_n_scored, s.min_n_scored, s.calibration_slope,
                s.mean_alpha, s.std_alpha, s.mean_y, s.std_y, s.n_bins)
end

@testset "The Return Forecast evaluation, series by series, at parity (#1389)" begin
    PO = PortfolioOptimisers
    gap(a) = (a[10, :] .= NaN; a[20, 4:end] .= NaN; a)
    ties(a) = round.(a ./ 0.5) .* 0.5
    FE_CASES = (; Base = (;), H5 = (; h = 5, step = 5, nfp = 3),
                Overlap = (; h = 3, l = 2, step = 1, nfp = 3),
                Weighted = (; h = 2, step = 2, weighted = true),
                Spearman = (; h = 2, step = 1, rank = true), Gap = (; edit = gap),
                Refit = (; planted = false, member = :target, h = 2, step = 2),
                EW = (; planted = false, member = :ew, h = 2, step = 1, nfp = 3),
                Ties = (; edit = ties, h = 2, step = 2))
    fixtures = Dict(p => evaluation_fixture(; planted = p) for p in (true, false))
    function build(; planted = true, member = :fixed, h = 1, l = 1, step = h, nfp = 4,
                   weighted = false, rank = false, edit = identity)
        fx = fixtures[planted]
        rd, csfm, rows = fx.rd, fx.csfm, fx.rows
        alpha = if member === :fixed
            return_forecast(FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                                        weights = [0.4, 0.6]), rd, csfm).hist
        elseif member === :ew
            return_forecast(ExpWeightedReturnForecast(; scores = fx.scores,
                                                      half_life = 10.0, min_obs = 1,
                                                      horizon = h, lag = l), rd, csfm).hist
        else
            # The oracle's evaluation refuses a member with no history of its own; ours
            # refits it along the grid, and the oracle scores the history ours refitted.
            PO.forecast_history(TargetReturnForecast(; scores = fx.scores, horizon = h,
                                                     lag = l, calibrate = false), rd, csfm;
                                step = step)
        end
        alpha = edit(copy(alpha))
        # The forecast the oracle scored, stored once for each member.
        stored = edit(parity_load("ForecastEvaluation",
                                  (; fixed = "Base", ew = "EW", target = "Refit")[member],
                                  "Alpha"))
        @test parity_compare(alpha, stored; scale = :array, name = "$(member) alpha").ok
        alpha = stored
        amsk = rd.pnl.amsk[rows, :]
        # A weight outside the active universe is missing, as the oracle's panel needs it.
        W = if weighted
            ifelse.(amsk, PO.panel_field(rd.pnl, "market_cap").vals[rows, :], NaN)
        else
            nothing
        end
        return (; alpha, X = csfm.csr.eps, emsk = rd.pnl.emsk[rows, :], W, B = csfm.Ms, h,
                l, step, nfp, rank)
    end
    function score(c; ties = :average)
        y = PO.forward_mean_returns(c.X, c.h, c.l)
        fe = forecast_evaluation(c.alpha, y; umsk = c.emsk, horizon = c.h, lag = c.l,
                                 step = c.step, ties = ties)
        rk = forecast_portfolio(fe; kind = :rank)
        zs = forecast_portfolio(fe; kind = :zscore)
        series = hcat(fe.dates, forecast_ic(fe, c.W), rk.ret, zs.ret, rk.turnover,
                      zs.turnover,
                      forecast_quantile_spread(fe; quantiles = (0.1, 0.25)).spread,
                      PO.forecast_summary_scored(fe,
                                                 PO.forecast_ic_weights(fe.alpha, nothing)),
                      forecast_coverage(fe))
        return (; fe, series,
                summary = fe_summary(forecast_evaluation_summary(fe, c.W;
                                                                 quantiles = (0.1, 0.25),
                                                                 mt = OverlapMeanTest())))
    end
    # Our series on the oracle's dates, and the dates the oracle dropped.
    function on_oracle_dates(ours, stored)
        k = [findfirst(==(t), ours[:, 1]) for t in stored[:, 1]]
        return ours[k, :], setdiff(ours[:, 1], stored[:, 1])
    end
    table(t) = hcat(t.spearman_mean_ic, t.spearman_ic_ir, t.spearman_t_stat,
                    t.pearson_mean_ic, t.pearson_ic_ir, t.pearson_t_stat, t.rank_ann_return,
                    t.rank_sharpe, t.zscore_ann_return, t.zscore_sharpe, t.mean_coverage)
    load(c, o) = parity_load("ForecastEvaluation", c, o)
    # The oracle's t-statistic is our ratio times the root of a whole count of dates.
    # Measured 2.5e-14 at most under the flags of `Pkg.test` and 1.1e-14 without them (H5,
    # Pearson; #1558): the square of a ratio of the oracle's value and ours, each summed in
    # its own order, doubles the round-off of both.
    function whole_root(t, ir, name)
        r = filter(isfinite, (t ./ ir) .^ 2)
        return parity_compare(r, round.(r); rtol = 1e-13, name = "$(name) whole root").ok
    end
    nonT = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for (name, kw) in pairs(FE_CASES)
        case = String(name)
        @testset "$(case)" begin
            c = build(; kw...)
            o = score(c)
            stored = load(case, "Series")
            ours, dropped = on_oracle_dates(o.series, stored)
            # The dates: ours keeps a date with no scorable asset, the oracle drops it.
            @test isempty(dropped) == (case != "Gap")
            @test all(t -> all(isnan, o.series[findfirst(==(t), o.series[:, 1]), 2:3]),
                      dropped)
            @test parity_compare(ours, stored; name = "$(case) series").ok
            fc = forecast_factor_correlation(o.fe, c.B, c.W; rank = c.rank)
            @test parity_compare(fc, load(case, "Factor"); scale = :array,
                                 name = "$(case) factor").ok
            tabs = vcat(table(forecast_holding_period(o.fe, c.X, c.W; n = c.nfp,
                                                      mt = OverlapMeanTest())),
                        table(forecast_decay(o.fe, c.X, c.W; n = c.nfp,
                                             mt = OverlapMeanTest())))
            st = load(case, "Tables")
            @test parity_compare(tabs[:, [1, 2, 4, 5, 7, 8, 9, 10, 11]],
                                 st[:, [1, 2, 4, 5, 7, 8, 9, 10, 11]];
                                 name = "$(case) tables").ok
            @test whole_root(st[:, 3], tabs[:, 2], "$(case) spearman")
            @test whole_root(st[:, 6], tabs[:, 5], "$(case) pearson")
            # R89 (#1512): the oracle's tables are per period whatever its factor is. An
            # evaluation annualised at 12 gives them with `ppy = 1` on the two verbs, and its
            # default tables are the linear rescale of them.
            fe12 = PO.ForecastEvaluationResult(o.fe.alpha, o.fe.y, o.fe.umsk, o.fe.dates,
                                               o.fe.target, o.fe.horizon, o.fe.lag,
                                               o.fe.step, o.fe.min_count, o.fe.ties, 12)
            per = vcat(table(forecast_holding_period(fe12, c.X, c.W; n = c.nfp, ppy = 1,
                                                     mt = OverlapMeanTest())),
                       table(forecast_decay(fe12, c.X, c.W; n = c.nfp, ppy = 1,
                                            mt = OverlapMeanTest())))
            @test parity_compare(per[:, 7:10], st[:, 7:10]; name = "$(case) per period").ok
            ann = vcat(table(forecast_holding_period(fe12, c.X, c.W; n = c.nfp,
                                                     mt = OverlapMeanTest())),
                       table(forecast_decay(fe12, c.X, c.W; n = c.nfp,
                                            mt = OverlapMeanTest())))
            @test isequal(isnan.(ann), isnan.(per))
            f = isfinite.(per)
            # The mean rescales bit-equal; the ratio measures 4.0e-16 at most (#1558).
            @test parity_compare(ann[:, [7, 9]][f[:, [7, 9]]],
                                 12 .* per[:, [7, 9]][f[:, [7, 9]]]; rtol = 1e-14,
                                 name = "$(case) annual mean").ok
            @test parity_compare(ann[:, [8, 10]][f[:, [8, 10]]],
                                 sqrt(12) .* per[:, [8, 10]][f[:, [8, 10]]]; rtol = 1e-14,
                                 name = "$(case) annual ratio").ok
            @test isequal(ann[:, setdiff(1:11, 7:10)], per[:, setdiff(1:11, 7:10)])
            cb = forecast_calibration(o.fe, c.W)
            curve = hcat(cb.curve.bin, cb.curve.mean_alpha, cb.curve.mean_y, cb.curve.count)
            @test parity_compare(curve, load(case, "Curve"); name = "$(case) curve").ok
            ss = vec(load(case, "Summary"))
            s = o.summary
            n = length(s)
            @test n == length(ss)
            keep = setdiff(1:n, [4, 9, 29, 30, 31, 32, n - 4])
            @test parity_compare(s[keep], ss[keep]; name = "$(case) summary").ok
            # The pooled mean of a standardised forecast is round-off zero, so it compares
            # against the pooled standard deviation beside it: measured 1.3e-15 of it at most
            # under the flags of `Pkg.test` (Spearman) and 2.2e-16 without them. The t
            # identities below measure 1.8e-15 at most (H5), each a ratio over a root (#1558).
            @test parity_compare(s[[n - 4, n - 3]], ss[[n - 4, n - 3]]; rtol = 1e-14,
                                 scale = :array, name = "$(case) pooled mean").ok
            ic = forecast_ic(o.fe, c.W)
            nsp, npe = count(isfinite, ic[:, 1]), count(isfinite, ic[:, 2])
            @test parity_compare(ss[[4, 9]], [s[3] * sqrt(nsp), s[8] * sqrt(npe)];
                                 rtol = 1e-14, name = "$(case) oracle t").ok
            if PO.forecast_ic_lags(o.fe) == 0
                @test parity_compare(s[[4, 9]], ss[[4, 9]]; rtol = 1e-14,
                                     name = "$(case) disjoint t").ok
            end
            # The coverage summary counts a silenced date; on the scored dates it is the
            # oracle's. The library's own coverage on the oracle's dates gives it.
            cov = ours[:, end]
            nsc = ours[:, end - 1]
            @test parity_compare([sum(cov) / length(cov), minimum(cov),
                                  sum(nsc) / length(nsc), minimum(nsc)], ss[29:32]).ok
            if case == "Gap"
                @test s[30] == 0 && s[32] == 0
            else
                @test parity_compare(s[29:32], ss[29:32]).ok
            end
            # R87 (#1512): the coverage over the scored dates is the oracle's coverage, and
            # the silenced dates are the dates the oracle dropped. The two means satisfy the
            # identity of `forecast_summary_coverage`.
            fs = forecast_evaluation_summary(o.fe, c.W; mt = OverlapMeanTest())
            @test fs.n_silenced == [length(dropped)]
            @test parity_compare([fs.mean_coverage_scored[1], fs.min_coverage_scored[1]],
                                 ss[29:30]; name = "$(case) scored coverage").ok
            nc = count(isfinite, forecast_coverage(o.fe, c.W))
            # Measured bit-equal in every case (#1558).
            @test parity_compare([fs.mean_coverage_scored[1]],
                                 [fs.mean_coverage[1] * nc / (nc - fs.n_silenced[1])];
                                 rtol = 1e-14, name = "$(case) coverage identity").ok
        end
    end

    @testset "TiesOrdinal: the oracle's rank is one keyword away" begin
        c = build(; FE_CASES.Ties...)
        o = score(c; ties = :ordinal)
        stored = load("TiesOrdinal", "Series")
        ours, _ = on_oracle_dates(o.series, stored)
        @test parity_compare(ours, stored; name = "TiesOrdinal series").ok
        ss = vec(load("TiesOrdinal", "Summary"))
        @test parity_compare(o.summary[nonT], ss[nonT]; name = "TiesOrdinal summary").ok
        # The midrank moves the rank statistics of a tied forecast, and nothing else.
        a = score(c)
        @test maximum(abs, filter(!isnan, a.series[:, 2] .- o.series[:, 2])) > 1e-3
        @test isequal(a.series[:, 3], o.series[:, 3])
    end

    @testset "The book carries its weights over a date with no scorable asset" begin
        c = build(; FE_CASES.Gap...)
        fe = score(c).fe
        for kind in (:rank, :zscore)
            p = forecast_portfolio(fe; kind = kind)
            k = findfirst(==(10), fe.dates)
            @test isnan(p.ret[k]) && isnan(p.turnover[k])
            @test p.w[k, :] == p.w[k - 1, :]
            # Measured 1.2e-16 (#1558).
            @test parity_compare([p.turnover[k + 1]],
                                 [sum(abs, p.w[k + 1, :] .- p.w[k - 1, :])]; rtol = 1e-14,
                                 name = "$(kind) turnover out of the gap").ok
        end
    end
end
