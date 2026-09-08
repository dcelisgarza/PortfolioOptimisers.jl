#=
Check `src/08_Moments/45_ReturnForecasts/07_ForecastEvaluation.jl` against the contract its
docstrings state, and against the reference implementation the map of issue #931 ports.
Issue #934.

THREE CONVENTIONS SHAPE THE PROBES.

1. THE TWO OBSERVATION AXES ARE RECONCILED BY THE TARGET, NOT BY THE EVALUATION. A Return
   Forecast history lives on the factor-model block's rows, and the block is a suffix of the
   carrier. `forecast_target_history` cuts the asset returns and the Panel Fields down to
   that suffix, and the idiosyncratic returns already live there, so the probes build a
   carrier that is strictly longer than the block and assert the cut.

2. THE EVALUATION DATES ARE ROW INDICES, NOT TIMESTAMPS. They index `alpha` and `y`, they
   start at the first observation some asset carries a finite pair at, they end at the last
   one, and they stride by `step`. An unscorable observation between the two bounds is kept,
   so the stride means the same thing everywhere in the sample.

3. THIS TICKET SHIPS THE PAIRING, NOT THE STATISTICS. The level-2 verbs are later tickets, so
   the perfect-forecast probe scores the pairing with `cs_spearman_correlation`, which map
   #643 already ships, rather than with a verb this file does not yet carry.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# The synthetic panel of issue #656, with a factor-model block fitted on a strict suffix of
# the carrier so every probe of the cut has something to cut.
function evaluation_fixture(; n_observations::Integer = 60, drop::Integer = 8)
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
    Ms[:, :, 1] = factor_exposure(xc, rd)
    for k in 1:size(Lo, 3)
        Ms[:, :, k + 1] = Lo[:, :, k]
    end
    nf = ["style"; ["ind$k" for k in 1:size(Lo, 3)]]
    fam = ["style"; fill("industry", size(Lo, 3))]
    vs = [pnl.amsk[t, i] ? 0.0004 * (1.5 + sin(0.3 * t + 0.7 * i)) : NaN
          for t in rows, i in 1:N]
    eps = [if pnl.amsk[t, i]
               0.01 * sin(0.7 * t + 0.29 * i) + 0.004 * cos(0.11 * t * i)
           else
               NaN
           end
           for t in rows, i in 1:N]
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
