#=
The checks of this subject are split over the files `test_12zb_parity_cs_prior_carry_fold*.jl`, so that each file of the
suite runs in about the same time. This file holds one part of them.

The carry fold of the Cross-Sectional Factor Prior (#1471, map #1375, ADR 0193).

An unwrapped prior folds as a carry: its first step seeds a `CrossSectionalCarryState`, and each
step computes the exposures, the regression and the idiosyncratic variance of the new
observations alone, from the panel rows that its Descriptors read. The factor prior and the
idiosyncratic variance fold, and the Return Forecast and the idiosyncratic correlation refit at the
call with no data. A factor that comes alive keeps every fitted observation, and folds the factor
prior again (#1627). A batch choice
that moves folds the move, and solves again each observation whose answer depends on the basis
(#1605, #1613). It folds the same way under a target that gives the same coefficients in every
basis, as `is_basis_invariant` answers (#1614), and under a `GeneralisedLinearModel` target (#1615).

A step folds the member states in place, as every fold does, so each stream reads the prior out
right after its step.

Every `Parity_CrossSectionalFactorPrior_CarrySeed*` file is an output of the oracle's online update
over the stream of #1468, rows 1-90, 91-170 and 171-250 of the exchange that
`parity_write(dir, grid_fixture(parity_large_panel()).rd; filled = true)` writes. Its factor prior
is an exponentially weighted mean and covariance of half-life 20 and `min_observations = 5`, whose
window of 60 rows cuts the factor returns of the first call alone. Each file stacks the three steps
vertically, as the `Online*` files of test_12za do: one row of `mu`, one 40 x 40 block of `sigma`,
or the factor returns of every row fitted so far. The cases are `SeedStyle` (`Base` with
`families = ["style" => nothing]`) and `SeedIndustry` (`FamOne`). The pinned choice with no window
is checked against the `Online*Fold*` files of test_12za.

A prior whose tree reads the Exogenous Series folds on the carry route too (#1479, ADR 0193): the
buffer and the carried rows hold the series, and the state derives each row of the returns net of
the observed factors one time. Its pinned choice under currency factors is checked against the
`OnlineCurrencyStyle_Fold*` files of test_12za.

A slot that reads the Return Forecast history makes the state carry it (#1484): each step fits the
member once for each new observation, and the history and the resolved `lambda` and `c` equal those
of the batch fit over the same rows.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))
include(joinpath(@__DIR__, "test06c_setup.jl"))

# Test-local rules of `lambda` and `c` that read the Return Forecast history (#1484). Each records
# the history it got, and its number moves with every row of it.
struct CarryHistoryShrinkage <:
       PortfolioOptimisers.AbstractSpannedShrinkageCalibrationAlgorithm
    seen::Vector{Any}
end
CarryHistoryShrinkage() = CarryHistoryShrinkage(Any[])
PortfolioOptimisers.reads_forecast_history(::CarryHistoryShrinkage) = true
function (r::CarryHistoryShrinkage)(key, pr, w, slv, ctx)
    h = ctx.cs.hist
    push!(r.seen, copy(h))
    return 1 / (1 + sum(abs, filter(isfinite, h)))
end
struct CarryHistoryScale <:
       PortfolioOptimisers.AbstractOrthogonalForecastScaleCalibrationAlgorithm
    seen::Vector{Any}
end
CarryHistoryScale() = CarryHistoryScale(Any[])
PortfolioOptimisers.reads_forecast_history(::CarryHistoryScale) = true
function (r::CarryHistoryScale)(key, pr, w, slv, ctx)
    h = ctx.cs.hist
    push!(r.seen, copy(h))
    return mean(abs, filter(isfinite, h)) * 100
end
# A test-local rule of `c` that records the fit of the prior it reads (#1572).
struct ContextRecordScale <:
       PortfolioOptimisers.AbstractOrthogonalForecastScaleCalibrationAlgorithm
    seen::Vector{Any}
end
ContextRecordScale() = ContextRecordScale(Any[])
PortfolioOptimisers.reads_forecast_history(::ContextRecordScale) = true
function (r::ContextRecordScale)(key, pr, w, slv, ctx)
    push!(r.seen, ctx.cs)
    return 1.0
end
# A test-local target that fits as a `LinearModel` and keeps the default answer of
# `is_basis_invariant`, so a move of the dropped member under it fits every observation again
# (#1614).
struct BasisBoundTarget <: PortfolioOptimisers.AbstractRegressionTarget end
function PortfolioOptimisers.factory(::BasisBoundTarget, w::PortfolioOptimisers.ObsWeights)
    return factory(LinearModel(), w)
end

@testset "The carry fold of the Cross-Sectional Factor Prior (#1471)" begin
    po = PortfolioOptimisers
    fx = parity_large_panel()
    rd = grid_fixture(fx)
    N = size(rd.X, 2)
    edges = (0, 90, 170, 250)
    rows(r, i) = po.port_opt_view(r, i, :)
    # Equal cell by cell, a `NaN` equal to a `NaN`.
    same(a, b) = size(a) == size(b) && all(((x, y),) -> isequal(x, y) || x == y, zip(a, b))
    function relerr(a, b)
        return maximum(abs, filter(isfinite, a - b); init = 0.0) /
               maximum(abs, filter(isfinite, b); init = 0.0)
    end
    # The prior after each step of a stream, and its call with no data at once.
    function stream(pe, r = rd, e = edges)
        out = []
        for k in 1:(length(e) - 1)
            pe = partial_fit!(pe, rows(r, (e[k] + 1):e[k + 1]))
            pr = try
                prior(pe)
            catch err
                err
            end
            push!(out, (; pe = pe, pr = pr))
        end
        return out
    end
    batch(pe, k, r = rd, e = edges) = prior(pe, rows(r, 1:e[k + 1]))
    function message(f)
        return try
            f()
            ""
        catch err
            sprint(showerror, err)
        end
    end
    function agrees(x, b)
        return same(x.mu, b.mu) &&
               same(x.sigma, b.sigma) &&
               same(x.X, b.X) &&
               same(x.o_X, b.o_X) &&
               same(x.fpr.X, b.fpr.X) &&
               same(x.rr.csr.f, b.rr.csr.f) &&
               same(x.rr.vs, b.rr.vs) &&
               same(x.rr.Ms, b.rr.Ms) &&
               same(x.rr.rw, b.rr.rw) &&
               isequal(x.rr.edof, b.rr.edof) &&
               x.rr.idx == b.rr.idx
    end
    # Equal to rounding, with the same `NaN` cells. The gap over the largest finite entry, and
    # zero when the two are equal, so a block with no finite entry other than zero compares.
    function close_to(p, q; tol = 1e-12)
        gap = maximum(abs, filter(isfinite, p - q); init = 0.0)
        return size(p) == size(q) &&
               isequal(isnan.(p), isnan.(q)) &&
               (iszero(gap) || relerr(p, q) < tol)
    end
    # A move of a batch choice keeps the factor returns of the old basis and selects their
    # columns, where the batch fit solves each row in the new basis, so the two differ by
    # rounding alone (#1605, #1613).
    function near(x, b; tol = 1e-12)
        return all(((p, q),) -> close_to(p, q; tol = tol),
                   ((x.mu, b.mu), (x.sigma, b.sigma), (x.X, b.X), (x.o_X, b.o_X),
                    (x.fpr.X, b.fpr.X), (x.rr.csr.f, b.rr.csr.f), (x.rr.vs, b.rr.vs),
                    (x.rr.Ms, b.rr.Ms), (x.rr.rw, b.rr.rw))) && x.rr.idx == b.rr.idx
    end
    dropped(pr) = pr.rr.nf[po.dropped_factor_indices(pr.rr.fcb)]
    style = (; grid_config("Base", rd)..., families = ["style" => nothing])
    industry = grid_config("FamOne", rd)

    @testset "The Descriptor scores of a Return Forecast fold their stateful Descriptors (#1587)" begin
        # A Descriptor of the scores that carries a state reads the rows of the step alone, so
        # the carry keeps the rows of the lag alone, and every step equals the batch fit to the
        # last bit. The state carries the Descriptor Scores with the state of each Descriptor.
        ind = grid_config("FamOne", rd).factors
        fk = (; lambda = 0.4, c = 0.6, minra = 5, pe = GRID_PE, ve = GRID_VE,
              families = ["industry" => nothing])
        e = (0, 90, 91, 92, 170, 250)
        for de in (Reversal(), RollingMomentum(; window = 40, skip = 5))
            ds = DescriptorScores(;
                                  descriptors = [Passthrough(; field = "net_income_ttm"),
                                                 de], group = "industry")
            for rfe in (FixedWeightedReturnForecast(; scores = ds, scale = 0.02),
                        ExpWeightedReturnForecast(; scores = ds),
                        TargetReturnForecast(; scores = ds, half_life = 10.0))
                pe = CrossSectionalFactorPrior(; fk..., factors = ind, rfe = rfe)
                @test po.lookback(pe) == po.lookback(de)
                @test po.carry_lookback(rfe) == 1
                @test po.cross_sectional_carry_rows(pe) == 2
                for (k, x) in enumerate(stream(pe, rd, e))
                    b = batch(pe, k, rd, e)
                    st = x.pe.cache
                    @test size(st.win.X, 1) == 2
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                    @test same(x.pr.rr.rf.mu, b.rr.rf.mu)
                    @test isa(st.fds.descriptors[2].cache, po.RollingLogReturnState)
                end
            end
        end
        # The first steps end inside the warm-up of the momentum factor, so they bring no
        # observation to score. Their rows still fold into the state of the Reversal score: a
        # state that started at a later row would sum from another first row, and its scores
        # would differ from the batch fit in the last bits.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1583)).rd
        momentum = ["market" => ConstantExposure(),
                    "momentum" => CompositeExposure(;
                                                    descriptors = [RollingMomentum(; window = 40,
                                                                                   skip = 5)]),
                    "industry" => OneHotExposure(; field = "industry", family = "industry")]
        rev = DescriptorScores(; descriptors = [Reversal(; window = 10), BookToPrice()])
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = momentum,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE,
                                       rfe = FixedWeightedReturnForecast(; scores = rev,
                                                                         scale = 0.02))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 10, 20, 50, 80, 81, 120, 160)
        s = stream(pe, syn, e)
        @test isnothing(s[1].pe.cache.fsc) && isnothing(s[2].pe.cache.Ms)
        @test isa(s[2].pe.cache.fds.descriptors[1].cache, po.RollingLogReturnState)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, syn, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.mu, b.mu) && same(x.pr.rr.rf.hist, b.rr.rf.hist)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # Two of the steps read out, so the loop is not vacuous.
        @test fitted == 2
        # A forecast that does not fold its rows reads the panel, so the state carries no
        # Descriptor Scores.
        kf = CrossSectionalFactorPrior(; fk..., factors = ind,
                                       rfe = TargetReturnForecast(;
                                                                  scores = DescriptorScores(;
                                                                                            descriptors = [Reversal()]),
                                                                  half_life = 10.0,
                                                                  cv = KFold()))
        @test !po.folds_forecast_rows(kf.rfe)
        @test isnothing(partial_fit!(kf, rows(rd, 1:90)).cache.fds)
    end

    @testset "A Target Return Forecast trains on rows of the warm-up and folds (#1588)" begin
        # Under `whole_history` the member trains on the last `lag + horizon - 1` rows before
        # the block, and the histories hold `pe.lag` of them. The state keeps the scores of the
        # other rows, which are rows of the warm-up of the momentum factor, before its first
        # row. A neutralised score is `NaN` before the block, so the rows train only under a
        # rule of `ofit` that does not neutralise. Before #1588 a horizon above one refitted
        # from every row, and a forecast lag above `pe.lag` lost the first of those rows: under
        # `UnadjustedForecast` its `mu` differed from the batch fit by 2 to 12 per cent.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1583)).rd
        momentum = ["market" => ConstantExposure(),
                    "momentum" => CompositeExposure(;
                                                    descriptors = [RollingMomentum(; window = 40,
                                                                                   skip = 5)]),
                    "industry" => OneHotExposure(; field = "industry", family = "industry")]
        rev = DescriptorScores(; descriptors = [Reversal(; window = 10), BookToPrice()])
        tf(; kw...) = TargetReturnForecast(; scores = rev, half_life = 10.0, kw...)
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        un = UnadjustedForecast()
        # Steps of one row cross the end of the warm-up, so the rows that the state keeps come
        # from several steps. The rolling variance fits every observation again at each step.
        e = (0, 20, 41, 42, 43, 44, 45, 46, 47, 80, 120, 160)
        for (cfg, a) in (((; ofit = un, rfe = tf(; horizon = 5)), 4),
                         ((; ofit = un, ve = wv, rfe = tf(; horizon = 5)), 4),
                         ((; ofit = OrthogonalPartCalibration(), rfe = tf(; horizon = 3, lag = 2)), 3),
                         ((; ofit = un, lag = 2, rfe = tf(; horizon = 2, lag = 4)), 3),
                         ((; ofit = un, rfe = tf(; lag = 3)), 2), ((; rfe = tf(; horizon = 5)), 4))
            pe = CrossSectionalFactorPrior(; lambda = 1, factors = momentum,
                                           families = ["industry" => nothing], minra = 5,
                                           pe = GRID_PE, ve = GRID_VE, cfg...)
            @test po.folds_forecast_rows(pe.rfe)
            @test po.cross_sectional_forecast_lead(pe.rfe, pe.lag) == a
            s = stream(pe, syn, e)
            fitted = 0
            for (k, x) in enumerate(s)
                b = try
                    batch(pe, k, syn, e)
                catch err
                    err
                end
                if isa(b, LowOrderPrior)
                    fitted += 1
                    rf, rb = x.pr.rr.rf, b.rr.rf
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                    @test same(rf.mu, rb.mu) && rf.model.n == rb.model.n
                    @test isequal(rf.calib, rb.calib) && isequal(rf.ocalib, rb.ocalib)
                else
                    @test typeof(x.pr) == typeof(b)
                end
            end
            @test fitted == 2
            st = last(s).pe.cache
            @test size(st.fsc.S, 1) - size(st.Ms, 1) == a
            @test size(st.win.X, 1) == po.cross_sectional_carry_rows(pe)
        end
        # The strict carry rule accepts the member, because it folds.
        @test po.folds_forecast_rows(CrossSectionalFactorPrior(; factors = momentum,
                                                               carry = FoldOnly(),
                                                               rfe = tf(; horizon = 5)).rfe)
        # The member that reads the block alone keeps no row of the warm-up.
        @test po.cross_sectional_forecast_lead(tf(; horizon = 5, whole_history = false),
                                               1) == 0
        @test po.cross_sectional_forecast_lead(ExpWeightedReturnForecast(; scores = rev),
                                               1) == 0
        @test po.cross_sectional_forecast_lead(tf(; horizon = 2, lag = 1), 3) == 0
    end

    @testset "Refusals" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        m = message(() -> prior(partial_fit!(pe, rows(rd, 1:2))))
        @test occursin("holds 2 observation(s) after the Descriptor warm-up", m)
        @test occursin("lag + 2 = 3", m)
        m = message(() -> partial_fit!(pe, ReturnsResult(; nx = rd.nx, X = rd.X[1:5, :])))
        @test occursin("the step carries no panel", m)
        e = partial_fit!(pe, rows(rd, 1:90))
        m = message(() -> po.port_opt_view(e, 1:5))
        @test occursin("not a slice of the ones it carries", m)
        m = message(() -> po.merge_states(e.cache, partial_fit!(pe, rows(rd, 91:170)).cache))
        @test occursin("cannot merge two states fitted on disjoint blocks", m)
        @test occursin("the matrix form of `partial_fit!` carries none",
                       message(() -> partial_fit!(pe, rd.X)))
    end
end
