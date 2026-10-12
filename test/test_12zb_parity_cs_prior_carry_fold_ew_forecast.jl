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
`parity_write(dir, grid_fixture(parity_large_panel()); filled = true)` writes. Its factor prior
is an exponentially weighted mean and covariance of half-life 20 and `min_observations = 5`, whose
window of 60 rows cuts the factor returns of the first call alone. Each file stacks the three steps
vertically, as the `Online*` files of test_12z do: one row of `mu`, one 40 x 40 block of `sigma`,
or the factor returns of every row fitted so far. The cases are `SeedStyle` (`Base` with
`families = ["style" => nothing]`) and `SeedIndustry` (`FamOne`). The pinned choice with no window
is checked against the `Online*Fold*` files of test_12z.

A prior whose tree reads the Exogenous Series folds on the carry route too (#1479, ADR 0193): the
buffer and the carried rows hold the series, and the state derives each row of the returns net of
the observed factors one time. Its pinned choice under currency factors is checked against the
`OnlineCurrencyStyle_Fold*` files of test_12z.

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

    @testset "An exponentially weighted Return Forecast folds its rows (#1574)" begin
        # The batch fit is a forward recursion over the rows whose target is known. A step
        # reads its new rows and the `lag + horizon - 1` rows before them, whose targets
        # mature at the step, from the carried state of the recursion. The Reversal and the
        # GrowthRate scores fold from their carried states (#1587, #1598), so the carry keeps
        # the rows of the lag of the prior alone.
        sharpe = IdiosyncraticSharpeUnit()
        ind = grid_config("FamOne", rd).factors
        scores(nm) = DescriptorScores(;
                                      descriptors = [Passthrough(;
                                                                 field = "net_income_ttm"),
                                                     Reversal(; window = 30),
                                                     GrowthRate(; field = "sales_ttm",
                                                                lag = 5)], neutralise = nm,
                                      nw = BlockRegressionWeights(), group = "industry")
        ew(nm, unit; kw...) = ExpWeightedReturnForecast(; scores = scores(nm),
                                                        half_life = 10, scale = 0.5,
                                                        unit = unit, kw...)
        fk = (; lambda = 0.4, c = 0.6, minra = 5, pe = GRID_PE, ve = GRID_VE)
        e = (0, 90, 91, 92, 93, 170, 250)
        # The first fit holds two rows, so the first steps carry fewer rows than the gap of
        # four. Passthrough scores and a short variance warm-up let the recursion advance in
        # them. The factor prior reads too few rows until the last step, so only the last
        # step reads out, and it reads the rows and the state of the early steps.
        e0 = (0, 3, 4, 5, 6, 7, 8, 120)
        early = (; grid_config("Base", rd)..., fk..., ofit = UnadjustedForecast(),
                 ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20), min_obs = 2),
                 rfe = ExpWeightedReturnForecast(;
                                                 scores = DescriptorScores(;
                                                                           descriptors = [Passthrough(;
                                                                                                      field = "net_income_ttm"),
                                                                                          Passthrough(;
                                                                                                      field = "sales_ttm")]),
                                                 half_life = 10, horizon = 2, lag = 3))
        st = partial_fit!(CrossSectionalFactorPrior(; early...), rows(rd, 1:3))
        st = partial_fit!(partial_fit!(st, rows(rd, 4:4)), rows(rd, 5:8)).cache
        @test size(st.fh, 1) == 7 && st.fst.n == 2
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        # The batch choice of `style` moves at the second step, and a rolling window does not
        # fold, so both fit every observation again from an empty state of the recursion.
        cases = (((; fk..., factors = ind, families = ["industry" => nothing],
                   rfe = ew(["industry", "style1"], sharpe)), e),
                 ((; fk..., factors = ind, lag = 2,
                   rfe = ew(["industry", "style1"], IdiosyncraticReturnUnit(); horizon = 3,
                            lag = 2, min_obs = 20, normalise = false)), e),
                 ((; style..., fk..., ofit = UnadjustedForecast(),
                   rfe = ew(nothing, sharpe; horizon = 2, lag = 3)), e), (early, e0),
                 ((; style..., fk..., ofit = ScoreNeutralisation(),
                   rfe = ew(nothing, IdiosyncraticReturnUnit())), edges),
                 ((; style..., fk..., ve = wv, rfe = ew(["style1"], sharpe; horizon = 2)),
                  edges))
        for (cfg, ee) in cases
            pe = CrossSectionalFactorPrior(; cfg...)
            @test po.lookback(pe) == (cfg === early ? 2 : 30)
            @test po.cross_sectional_carry_rows(pe) == pe.lag + 1
            for (k, x) in enumerate(stream(pe, rd, ee))
                # A block too short for a fit refuses on both sides.
                b = try
                    batch(pe, k, rd, ee)
                catch err
                    err
                end
                st = x.pe.cache
                if b isa Exception
                    @test x.pr isa Exception
                    continue
                end
                rf, rb = x.pr.rr.rf, b.rr.rf
                @test size(st.win.X, 1) <= 6
                @test size(st.fh) == size(st.vs) == size(rb.hist)
                @test relerr(x.pr.mu, b.mu) < 1e-13 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-13 &&
                      relerr(rf.hist, rb.hist) < 1e-13
                @test isequal(isnan.(rf.hist), isnan.(rb.hist))
                # The state of the recursion equals the one of the batch fit.
                @test rf.n == rb.n == st.fst.n
                @test relerr(rf.A, rb.A) < 1e-13 && relerr(rf.c, rb.c) < 1e-13
                @test isequal(isnan.(rf.coef), isnan.(rb.coef)) &&
                      relerr(rf.coef, rb.coef) < 1e-12
            end
        end
        # A step copies the carried state, so the state of an earlier step stays as it is.
        pe = CrossSectionalFactorPrior(; first(first(cases))...)
        pe = partial_fit!(partial_fit!(pe, rows(rd, 1:90)), rows(rd, 91:91))
        A0 = copy(pe.cache.fst.A)
        st0 = pe.cache
        partial_fit!(pe, rows(rd, 92:92))
        @test st0.fst.A == A0
    end
end
