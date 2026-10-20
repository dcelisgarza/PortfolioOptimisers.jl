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

    @testset "An observed factor (#1479)" begin
        # The carry derives each row of the returns net of the observed factors one time, from
        # the observed exposures of the row `lag` observations before it, and carries the row.
        # The batch fit derives the first `lag` rows of the sample from the exposure of the same
        # row, so a window that derived its first rows again would give other values.
        base = grid_config("Base", rd)
        ccy = grid_config("Currency", rd)
        msens = "msens" => CompositeExposure(;
                                             descriptors = [EWMacroSensitivity(; series = "MACRO",
                                                                               half_life = 10)],
                                             outlier = nothing, scoring = nothing, family = "msens")
        # An observed member over a Descriptor of the returns reads the returns net of the
        # Currency Factors, a second derived series.
        rev = "rev" => ObservedExposure(;
                                        xe = CompositeExposure(; descriptors = [Reversal()],
                                                               outlier = nothing, scoring = nothing,
                                                               family = "rev"), series = "MACRO",
                                        family = "rev")
        mixed = (; ccy..., factors = [ccy.factors; rev])
        rfr = FixedWeightedReturnForecast(;
                                          scores = DescriptorScores(;
                                                                    descriptors = [Reversal()]),
                                          scale = 0.02)
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        exact = (; Currency = ccy, CurrencyLx = grid_config("CurrencyLx", rd),
                 Macro = grid_config("Macro", rd),
                 Sensitivity = (; base..., factors = [base.factors; msens]),
                 Entropy = (; ccy..., pe = EntropyPoolingPrior(; pe = GRID_PE)),
                 Rolling = (; ccy..., ve = wv))
        # Measured: every case gives a difference of exactly zero at each step.
        for cfg in exact
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            @test po.reads_exogenous_series(pe)
            @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        end
        # The `Reversal` exposure of the observed member folds from its carried state, so the
        # carry keeps the rows of the lag alone (#1593). A fixed weighted forecast scores the net
        # returns of the rows it reads, and its `Reversal` score folds too (#1587).
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      mixed...)) == 2
        for cfg in (mixed, (; mixed..., lag = 2), (; mixed..., rfe = rfr),
                    (; mixed..., rfe = rfr, lag = 2))
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            for (k, x) in enumerate(stream(pe))
                b = batch(pe, k)
                @test relerr(x.pr.mu, b.mu) < 1e-12 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-12 &&
                      relerr(x.pr.fpr.X, b.fpr.X) < 1e-12
                @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
                @test isnothing(pe.rfe) || relerr(x.pr.rr.rf.hist, b.rr.rf.hist) < 1e-12
            end
        end
        # A constrained Family reduces the factor axis, and the observed factor returns fill its
        # trailing columns. So the carry of a forecast that folds, and of the history of one
        # that reads the panel, builds the model on the reduced axis, as the batch fit does
        # (#1585). Before, the seed threw a `DimensionMismatch`.
        famccy = [grid_config("FamOne", rd).factors; "ccy" => CurrencyExposure()]
        fold = (; grid_config("FcFamily", rd)..., factors = famccy)
        pe = CrossSectionalFactorPrior(; fold...)
        @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        panel = (; grid_config("FcTarget", rd)..., factors = famccy,
                 families = ["industry" => nothing])
        pc = CrossSectionalFactorPrior(; panel..., lambda = CarryHistoryShrinkage())
        pb = CrossSectionalFactorPrior(; panel..., lambda = CarryHistoryShrinkage())
        @test all(((k, x),) -> agrees(x.pr, batch(pb, k)), enumerate(stream(pc)))
        # A macro sensitivity is a recursion from the first row, and it folds from its state, so
        # the carry keeps the rows of the lag alone (#1608).
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      exact.Sensitivity...)) ==
              2
        # One observation at a time after the first fit.
        pe = CrossSectionalFactorPrior(; lambda = 1, ccy...)
        e = (0, 90, 91, 92, 93, 250)
        @test all(((k, x),) -> agrees(x.pr, batch(pe, k, rd, e)),
                  enumerate(stream(pe, rd, e)))
        # The window carries the series and the derived rows of its two rows, and the buffer
        # records every column of the series.
        st = last(stream(pe)).pe.cache
        @test st.win.ne == rd.ne && isequal(st.win.E, rd.E[249:250, :])
        @test size(st.der.Xl, 1) == 2 && isnothing(st.der.Xn)
        @test isequal(po.exogenous_buffer_kwargs(po.returns_buffer(st)).E, rd.E)
        # On the online step of an optimiser the prior's buffer owns the series, and the Fold
        # Context reads it back.
        o = po.update_online_estimator(InverseVolatility(; pe = pe))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        @test isa(o.pe.cache, po.CrossSectionalCarryState) && isnothing(o.cache.E)
        bo, brd = po.batch_from_state(o)
        @test brd.ne == rd.ne && isequal(brd.E, rd.E)
        @test agrees(bo.pe, batch(pe, 3))
        @test optimise(o).w == optimise(InverseVolatility(; pe = pe), rd).w
        # A step with no series, and a step with other names.
        nor = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl)
        @test occursin("this step carries no `rd.E`",
                       message(() -> partial_fit!(pe, rows(nor, 1:90))))
        e1 = partial_fit!(pe, rows(rd, 1:90))
        @test occursin("this step carries no `rd.E`",
                       message(() -> partial_fit!(e1, rows(nor, 91:170))))
        ren = ReturnsResult(; nx = rd.nx, X = rd.X, ne = [rd.ne[1:(end - 1)]; "OTHER"],
                            E = rd.E, pnl = rd.pnl)
        @test occursin("a later block must carry the same names",
                       message(() -> partial_fit!(e1, rows(ren, 91:170))))
        # The carry keeps every fitted row, so the step that fits a row refuses an infinite
        # observed return on it, as the batch fit does. A NaN marks a gap, and the step fits
        # it as the batch fit does (#1530). Row 1 has no lagged exposure, so the regression
        # never reads it.
        for (row, v, refused) in ((200, Inf, true), (200, NaN, false), (1, NaN, false))
            E = copy(rd.E)
            E[row, 1] = v
            rn = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = E, pnl = rd.pnl)
            if refused
                e2 = partial_fit!(partial_fit!(pe, rows(rn, 1:90)), rows(rn, 91:170))
                err = try
                    partial_fit!(e2, rows(rn, 171:250))
                catch x
                    x
                end
                @test isa(err, IsNonFiniteError)
                @test occursin("first at observation 200 of the returns data",
                               sprint(showerror, err))
                @test_throws IsNonFiniteError batch(pe, 3, rn)
            else
                @test agrees(last(stream(pe, rn)).pr, batch(pe, 3, rn))
            end
        end
    end
end
