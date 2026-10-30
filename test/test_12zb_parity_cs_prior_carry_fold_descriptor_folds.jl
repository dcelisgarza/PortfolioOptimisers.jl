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
struct BasisBoundTarget <: PortfolioOptimisers.AbstractUnpenalisedRegressionTarget end
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

    @testset "A RollingLogReturn folds one row from its carried state (#1583)" begin
        # The state takes the cumulative sums from the first observation with the arithmetic
        # of the batch call, so each folded row equals the batch call to the last bit, in
        # blocks of any size. A second step from the same state gives the same rows, because
        # a step copies the buffers of the state and changes no carried row.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1583)).rd
        blocks = ((1, 1), (2, 30), (31, 31), (32, 100), (101, 160))
        for de in (RollingMomentum(; window = 40, skip = 5), Reversal(),
                   RollingLogReturn(; window = 10, sign = -1, exponentiate = true))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.cs[end],
                           r.de.cache.cs[end])
                st = r.de
            end
            @test length(st.cache.cs) == de.window + de.skip + 1
            c = copy(st.cache)
            @test all(i -> c.cs[i] == st.cache.cs[i] && c.cs[i] !== st.cache.cs[i],
                      eachindex(c.cs))
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(c, st.cache)))
            @test occursin("carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        # On the carry fold the Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and the exposures of every step equal those of the batch fit.
        momentum = ["market" => ConstantExposure(),
                    "momentum" => CompositeExposure(;
                                                    descriptors = [RollingMomentum(; window = 40,
                                                                                   skip = 5),
                                                                   BookToPrice()]),
                    "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = momentum,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test po.lookback(pe) == 46
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
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
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test isa(last(last(s).pe.cache.xf[2]).descriptors[1].cache,
                  po.RollingLogReturnState)
    end

    @testset "An EW mean Descriptor folds one row from its carried state (#1586)" begin
        # The state runs the recursion from the first observation with the arithmetic of the
        # batch call, so each folded row equals the batch call to the last bit, in blocks of
        # any size. A skip reads the log returns of the ring of the state. A second step from
        # the same state gives the same rows, because a step copies the state.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1586)).rd
        blocks = ((1, 1), (2, 30), (31, 31), (32, 100), (101, 160))
        for de in (EWMomentum(; half_life = 10, skip = 5),
                   EWMean(; decay = 0.9, min_obs = 3, exponentiate = true),
                   EWShareTurnover(; half_life = 5), EWAmihudIlliquidity(; half_life = 7),
                   DaysToCover(; half_life = 4),
                   EWVolumeRatio(; num = nothing, den = "adj_close", decay = 0.7, min_obs = 2))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.s, r.de.cache.s)
                st = r.de
            end
            @test count(isfinite, full) > 8000
            c = copy(st.cache)
            @test c.s == st.cache.s && c.s !== st.cache.s && c.n == st.cache.n
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(c, st.cache)))
            @test occursin("carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        @test length(partial_fit!(EWMomentum(; skip = 5), syn).cache.buf) == 6
        # The state of a ratio holds no ring, so an EWMean refuses it.
        ratio = partial_fit!(DaysToCover(; half_life = 4), rows(syn, 1:5)).cache
        @test occursin("holds no ring of log returns",
                       message(() -> po.descriptor_step(EWMean(; decay = 0.5, min_obs = 1,
                                                               cache = ratio),
                                                        rows(syn, 6:7))))
        @test occursin("hold one entry per asset, got 2 and 1",
                       message(() -> po.EWMeanState(; s = [0.0, 0.0], n = [0])))
        @test po.show_fields(EWMean(; decay = 0.5, min_obs = 1)) ==
              (:decay, :min_obs, :skip, :exponentiate)
        # On the carry fold each Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and the exposures of every step equal those of the batch fit.
        ew = ["market" => ConstantExposure(),
              "momentum" => CompositeExposure(;
                                              descriptors = [EWMomentum(; half_life = 10, skip = 5),
                                                             BookToPrice()]),
              "liquidity" => CompositeExposure(;
                                               descriptors = [EWShareTurnover(; half_life = 5),
                                                              DaysToCover(; half_life = 4)]),
              "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = ew,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test isnothing(po.lookback(pe))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
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
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, po.EWMeanState),
                  last(last(s).pe.cache.xf[3]).descriptors)
    end

    @testset "A lag Descriptor folds one row from its carried state (#1598)" begin
        # The state carries the lagged quantity of the last `lag` observations with the bits
        # of the batch call, so each folded row equals the batch call to the last bit, in
        # blocks of any size, also blocks shorter than the lag. A second step from the same
        # state gives the same rows, because a step copies the state.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1598)).rd
        blocks = ((1, 1), (2, 3), (4, 30), (31, 31), (32, 100), (101, 160))
        for de in (GrowthRate(; field = "sales_ttm", lag = 7), SalesGrowthRate(; lag = 1),
                   ChangeToScale(; field = "net_income_ttm", scale = "market_cap", lag = 5),
                   EarningsChangeToPrice(; lag = 40),
                   ChangeInIntensity(; field = "capex_ttm", scale = "total_assets", lag = 9),
                   CapexToAssetsChangeInIntensity(; lag = 3))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.q[end], r.de.cache.q[end])
                st = r.de
            end
            @test count(isfinite, full) > 6000
            @test length(st.cache.q) == de.lag
            # A merge of the states of two consecutive blocks is the state of both blocks.
            a = partial_fit!(de, rows(syn, 1:50)).cache
            m = po.merge_states(a, partial_fit!(de, rows(syn, 51:52)).cache)
            @test all(((x, y),) -> same(x, y),
                      zip(m.q, partial_fit!(de, rows(syn, 1:52)).cache.q))
            c = copy(a)
            @test same(c.q[end], a.q[end]) && c.q[end] !== a.q[end]
            @test occursin("carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        g = partial_fit!(GrowthRate(; field = "sales_ttm", lag = 4), rows(syn, 1:9))
        @test occursin("carries the rows of a lag of 4, and the lag of the Descriptor is 3",
                       message(() -> po.descriptor_step(GrowthRate(; field = "sales_ttm",
                                                                   lag = 3,
                                                                   cache = g.cache),
                                                        rows(syn, 10:11))))
        @test occursin("got the lags 4 and 3",
                       message(() -> po.merge_states(g.cache,
                                                     partial_fit!(GrowthRate(;
                                                                             field = "sales_ttm",
                                                                             lag = 3),
                                                                  rows(syn, 1:2)).cache)))
        q = po.DataStructures.CircularBuffer{Vector{Float64}}(2)
        push!(q, [0.0, 0.0])
        push!(q, [0.0])
        @test occursin("got rows of [2, 1] assets",
                       message(() -> po.LagDescriptorState(; q = q)))
        @test po.show_fields(EarningsChangeToPrice()) == (:field, :scale, :lag, :gt0)
        # On the carry fold each lag Descriptor counts as one row, so the carry keeps the
        # rows of the lag of the prior alone, and every step equals the batch fit.
        lagf = ["market" => ConstantExposure(),
                "growth" => CompositeExposure(;
                                              descriptors = [SalesGrowthRate(; lag = 20),
                                                             EarningsChangeToPrice(; lag = 15)]),
                "intensity" => CompositeExposure(;
                                                 descriptors = [CapexToAssetsChangeInIntensity(; lag = 25),
                                                                BookToPrice()]),
                "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = lagf,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test po.lookback(pe) == 27
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
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
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, po.LagDescriptorState),
                  last(last(s).pe.cache.xf[2]).descriptors)
        @test isa(first(last(last(s).pe.cache.xf[3]).descriptors).cache,
                  po.LagDescriptorState)
    end

    @testset "An EW volatility Descriptor folds one row from its carried state (#1607)" begin
        # The state carries the state of the variance estimator and, for a residual form, the
        # state of the beta recursion. Both run the arithmetic of the batch call from the first
        # observation, so each folded row equals the batch call to the last bit, in blocks of
        # any size, also with a regime adjustment and HAC lags. A second step from the same
        # state gives the same rows, because a step copies the state.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1607)).rd
        blocks = ((1, 1), (2, 30), (31, 31), (32, 100), (101, 160))
        for de in (EWVolatility(; half_life = 10),
                   EWDownsideVolatility(; half_life = 5, mar = 0.001),
                   EWResidualVolatility(; half_life = 5, beta_half_life = 8),
                   EWResidualDownsideVolatility(; half_life = 7, beta_half_life = 3),
                   EWVolatility(;
                                ce = RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 3,
                                                                       regime_min_obs = 4)),
                   EWResidualVolatility(;
                                        ce = RegimeAdjustedExpWeightedVariance(; decay = 0.8,
                                                                               min_obs = 2,
                                                                               hac_lags = 2)))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.ve.variance,
                           r.de.cache.ve.variance)
                st = r.de
            end
            @test count(isfinite, full) > 8000
            c = copy(st.cache)
            @test c.ve.variance == st.cache.ve.variance &&
                  c.ve.variance !== st.cache.ve.variance
            @test isnothing(c.beta) == isnothing(st.cache.beta)
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(c, st.cache)))
            # A residual form checks its beta state first.
            @test occursin(r"the state holds 60 assets, and `X` holds 5|the beta state carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        # The beta recursion folds by itself too: a step from the carried state equals the
        # batch call, also where `min_obs` counts observations across the cut.
        rm = po.market_return_series(syn, "market_cap")
        B, Vm = po.ew_beta_series(syn.X, rm, 0.8, 7, 1e-12, syn.pnl.amsk)
        b1 = po.ew_beta_series!(po.ew_beta_state(nothing, syn.X, rm, 0.8), syn.X[1:4, :],
                                rm[1:4], 0.8, 7, 1e-12, syn.pnl.amsk[1:4, :])
        b2 = po.ew_beta_series!(po.ew_beta_state(b1.st, syn.X, rm, 0.8), syn.X[5:160, :],
                                rm[5:160], 0.8, 7, 1e-12, syn.pnl.amsk[5:160, :])
        @test same([b1.B; b2.B], B) && same([b1.Vm; b2.Vm], Vm) && b2.st.t == 160
        @test occursin("cannot merge two states fitted on disjoint blocks",
                       message(() -> po.merge_states(b1.st, b2.st)))
        @test occursin("the beta state carries 60 assets, and the step brings 5",
                       message(() -> po.ew_beta_state(b1.st, syn.X[:, 1:5], rm, 0.8)))
        @test occursin("hold one entry per asset",
                       message(() -> po.EWBetaState(; b = [0.0], mu = [0.0, 0.0],
                                                    cv = [0.0], n = [0], act = trues(1),
                                                    mu_m = 0.0, var_m = 0.0, t = 0)))
        # A state of the other form is refused.
        res = partial_fit!(EWResidualVolatility(), rows(syn, 1:5)).cache
        vol = partial_fit!(EWVolatility(), rows(syn, 1:5)).cache
        @test occursin("holds a beta state",
                       message(() -> po.descriptor_step(EWVolatility(; cache = res),
                                                        rows(syn, 6:7))))
        @test occursin("holds no beta state",
                       message(() -> po.descriptor_step(EWResidualVolatility(; cache = vol),
                                                        rows(syn, 6:7))))
        @test po.EWVolatilityState(; ve = vol.ve).ve === vol.ve
        @test po.show_fields(EWVolatility()) == (:ce, :alg, :mar)
        @test po.show_fields(EWResidualVolatility()) ==
              (:mcap, :ce, :beta_decay, :alg, :mar, :min_val)
        # A `ce` with no fold has no state: the Descriptor answers no step, and its batch
        # call is the variance series of `ce`.
        wv = EWVolatility(; ce = WindowedVariance(; window = 20))
        @test isnothing(po.descriptor_step(wv, rows(syn, 1:5)))
        @test isnothing(po.carry_lookback(wv))
        @test isnothing(po.ew_volatility_fold(wv, syn, nothing).st)
        # On the carry fold each Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and every step equals the batch fit.
        vf = ["market" => ConstantExposure(),
              "volatility" => CompositeExposure(;
                                                descriptors = [EWVolatility(; half_life = 10),
                                                               EWDownsideVolatility(; half_life = 5)]),
              "residual" => CompositeExposure(;
                                              descriptors = [EWResidualVolatility(; half_life = 5,
                                                                                  beta_half_life = 8),
                                                             BookToPrice()]),
              "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = vf,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test isnothing(po.lookback(pe))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
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
                @test dropped(x.pr) == dropped(b)
                @test same(x.pr.rr.Ms, b.rr.Ms)
                # The dropped member of the industry family moves at the fourth step. A move
                # folds, and equals the batch fit to rounding alone (#1605).
                if k < 4
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                else
                    @test near(x.pr, b)
                end
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test [only(dropped(x.pr)) for x in s] ==
              ["industry=Banks", "industry=Banks", "industry=Banks", "industry=Real Estate",
               "industry=Real Estate", "industry=Real Estate"]
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, po.EWVolatilityState),
                  last(last(s).pe.cache.xf[2]).descriptors)
        @test isa(first(last(last(s).pe.cache.xf[3]).descriptors).cache,
                  po.EWVolatilityState)
        # A `ce` with no fold makes the carry keep every row, and the strict carry rule
        # refuses it in the constructor.
        wf = ["market" => ConstantExposure(),
              "volatility" => CompositeExposure(; descriptors = [wv])]
        @test isnothing(po.cross_sectional_carry_rows(CrossSectionalFactorPrior(;
                                                                                factors = wf,
                                                                                minra = 5)))
        @test occursin("the factor \"volatility\" (CompositeExposure) has no finite look-back",
                       message(() -> CrossSectionalFactorPrior(; factors = wf, minra = 5,
                                                               carry = FoldOnly())))
    end

    @testset "An EW beta Descriptor folds one row from its carried state (#1608)" begin
        # The state runs each recursion from the first observation with the arithmetic of the
        # batch call, so each folded row equals the batch call to the last bit, in blocks of
        # any size. With `agg_obs > 1` the state carries the observations of the window that
        # is not complete, so a block that ends inside a window gives the same rows, and the
        # shrinkage of a group reads the labels and the weights of the row that closes the
        # window. A second step from the same state gives the same rows, because a step
        # copies the state.
        blocks = ((1, 1), (2, 3), (4, 30), (31, 31), (32, 33), (34, 100), (101, 101),
                  (102, 250))
        # A copy is equal and shares no array with the original, at every level.
        function copied(c, s)
            return all(fieldnames(typeof(c))) do f
                x, y = getfield(c, f), getfield(s, f)
                if isa(x, po.AbstractPartialFitState)
                    copied(x, y)
                else
                    isequal(x, y) && (!isa(x, AbstractArray) || x !== y)
                end
            end
        end
        for de in
            (EWMarketBeta(; half_life = 10), EWMarketBeta(; half_life = 10, agg_obs = 3),
             EWMarketBeta(; half_life = 10, group = "industry", min_group_size = 3),
             EWBeta(; decay = 0.8, min_obs = 30, agg_obs = 4, group = "industry",
                    min_group_size = 3), EWDownsideBeta(; half_life = 7, mar = 0.001),
             EWMacroSensitivity(; series = "MACRO", half_life = 8),
             EWMacroSensitivity(; series = "MACRO", half_life = 8, agg_obs = 3))
            full = descriptor(de, rd)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(rd, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(rd, a:b)).D, r.D)
                @test copied(partial_fit!(st, rows(rd, a:b)).cache, r.de.cache)
                st = r.de
            end
            @test count(isfinite, full) > 4000
            @test copied(copy(st.cache), st.cache)
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(st.cache, st.cache)))
            @test occursin("carries $N assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(rd, 1:2),
                                                                             1:5))))
        end
        # A step refuses the gap and the infinite value of the batch call, and names the same
        # observations, counted from the first one that the state folded.
        j = findfirst(==("MACRO"), rd.ne)
        withE(E) = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = E, pnl = rd.pnl)
        E = copy(rd.E)
        E[120, j] = NaN
        dm = EWMacroSensitivity(; series = "MACRO", half_life = 8)
        @test occursin("none at observation 120", message(() -> descriptor(dm, withE(E))))
        @test occursin("none at observation 120",
                       message(() -> partial_fit!(partial_fit!(dm, rows(withE(E), 1:100)),
                                                  rows(withE(E), 101:140))))
        E[120, j] = 0.0
        E[118:120, j] .= NaN
        d3 = EWMacroSensitivity(; series = "MACRO", half_life = 8, agg_obs = 3)
        @test occursin("none at observations 118 to 120",
                       message(() -> descriptor(d3, withE(E))))
        @test occursin("none at observations 118 to 120",
                       message(() -> partial_fit!(partial_fit!(d3, rows(withE(E), 1:119)),
                                                  rows(withE(E), 120:140))))
        # The window of observations 148 to 150 holds two finite values, so the gap is
        # accepted, and the infinite value is refused.
        E[118:120, j] .= 0.0
        E[150, j] = Inf
        @test occursin("infinite at observation 150",
                       message(() -> descriptor(d3, withE(E))))
        @test occursin("infinite at observation 150",
                       message(() -> partial_fit!(partial_fit!(d3, rows(withE(E), 1:100)),
                                                  rows(withE(E), 101:200))))
        # The state of the other Descriptor is refused.
        b = partial_fit!(EWMarketBeta(; half_life = 10), rows(rd, 1:20)).cache
        m = partial_fit!(dm, rows(rd, 1:20)).cache
        @test occursin("holds no reference return, so it belongs to another Descriptor",
                       message(() -> partial_fit!(EWMacroSensitivity(; series = "MACRO",
                                                                     cache = b),
                                                  rows(rd, 21:22))))
        @test occursin("holds a reference return, so it belongs to another Descriptor",
                       message(() -> partial_fit!(EWBeta(; decay = 0.9, min_obs = 5,
                                                         cache = m), rows(rd, 21:22))))
        @test occursin("hold one market return and one reference return each",
                       message(() -> po.EWBlockState(; st = b.st, X = zeros(1, 2),
                                                     m = Float64[])))
        @test occursin("hold one entry per asset, got 2 and 1",
                       message(() -> po.EWDownsideBetaState(; cd = [0.0, 0.0], n = [0],
                                                            vd = 0.0, t = 0)))
        @test po.show_fields(EWDownsideBeta()) == (:mcap, :decay, :min_obs, :mar, :min_val)
        # Two states of the macro sensitivity do not merge: the second block starts from the first.
        @test_throws ArgumentError po.merge_states(m, copy(m))
        # A carried Descriptor pads the rows before its own with `NaN`, keeps them when they fill the returns.
        Nd = length(rd.nx)
        carried = po.CarriedDescriptor(ones(2, Nd))
        padded = descriptor(carried, rows(rd, 1:5))
        @test size(padded) == (5, Nd)
        @test all(isnan, padded[1:3, :]) && padded[4:5, :] == ones(2, Nd)
        @test descriptor(carried, rows(rd, 1:2)) == ones(2, Nd)
        @test_throws DimensionMismatch descriptor(carried, rows(rd, 1:1))
        # On the carry fold each Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and the exposures of every step equal those of the batch fit.
        ewb = ["market" => ConstantExposure(),
               "beta" => CompositeExposure(;
                                           descriptors = [EWMarketBeta(; half_life = 10,
                                                                       agg_obs = 3,
                                                                       group = "industry",
                                                                       min_group_size = 3),
                                                          EWDownsideBeta(; half_life = 7)]),
               "macro" => CompositeExposure(;
                                            descriptors = [EWMacroSensitivity(; series = "MACRO",
                                                                              half_life = 8,
                                                                              agg_obs = 2),
                                                           BookToPrice()]),
               "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = ewb,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test isnothing(po.lookback(pe))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160, 250)
        s = stream(pe, rd, e)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, rd, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, Union{po.EWBlockState, po.EWDownsideBetaState}),
                  last(last(s).pe.cache.xf[2]).descriptors)
        @test isa(first(last(last(s).pe.cache.xf[3]).descriptors).cache, po.EWBlockState)
    end
end
