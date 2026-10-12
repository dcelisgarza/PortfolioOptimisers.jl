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

    @testset "Folds the members that fold, and refits the rest" begin
        # A rolling window does not fold, so every step fits every observation again over
        # the carried histories, the second pass of the weights too.
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        for cfg in (style, grid_config("Blend", rd))
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg..., ve = wv)
            @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        end
        # A factor prior that does not fold refits over the carried factor returns.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...,
                                       pe = EntropyPoolingPrior(; pe = GRID_PE))
        sx = stream(pe)
        @test all(x -> isa(x.pe.cache.pe, EntropyPoolingPrior), sx)
        @test all(k -> near(sx[k].pr, batch(pe, k)), 1:3)
    end

    @testset "Parity with the oracle's online update" begin
        # The prior of each step against the stored block of that step. `mu` compares cell by
        # cell. `sigma` compares against its largest entry: an off-diagonal entry of two assets
        # near zero correlation is a cancellation in `L F L'` (#1376), measured maxrel 2.7e-12
        # cell by cell. It was 1.3e-10 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576). The factor returns compare cell by cell with an absolute floor: the
        # industry level "Utilities" is empty after its only asset delists, so its factor return
        # is a round-off zero on both sides, measured maxrel 1.9 and maxabs 1.5e-16. The floor
        # is at most 6e-14 of the largest factor return, and every other cell measured maxrel
        # 5.7e-13.
        function check(out, unit_case, src)
            load(o) = parity_load("CrossSectionalFactorPrior", unit_case, "$(src)$(o)")
            mu, S, F = load("Mu"), load("Sigma"), load("FactorReturns")
            r0 = 0
            for k in 1:3
                pr = out[k].pr
                n = edges[k + 1] - 1
                @test parity_compare(pr.mu, mu[k, :]; name = "$(unit_case) mu $(k)").ok
                @test parity_compare(pr.sigma, S[((k - 1) * N + 1):(k * N), :];
                                     scale = :array, name = "$(unit_case) sigma $(k)").ok
                @test parity_compare(pr.fpr.X, F[(r0 + 1):(r0 + n), :]; atol = 1e-15,
                                     name = "$(unit_case) factor returns $(k)").ok
                r0 += n
            end
        end
        pinned(cfg) = stream(CrossSectionalFactorPrior(; lambda = 1, cfg...,
                                                       choice = PinnedChoice()))
        # The pinned choice reproduces the oracle's online update, which pins the dropped
        # member at its first call. Measured over the three steps: mu maxrel 1.7e-14, sigma
        # maxscaled 1.1e-15, factor returns maxrel 4.6e-14 on `Style`; mu maxrel 5.3e-13,
        # sigma maxscaled 1.2e-15, factor returns maxabs 1.5e-16 on `Industry`. The sigma
        # values were 4.2e-13 and 6.5e-13 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576).
        check(pinned(style), "OnlineStyle", "Fold")
        check(pinned(industry), "OnlineIndustry", "Fold")
        # Currency factors on the carry route (#1479). The oracle's online update reads the
        # Currency Excess Returns of each batch, and it pins `style1` as in `Style`. Measured
        # over the three steps: mu maxrel 2.3e-13, sigma maxscaled 5.6e-16 (4.2e-13 before
        # #1576), factor returns maxrel 9.3e-14.
        cp = pinned((; grid_config("Currency", rd)..., families = ["style" => nothing]))
        @test all(x -> x.pe.cache.families == ["style" => "style1"], cp)
        check(cp, "OnlineCurrencyStyle", "Fold")
        # A seed window on the factor prior cuts the factor returns of the first fit to the
        # last 60, and every later step folds every row, as the oracle's does. Measured: mu
        # maxrel 7.2e-15, sigma maxscaled 1.5e-15 (3.7e-13 before #1576), factor returns maxrel
        # 4.6e-14 on `Style`; mu maxrel 1.4e-13, sigma maxscaled 6.1e-16 (7.3e-13 before
        # #1576), factor returns maxabs 1.5e-16 on `Industry`.
        decay = 2.0^(-1 / 20)
        seed = EmpiricalPrior(;
                              me = WindowedExpectedReturns(;
                                                           me = ExpWeightedExpectedReturns(;
                                                                                           decay = decay,
                                                                                           min_obs = 5),
                                                           window = 60,
                                                           rule = SeedWindow()),
                              ce = WindowedCovariance(;
                                                      ce = ExpWeightedCovariance(;
                                                                                 decay = decay,
                                                                                 min_obs = 5,
                                                                                 centring = PreCentred()),
                                                      window = 60, rule = SeedWindow()))
        ss = pinned((; style..., pe = seed))
        check(ss, "CarrySeedStyle", "Fold")
        check(pinned((; industry..., pe = seed)), "CarrySeedIndustry", "Fold")
        # The first fit is the batch fit, and the window of the first fit alone moves the
        # later calls with no data away from it: sigma by 19 % of its largest entry at the last step.
        spe = CrossSectionalFactorPrior(; lambda = 1, style..., pe = seed)
        @test agrees(ss[1].pr, batch(spe, 1))
        @test relerr(ss[3].pr.sigma, batch(spe, 3).sigma) > 0.1
    end

    @testset "On the online step of an optimiser" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        o = po.update_online_estimator(InverseVolatility(; pe = pe))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
            bo, brd = po.batch_from_state(o)
            @test isa(o.pe.cache, po.CrossSectionalCarryState)
            @test near(bo.pe, batch(pe, k))
            @test isequal(brd.X, rd.X[1:edges[k + 1], :])
        end
        @test optimise(o).w == optimise(InverseVolatility(; pe = pe), rd).w
        # `partial_fit` copies the state, so the earlier prior answers as it did.
        e1 = partial_fit!(pe, rows(rd, 1:90))
        p1 = prior(e1)
        e2 = partial_fit(e1, rows(rd, 91:170))
        @test agrees(prior(e1), p1) && near(prior(e2), batch(pe, 2))
    end

    @testset "A step appends to the histories in place (#1564)" begin
        # A step writes its rows into the spare rows of the backing of each history, and each
        # state keeps the view of its own rows.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        e = (0, 90, 91, 92, 93)
        s = stream(pe, rd, e)
        st2, st4 = s[2].pe.cache, s[4].pe.cache
        @test st4.Ms isa SubArray && parent(st4.Ms) === parent(st2.Ms)
        @test parent(st4.csr.f) === parent(st2.csr.f) && parent(st4.vs) === parent(st2.vs)
        @test size(st2.Ms, 1) + 2 == size(st4.Ms, 1)
        # An earlier estimator reads its own rows after a later step. The factor prior folds
        # in place, so its moments read the later state, and the rows are compared alone.
        function rows_agree(x, b)
            return same(x.rr.csr.f, b.rr.csr.f) &&
                   same(x.rr.vs, b.rr.vs) &&
                   same(x.rr.Ms, b.rr.Ms) &&
                   same(x.rr.rw, b.rr.rw) &&
                   x.rr.idx == b.rr.idx
        end
        @test all(k -> rows_agree(prior(s[k].pe), s[k].pr), 1:4)
        # A second step of an earlier state copies its histories, so the later state and the
        # Result read out of it keep their rows.
        H = deepcopy((st4.Ms, st4.X, st4.csr.f, st4.vs, st4.W))
        rdz = deepcopy(rd)
        rdz.X[92:93, :] .*= 1.5
        b = partial_fit!(s[2].pe, rows(rdz, 92:93))
        @test parent(b.cache.Ms) !== parent(st4.Ms)
        @test isequal(H, (st4.Ms, st4.X, st4.csr.f, st4.vs, st4.W))
        @test agrees(s[4].pr, batch(pe, 4, rd, e))
    end

    @testset "A step joins the observed exposures of its new rows alone (#1589)" begin
        # Under observed factors the exposure history and the observed exposures are views of
        # one backing, the estimated factors first. A step appends its joined rows in place, and
        # the call with no data reads the joined history as a view of the backing.
        pe = CrossSectionalFactorPrior(; lambda = 1, grid_config("Currency", rd)...)
        e = (0, 90, 91, 92, 93)
        s = stream(pe, rd, e)
        st2, st4 = s[2].pe.cache, s[4].pe.cache
        P = parent(st4.Ms)
        @test P === parent(st4.obs.Z) && P === parent(st2.Ms)
        @test size(P, 3) == size(st4.Ms, 3) + size(st4.obs.Z, 3)
        @test parent(s[4].pr.rr.Ms) === P
        @test all(k -> agrees(s[k].pr, batch(pe, k, rd, e)), 1:4)
        # A copy keeps the two histories in one backing, and shares no array with the state.
        c = copy(st4)
        @test parent(c.Ms) === parent(c.obs.Z) && parent(c.Ms) !== P
        # A second step of an earlier state copies its histories, so the later Result keeps
        # its rows.
        H = deepcopy(s[4].pr.rr.Ms)
        rdz = deepcopy(rd)
        rdz.X[92:93, :] .*= 1.5
        b = partial_fit!(s[2].pe, rows(rdz, 92:93))
        @test parent(b.cache.Ms) === parent(b.cache.obs.Z) && parent(b.cache.Ms) !== P
        @test isequal(H, s[4].pr.rr.Ms)
        @test agrees(s[4].pr, batch(pe, 4, rd, e))
    end

    @testset "An observed member computes the exposures of its new rows alone (#1593)" begin
        # An `ObservedExposure` folds the member that it wraps, as an estimated member does, so
        # a Descriptor that carries a state reads the new rows alone, and the carry keeps the
        # rows of the lag alone. A Descriptor that carries no state computes the new rows from
        # the rows that it reads. The derivation of the net returns of a row reads the observed
        # exposures of the row `lag` observations before it, so the state carries those of the
        # last `lag` rows.
        ccy = grid_config("Currency", rd)
        function observed(d, nm)
            return nm => ObservedExposure(;
                                          xe = CompositeExposure(; descriptors = [d],
                                                                 outlier = nothing,
                                                                 scoring = nothing, family = nm),
                                          series = "MACRO", family = nm)
        end
        e = (0, 90, 91, 92, 93, 250)
        for (d, L) in ((Reversal(), 1), (EWMomentum(; half_life = 10, skip = 5), 1),
                       (GrowthRate(; field = "sales_ttm", lag = 5), 1),
                       (RollingMax(; window = 6), 6)), lag in (1, 2)

            pe = CrossSectionalFactorPrior(; lambda = 1, ccy...,
                                           factors = [ccy.factors; observed(d, "obs")],
                                           lag = lag)
            @test po.cross_sectional_carry_rows(pe) == L + lag
            s = stream(pe, rd, e)
            for (k, x) in enumerate(s)
                b = batch(pe, k, rd, e)
                @test relerr(x.pr.mu, b.mu) < 1e-12 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-12 &&
                      relerr(x.pr.fpr.X, b.fpr.X) < 1e-12
                @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
            end
            st = last(s).pe.cache
            @test size(st.win.X, 1) == L + lag
            @test size(st.der.Zo) == (lag, N, size(st.obs.Z, 3))
            @test isequal(st.der.Zo, st.obs.Z[(end - lag + 1):end, :, :])
            # The state keeps the folded member at its place in the factor list.
            @test [first(p) for p in st.xf] == [first(p) for p in pe.factors]
            if L == 1
                @test !isnothing(last(st.xf[end]).xe.descriptors[1].cache)
            end
        end
    end
end
