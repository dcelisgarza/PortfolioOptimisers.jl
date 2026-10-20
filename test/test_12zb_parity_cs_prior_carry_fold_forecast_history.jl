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

    @testset "A Calibration Rule resolves at each step (#1481)" begin
        # A rule that reads the number of fitted observations moves at each step. The carry
        # resolves it at the call with no data, as the batch fit over the same rows does.
        nobs(k, p, w, s, x) = size(x.cs.csfm.vs, 1) / 1000
        half(k, p, w, s, x) = 0.5
        pe = CrossSectionalFactorPrior(; lambda = 1,
                                       merge(grid_config("FcFixed", rd),
                                             (; lambda = half, c = nobs))...)
        cs = Float64[]
        for (k, x) in enumerate(stream(pe))
            b = batch(pe, k)
            @test x.pr.rr.c == b.rr.c == size(b.rr.vs, 1) / 1000
            @test x.pr.rr.lambda == 0.5
            @test agrees(x.pr, b)
            push!(cs, x.pr.rr.c)
        end
        @test allunique(cs)
    end

    @testset "The state carries the Return Forecast history (#1484)" begin
        # A member that publishes no history is fitted again at each block row. The state
        # carries those rows, so a step fits the member once for each new observation, and
        # the history equals the one the batch fit makes over the same rows.
        # A stream whose batch choice moves equals the batch fit to rounding, `exact = false`.
        function history_case(cfg, e; exact = true)
            rl, rc = CarryHistoryShrinkage(), CarryHistoryScale()
            bl, bc = CarryHistoryShrinkage(), CarryHistoryScale()
            pc = CrossSectionalFactorPrior(; cfg..., lambda = rl, c = rc)
            pb = CrossSectionalFactorPrior(; cfg..., lambda = bl, c = bc)
            xs = stream(pc, rd, e)
            eqn(p, q) = exact ? p == q : isapprox(p, q; rtol = 1e-12)
            eqh(p, q) = exact ? same(p, q) : close_to(p, q)
            for (k, x) in enumerate(xs)
                b = batch(pb, k, rd, e)
                @test eqn(x.pr.rr.lambda, b.rr.lambda)
                @test eqn(x.pr.rr.c, b.rr.c)
                @test exact ? agrees(x.pr, b) : near(x.pr, b)
                @test eqh(rl.seen[k], bl.seen[k])
                @test eqh(rc.seen[k], bc.seen[k])
                @test same(rl.seen[k], rc.seen[k])
            end
            return (; xs = xs, seen = rl.seen)
        end
        target = grid_config("FcTarget", rd)
        # Blocks of rows, then one observation at a time.
        for e in (edges, (0, 90, 91, 92, 93))
            (; xs, seen) = history_case(target, e)
            for (k, x) in enumerate(xs)
                H = x.pe.cache.hist
                # Every block row but the last, which the call with no data appends.
                @test size(H, 1) == size(seen[k], 1) - 1
                @test same(H, seen[k][1:(end - 1), :])
                # A step that fits the new observations alone keeps the carried rows.
                if k > 1
                    Hp = xs[k - 1].pe.cache.hist
                    @test same(H[1:size(Hp, 1), :], Hp)
                end
            end
            @test allunique([x.pr.rr.lambda for x in xs])
        end
        # A batch choice that moves folds the move, which keeps the old rows of the history
        # (#1605, #1613). Measured when the move fitted every observation again: the move at
        # the second step changes the old rows by up to 1.9e-17, so the carry equals the batch
        # fit to rounding.
        (; xs) = history_case((; target..., families = ["style" => nothing]), edges;
                              exact = false)
        @test [only(x.pe.cache.families).second for x in xs] ==
              ["style1", "style2", "style2"]
        H1, H2 = xs[1].pe.cache.hist, xs[2].pe.cache.hist
        @test same(H1, H2[1:size(H1, 1), :])
        # A member that publishes its history carries no row: the call with no data reads the
        # history of the Result it fits.
        (; xs) = history_case(grid_config("FcEW", rd), edges)
        @test all(x -> isnothing(x.pe.cache.hist), xs)
        @test !PortfolioOptimisers.cross_sectional_carries_history(CrossSectionalFactorPrior(;
                                                                                             grid_config("FcEW",
                                                                                                         rd)...,
                                                                                             c = CarryHistoryScale()))
        # A prior whose slots read no history carries none.
        @test isnothing(last(stream(CrossSectionalFactorPrior(; target...))).pe.cache.hist)
        # The shipped rules that read the history split each carried row against the exposures
        # and the weights of its row, so they resolve on the carry fold as on the batch fit.
        rules = (; lambda = PrecisionBlend(; err = ForecastHistoryError()),
                 c = ForecastCalibrationSlope(; wu = ThresholdWarmUp(; min_obs = 2)))
        # The batch choice moves under the second config, so it equals the batch fit to
        # rounding, as in `history_case`.
        for (cfg, exact) in
            ((target, true), ((; target..., families = ["style" => nothing]), false))
            pr_rules = CrossSectionalFactorPrior(; cfg..., rules...)
            for (k, x) in enumerate(stream(pr_rules))
                b = batch(pr_rules, k)
                @test if exact
                    x.pr.rr.lambda == b.rr.lambda
                else
                    isapprox(x.pr.rr.lambda, b.rr.lambda; rtol = 1e-12)
                end
                @test if exact
                    x.pr.rr.c == b.rr.c
                else
                    isapprox(x.pr.rr.c, b.rr.c; rtol = 1e-12)
                end
                @test 0 <= x.pr.rr.lambda <= 1
                @test x.pr.rr.c != 1
            end
        end
    end

    @testset "The Choice Rule" begin
        # A batch choice that moves folds the move, and equals the batch fit to rounding.
        sb = stream(CrossSectionalFactorPrior(; lambda = 1, style...))
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test all(k -> near(sb[k].pr,
                            batch(CrossSectionalFactorPrior(; lambda = 1, style...), k)),
                  1:3)
        # A pinned choice is recorded in the state, and `families` stays as it is. Its
        # call with no data is the batch fit with the member stated.
        sp = stream(CrossSectionalFactorPrior(; lambda = 1, style...,
                                              choice = PinnedChoice()))
        @test all(x -> x.pe.families == ["style" => nothing], sp)
        @test all(x -> x.pe.cache.families == ["style" => "style1"], sp)
        stated = CrossSectionalFactorPrior(; lambda = 1, style...,
                                           families = ["style" => "style1"])
        @test all(k -> agrees(sp[k].pr, batch(stated, k)), 1:3)
        # A factor that is empty at the first fit and comes alive folds (#1627), and equals
        # the batch fit to rounding.
        rdz = deepcopy(rd)
        po.panel_field(rdz.pnl, "style1").vals[1:120, :] .= 0.0
        pe = CrossSectionalFactorPrior(; lambda = 1, grid_config("Base", rdz)...)
        sz = stream(pe, rdz)
        @test [x.pe.cache.lv for x in sz] == [[true, false, true], trues(3), trues(3)]
        @test sz[2].pe.cache.csr.f isa SubArray
        @test all(k -> near(sz[k].pr, batch(pe, k, rdz)), 1:3)
    end
end
