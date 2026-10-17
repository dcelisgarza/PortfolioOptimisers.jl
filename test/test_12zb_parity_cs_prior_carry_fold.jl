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
# A test-local Descriptor that reads the implied volatility of its own row, so the carry
# answers the batch fit only when the rows it carries hold the surface (#1647).
struct ImpliedVolatilityLevel <: PortfolioOptimisers.AbstractDescriptorEstimator end
function PortfolioOptimisers.descriptor(::ImpliedVolatilityLevel, rd::ReturnsResult)
    return ifelse.(rd.pnl.amsk, rd.iv, NaN)
end
PortfolioOptimisers.lookback(::ImpliedVolatilityLevel) = 1

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

    @testset "The carry fold equals the batch fit at each step, the grid" begin
        # Measured: every configuration gives a difference of exactly zero, because each
        # output of an observation reads no later observation, and the step runs the code of
        # the batch fit on the new observations.
        for name in
            ("Base", "Lag2", "Blend", "Neutralised", "FamStated", "FamTwo", "FcFixed",
             "FcCustom")
            pe = CrossSectionalFactorPrior(; lambda = 1, grid_config(name, rd)...)
            for (k, x) in enumerate(stream(pe))
                @test agrees(x.pr, batch(pe, k))
            end
        end
        # One observation at a time after the first fit. The batch choice of `style` moves at
        # the last step, and the fold of the move equals the batch fit to rounding (#1605,
        # #1613), so the configurations of `style` compare with `near` from here on.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        e = (0, 90, 91, 92, 93, 250)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test near(x.pr, batch(pe, k, rd, e))
        end
    end

    @testset "The carried rows hold the implied volatilities that a Descriptor reads (#1647)" begin
        iv = 0.15 .+ 0.1 .* rand(StableRNG(1647), size(rd.X)...)
        rdi = ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, ne = rd.ne,
                            E = rd.E, ts = rd.ts, iv = iv, ivpa = 1.1, pnl = rd.pnl)
        base = grid_config("Base", rdi)
        fl = vcat(base.factors,
                  ["ivl" => CompositeExposure(; descriptors = [ImpliedVolatilityLevel()],
                                              outlier = nothing, scoring = nothing,
                                              family = "ivl")])
        pe = CrossSectionalFactorPrior(; lambda = 1, base..., factors = fl)
        @test po.cross_sectional_carry_rows(pe) isa Integer
        s = stream(pe, rdi)
        @test all(k -> agrees(s[k].pr, batch(pe, k, rdi)), 1:3)
        # The buffer records the surface, so a member above the prior reads it back.
        kw = po.implied_volatility_buffer_kwargs(po.prior_returns_buffer(s[end].pe))
        @test kw.iv == iv && kw.ivpa == 1.1
        # The first step fixes whether the steps carry `iv`, and pins `ivpa`.
        r = rows(rdi, 91:91)
        p = partial_fit!(pe, rows(rdi, 1:90))
        m = message(() -> partial_fit!(po.copy_states(p),
                                       ReturnsResult(; nx = r.nx, X = r.X, ne = r.ne,
                                                     E = r.E, ts = r.ts, iv = r.iv,
                                                     ivpa = 2.0, pnl = r.pnl)))
        @test occursin("`ivpa`", m)
        m = message(() -> partial_fit!(po.copy_states(p),
                                       ReturnsResult(; nx = r.nx, X = r.X, ne = r.ne,
                                                     E = r.E, ts = r.ts, pnl = r.pnl)))
        @test occursin("implied volatilities", m)
        # An optimiser over the unwrapped prior steps when its clustering reads the surface,
        # and gives the batch weights. The clustering reads the returns data, so the plain
        # method of `ImpliedVolatility` reads the assets whose returns have no gap.
        cle = ClustersEstimator(; ce = ImpliedVolatility(; ppy = 252))
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe, cle = cle))
        rdc = po.port_opt_view(rdi, :,
                               findall(j -> all(isfinite, view(rdi.X, :, j)),
                                       axes(rdi.X, 2)))
        o = foldl((o, k) -> partial_fit!(o, rows(rdc, (edges[k] + 1):edges[k + 1])), 1:3;
                  init = hrp)
        @test po.returns_result(o).iv == rdc.iv
        @test optimise(o).w == optimise(hrp, rdc).w
        # Under `x_src = PriorSource()` the clustering reads the fitted rows of the prior, the last
        # rows after the Descriptor warm-up and the exposure lag, and the last rows of the
        # surface pair them (#1655).
        cle = ClustersEstimator(; ce = ImpliedVolatility(; ppy = 252),
                                x_src = PriorSource())
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe, cle = cle))
        res = optimise(hrp, rdc)
        @test size(res.pr.X, 1) < size(rdc.X, 1)
        n = size(res.pr.X, 1)
        @test res.clr.res.order ==
              clusterise(cle, res.pr.X; iv = rdc.iv[(end - n + 1):end, :], ivpa = rdc.ivpa).res.order
        o = foldl((o, k) -> partial_fit!(o, rows(rdc, (edges[k] + 1):edges[k + 1])), 1:3;
                  init = hrp)
        @test optimise(o).w == res.w
    end
end
