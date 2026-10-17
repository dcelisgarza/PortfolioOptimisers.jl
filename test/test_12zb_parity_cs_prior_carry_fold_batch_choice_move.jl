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

    @testset "A move of a batch choice folds and solves no row of full rank again (#1605), the grid and the targets" begin
        # The raw factor returns of an observation of full rank do not depend on the dropped
        # member, so a move selects their columns and folds the factor prior again. The
        # residuals of the old observations are the ones of the step before, bit for bit. Under
        # `SolvedUnseenMember()`, the rule of the `grid_config` cases, the move also solves each
        # observation with an Unseen Member again (#1613). A `LinearModel` target gives the
        # same raw factor returns in every basis, so a move under it folds too, and solves
        # each such observation again through the target (#1614).
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember()),
            pf in
            (GRID_PE, EmpiricalPrior(; me = SimpleExpectedReturns(), ce = Covariance())),
            cre in (CrossSectionalLinearRegression(), CrossSectionalTargetRegression())

            pe = CrossSectionalFactorPrior(; lambda = 1, style..., unseen = unseen, pe = pf,
                                           cre = cre)
            sb = stream(pe)
            @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
            @test all(k -> dropped(batch(pe, k)) == dropped(sb[k].pr), 1:3)
            @test all(k -> near(sb[k].pr, batch(pe, k)), 1:3)
            # A refit leaves a plain `Matrix`, and a step appends to a view of a backing.
            @test sb[2].pe.cache.csr.f isa SubArray
            e1, e2 = sb[1].pe.cache.csr.eps, sb[2].pe.cache.csr.eps
            @test same(e2[1:size(e1, 1), :], e1)
            @test same(sb[2].pe.cache.W[1:size(e1, 1), :], sb[1].pe.cache.W)
        end
        # The two solve rules that the loop above leaves out, and a Return Forecast that fits
        # nothing, fold a move the same way.
        for cfg in
            ((; style..., cre = CrossSectionalLinearRegression(; alg = MinimumNormSolve())),
             (; style..., cre = CrossSectionalLinearRegression(; alg = UncheckedSolve())),
             (; grid_config("FcCustom", rd)..., families = ["style" => nothing]))
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            sb = stream(pe)
            @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
            @test all(k -> near(sb[k].pr, batch(pe, k)), 1:3)
            @test sb[2].pe.cache.csr.f isa SubArray
        end
        @test CrossSectionalFactorPrior(; style...).unseen === SolvedUnseenMember()
        # A target that keeps the default answer of `is_basis_invariant` can penalise its
        # coefficients, so a move under it fits every observation again, and equals the batch
        # fit exactly (#1614). A refit leaves a plain `Matrix`.
        @test po.is_basis_invariant(LinearModel())
        @test po.is_basis_invariant(GeneralisedLinearModel())
        @test !po.is_basis_invariant(BasisBoundTarget())
        # Such a target cannot take the projection of `PseudoInverseFallback()`, so it names
        # another solve algorithm (#1625).
        pe = CrossSectionalFactorPrior(; lambda = 1, style...,
                                       cre = CrossSectionalTargetRegression(;
                                                                            tgt = BasisBoundTarget(),
                                                                            alg = DependentColumnDrop()))
        sb = stream(pe)
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test sb[2].pe.cache.csr.f isa Matrix
        @test all(k -> agrees(sb[k].pr, batch(pe, k)), 1:3)
        # A `GeneralisedLinearModel` target folds a move too (#1615). Its fit is iterative, but
        # each iteration is a weighted least-squares fit and the start reads the response alone,
        # so the two bases take the same iterates and stop at the same one. Measured: the
        # default `Normal()` family gives 5.9e-16, and a log link on gross returns gives
        # 1.4e-15 under `Normal()` and 6.3e-16 under `Gamma()`. A tolerance near the rounding
        # of the deviance lets rounding stop the two bases at different iterates: under
        # `Gamma()` with `rtol = atol = 1e-12` the gap is 7.6e-10, on the factor returns of
        # one row, and `1e-14` makes `GLM` throw. So a tighter `kwargs` brings each fit nearer to
        # the maximum-likelihood answer, and it does not shrink the gap.
        glm(args, kw = (;)) = CrossSectionalTargetRegression(;
                                                             tgt = GeneralisedLinearModel(;
                                                                                          args = args,
                                                                                          kwargs = kw))
        rdp = deepcopy(rd)
        rdp.X .+= 1
        loglink = po.GLM.LogLink()
        tight = (; rtol = 1e-12, atol = 1e-12)
        for (r, cre, tol) in ((rd, glm((Distributions.Normal(),)), 1e-12),
                              (rdp, glm((Distributions.Normal(), loglink)), 1e-12),
                              (rdp, glm((Distributions.Gamma(), loglink)), 1e-12),
                              (rdp, glm((Distributions.Gamma(), loglink), tight), 1e-8))
            pe = CrossSectionalFactorPrior(; lambda = 1, style..., cre = cre)
            sb = stream(pe, r)
            @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
            @test sb[2].pe.cache.csr.f isa SubArray
            @test all(k -> near(sb[k].pr, batch(pe, k, r); tol = tol), 1:3)
        end
    end
end
