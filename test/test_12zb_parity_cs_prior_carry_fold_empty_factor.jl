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

    @testset "A step where an Empty Factor comes alive folds (#1627)" begin
        # `style1` is zero on the first 120 rows, so its factor comes alive at the second
        # block. It had a zero exposure at every pair of positive weight of each fitted
        # observation, so the step keeps every fitted output, gives the factor a return of zero
        # there, and folds the factor prior again. The batch fit solves those observations
        # with a zero column, through the rank-deficient branch of `PseudoInverseFallback()`
        # and `MinimumNormSolve()`, so the two agree to rounding: measured 8.2e-16 or better.
        # A target under `PseudoInverseFallback()` projects its answer onto the row space of
        # the weighted design, and agrees to rounding too (2.5e-16). `DependentColumnDrop()`
        # drops the zero column by its pivot and fits the columns of the step in their
        # order, so it agrees to the bit under both estimators (#1625).
        rdz = deepcopy(rd)
        po.panel_field(rdz.pnl, "style1").vals[1:120, :] .= 0.0
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember()),
            cre in (CrossSectionalLinearRegression(),
                    CrossSectionalLinearRegression(; alg = MinimumNormSolve()),
                    CrossSectionalLinearRegression(; alg = DependentColumnDrop()),
                    CrossSectionalTargetRegression(),
                    CrossSectionalTargetRegression(; alg = DependentColumnDrop()),
                    CrossSectionalTargetRegression(; tgt = GeneralisedLinearModel()))

            pe = CrossSectionalFactorPrior(; lambda = 1, grid_config("Base", rdz)...,
                                           unseen = unseen, cre = cre)
            sz = stream(pe, rdz)
            @test [x.pe.cache.lv for x in sz] == [[true, false, true], trues(3), trues(3)]
            # A refit leaves a plain `Matrix`, and a step appends to a view of a backing.
            @test sz[2].pe.cache.csr.f isa SubArray
            s1, s2 = sz[1].pe.cache, sz[2].pe.cache
            n = size(s1.csr.f, 1)
            @test same(s2.csr.eps[1:n, :], s1.csr.eps) && same(s2.W[1:n, :], s1.W)
            @test all(iszero, s2.csr.f[1:n, 2])
            fit = cre.alg isa DependentColumnDrop ? agrees : near
            @test all(k -> fit(sz[k].pr, batch(pe, k, rdz)), 1:3)
        end
        # `RankDeficiencyRefusal()` refuses a design with a zero column, and `UncheckedSolve()`
        # throws on a square one or hands it to a target unchecked, so under either one the
        # step fits every observation again, and gives the answer or the refusal of the batch
        # fit.
        unchecked(alg) = CrossSectionalFactorPrior(; lambda = 1,
                                                   grid_config("Base", rdz)...,
                                                   unseen = ZeroUnseenMember(),
                                                   cre = CrossSectionalLinearRegression(;
                                                                                        alg = alg))
        @test !po.cross_sectional_alive_folds(CrossSectionalLinearRegression(;
                                                                             alg = UncheckedSolve()))
        @test !po.cross_sectional_alive_folds(CrossSectionalLinearRegression(;
                                                                             alg = RankDeficiencyRefusal()))
        pe = unchecked(UncheckedSolve())
        sz = stream(pe, rdz)
        @test sz[2].pe.cache.csr.f isa Matrix
        @test all(k -> agrees(sz[k].pr, batch(pe, k, rdz)), 1:3)
        pe = unchecked(RankDeficiencyRefusal())
        p1 = partial_fit!(pe, rows(rdz, 1:90))
        m = message(() -> partial_fit!(p1, rows(rdz, 91:170)))
        @test occursin("has rank 2 over 3 factors", m)
        @test m == message(() -> batch(pe, 2, rdz))
        @test po.cross_sectional_alive_folds(CrossSectionalLinearRegression())
        @test po.cross_sectional_alive_folds(CrossSectionalTargetRegression())
        @test !po.cross_sectional_alive_folds(CrossSectionalTargetRegression(;
                                                                             alg = UncheckedSolve()))
        # The only asset of Utilities joins it at data row 200, so that level comes alive at
        # the step of row 201, which the exposures of row 200 explain, under the industry
        # family of a batch choice. The asset delists after row 225, where Utilities becomes an
        # Unseen Member. The same stream folds under a Target Return Forecast and under an
        # observed factor. Measured: 4.0e-14 or better.
        dlf = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" =>
                   CompositeExposure(; descriptors = [LogMarketCap()], family = "style")]
        function late_panel(L; from = nothing, x = 4.0)
            d = parity_panel(; T = 300, N = 60, seed = 1601).rd
            po.panel_field(d.pnl, "industry").codes[1:(L - 1), 3] .= 1
            if !isnothing(from)
                lvl = ["Banks", "Energy", "Software"]
                ind = [i == 3 ? "Utilities" : lvl[mod(i - 1, 3) + 1]
                       for i in 1:size(d.X, 2)]
                po.panel_field(d.pnl, "market_cap").vals[from:end, ind .== "Energy"] .*= x
            end
            return d
        end
        rev = "rev" => ObservedExposure(;
                                        xe = CompositeExposure(; descriptors = [Reversal()],
                                                               outlier = nothing, scoring = nothing,
                                                               family = "rev"), series = "MACRO",
                                        family = "rev")
        dsl = DescriptorScores(;
                               descriptors = [Passthrough(; field = "net_income_ttm"),
                                              Passthrough(; field = "sales_ttm")])
        dl, de = late_panel(200), (0, 150, 200, 201, 260, 300)
        for (unseen, cre, kw) in
            ((ZeroUnseenMember(), CrossSectionalLinearRegression(), (;)),
             (SolvedUnseenMember(), CrossSectionalLinearRegression(), (;)),
             (ZeroUnseenMember(), CrossSectionalTargetRegression(), (;)),
             (SolvedUnseenMember(), CrossSectionalTargetRegression(), (;)),
             (ZeroUnseenMember(), CrossSectionalLinearRegression(),
              (; lambda = 0.4, c = 0.6,
               rfe = TargetReturnForecast(; scores = dsl, scale = 0.02))),
             (SolvedUnseenMember(), CrossSectionalTargetRegression(),
              (; factors = [dlf; rev])))
            pe = CrossSectionalFactorPrior(; factors = dlf,
                                           families = ["industry" => nothing], minra = 5,
                                           pe = GRID_PE, ve = GRID_VE, unseen = unseen,
                                           cre = cre, kw...)
            xs = stream(pe, dl, de)
            @test [count(x.pe.cache.lv) for x in xs] == [4, 4, 5, 5, 5]
            @test all(x -> x.pe.cache.csr.f isa SubArray, xs[2:end])
            @test all(k -> near(xs[k].pr, batch(pe, k, dl, de)), eachindex(xs))
        end
        # A move of the dropped member and a factor that comes alive at one step: the move
        # folds first, and the step then folds the new factor. Measured: 1.2e-15 or better.
        dm, dem = late_panel(170; from = 150), (0, 150, 190, 226, 300)
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember()),
            cre in (CrossSectionalLinearRegression(), CrossSectionalTargetRegression())

            pe = CrossSectionalFactorPrior(; factors = dlf,
                                           families = ["industry" => nothing], minra = 5,
                                           pe = GRID_PE, ve = GRID_VE, unseen = unseen,
                                           cre = cre)
            xs = stream(pe, dm, dem)
            @test [count(x.pe.cache.lv) for x in xs] == [4, 5, 5, 5]
            @test [only(dropped(x.pr)) for x in xs] ==
                  ["industry=Software"; fill("industry=Energy", 3)]
            @test xs[2].pe.cache.csr.f isa SubArray
            @test all(k -> near(xs[k].pr, batch(pe, k, dm, dem)), eachindex(xs))
        end
    end
end
