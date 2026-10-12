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

    @testset "A move of a batch choice folds and solves no row of full rank again (#1605), the panel of #1606, the rows solved again" begin
        # The panel of #1606: the only asset in Utilities delists after data row 225, so row
        # 226 has an Unseen Member. A larger market cap of Energy moves the automatic member
        # from Software to Energy. From row 150 the move comes before the step of row 226, and
        # from row 229 it comes after it, so the move solves row 226 again under
        # `SolvedUnseenMember()`.
        dlf = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" =>
                   CompositeExposure(; descriptors = [LogMarketCap()], family = "style")]
        function energy_panel(from, x)
            d = parity_panel(; T = 300, N = 60, seed = 1601).rd
            lvl = ["Banks", "Energy", "Software"]
            ind = [i == 3 ? "Utilities" : lvl[mod(i - 1, 3) + 1] for i in 1:size(d.X, 2)]
            po.panel_field(d.pnl, "market_cap").vals[from:end, ind .== "Energy"] .*= x
            return d
        end
        unseen_rows(st) = [s
                           for s in axes(st.W, 1)
                           if !isnothing(po.unseen_member_change(st.fcb,
                                                                 view(st.Ms, s, :, :),
                                                                 view(st.W, s, :) .> 0, s))]
        # Row 226 is the fitted observation 225, and the step before the move fits it. Under
        # `SolvedUnseenMember()` its answer depends on the dropped member (#1613).
        pe = CrossSectionalFactorPrior(; factors = dlf, families = ["industry" => nothing],
                                       minra = 5, unseen = SolvedUnseenMember())
        dl, de = energy_panel(229, 100.0), (0, 228, 260, 300)
        xs = stream(pe, dl, de)
        s1 = xs[1].pe.cache
        @test unseen_rows(s1) == [225]
        b = batch(pe, 2, dl, de)
        # The columns of the old answer that the new basis keeps are not the batch answer at
        # that row, and the row that the move solves again is.
        Tf = size(s1.Ms, 1)
        fr = po.cross_sectional_expand(s1.fcb, (pe.lag + 1):Tf, pe.lag, s1.csr.f)
        old = fr[225, po.retained_factor_indices(b.rr.fcb)]
        @test relerr(old, b.rr.csr.f[225, :]) > 1e-3
        @test relerr(xs[2].pr.rr.csr.f[225, :], b.rr.csr.f[225, :]) < 1e-12
        @test same(xs[2].pe.cache.csr.eps[1:size(s1.csr.eps, 1), :], s1.csr.eps)
        @test same(xs[2].pe.cache.W[1:size(s1.W, 1), :], s1.W)
        # Under a `LinearModel` target the move solves that row again through the target, and
        # under the default `PseudoInverseFallback()` the target gives the answer of least
        # norm, which is the answer of the linear member (#1614, #1625). Under
        # `DependentColumnDrop()` the rank test drops the column of the Unseen Member by its
        # pivot in both bases on this panel, so it returns zero there (#1620).
        for (alg, own) in ((PseudoInverseFallback(), b),
                           (DependentColumnDrop(),
                            batch(CrossSectionalFactorPrior(; factors = dlf,
                                                            families = ["industry" => nothing], minra = 5,
                                                            unseen = SolvedUnseenMember(),
                                                            cre = CrossSectionalLinearRegression(;
                                                                                                 alg = DependentColumnDrop())),
                                  2, dl, de)))
            pt = CrossSectionalFactorPrior(; factors = dlf,
                                           families = ["industry" => nothing], minra = 5,
                                           unseen = SolvedUnseenMember(),
                                           cre = CrossSectionalTargetRegression(;
                                                                                alg = alg))
            xt = stream(pt, dl, de)
            bt = batch(pt, 2, dl, de)
            @test xt[2].pe.cache.csr.f isa SubArray
            kept = bt.rr.nf[po.retained_factor_indices(bt.rr.fcb)]
            u = bt.rr.csr.f[225, findfirst(==("industry=Utilities"), kept)]
            @test isa(alg, DependentColumnDrop) ? iszero(u) : !iszero(u)
            @test relerr(xt[2].pr.rr.csr.f[225, :], bt.rr.csr.f[225, :]) < 1e-12
            @test relerr(bt.rr.csr.f[225, :], own.rr.csr.f[225, :]) < 1e-12
        end
        # A `GeneralisedLinearModel` target solves that row again through the target too
        # (#1615). Measured: each step equals the batch fit to 4.2e-16 or better.
        pg = CrossSectionalFactorPrior(; factors = dlf, families = ["industry" => nothing],
                                       minra = 5, unseen = SolvedUnseenMember(),
                                       cre = CrossSectionalTargetRegression(;
                                                                            tgt = GeneralisedLinearModel()))
        xg = stream(pg, dl, de)
        @test xg[2].pe.cache.csr.f isa SubArray
        @test only(dropped(xg[2].pr)) == "industry=Energy"
        @test all(k -> near(xg[k].pr, batch(pg, k, dl, de)), eachindex(xg))
        # A beta that shrinks to the mean of its industry is a function of the industry
        # columns where every industry shrinks fully, so the design of such an observation has
        # a dependent factor set under every rule. Its answer of least norm reads the basis, so
        # the move solves it again in the new basis (#1616). Without that, the factor returns
        # differed by 1 %. The rows that the scan solves again are the rows that it writes
        # over `NaN`. A zero column has a return of zero in every basis, so under
        # `ZeroUnseenMember()` the stream with no beta solves no row again, and under
        # `SolvedUnseenMember()` it solves the rows with an Unseen Member alone. A `LinearModel`
        # target fits the same streams: it fits the columns that the rank test keeps, where its
        # own factorisation refused the design with a `PosDefException` (#1620), and projects
        # the answer onto the row space, as `PseudoInverseFallback()` does (#1625).
        function solved(pe, st)
            f = fill(NaN, size(st.csr.f))
            po.cross_sectional_move_solve(pe, st, (; fcb = st.fcb, f = f, lv = st.lv))
            return findall(r -> all(isfinite, r), collect(eachrow(f)))
        end
        beta = "beta" => CompositeExposure(;
                                           descriptors = [EWMarketBeta(; half_life = 10, agg_obs = 5,
                                                                       group = "industry")],
                                           family = "style")
        dl, de = energy_panel(150, 4.0), (0, 150, 190, 225, 226, 300)
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember()), fs in (dlf, [dlf; beta]),
            cre in (CrossSectionalLinearRegression(), CrossSectionalTargetRegression())

            pe = CrossSectionalFactorPrior(; factors = fs,
                                           families = ["industry" => nothing], minra = 5,
                                           unseen = unseen, pe = GRID_PE, cre = cre)
            xs = stream(pe, dl, de)
            @test only(dropped(xs[2].pr)) == "industry=Energy"
            @test xs[2].pe.cache.csr.f isa SubArray
            @test all(k -> near(xs[k].pr, batch(pe, k, dl, de)), eachindex(xs))
            st = xs[end].pe.cache
            rs = solved(pe, st)
            if fs === dlf
                @test rs == (unseen === ZeroUnseenMember() ? Int[] : unseen_rows(st))
            else
                @test unseen === SolvedUnseenMember() || length(rs) == 131
                @test issubset(unseen === ZeroUnseenMember() ? Int[] : unseen_rows(st), rs)
            end
        end
        # The pivot of `DependentColumnDrop()` reads the basis too, so the move solves each
        # rank-deficient row again under it, and the two estimators agree (#1625).
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember())
            pes = map((CrossSectionalLinearRegression(; alg = DependentColumnDrop()),
                       CrossSectionalTargetRegression(; alg = DependentColumnDrop()))) do cre
                return CrossSectionalFactorPrior(; factors = [dlf; beta],
                                                 families = ["industry" => nothing],
                                                 minra = 5, unseen = unseen, pe = GRID_PE,
                                                 cre = cre)
            end
            for pe in pes
                xs = stream(pe, dl, de)
                @test xs[2].pe.cache.csr.f isa SubArray
                @test all(k -> near(xs[k].pr, batch(pe, k, dl, de)), eachindex(xs))
            end
            k = length(de) - 1
            @test near(batch(pes[2], k, dl, de), batch(pes[1], k, dl, de))
        end
        # `RankDeficiencyRefusal()` refuses in the stream at the step whose batch fit refuses,
        # with the same message: under `SolvedUnseenMember()`, the step that fits row 226,
        # whose Unseen Member makes it rank-deficient. Both name the position of the
        # observation among the fitted observations, not its position within the step (#1629).
        dr = de[1:5]
        for cre in (CrossSectionalLinearRegression(; alg = RankDeficiencyRefusal()),
                    CrossSectionalTargetRegression(; alg = RankDeficiencyRefusal()))
            pe = CrossSectionalFactorPrior(; factors = dlf,
                                           families = ["industry" => nothing], minra = 5,
                                           unseen = SolvedUnseenMember(), pe = GRID_PE,
                                           cre = cre)
            ps, ms = pe, String[]
            for k in 1:(length(dr) - 1)
                m = try
                    ps = partial_fit!(ps, rows(dl, (dr[k] + 1):dr[k + 1]))
                    ""
                catch err
                    sprint(showerror, err)
                end
                push!(ms, m)
            end
            mb = [message(() -> batch(pe, k, dl, dr)) for k in eachindex(ms)]
            @test isempty.(ms) == isempty.(mb) == [true, true, true, false]
            @test occursin("observation 225 has rank 4 over 5 factors and 59 eligible assets",
                           mb[end])
            @test ms[end] == mb[end]
        end
        # A new dropped member with a zero benchmark-weighted exposure at an observation has
        # no finite ratio, so the rebase answers `nothing`, and the fit of every observation
        # refuses the member as the batch fit does.
        fcb = FactorFamilyBasis(; fnm = ["ind"], fi = [[1, 2, 3]], di = [1],
                                ratios = [0.5 0.0; 0.25 2.0], K = 3)
        @test isnothing(po.cross_sectional_rebase(fcb, ["ind" => "c"], ["a", "b", "c"]))
        rb = po.cross_sectional_rebase(fcb, ["ind" => "b"], ["a", "b", "c"])
        @test rb.di == [2] && rb.ratios ≈ [2.0 0.0; 4.0 8.0]
        @test po.cross_sectional_rebase(fcb, ["ind" => "a"], ["a", "b", "c"]).ratios ==
              fcb.ratios
    end
end
