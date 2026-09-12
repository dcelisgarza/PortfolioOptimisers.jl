#=
The closing verification of map #861, issue #874: every layer above the moments takes the
online step.

One walk-forward runs end to end over a synthetic panel whose universe moves inside the
window, and the online run reaches the batch expanding-window run fold for fold. The six
points of the ticket are the six testsets. The panel is #677's draw from the generator of
`test06c_setup.jl`, carried through its returns and its active mask: the step refuses the
full panel by name, twice, and the first testset pins both refusals as the stated edge of the
map (#1007). The universe moves because the prior carries a `CoveragePolicy` (#866, #977); a
plain prior over an expanding window from row 1 reduces to a Coverage Universe that never
admits a late listing, in batch and online alike, and the second testset pins that too.

Every identity below the JuMP head is bit-exact, because the available-case batch arm of the
policy family *is* the fold (#977), a refit member reads the same rows, and a capped buffer
reads exactly the rolling window. The JuMP head agrees to the solver's tolerance.

The measured gain is where the batch fit is itself a recursion over the rows. The plain
family's batch covariance is one BLAS product, and its online step is no faster at any size
(ratios of 1.0 to 1.6 measured at 8 × 300, 30 × 1500 and 60 × 3000 assets × observations):
the accuracy is the real benefit there, as #316 recorded. The policy family's batch arm folds
row by row, and its online loop runs at 0.55 of batch on this panel through the hierarchical
head, 0.31 at 30 × 1500 and 0.20 at 60 × 3000; through the JuMP head the solve dominates, and
the ratio is 0.97 here and 0.46 at 60 × 3000. The last testset asserts the hierarchical
head's gain and prints the JuMP head's.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))
using Clarabel, Statistics, Dates

const ONL_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = false, allow_almost = false),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                       "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))

# #677's draw, unchanged: eight assets over 300 observations, three listing after row 1 and
# three delisting before row 300, no scattered holes.
const ONL_PANEL = synthetic_asset_panel(; n_assets = 8, n_observations = 300,
                                        n_industries = 3, late_listing_proba = 0.4,
                                        delisting_proba = 0.4, missing_ratio = 0.0,
                                        rng = StableRNG(3))
const ONL_AMSK = ONL_PANEL.rd.pnl.amsk
# The panel the step takes: the returns and the active mask, the estimation mask set to the
# active one, and the generator's Panel Fields dropped. The first testset says why.
const ONL_RD = ReturnsResult(; nx = ONL_PANEL.rd.nx, X = ONL_PANEL.rd.X,
                             ts = ONL_PANEL.rd.ts,
                             pnl = AssetPanel(; amsk = ONL_AMSK, emsk = copy(ONL_AMSK)))
# The same panel beside the generator's own factor returns, for the refit member.
const ONL_RDF = ReturnsResult(; nx = ONL_RD.nx, X = ONL_RD.X, ts = ONL_RD.ts,
                              nf = ONL_PANEL.truth.nf, F = ONL_PANEL.truth.f,
                              pnl = ONL_RD.pnl)
const ONL_CVG = CoveragePolicy()
# The prior whose moments carry an exact seam and admit a listing: the available-case arm.
const ONL_PE = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = ONL_CVG),
                              ce = PortfolioOptimisersCovariance(;
                                                                 ce = Covariance(;
                                                                                 cvg = ONL_CVG)))
const ONL_W, ONL_T, ONL_P = 100, 40, 3
const ONL_BATCH = IndexWalkForward(ONL_W, ONL_T; purged_size = ONL_P, expand_train = true)
const ONL_ONLINE = IndexWalkForward(ONL_W, ONL_T; purged_size = ONL_P, ff = OnlineStep())
const ONL_TRAIN = [1:97, 1:137, 1:177, 1:217, 1:257]

onl_jump(pe) = JuMPOptimiser(; pe = pe, slv = ONL_SLV)
onl_hier(pe) = HierarchicalOptimiser(; pe = pe)
onl_weights(res) = [p.res.w for p in res.pred]
onl_masks(res) = [p.res.imsk for p in res.pred]
# The Investable Mask of a training window under the policy, written out by hand: an asset
# active at the window's last row, whatever its share (`min_coverage = 0`). `nothing` when
# every asset is, which is what the result carries for an unreduced universe.
function onl_policy_mask(tr)
    m = BitVector(ONL_AMSK[last(tr), :])
    return all(m) ? nothing : m
end
# The Coverage Universe of the same window, for a plain prior: finite and active at every
# row of it.
function onl_coverage_mask(tr)
    return BitVector([all(view(ONL_AMSK, tr, j)) for j in axes(ONL_AMSK, 2)])
end

@testset "1. The panel: #677's draw, and the two shapes the step refuses by name" begin
    T, N = size(ONL_RD.X)
    @test (T, N) == (300, 8)
    @test [findfirst(view(ONL_AMSK, :, j)) for j in 1:N] == [1, 34, 42, 1, 67, 10, 175, 1]
    @test [findlast(view(ONL_AMSK, :, j)) for j in 1:N] ==
          [300, 300, 300, 252, 300, 291, 300, 264]
    @test all(isequal.(isfinite.(ONL_RD.X), ONL_AMSK))
    # The two panels the step refuses, each by name (#1007). The generator's panel carries
    # twenty-one time-varying Panel Fields, whose rows are sample the buffers cannot hold;
    # and its estimation mask is narrower than its active mask, which the exact folds of the
    # moment layer take no keyword for.
    mr = MeanRisk(; opt = onl_jump(ONL_PE))
    err = try
        cross_val_predict(mr, ONL_PANEL.rd, ONL_ONLINE)
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("time-varying Asset Panel through its masks alone", err.msg)
    @test occursin("21 Panel Field(s)", err.msg)
    narrow = ReturnsResult(; nx = ONL_RD.nx, X = ONL_RD.X, ts = ONL_RD.ts,
                           pnl = AssetPanel(; amsk = ONL_AMSK,
                                            emsk = copy(ONL_PANEL.rd.pnl.emsk)))
    # The refusal is at warm-up, so it counts the disagreement over the warm-up window.
    narrow_cells = count(view(narrow.pnl.emsk, ONL_TRAIN[1], :) .!=
                         view(ONL_AMSK, ONL_TRAIN[1], :))
    @test narrow_cells == 31
    err = try
        cross_val_predict(mr, narrow, ONL_ONLINE)
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("the estimation mask does not travel the online step", err.msg)
    @test occursin("$narrow_cells cell(s)", err.msg)
    # The batch loop takes both panels. The refusals are the step's, not the walk-forward's.
    @test length(cross_val_predict(mr, ONL_PANEL.rd, ONL_BATCH).pred) == length(ONL_TRAIN)
    # The windows, and the universe inside them: asset 7 lists at row 175, inside the third
    # window, and asset 4 delists at row 252, inside the fifth.
    (; train_idx, test_idx) = split(ONL_BATCH, ONL_RD)
    @test train_idx == ONL_TRAIN
    @test split(ONL_ONLINE, ONL_RD).train_idx == ONL_TRAIN
    @test [first(t):last(t) for t in test_idx] ==
          [101:140, 141:180, 181:220, 221:260, 261:300]
    @test [onl_policy_mask(tr) for tr in ONL_TRAIN] ==
          [BitVector([1, 1, 1, 1, 1, 1, 0, 1]), BitVector([1, 1, 1, 1, 1, 1, 0, 1]),
           nothing, nothing, BitVector([1, 1, 1, 0, 1, 1, 1, 1])]
    # A plain prior over the same windows never admits a listing, because the expanding
    # window starts at row 1 and the Coverage Universe is monotone (#866).
    @test [findall(onl_coverage_mask(tr)) for tr in ONL_TRAIN] ==
          [[1, 4, 8], [1, 4, 8], [1, 4, 8], [1, 4, 8], [1, 8]]
end

@testset "2. The online walk-forward equals the batch expanding one, fold for fold" begin
    # Through a JuMP optimiser and a hierarchical one over the policy prior, whose moments
    # carry an exact seam: the JuMP head to the solver's tolerance, the hierarchical one to
    # the bit, because the available-case batch arm is the fold.
    for (opt, tol) in ((MeanRisk(; opt = onl_jump(ONL_PE)), 1e-5),
                       (HierarchicalRiskParity(; opt = onl_hier(ONL_PE)), 0.0))
        b = cross_val_predict(opt, ONL_RD, ONL_BATCH)
        o = cross_val_predict(opt, ONL_RD, ONL_ONLINE)
        @test length(o.pred) == length(b.pred) == length(ONL_TRAIN)
        for (pb, po, tr) in zip(b.pred, o.pred, ONL_TRAIN)
            @test isa(pb.res.retcode, PortfolioOptimisers.OptimisationSuccess)
            @test isa(po.res.retcode, PortfolioOptimisers.OptimisationSuccess)
            if iszero(tol)
                @test po.res.w == pb.res.w
            else
                @test isapprox(po.res.w, pb.res.w; atol = tol)
            end
            # The universe moves, and both routes agree on it: a listing enters at the
            # third fold and a delisting leaves at the fifth.
            @test po.res.imsk == pb.res.imsk == onl_policy_mask(tr)
            @test isapprox(sum(po.res.w), 1)
            @test isnothing(po.res.imsk) || all(iszero, view(po.res.w, .!po.res.imsk))
            @test isapprox(po.rd.X, pb.rd.X; atol = tol)
        end
        @test !isnothing(o.pred[1].res.imsk) && !o.pred[1].res.imsk[7]
        @test isnothing(o.pred[3].res.imsk)
        @test !isnothing(o.pred[5].res.imsk) && !o.pred[5].res.imsk[4]
    end
    # The plain prior reaches the batch weights too, on the Coverage Universe both routes
    # reduce to.
    for (opt, tol) in ((MeanRisk(; opt = onl_jump(EmpiricalPrior())), 1e-5),
                       (HierarchicalRiskParity(; opt = onl_hier(EmpiricalPrior())), 1e-15))
        b = cross_val_predict(opt, ONL_RD, ONL_BATCH)
        o = cross_val_predict(opt, ONL_RD, ONL_ONLINE)
        for (pb, po, tr) in zip(b.pred, o.pred, ONL_TRAIN)
            @test isapprox(po.res.w, pb.res.w; atol = tol)
            @test po.res.imsk == pb.res.imsk == onl_coverage_mask(tr)
        end
    end
    # A high-order prior folds its co-moments beside the low order. A plain co-moment
    # folded over a changing universe answers a `NaN` frame at every pair naming an asset
    # outside the Coverage Universe, and its read-out repairs the block and leaves the
    # frame, as the low order's does (#874 found the fourth order's read-out reaching
    # LAPACK instead). With a policy on every order the universe moves and the identity is
    # bit-exact; with a policy on the low order alone the higher orders narrow it back to
    # the Coverage Universe (#983).
    plain = HighOrderPriorEstimator(; pe = EmpiricalPrior())
    every = HighOrderPriorEstimator(; pe = ONL_PE, ske = Coskewness(; cvg = ONL_CVG),
                                    kte = Cokurtosis(; cvg = ONL_CVG))
    mixed = HighOrderPriorEstimator(; pe = ONL_PE)
    for (pe, tol, mask) in
        ((plain, 1e-15, onl_coverage_mask), (every, 0.0, onl_policy_mask),
         (mixed, 0.0, onl_coverage_mask))
        opt = HierarchicalRiskParity(; opt = onl_hier(pe))
        b = cross_val_predict(opt, ONL_RD, ONL_BATCH)
        o = cross_val_predict(opt, ONL_RD, ONL_ONLINE)
        for (pb, po, tr) in zip(b.pred, o.pred, ONL_TRAIN)
            @test isapprox(po.res.w, pb.res.w; atol = tol)
            @test po.res.imsk == pb.res.imsk == mask(tr)
        end
    end
    # Through a JuMP head that reads the third order.
    mrk = MeanRisk(; opt = onl_jump(plain), r = NegativeSkewness())
    b = cross_val_predict(mrk, ONL_RD, ONL_BATCH)
    o = cross_val_predict(mrk, ONL_RD, ONL_ONLINE)
    for (pb, po) in zip(b.pred, o.pred)
        @test isapprox(po.res.w, pb.res.w; atol = 1e-5)
    end
end

@testset "3. A refit member equals the batch loop to the last bit" begin
    # A factor prior has no exact seam. Wrapped in `Online`, it refits from the buffer at
    # every read-out, and the buffer holds the rows the batch fold reads, so the two fit
    # the same sample and every weight agrees to the bit, the JuMP head's included.
    for (batch, online) in ((InverseVolatility(; pe = FactorPrior()),
                             InverseVolatility(; pe = PortfolioOptimisers.Online(FactorPrior()))),
                            (HierarchicalRiskParity(; opt = onl_hier(FactorPrior())),
                             HierarchicalRiskParity(; opt = onl_hier(PortfolioOptimisers.Online(FactorPrior())))),
                            (MeanRisk(; opt = onl_jump(FactorPrior())),
                             MeanRisk(; opt = onl_jump(PortfolioOptimisers.Online(FactorPrior())))))
        b = cross_val_predict(batch, ONL_RDF, ONL_BATCH)
        o = cross_val_predict(online, ONL_RDF, ONL_ONLINE)
        @test length(o.pred) == length(ONL_TRAIN)
        @test onl_weights(o) == onl_weights(b)
        @test onl_masks(o) == onl_masks(b)
    end
end

@testset "4. The capped buffer equals the rolling batch walk-forward" begin
    # A rolling scheme of window `w + p` with purge `p` trains over `w` rows; the online
    # scheme with the prior capped at `w` reads out over exactly those rows.
    rolling = IndexWalkForward(ONL_W + ONL_P, ONL_T; purged_size = ONL_P)
    stepped = IndexWalkForward(ONL_W + ONL_P, ONL_T; purged_size = ONL_P, ff = OnlineStep())
    @test all(length.(split(rolling, ONL_RD).train_idx) .== ONL_W)
    cap(pe) = PortfolioOptimisers.Online(pe; max_history = ONL_W)
    for (batch, online) in ((MeanRisk(; opt = onl_jump(EmpiricalPrior())),
                             MeanRisk(; opt = onl_jump(cap(EmpiricalPrior())))),
                            (HierarchicalRiskParity(; opt = onl_hier(EmpiricalPrior())),
                             HierarchicalRiskParity(; opt = onl_hier(cap(EmpiricalPrior())))),
                            (HierarchicalRiskParity(; opt = onl_hier(ONL_PE)),
                             HierarchicalRiskParity(; opt = onl_hier(cap(ONL_PE)))))
        b = cross_val_predict(batch, ONL_RD, rolling)
        o = cross_val_predict(online, ONL_RD, stepped)
        @test length(o.pred) == length(b.pred) == n_splits(rolling, ONL_RD)
        @test onl_weights(o) == onl_weights(b)
        @test onl_masks(o) == onl_masks(b)
    end
end

@testset "5. The online searches pick the batch candidate" begin
    r = ConditionalValueatRisk()
    # The grid tunes the weight bounds, which bind and so separate the candidates.
    grid = ["opt.wb" =>
                [WeightBounds(; lb = 0.0, ub = 1.0), WeightBounds(; lb = 0.0, ub = 0.2),
                 WeightBounds(; lb = 0.1, ub = 1.0)]]
    gs(cv) = GridSearchCrossValidation(grid; cv = cv, r = r, train_score = true)
    rs(cv) = RandomisedSearchCrossValidation(grid; cv = cv, r = r, n_iter = 2,
                                             seed = 20260912)
    for opt in (MeanRisk(; opt = onl_jump(ONL_PE)),
                HierarchicalRiskParity(; opt = onl_hier(ONL_PE)))
        b = search_cross_validation(opt, gs(ONL_BATCH), ONL_RD)
        o = search_cross_validation(opt, gs(ONL_ONLINE), ONL_RD)
        @test size(o.test_scores) == size(b.test_scores) == (length(ONL_TRAIN), 3)
        @test o.test_scores == b.test_scores
        @test o.train_scores == b.train_scores
        @test o.idx == b.idx
        @test o.val_grid[o.idx] == b.val_grid[b.idx]
        # The winner is not a knife-edge: the two matrices agree exactly, and the winning
        # column stands clear of the runner-up.
        means = vec(mean(b.test_scores; dims = 1))
        sorted = sort(means; rev = true)
        @test sorted[1] - sorted[2] > 1e-4
        # The randomised search under the same seed draws the same candidates from the
        # same grid, and inherits the identity.
        rb = search_cross_validation(opt, rs(ONL_BATCH), ONL_RD)
        ro = search_cross_validation(opt, rs(ONL_ONLINE), ONL_RD)
        @test ro.val_grid == rb.val_grid
        @test size(ro.test_scores) == (length(ONL_TRAIN), 2)
        @test ro.test_scores == rb.test_scores
        @test ro.idx == rb.idx
    end
end

@testset "6. The measured gain" begin
    # Wall time of the online loop against the batch loop on this panel, the minimum of
    # three runs each after one warm run. The hierarchical head's step is the moment fit,
    # which the policy family folds row by row in batch, so its gain is the seam's own and
    # is asserted; the JuMP head's step is the solve (ADR 0139 measured the build at 4 to
    # 18 % of it), so on eight assets its ratio sits at one and is printed only.
    function ratio(opt)
        cross_val_predict(opt, ONL_RD, ONL_BATCH)
        cross_val_predict(opt, ONL_RD, ONL_ONLINE)
        tb = minimum(@elapsed(cross_val_predict(opt, ONL_RD, ONL_BATCH)) for _ in 1:3)
        to = minimum(@elapsed(cross_val_predict(opt, ONL_RD, ONL_ONLINE)) for _ in 1:3)
        return tb, to, to / tb
    end
    tb, to, rh = ratio(HierarchicalRiskParity(; opt = onl_hier(ONL_PE)))
    println("online / batch, hierarchical head over the policy prior: ",
            round(to * 1e3; digits = 2), " ms / ", round(tb * 1e3; digits = 2), " ms = ",
            round(rh; digits = 3))
    @test rh < 1
    tb, to, rj = ratio(MeanRisk(; opt = onl_jump(ONL_PE)))
    println("online / batch, JuMP head over the policy prior: ",
            round(to * 1e3; digits = 2), " ms / ", round(tb * 1e3; digits = 2), " ms = ",
            round(rj; digits = 3))
end
