#=
Walk-forward parity with the oracle, on a point-in-time universe (#677).

`test_57_point_in_time_universe.jl` pins the contract against a hand-reduced oracle. This file
asks the other question: on the same walk-forward, does this library reach the oracle's own
numbers? Every expected value below was measured by RUNNING the oracle at its `v1.0.4` tag,
not by reading it, on the two panels the generator of #656 draws at `StableRNG(3)`.

Where the two can be compared, and where they cannot
====================================================

The oracle reduces to an investable subset in its **convex** family only. Its hierarchical
base shadows the input cleaner with a version that omits the mask, so a gapped panel does not
reach a reduction there at all -- it reaches its correlation step and raises
`ValueError: Input X contains NaN`. Measured on all four folds of the gapped panel below. That
is the capability this map adds: the same run answers here.

So the comparison is built in three layers, from the one that admits the tightest tolerance to
the one that admits the loosest:

 1. **The clustering inputs.** The distance matrix each side clusters, and the heights of the
    linkage it builds, agree to floating-point noise. This is the part of the hierarchical
    route that carries no tie-break.
 2. **The hierarchical allocation, with the dendrogram's leaf order pinned to the oracle's.**
    Equal to round-off, on both panels and every fold (maxrel 6.3e-16). The leaf order has
    to be pinned because the two sides order the *same* dendrogram differently -- see the
    note below -- and the allocation
    is what this library is being asked to reproduce, not the third-party ordering routine.
 3. **The convex allocation.** Both sides solve the same quadratic programme through a
    different build of a solver, so the weights agree to the solver's precision rather than to
    machine precision, and the meaningful comparison is the objective each attains. This
    library's solution attains a **lower** portfolio variance on every fold of both panels.

The leaf-ordering divergence
============================

Both sides build the same dendrogram: the heights agree cell by cell to `8.5e-16` relative, a
rounding of the linkage recursion. They then order its branches differently on three of
the four folds of the complete panel, and each ordering is a valid ordering of that same tree --
one is reachable from the other by flipping internal nodes. Since #1494 the library's default
`branchorder = :optimal` is the exact optimal leaf ordering, and the other side's routine is not
optimal, so the two orders differ by design. The difference moves an HRP weight by as much as
`0.09` (#947). The parity of layer 2 is stated with the order pinned, so that this file measures
the allocation rather than the leaf order.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))
include(joinpath(@__DIR__, "parity_harness.jl"))
using Clarabel, Clustering, Statistics, Dates

const REFP_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                        check_sol = (; allow_local = true, allow_almost = true),
                        settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                        "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                        "tol_infeas_abs" => 1e-12,
                                        "tol_infeas_rel" => 1e-12))

# The two panels. `gap` moves its universe inside the window; `full` is the same draw with the
# listing and the delisting probabilities set to zero, so the oracle has an answer for every
# family on it and the comparison isolates the walk-forward and the optimiser from the
# reduction.
const REFP_GAP = synthetic_asset_panel(; n_assets = 8, n_observations = 300,
                                       n_industries = 3, late_listing_proba = 0.4,
                                       delisting_proba = 0.4, missing_ratio = 0.0,
                                       rng = StableRNG(3)).rd
const REFP_FULL = synthetic_asset_panel(; n_assets = 8, n_observations = 300,
                                        n_industries = 3, late_listing_proba = 0.0,
                                        delisting_proba = 0.0, missing_ratio = 0.0,
                                        rng = StableRNG(3)).rd
const REFP_CV = IndexWalkForward(120, 40)
const REFP_TRAIN = [1:120, 41:160, 81:200, 121:240]

# The assets the oracle optimises over, per fold. On the complete panel that is every
# asset; on the gapped one it is the subset the oracle was handed by hand, because its
# hierarchical family cannot derive it.
const REFP_KEEP = Dict("full" => [collect(1:8) for _ in 1:4],
                       "gap" => [[1, 4, 8], [1, 2, 4, 6, 8], [1, 2, 3, 4, 5, 6, 8],
                                 [1, 2, 3, 4, 5, 6, 8]])

# The oracle's dendrogram leaf order, per fold, as positions within `REFP_KEEP`.
const REFP_ORDER = Dict("full" => [[7, 6, 2, 5, 4, 3, 8, 1], [7, 8, 1, 2, 5, 4, 3, 6],
                                   [7, 8, 3, 2, 5, 4, 1, 6], [7, 8, 3, 4, 5, 2, 1, 6]],
                        "gap" => [[3, 1, 2], [5, 4, 3, 2, 1], [6, 1, 4, 5, 2, 3, 7],
                                  [6, 1, 2, 5, 4, 3, 7]])

# The heights of the oracle's linkage, per fold.
const REFP_HEIGHT = Dict("full" =>
                             [[0.28730777867496443, 0.38571468075473603, 0.4673070260805271,
                               0.55059746754279781, 0.59395677279713754,
                               0.62426186018601704, 0.77835894302934849],
                              [0.28713174063509389, 0.39151923028154656,
                               0.46420379095153375, 0.54454361870617385,
                               0.57015939642555091, 0.60164192098420388,
                               0.78932788073078808],
                              [0.26270664686781631, 0.39395649305394614,
                               0.42203007017795219, 0.48325897402791118, 0.5266441886651686,
                               0.61307669772073758, 0.82383273347103692],
                              [0.23882382069163213, 0.35944132414609897,
                               0.42439743106937344, 0.42654761744367853,
                               0.50932732862639785, 0.60248587755092353,
                               0.8061536572615331]],
                         "gap" => [[0.47978571495280664, 0.57768478814210278],
                                   [0.39367673307130807, 0.51611599678645714,
                                    0.53492449427293465, 0.61819695861325874],
                                   [0.26270664686781631, 0.39395649305394614,
                                    0.42203007017795219, 0.48325897402791118, 0.5266441886651686,
                                    0.61307669772073758],
                                   [0.23882382069163213, 0.35944132414609897,
                                    0.42439743106937344, 0.42654761744367853,
                                    0.50932732862639785, 0.60248587755092353]])

# The oracle's Hierarchical Risk Parity weights, per fold, on the assets of `REFP_KEEP`.
const REFP_HRP = Dict("full" => [[0.076390784075419924, 0.089096067878742732,
                                  0.093107791547405824, 0.038879879913144358,
                                  0.070635201103435477, 0.1320137965934281, 0.35706074794069786,
                                  0.14281573094772573],
                                 [0.078995341833409602, 0.11677712861509344, 0.096381077219134081,
                                  0.032678014955128155, 0.06014234681996266, 0.07378537966549184,
                                  0.33894974253074178, 0.20229096836103849],
                                 [0.063310982185124834, 0.093708159328545981, 0.09036886506759087,
                                  0.038028163956322907, 0.065776189163092094,
                                  0.096243051256758014, 0.35437229975711193, 0.19819228928545338],
                                 [0.059466380456061334, 0.073065731706789733, 0.10209517538874027,
                                  0.048814373512512936, 0.063513460827334334,
                                  0.088533568698107273, 0.36121042798892233, 0.20330088142153183]],
                      "gap" =>
                          [[0.22573438335044638, 0.13239067111403569, 0.64187494553551794],
                           [0.10321349085416587, 0.15257830166375858, 0.096955433685417064,
                            0.24370837519315056, 0.40354439860350794],
                           [0.1005474603712317, 0.127742832941901, 0.14009169119754195,
                            0.075317095843209017, 0.10685103894792429, 0.25166407253481921,
                            0.19778580816337288],
                           [0.093897366798357612, 0.15420870045697993, 0.14004792589717605,
                            0.062169971148710854, 0.11508192299491267, 0.23425395124266593,
                            0.20034016146119696]])

# The oracle's minimum-variance weights, per fold, on the assets of `REFP_KEEP`.
const REFP_MR = Dict("full" => [[0.079620316071204164, 0.026102935248386226,
                                 6.5057722590900107e-7, 7.8994606463973309e-8,
                                 2.0087278651385208e-7, 0.1346271321786163, 0.53192634108756043,
                                 0.2277223449696138],
                                [0.051556395269849599, 0.038838476104729125,
                                 0.00023312723656529827, 5.7244034543309455e-7,
                                 1.555001113044431e-6, 0.13593051907309181, 0.53767575841335058,
                                 0.23576359646095485],
                                [0.031480053369017894, 3.1095498506425611e-6,
                                 4.0645628266337565e-6, 5.3913761607981901e-7,
                                 1.0597228702377962e-6, 0.18361988100533058, 0.54975164756048223,
                                 0.23513964509200572],
                                [0.032952622390051207, 1.1109561665748777e-6, 0.01307181529522829,
                                 4.0538617472753613e-7, 6.9178543939829565e-7,
                                 0.14967668525712088, 0.58015058830615462, 0.22414608062366417]],
                     "gap" =>
                         [[0.24460940731308584, 0.02075258172558141, 0.73463801096133263],
                          [0.050183753814430181, 0.12653895773631341, 8.177130945305387e-8,
                           0.28672426361713121, 0.53655294306081591],
                          [2.9541699352039129e-7, 0.0040609564023445944,
                           5.2740311242331317e-7, 1.059816555153692e-7,
                           4.8275592104562452e-7, 0.4044488239910522, 0.59148880804892068],
                          [7.4443223113623013e-8, 1.1380417359716253e-7,
                           0.0084276848929051712, 2.8443556135639853e-8,
                           9.2032539761969501e-8, 0.36555338055547026, 0.62601862582813206]])

refp_rd(tag) = tag == "full" ? REFP_FULL : REFP_GAP

# The training window of one fold, reduced to the assets the oracle optimised over.
function refp_window(tag, i)
    keep = REFP_KEEP[tag][i]
    return refp_rd(tag).X[REFP_TRAIN[i], keep], keep
end

@testset "The clustering inputs agree with the oracle" begin
    # The distance each side clusters. The oracle's is `sqrt(0.5 * (1 - corr))`, which is
    # this library's canonical distance over a Pearson correlation, so the two are the same
    # quantity reached by two routes.
    de = Distance(; alg = CanonicalDistance())
    for tag in ("full", "gap"), i in 1:4
        Xtr, _ = refp_window(tag, i)
        D = PortfolioOptimisers.distance(de, PortfolioOptimisersCovariance(), Xtr)
        @test isapprox(D, sqrt.(0.5 .* (1 .- cor(Xtr))); atol = 1e-15)
        # And the dendrogram built from it has the oracle's heights. Measured maxrel
        # 4.2e-16, and 8.5e-16 under the flags of CI (`--check-bounds=yes`): a rounding
        # of the Ward recursion, whose summation order differs.
        clr = clusterise(ClustersEstimator(), Xtr)
        @test parity_compare(clr.res.heights, REFP_HEIGHT[tag][i]; rtol = 1e-14,
                             name = "$(tag) fold $(i) heights").ok
    end
end

@testset "Hierarchical Risk Parity reproduces the oracle to round-off" begin
    # With the dendrogram's leaf order pinned to the oracle's, the allocation is the
    # reference's own, on the complete panel and on the gapped one alike. Measured maxrel
    # 3.6e-16, and 6.3e-16 under the flags of CI, a few ulps.
    for tag in ("full", "gap"), i in 1:4
        Xtr, keep = refp_window(tag, i)
        clr = clusterise(ClustersEstimator(), Xtr)
        pinned = Clusters(Clustering.Hclust(clr.res.merges, clr.res.heights,
                                            REFP_ORDER[tag][i], clr.res.linkage), clr.S,
                          clr.D, clr.P, clr.k)
        w = optimise(HierarchicalRiskParity(; r = Variance(),
                                            opt = HierarchicalOptimiser(; cle = pinned)),
                     ReturnsResult(; nx = refp_rd(tag).nx[keep], X = Xtr)).w
        @test parity_compare(w, REFP_HRP[tag][i]; rtol = 1e-14,
                             name = "$(tag) fold $(i) HRP").ok
        @test isapprox(sum(w), 1)
    end
end

@testset "The minimum-variance programme reaches the oracle's optimum or better" begin
    mr = MeanRisk(; obj = MinimumRisk(), r = Variance(),
                  opt = JuMPOptimiser(; slv = REFP_SLV))
    for tag in ("full", "gap")
        mpr = PortfolioOptimisers.fit_and_predict(mr, refp_rd(tag), REFP_CV)
        for i in 1:4
            Xtr, keep = refp_window(tag, i)
            w = mpr.pred[i].res.w[keep]
            ref = REFP_MR[tag][i]
            # The weights agree to the solver's precision. The two sides run different
            # builds of the solver on the same programme, so this tolerance is sized to the
            # solve and not to the arithmetic: the largest disagreement measured over the
            # eight folds is `5.6e-4` in the two-norm and `3.5e-4` on a single weight. The
            # check is absolute: a weight near zero on one side has no relative meaning.
            @test parity_compare(w, ref; rtol = 0.0, atol = 1e-3,
                                 name = "$(tag) fold $(i) min-variance").ok
            @test isapprox(sum(w), 1)
            # The objective is where the comparison is meaningful, and it is where this
            # library does not lose: the variance it attains is never the higher one.
            sigma = cov(PortfolioOptimisersCovariance(), Xtr)
            @test dot(w, sigma, w) <= dot(ref, sigma, ref)
            # The two objectives agree to `2.5e-6` relative at worst.
            @test parity_compare([dot(w, sigma, w)], [dot(ref, sigma, ref)]; rtol = 1e-5,
                                 name = "$(tag) fold $(i) objective").ok
        end
    end
end

@testset "The gapped panel is answered here and refused there" begin
    # The oracle raises `ValueError: Input X contains NaN` on every fold of the gapped
    # panel through its hierarchical family, with its exponentially weighted moments routed
    # the active mask and everything else at its defaults: the mask never reaches its
    # correlation step. This library answers the same run, which is the capability the map
    # was charted to add, so what is asserted here is that the answer exists and obeys the
    # contract -- there is no oracle number to compare it against.
    for opt in
        (HierarchicalRiskParity(; r = Variance()), HierarchicalEqualRiskContribution(),
         SchurComplementHierarchicalRiskParity())
        mpr = PortfolioOptimisers.fit_and_predict(opt, REFP_GAP, REFP_CV)
        @test length(mpr.pred) == 4
        for (i, pred) in pairs(mpr.pred)
            @test length(pred.res.w) == 8
            @test isapprox(sum(pred.res.w), 1)
            @test all(iszero, pred.res.w[setdiff(1:8, REFP_KEEP["gap"][i])])
        end
    end
end
