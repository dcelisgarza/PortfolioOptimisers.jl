using Test, PortfolioOptimisers

const PO = PortfolioOptimisers

# The frontier sweep's flat order is load-bearing, not cosmetic (ADR 0062).
# `NearOptimalCentering` solves its anchors as one `MeanRisk` sweep over the same two
# frontiers and pairs anchor `i` with sweep point `i`, so the two sweeps must enumerate
# their points in the same order. `frontier_sweep_axes` is the one place that order is
# stated. These tests need no solver and no model: they drive the seam on synthetic
# registries.

# A registry entry is `(bound_var_key, bound_key) => (expr, points, …)`. `frontier_axis`
# reads only the first key and the points, so the expression is `nothing` here.
ret_registry = [(:ret_lb_var_1, :ret_lb_1) => (nothing, [10.0, 20.0], 1)]
risk_registry = [(:sd_ub_var, :sd_ub) => (nothing, [1.0, 2.0, 3.0], true)]

@testset "Frontier sweep order: one axis enumerates its own points" begin
    axis = PO.frontier_axis(ret_registry)
    pts = PO.frontier_sweep_axes(axis, nothing)
    @test length(pts) == 2
    # Each point is a tuple of one `(keys, bounds)` pair, and each pair holds one entry.
    @test [only(p[1][1]) for p in pts] == [:ret_lb_var_1, :ret_lb_var_1]
    @test [only(p[1][2]) for p in pts] == [10.0, 20.0]
end

@testset "Frontier sweep order: several entries on one axis are a product" begin
    two = [(:ret_lb_var_1, :ret_lb_1) => (nothing, [10.0, 20.0], 1),
           (:ret_lb_var_2, :ret_lb_2) => (nothing, [30.0, 40.0, 50.0], 2)]
    pts = PO.frontier_sweep_axes(PO.frontier_axis(two), nothing)
    # Two bounds of 2 and 3 points cost 6 solves, which is what `frontier_sweep_points`
    # counts.
    @test length(pts) == 6
    # Iterate exactly as `frontier_sweep!` does, so this asserts the traversal and not the
    # shape a comprehension over a product would give back.
    order = Vector{Float64}[]
    for p in pts
        push!(order, collect(p[1][2]))
    end
    @test order == [[10.0, 30.0], [20.0, 30.0], [10.0, 40.0], [20.0, 40.0], [10.0, 50.0],
                    [20.0, 50.0]]
end

@testset "Frontier sweep order: the risk axis varies fastest" begin
    ret_axis = PO.frontier_axis(ret_registry)
    risk_axis = PO.frontier_axis(risk_registry)
    pts = PO.frontier_sweep_axes(ret_axis, risk_axis)
    @test length(pts) == 6
    # `p[1]` is the risk axis and `p[2]` the return axis. The return bound is held while the
    # risk bound runs, so the flat order is return-outer and risk-inner.
    order = Tuple{Float64, Float64}[]
    for p in pts
        push!(order, (only(p[2][2]), only(p[1][2])))
    end
    @test order ==
          [(10.0, 1.0), (10.0, 2.0), (10.0, 3.0), (20.0, 1.0), (20.0, 2.0), (20.0, 3.0)]
end

@testset "Frontier sweep order: a sweep with no frontier still has its points" begin
    # The unconstrained `NearOptimalCentering` sweep: no bound to write, one solve per
    # anchor.
    pts = Iterators.repeated((), 4)
    @test length(pts) == 4
    @test all(p -> p === (), pts)
end

# A `:risk_frontier` entry is `(bound_var_key, bound_key) => (expr, points, flag, owner)`.
# The registry is **not** parallel to the risk measure vector: a measure registers an entry
# only when its `settings.ub` is a `Front_NumVec`. So an entry names its owner rather than
# standing at the owner's position, and the rebuild reads both the owner and the expression
# off the entry it is rebuilding.
@testset "Frontier rebuild: an entry is rebuilt from itself, not from entry 1" begin
    X = [0.01 0.02 -0.01; -0.02 0.01 0.03; 0.03 -0.01 0.02; -0.01 0.02 -0.02;
         0.02 -0.03 0.01; 0.01 0.01 0.01; -0.03 0.02 -0.01; 0.02 0.01 0.02]
    pr = prior(EmpiricalPrior(), X)
    w_min = fill(1 / 3, 3)
    w_max = [0.7, 0.2, 0.1]
    front = Frontier(; N = 3)
    registry = [(:a_ub_var, :a_ub) => (:EXPR_ONE, front, true, 1),
                (:b_ub_var, :b_ub) => (:EXPR_TWO, front, false, 2)]
    rs = factory([Variance(), ConditionalValueatRisk()], pr)

    entry = PO._rebuild_risk_frontier(pr, nothing, rs[2], registry, w_min, w_max, 2)
    # The keys, the expression, the polarity and the owner are entry 2's own. Only the bound
    # changes. Reading the expression off entry 1 bounds the first measure twice and leaves
    # the second measure unbounded.
    @test entry.first == (:b_ub_var, :b_ub)
    @test entry.second[1] === :EXPR_TWO
    @test entry.second[3] === false
    @test entry.second[4] === 2
    # The span is measure 2's own risk at the two corner portfolios.
    @test collect(entry.second[2]) ≈
          range(expected_risk(rs[2], w_min, X), expected_risk(rs[2], w_max, X); length = 3)

    entry1 = PO._rebuild_risk_frontier(pr, nothing, rs[1], registry, w_min, w_max, 1)
    @test entry1.second[1] === :EXPR_ONE
    @test entry1.second[3] === true
    @test entry1.second[4] === 1
end

@testset "Frontier rebuild: registry positions map to their owning measures" begin
    front = Frontier(; N = 2)
    # Measure 1 carries no bound, so the only entry belongs to measure 2. Indexing the
    # measure vector with the registry position picks measure 1 and builds the span from the
    # wrong measure.
    registry = [(:b_ub_var, :b_ub) => (:EXPR_TWO, front, true, 2)]
    @test PO.risk_frontier_owners(registry, [1]) == [2]

    # Three measures, two of them bounded, and the second bounded one is the third measure.
    registry = [(:a_ub_var, :a_ub) => (:EXPR_ONE, front, true, 1),
                (:c_ub_var, :c_ub) => (:EXPR_THREE, front, true, 3)]
    @test PO.risk_frontier_owners(registry, [1, 2]) == [1, 3]
    @test PO.risk_frontier_owners(registry, [2]) == [3]
end
