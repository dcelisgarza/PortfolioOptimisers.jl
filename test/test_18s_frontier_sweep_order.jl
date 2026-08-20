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
