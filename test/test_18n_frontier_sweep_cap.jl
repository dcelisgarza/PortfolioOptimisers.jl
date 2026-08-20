include(joinpath(@__DIR__, "test18_setup.jl"))

# The frontier sweep is a product: every swept return term and every swept risk measure joins
# the same `Iterators.product`, so `k` bounds of `N` points each cost `N^k` full solves.
# `Frontier`'s constructor only caps the `N` of one bound, so the product is capped separately
# at Model Assembly by `assert_frontier_sweep_cap`.

const PO = PortfolioOptimisers

@testset "Frontier sweep cap: counting one bound" begin
    @test PO.frontier_point_count(Frontier(; N = 7)) == 7
    @test PO.frontier_point_count([0.1, 0.2, 0.3]) == 3
    @test PO.frontier_point_count(range(0.1, 0.2; length = 4)) == 4
end

@testset "Frontier sweep cap: two risk frontiers at the ceiling" begin
    # This is the configuration that passes today: `Frontier`'s constructor accepts each `N`
    # on its own, and nothing sees the 10^10-solve product.
    cap = PO.RESOURCE_LIMITS[].max_frontier
    r = [Variance(; settings = RiskMeasureSettings(; ub = Frontier(; N = cap))),
         ConditionalValueatRisk(;
                                settings = RiskMeasureSettings(; ub = Frontier(; N = cap)))]
    mr = MeanRisk(; r = r, obj = MaximumReturn(), opt = JuMPOptimiser(; pe = pr, slv = slv))
    err = try
        optimise(mr)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    # The message names the product, the factors that made it, and the knob.
    @test err.val == big(cap)^2
    @test occursin("$(big(cap)^2)", err.msg)
    @test length(collect(eachmatch(Regex("_ub = $cap"), err.msg))) == 2
    @test occursin("max_frontier", err.msg)
    @test occursin("set_resource_limits!", err.msg)
    @test occursin("with_resource_limits", err.msg)
end

@testset "Frontier sweep cap: two return frontiers at the ceiling" begin
    cap = PO.RESOURCE_LIMITS[].max_frontier
    ret = [ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap))),
           ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap)))]
    mr = MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret))
    err = try
        optimise(mr)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test err.val == big(cap)^2
    @test occursin("ret_lb_1", err.msg)
    @test occursin("ret_lb_2", err.msg)
    @test occursin("max_frontier", err.msg)
end

@testset "Frontier sweep cap: the two sides multiply together" begin
    # One bound on each side, each well inside the per-`Frontier` cap, whose product is not.
    cap = PO.RESOURCE_LIMITS[].max_frontier
    mr = MeanRisk(;
                  r = Variance(;
                               settings = RiskMeasureSettings(; ub = Frontier(; N = cap))),
                  opt = JuMPOptimiser(; pe = pr, slv = slv,
                                      ret = ArithmeticReturn(;
                                                             settings = JuMPReturnsSettings(;
                                                                                            lb = Frontier(;
                                                                                                          N = cap)))))
    err = try
        optimise(mr)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test err.val == big(cap)^2
    @test occursin("ret_lb_1 = $cap", err.msg)
    @test length(collect(eachmatch(Regex("_ub = $cap"), err.msg))) == 1
end

@testset "Frontier sweep cap: NearOptimalCentering is guarded through the same seam" begin
    cap = PO.RESOURCE_LIMITS[].max_frontier
    ret = [ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap))),
           ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap)))]
    noc = NearOptimalCentering(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret))
    @test_throws DomainError optimise(noc)
end

@testset "Frontier sweep cap: a stated vector counts by its length" begin
    # A vector of bounds is swept exactly like a `Frontier`, and `Frontier`'s constructor
    # never sees it at all, so the product is the only guard it meets.
    PO.with_resource_limits(; max_frontier = 3) do
        mr = MeanRisk(;
                      r = Variance(;
                                   settings = RiskMeasureSettings(;
                                                                  ub = [1e-4, 2e-4, 3e-4,
                                                                        4e-4])),
                      obj = MaximumReturn(), opt = JuMPOptimiser(; pe = pr, slv = slv))
        err = try
            optimise(mr)
            nothing
        catch e
            e
        end
        @test isa(err, DomainError)
        @test err.val == 4
        @test occursin("RESOURCE_LIMITS[].max_frontier = 3", err.msg)
    end
end

@testset "Frontier sweep cap: an unswept model never meets the guard" begin
    # No frontier registry, no factors, no check — the ordinary path is untouched even at the
    # tightest ceiling the caps admit.
    PO.with_resource_limits(; max_frontier = 1) do
        res = optimise(MeanRisk(; r = Variance(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv)))
        @test isa(res.retcode, PO.OptimisationSuccess)
        @test isa(PO.frontier_sweep_points(res.model),
                  Tuple{BigInt, Vector{Pair{Symbol, Int}}})
        @test PO.frontier_sweep_points(res.model) == (big(1), Pair{Symbol, Int}[])
    end
end

@testset "Frontier sweep cap: a product inside the ceiling still sweeps" begin
    # Both registries populated, 3 x 3 = 9 sweep points against a ceiling of 12. The point is
    # that the two sides multiply rather than being capped one at a time: 3 and 3 each pass a
    # per-bound check trivially, and 9 is what the guard actually compares.
    PO.with_resource_limits(; max_frontier = 12) do
        mr = MeanRisk(;
                      r = Variance(;
                                   settings = RiskMeasureSettings(; ub = Frontier(; N = 3))),
                      opt = JuMPOptimiser(; pe = pr, slv = slv,
                                          ret = ArithmeticReturn(;
                                                                 settings = JuMPReturnsSettings(;
                                                                                                lb = Frontier(;
                                                                                                              N = 3)))))
        res = optimise(mr)
        @test length(res.w) == 9
        total, factors = PO.frontier_sweep_points(res.model)
        @test total == 9
        # The return term is counted first, then the risk measure; both count 3.
        @test last.(factors) == [3, 3]
        @test first(factors).first === :ret_lb_1
        @test endswith(string(last(factors).first), "_ub")
        # One more point on either side would have been refused.
        @test_throws DomainError optimise(MeanRisk(;
                                                   r = Variance(;
                                                                settings = RiskMeasureSettings(;
                                                                                               ub = Frontier(;
                                                                                                             N = 4))),
                                                   opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                       ret = ArithmeticReturn(;
                                                                                              settings = JuMPReturnsSettings(;
                                                                                                                             lb = Frontier(;
                                                                                                                                           N = 4))))))
    end
end
