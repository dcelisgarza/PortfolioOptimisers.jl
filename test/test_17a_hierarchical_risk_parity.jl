include(joinpath(@__DIR__, "test17_setup.jl"))

@testset "HierarchicalRiskParity" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity1.csv.gz"), DataFrame)
    for (i, r) in pairs(rs)
        res = optimise(HierarchicalRiskParity(; r = r, opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        success = isapprox(res.w, df[!, i]; rtol = 5e-7)
        if !success
            println("Counter: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
    end
end
@testset "HierarchicalRiskParity vector rm" begin
    sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1.2e2),
            LogSumExpScalariser(; gamma = 1e6), MinScalariser()]
    df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity2.csv.gz"), DataFrame)
    for (i, sca) in pairs(sces)
        res = optimise(HierarchicalRiskParity(;
                                              r = [ConditionalValueatRisk(),
                                                   Variance(;
                                                            settings = RiskMeasureSettings(;
                                                                                           scale = 2e2))],
                                              opt = opt, sca = sca))
        @test isa(res.retcode, OptimisationSuccess)
        success = isapprox(res.w, df[!, i])
        if !success
            println("Counter: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
    end
end
@testset "The docstrings of 02_HierarchicalRiskParity.jl against numbers" begin
    using LinearAlgebra, StableRNGs
    PO = PortfolioOptimisers
    rng = StableRNG(880)
    T, N = 300, 11
    X = randn(rng, T, N) * Diagonal(0.005 .+ 0.02 .* rand(rng, N)) .+ 0.003 .* randn(rng, T)
    rdx = ReturnsResult(; nx = ["a$i" for i in 1:N], X = X)
    E = Matrix{Float64}(I, N, N)
    # Recursive bisection written from the `# Mathematical definition`: halves of
    # floor(n / 2) and the rest, inverse-risk weights inside a half, the split factor
    # clamped to the half totals of the bounds, the bounds of the right half last.
    function bisect(order, pair; lb = zeros(N), ub = ones(N))
        w = ones(N)
        parts = [order]
        while !isempty(parts)
            halves = Vector{Vector{Int}}()
            for p in parts
                length(p) > 1 || continue
                m = div(length(p), 2)
                L = p[1:m]
                R = p[(m + 1):end]
                l, r = pair(L, R)
                a = r / (l + r)
                c = w[L[1]]
                a = min(sum(ub[L]) / c, max(sum(lb[L]) / c, a))
                a = 1 - min(sum(ub[R]) / c, max(sum(lb[R]) / c, 1 - a))
                w[L] .*= a
                w[R] .*= 1 - a
                push!(halves, L, R)
            end
            parts = halves
        end
        return w
    end
    function naive(rk, C)
        v = zeros(N)
        v[C] = inv.(rk[C])
        return v / sum(v)
    end
    function single_pair(r, Xm)
        rk = [expected_risk(r, E[:, i], Xm) for i in 1:N]
        return (L, R) -> (expected_risk(r, naive(rk, L), Xm),
                          expected_risk(r, naive(rk, R), Xm))
    end
    # One measure: the allocation, and the weights sum to one before the finaliser.
    res = optimise(HierarchicalRiskParity(), rdx)
    ord = res.clr.res.order
    wi = bisect(ord, single_pair(res.r, X))
    @test isapprox(res.w, wi; atol = 1e-15)
    @test isapprox(sum(wi), 1; atol = 1e-15)
    # Bounds that the recursion meets by itself, and one asset capped at a leaf.
    lb = fill(0.03, N)
    ub = fill(0.14, N)
    ub[3] = 0.05
    resb = optimise(HierarchicalRiskParity(;
                                           opt = HierarchicalOptimiser(;
                                                                       wb = WeightBounds(;
                                                                                         lb = lb,
                                                                                         ub = ub))),
                    rdx)
    wib = bisect(ord, single_pair(resb.r, X); lb = lb, ub = ub)
    @test isapprox(resb.w, wib; atol = 1e-15)
    @test all(lb .- 1e-15 .<= wib .<= ub .+ 1e-15)
    # Feasible bounds hold at every asset before the finaliser, over any leaf order.
    S = res.pr.sigma
    rngb = StableRNG(1)
    worst = 0.0
    for _ in 1:200
        u = 0.05 .+ 0.4 .* rand(rngb, N)
        l = u .* 0.5 .* rand(rngb, N)
        sum(l) <= 1 <= sum(u) || continue
        rk = diag(S)
        w = bisect(PO.Random.randperm(rngb, N),
                   (L, R) -> (dot(naive(rk, L), S, naive(rk, L)),
                              dot(naive(rk, R), S, naive(rk, R))); lb = l, ub = u)
        worst = max(worst, maximum(w .- u), maximum(l .- w))
    end
    @test worst <= 1e-15
    # The bounds of the right half hold when the two halves cannot both hold.
    sf = PO.split_factor_weight_constraints
    wb4 = WeightBounds(; lb = zeros(4), ub = [0.1, 0.1, 0.1, 0.1])
    a = sf(0.5, wb4, fill(0.5, 4), [1, 2], [3, 4])
    @test 0.5 * (1 - a) <= 0.2 + 1e-15
    @test 0.5 * a > 0.2
    # A part of zero weight returns the factor unchanged.
    @test sf(0.4, WeightBounds(; lb = fill(0.01, 4), ub = ones(4)), zeros(4), [1, 2],
             [3, 4]) == 0.4
    # End to end: a zero cap on the first two leaves gives their part zero weight, and
    # its own split meets a zero common weight.
    ubz = ones(N)
    ubz[ord[1:2]] .= 0
    rz = optimise(HierarchicalRiskParity(;
                                         opt = HierarchicalOptimiser(;
                                                                     wb = WeightBounds(;
                                                                                       lb = zeros(N),
                                                                                       ub = ubz))),
                  rdx)
    @test isa(rz.retcode, OptimisationSuccess)
    @test all(isfinite, rz.w)
    @test all(iszero, rz.w[ord[1:2]])
    @test isapprox(sum(rz.w), 1)
    # A vector of measures: each measure scales its own pair, and the scalariser combines
    # the pairs; a maximum or a minimum selects one measure by the total of its pair.
    lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
    combine = [SumScalariser() => ps -> (sum(first, ps), sum(last, ps)),
               MaxScalariser() => ps -> ps[argmax(map(sum, ps))],
               MinScalariser() => ps -> ps[argmin(map(sum, ps))],
               LogSumExpScalariser(; gamma = 120) =>
                   ps -> (lse(120 .* first.(ps)) / 120, lse(120 .* last.(ps)) / 120)]
    rv = [ConditionalValueatRisk(),
          Variance(; settings = RiskMeasureSettings(; scale = 2e2))]
    for (sca, f) in combine
        rr = optimise(HierarchicalRiskParity(; r = rv, sca = sca), rdx)
        pairs_k = [single_pair(r, X) for r in rr.r]
        scales = [r.settings.scale for r in rr.r]
        wv = bisect(rr.clr.res.order,
                    (L, R) -> f([p(L, R) .* s for (p, s) in zip(pairs_k, scales)]))
        @test isapprox(rr.w, wv; atol = 1e-15)
    end
    # The scale of a single measure cancels in the split factor, and a vector of one
    # measure is that measure.
    @test optimise(HierarchicalRiskParity(;
                                          r = Variance(;
                                                       settings = RiskMeasureSettings(;
                                                                                      scale = 37.0))),
                   rdx).w == res.w
    @test optimise(HierarchicalRiskParity(; r = [Variance()]), rdx).w == res.w
    # `branchorder` reaches `_optimise` and changes nothing.
    @test optimise(HierarchicalRiskParity(), rdx; branchorder = :r).w == res.w
    # A float sample of another width keeps its type.
    r32 = optimise(HierarchicalRiskParity(), ReturnsResult(; nx = rdx.nx, X = Float32.(X)))
    @test eltype(r32.w) == Float32
    # An integer sample gives the weights of its Float64 copy.
    Xi = rand(StableRNG(5), -20:20, 60, 6)
    nxi = ["a$i" for i in 1:6]
    wint = optimise(HierarchicalRiskParity(), ReturnsResult(; nx = nxi, X = Xi)).w
    @test wint == optimise(HierarchicalRiskParity(),
                           ReturnsResult(; nx = nxi, X = Float64.(Xi))).w
    # Validation.
    @test_throws IsEmptyError HierarchicalRiskParity(; r = PO.OptimisationRiskMeasure[])
    @test_throws ArgumentError HierarchicalRiskParity(; r = NoRisk())
    @test_throws ArgumentError HierarchicalRiskParity(; r = [Variance(), NoRisk()])
    @test_throws ArgumentError HierarchicalRiskParity(;
                                                      fb = TimeDependent([EqualWeighted()],
                                                                         :nearest))
    # A view reads the returns of a prior result in `opt.pe`, not the matrix it receives.
    hpr = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = res.pr))
    @test size(PO.port_opt_view(hpr, 2:5, randn(rng, 10, N)).opt.pe.X) == (T, 4)
    # The Non-Investable Axis lands on the sets of `opt`, and the other fields stay.
    hs = HierarchicalRiskParity(;
                                opt = HierarchicalOptimiser(;
                                                            sets = UniverseSets(;
                                                                                dict = Dict("nx" =>
                                                                                                rdx.nx[2:end]))))
    hn = PO.non_investable_universe(hs, ["a1"])
    @test hn.opt.sets.dict[hn.opt.sets.nikey] == ["a1"]
    @test hn.r === hs.r
    # Previous weights: none by default, and a turnover measure or fee needs them.
    w0 = fill(inv(N), N)
    @test !PO.needs_previous_weights(HierarchicalRiskParity())
    @test PO.needs_previous_weights(HierarchicalRiskParity(;
                                                           r = TurnoverRiskMeasure(;
                                                                                   w = w0)))
    @test PO.needs_previous_weights(HierarchicalRiskParity(;
                                                           opt = HierarchicalOptimiser(;
                                                                                       fees = Fees(;
                                                                                                   tn = Turnover(;
                                                                                                                 w = w0)))))
end
