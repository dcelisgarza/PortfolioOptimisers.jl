using Clarabel, JuMP
const PO = PortfolioOptimisers

# The sweep of `src/17_Optimisation/05_JuMP/06_NearOptimalCentering_a.jl`. Each solve is
# checked against an independent JuMP model of problem 11.31 of cajas2025: the analytic
# centre of Section 8.5.3 of boydvandenberghe2004 over the near optimal region, with the
# margins of Equation 11.32 and one barrier term per finite weight bound.

const NOCS_X = let rng = StableRNG(688)
    T, N = 300, 5
    F = randn(rng, T, 2)
    0.0008 .+ F * (0.01 .* randn(rng, 2, N)) .+ 0.01 .* randn(rng, T, N) .+
    (0.0004 .* (1:N))'
end
const NOCS_PR = prior(EmpiricalPrior(), NOCS_X)
const NOCS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                        check_sol = (; allow_local = true, allow_almost = true),
                        settings = Dict("verbose" => false, "tol_gap_abs" => 1e-11,
                                        "tol_gap_rel" => 1e-11, "tol_feas" => 1e-11))
nocs_opt(; kwargs...) = JuMPOptimiser(; pe = NOCS_PR, slv = NOCS_SLV, kwargs...)
nocs_ok(res) = isa(res.retcode, OptimisationSuccess)
function nocs_anchors(opt)
    w_min = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w
    w_max = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt)).w
    return w_min, w_max
end
# The analytic centre of the near optimal region of a standard deviation, built by hand. The
# margins are those of the anchors, the default number of bins is T / N, and an asset with a
# fixed weight is an equality with no barrier term.
function nocs_oracle(w_min, w_opt, w_max; lb = zeros(5), ub = ones(5),
                     m = size(NOCS_X, 1) / size(NOCS_X, 2))
    mu, S = NOCS_PR.mu, NOCS_PR.sigma
    sd(w) = sqrt(dot(w, S, w))
    e1 = dot(mu, w_opt) - (dot(mu, w_max) - dot(mu, w_min)) / m
    e2 = sd(w_opt) + (sd(w_max) - sd(w_min)) / m
    N = length(mu)
    G = cholesky(Symmetric(S)).U
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    set_attribute(model, "tol_gap_abs", 1e-11)
    set_attribute(model, "tol_gap_rel", 1e-11)
    set_attribute(model, "tol_feas", 1e-11)
    @variable(model, w[1:N])
    @variable(model, g)
    @variable(model, a)
    @variable(model, b)
    @constraint(model, sum(w) == 1)
    @constraint(model, [g; G * w] in SecondOrderCone())
    @constraint(model, [a, 1, dot(mu, w) - e1] in MOI.ExponentialCone())
    @constraint(model, [b, 1, e2 - g] in MOI.ExponentialCone())
    obj = AffExpr(0.0) + a + b
    for i in 1:N
        if lb[i] == ub[i]
            @constraint(model, w[i] == lb[i])
            continue
        end
        if isfinite(lb[i])
            t = @variable(model)
            @constraint(model, [t, 1, w[i] - lb[i]] in MOI.ExponentialCone())
            add_to_expression!(obj, t)
        end
        if isfinite(ub[i])
            t = @variable(model)
            @constraint(model, [t, 1, ub[i] - w[i]] in MOI.ExponentialCone())
            add_to_expression!(obj, t)
        end
    end
    @objective(model, Max, obj)
    optimize!(model)
    return value.(w)
end

@testset "NearOptimalCentering sweep: the centre agrees with problem 11.31" begin
    opt = nocs_opt()
    w_min, w_max = nocs_anchors(opt)
    res = optimise(NearOptimalCentering(; opt = opt))
    @test nocs_ok(res)
    # The default objective is MinimumRisk, so w_opt is w_min, and m = T / N = 60.
    @test isapprox(res.w, nocs_oracle(w_min, w_min, w_max); atol = 1e-6)
    # The user's objective sets w_opt, and `bins` sets m.
    w_tan = optimise(MeanRisk(; obj = MaximumRatio(), opt = opt)).w
    res_r = optimise(NearOptimalCentering(; opt = opt, obj = MaximumRatio()))
    @test isapprox(res_r.w, nocs_oracle(w_min, w_tan, w_max); atol = 5e-5)
    res_b = optimise(NearOptimalCentering(; opt = opt, obj = MaximumRatio(), bins = 7.5))
    @test isapprox(res_b.w, nocs_oracle(w_min, w_tan, w_max; m = 7.5); atol = 5e-5)
    @test !isapprox(res_b.w, res_r.w; atol = 0.1)
    # Supplied anchors skip the sub-problems and give the same centre.
    res_s = optimise(NearOptimalCentering(; opt = opt, w_min = w_min, w_opt = w_min,
                                          w_max = w_max))
    @test isapprox(res_s.w, res.w; atol = 1e-6)
    # On a problem with no other constraint, the constrained variant builds the same model.
    res_c = optimise(NearOptimalCentering(; opt = opt,
                                          alg = ConstrainedNearOptimalCentering()))
    @test isapprox(res_c.w, res.w; atol = 1e-8)
    # A positive scale of an inequality does not move its analytic centre. A lone measure
    # drops its scale, and a vector applies it to the risk and to the targets alike.
    @test isapprox(optimise(NearOptimalCentering(; opt = nocs_opt(; sc = 100))).w, res.w;
                   atol = 1e-6)
    rs = StandardDeviation(; settings = RiskMeasureSettings(; scale = 2.0))
    @test isapprox(optimise(NearOptimalCentering(; opt = opt, r = rs)).w, res.w;
                   atol = 1e-6)
    @test isapprox(optimise(NearOptimalCentering(; opt = opt, r = [rs])).w, res.w;
                   atol = 1e-6)
end

@testset "NearOptimalCentering sweep: the barrier reads both weight bounds" begin
    @test PO.barrier_weight_bounds([0.0, -0.2, -Inf, 0.0], [1.0, Inf, 1.0, 0.0]) ==
          ([1, 2], [1, 3])
    @test PO.barrier_weight_bounds(range(0.0, 0.0; length = 3), fill(Inf, 3)) ==
          ([1, 2, 3], Int[])
    # A short book. The lower margin is w - lb, so the centre is not held to w > 0.
    opt = nocs_opt(; wb = WeightBounds(; lb = -0.2, ub = 1.0), sbgt = 1000.0)
    w_min, w_max = nocs_anchors(opt)
    res = optimise(NearOptimalCentering(; opt = opt))
    @test nocs_ok(res)
    @test isapprox(res.w, nocs_oracle(w_min, w_min, w_max; lb = fill(-0.2, 5)); atol = 1e-5)
    # Near the maximum-return anchor the region holds no long-only portfolio. A barrier
    # term -ln(w) made this model infeasible.
    res = optimise(NearOptimalCentering(; opt = opt, obj = MaximumReturn(), bins = 50))
    @test nocs_ok(res)
    @test isapprox(res.w, nocs_oracle(w_min, w_max, w_max; lb = fill(-0.2, 5), m = 50);
                   atol = 1e-5)
    @test count(<(0), res.w) == 3
    # An infinite bound states no inequality, so it has no barrier term. A term with an
    # infinite margin made both models fail.
    opt = nocs_opt(; wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0)
    w_min, w_max = nocs_anchors(opt)
    res = optimise(NearOptimalCentering(; opt = opt))
    @test nocs_ok(res)
    # The weights reach 6.5 under the short budget of 1000. Two CI host kinds miss by 8.7e-5
    # and 9.2e-5 (8 of 18 runs from 40565553296 to 40988847855, R14 of #1817).
    @test isapprox(res.w,
                   nocs_oracle(w_min, w_min, w_max; lb = fill(-Inf, 5), ub = fill(Inf, 5));
                   atol = 1e-4)
    opt = nocs_opt(; wb = WeightBounds(; lb = 0.0, ub = Inf))
    w_min, w_max = nocs_anchors(opt)
    res = optimise(NearOptimalCentering(; opt = opt))
    @test nocs_ok(res)
    @test isapprox(res.w, nocs_oracle(w_min, w_min, w_max; ub = fill(Inf, 5)); atol = 5e-5)
    # A fixed weight is an equality, so it has no barrier term either.
    ub = [1.0, 1.0, 1.0, 1.0, 0.0]
    opt = nocs_opt(; wb = WeightBounds(; lb = 0.0, ub = ub))
    w_min, w_max = nocs_anchors(opt)
    res = optimise(NearOptimalCentering(; opt = opt))
    @test nocs_ok(res)
    @test isapprox(res.w, nocs_oracle(w_min, w_min, w_max; ub = ub); atol = 5e-5)
end

@testset "NearOptimalCentering sweep: the return frontier" begin
    # A stated vector of levels passes unchanged, and a Frontier becomes N even levels
    # between the ends of its own span.
    rf = [:a => (:e1, Frontier(; N = 3), 1), :b => (:e2, [0.1, 0.2], 2)]
    out = PO.compute_ret_lbs(rf, [1 => (0.0, 1.0)])
    @test out[1].second[2] == range(0.0, 1.0; length = 3)
    @test out[2] === rf[2]
    @test PO.compute_ret_lbs(rf, nothing) === rf
    @test_throws ArgumentError PO.compute_ret_lbs(rf, [2 => (0.0, 1.0)])
    lbf = JuMPReturnsSettings(; lb = Frontier(; N = 3))
    @test PO.frontier_return_terms([ArithmeticReturn(; settings = lbf), ArithmeticReturn(),
                                    ArithmeticReturn(; settings = lbf)]) == [1, 3]
    @test PO.frontier_return_terms(ArithmeticReturn()) == Int[]
    # The default variant holds no return floor, so a Frontier only gives one anchor of the
    # objective for each point, and each point is the centre around its own anchor.
    opt = nocs_opt(; ret = ArithmeticReturn(; settings = lbf))
    res = optimise(NearOptimalCentering(; opt = opt))
    @test nocs_ok(res)
    @test length(res.w) == 3
    w_min, w_max = nocs_anchors(nocs_opt())
    w_opts = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w
    for j in 1:2
        st = optimise(NearOptimalCentering(; opt = opt, w_min = w_min, w_max = w_max,
                                           w_opt = w_opts[j]))
        @test isapprox(st.w, res.w[j]; atol = 1e-5)
    end
    # A sweep over a Frontier and a stated vector has 3 x 2 points, and each point is the
    # centre of a standalone solve at its own levels. The top level of the Frontier is the
    # maximum return, where the region is one corner, so it is left out.
    ret = [ArithmeticReturn(; settings = lbf),
           ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = [0.0, 0.0005]))]
    res = optimise(NearOptimalCentering(; opt = nocs_opt(; ret = ret),
                                        alg = ConstrainedNearOptimalCentering()))
    @test nocs_ok(res)
    @test length(res.w) == 6
    w_min, _ = nocs_anchors(nocs_opt())
    levels = range(dot(NOCS_PR.mu, w_min), maximum(NOCS_PR.mu); length = 3)
    for j in 1:2
        rj = [ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = levels[j])),
              ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = 0.0))]
        st = optimise(NearOptimalCentering(; opt = nocs_opt(; ret = rj),
                                           alg = ConstrainedNearOptimalCentering()))
        @test isapprox(st.w, res.w[j]; atol = 1e-5)
    end
end

@testset "NearOptimalCentering sweep: the estimator and its result" begin
    opt = nocs_opt()
    for k in (:w_min, :w_min_ini, :w_opt, :w_opt_ini, :w_max, :w_max_ini)
        @test_throws IsEmptyError NearOptimalCentering(; opt = opt, (k => Float64[],)...)
    end
    @test_throws IsEmptyError NearOptimalCentering(; opt = opt, r = RiskMeasure[])
    for bins in (0, -1.0, Inf, NaN)
        @test_throws DomainError NearOptimalCentering(; opt = opt, bins = bins)
    end
    pwf = PreviousWeightsFunction(; f = c -> c.n)
    @test !PO.needs_previous_weights(NearOptimalCentering(; opt = opt))
    @test PO.needs_previous_weights(NearOptimalCentering(; opt = opt,
                                                         w_min = TimeDependent(pwf)))
    @test PO.needs_previous_weights(NearOptimalCentering(; opt = opt,
                                                         fb = PreviousWeights()))
    # The view reads the matrix of the prior, not the matrix it is given.
    wv = [0.1, 0.2, 0.3, 0.4, 0.5]
    noc = NearOptimalCentering(; opt = opt, w_min = wv, w_max_ini = wv, bins = 4)
    v = PO.port_opt_view(noc, [1, 3], zeros(2, 2))
    @test v.opt.pe.X == NOCS_X[:, [1, 3]]
    @test v.w_min == [0.1, 0.3]
    @test v.w_max_ini == [0.1, 0.3]
    @test isnothing(v.w_opt)
    @test v.bins == 4
    # A new return code rebuilds the shared core and keeps the codes of the sub-problems.
    res = optimise(NearOptimalCentering(; opt = opt))
    rr = PO.set_retcode(res, OptimisationFailure(; res = "x"))
    @test isa(rr.retcode, OptimisationFailure)
    @test rr.w_min_retcode === res.w_min_retcode
    @test rr.noc_retcode === res.noc_retcode
    @test rr.w == res.w
end
