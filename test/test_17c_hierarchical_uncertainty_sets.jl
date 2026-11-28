#=
The hierarchical heads fit the uncertainty set of an `UncertaintySetVariance` (#2324).

THE ORACLE. For weights that are not negative, the worst case over a box is the variance under the
upper bound of the box. So a head under the estimator gives the weights of the same head under
`Variance(; sigma = ub)`, which reads no set.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra

const PO = PortfolioOptimisers
const HU_X = 0.01 .* randn(StableRNG(1), 300, 8) .* (1:8)' .+
             0.005 .* randn(StableRNG(2), 300)
const HU_RD = ReturnsResult(; nx = string.("A", 1:8), X = HU_X)
const HU_PR = prior(EmpiricalPrior(), HU_RD)
const HU_UB = sigma_ucs(DeltaUncertaintySet(), HU_RD).ub
# Each estimator fits with no draw, or with a seed, so it gives one set at every fit.
const HU_UES = (DeltaUncertaintySet(), DeltaUncertaintySet(; pe = nothing),
                NormalUncertaintySet(; seed = 2324, n_sim = 200,
                                     alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                              diagonal = false)),
                NormalUncertaintySet(; seed = 2324, n_sim = 200))
hu_usv(u; kwargs...) = UncertaintySetVariance(; ucs = u, kwargs...)
function hu_hrp(r, rd = HU_RD; kwargs...)
    return optimise(HierarchicalRiskParity(; r = r,
                                           opt = HierarchicalOptimiser(; kwargs...)), rd)
end
function hu_hec(ri, ro = ri, rd = HU_RD; kwargs...)
    return optimise(HierarchicalEqualRiskContribution(; ri = ri, ro = ro,
                                                      opt = HierarchicalOptimiser(;
                                                                                  kwargs...)),
                    rd)
end

@testset "A hierarchical head allocates on the worst case of the estimator" begin
    vub = Variance(; sigma = HU_UB)
    for ue in HU_UES[1:2]
        u = hu_usv(ue)
        for f in (hu_hrp, hu_hec)
            w = f(u).w
            @test isapprox(w, f(vub).w; rtol = 1e-12)
            @test maximum(abs, w - f(Variance()).w) > 1e-6
            # A vector, with the nominal variance beside the worst case.
            @test isapprox(f([u, Variance()]).w, f([vub, Variance()]).w; rtol = 1e-12)
        end
        # The inner and the outer measure each fit their own set.
        @test isapprox(hu_hec(Variance(), u).w, hu_hec(Variance(), vub).w; rtol = 1e-12)
        @test isapprox(hu_hec(u, Variance()).w, hu_hec(vub, Variance()).w; rtol = 1e-12)
    end
    # Every family of sets: the estimator gives the weights of its own fitted set.
    for ue in HU_UES
        set = sigma_ucs(ue, HU_RD, HU_PR)
        for wrap in (hu_usv, u -> RiskRatio(; r1 = hu_usv(u), r2 = StandardDeviation()),
                     u -> RiskRatio(; r1 = StandardDeviation(), r2 = hu_usv(u)),
                     u -> RiskTrackingRiskMeasure(; r = hu_usv(u),
                                                  tr = WeightsTracking(; w = fill(0.125, 8)),
                                                  alg = DependentVariableTracking()))
            r, rs = wrap(ue), wrap(set)
            @test PO.holds_ucs_estimator(r) && !PO.holds_ucs_estimator(rs)
            for f in (hu_hrp, hu_hec)
                @test isapprox(f(r).w, f(rs).w; rtol = 1e-12)
            end
        end
    end
    # The Result holds the fitted set, and reports the worst case.
    res = hu_hrp(hu_usv(DeltaUncertaintySet()))
    @test isa(res.r.ucs, BoxUncertaintySet)
    @test isapprox(expected_risk(res.r, res.w, res.pr), dot(res.w, HU_UB, res.w);
                   rtol = 1e-12)
    # One object in both slots fits one set, also with an estimator that draws.
    u = hu_usv(NormalUncertaintySet(; n_sim = 200))
    res = hu_hec(u, u)
    @test res.ri.ucs === res.ro.ucs && isa(res.ri.ucs, BoxUncertaintySet)
end

@testset "The set fits on the investable assets, and on the prior result alone" begin
    Xn = copy(HU_X)
    Xn[1:100, 3] .= NaN
    rdn = ReturnsResult(; nx = HU_RD.nx, X = Xn)
    keep = setdiff(1:8, 3)
    rdk = ReturnsResult(; nx = HU_RD.nx[keep], X = HU_X[:, keep])
    for ue in HU_UES
        for f in (hu_hrp, (r, rd) -> hu_hec(r, r, rd))
            res = f(hu_usv(ue), rdn)
            @test iszero(res.w[3])
            @test isapprox(res.w[keep], f(hu_usv(ue), rdk).w; rtol = 1e-12)
        end
    end
    # An estimator that reads the prior result needs no returns data. An estimator with its
    # own prior does.
    for f in (r -> hu_hrp(r, ReturnsResult(); pe = HU_PR),
              r -> hu_hec(r, r, ReturnsResult(); pe = HU_PR))
        @test isapprox(f(hu_usv(HU_UES[2])).w, f(Variance(; sigma = HU_UB)).w; rtol = 1e-12)
        @test_throws IsNothingError f(hu_usv(HU_UES[1]))
    end
end

@testset "A measure with no estimator of a set takes the plain factory" begin
    ue = DeltaUncertaintySet()
    set = sigma_ucs(ue, HU_RD)
    mset = mu_ucs(ue, HU_RD)
    w = fill(0.125, 8)
    ar(u) = ArithmeticReturn(; ucs = u)
    for (wrap, s) in ((hu_usv, set), (u -> [Variance(), hu_usv(u)], set),
                      (u -> MeanReturnRiskRatio(; rk = [Variance(), hu_usv(u)]), set),
                      (u -> NonOptimisationRiskRatio(; r1 = hu_usv(u), r2 = Variance()), set),
                      (u -> ExpectedReturnRiskRatio(; rk = hu_usv(u)), set),
                      (u -> ExpectedReturnRiskRatio(; rt = ar(u)), mset),
                      (u -> ExpectedReturn(; rt = ar(u)), mset),
                      (u -> ExpectedReturn(; rt = [LogarithmicReturn(), ar(u)]), mset))
        @test PO.holds_ucs_estimator(wrap(ue))
        @test !PO.holds_ucs_estimator(wrap(s)) && !PO.holds_ucs_estimator(wrap(nothing))
        r1 = PO.resolve_ucs_measure(wrap(ue), HU_PR, nothing, HU_RD)
        r2 = PO.resolve_ucs_measure(wrap(s), HU_PR, nothing, HU_RD)
        @test !PO.holds_ucs_estimator(r1)
        v = expected_risk(r1, w, HU_PR)
        @test isapprox(v, expected_risk(r2, w, HU_PR); rtol = 1e-12)
        @test !isapprox(v, expected_risk(wrap(nothing), w, HU_PR); rtol = 1e-6)
    end
    for r in (Variance(), StandardDeviation(; sigma = HU_PR.sigma), RiskRatio(),
              ConditionalValueatRisk(), ExpectedReturn(), MeanReturnRiskRatio(),
              RiskTrackingRiskMeasure(; r = Variance(), tr = WeightsTracking(; w = w)),
              [Variance(), ConditionalValueatRisk()], [Variance(), Variance()])
        @test !PO.holds_ucs_estimator(r)
    end
    # The types decide the answer, so a single measure pays nothing for the guard. The type of
    # the `Val` is inferred only when the answer is a constant.
    answer(r) = Val(PO.holds_ucs_estimator(r))
    for r in (Variance(), RiskRatio(), MeanReturnRiskRatio(), hu_usv(set))
        @test @inferred(answer(r)) === Val(false)
    end
    @test @inferred(answer(hu_usv(ue))) === Val(true)
    # The ellipsoid value agrees with the square root that the model reads.
    ell = sigma_ucs(HU_UES[3], HU_RD, HU_PR)
    v = normalize(randn(StableRNG(3), 8), 1)
    G = transpose(matrix_square_root(EigenFallbackSquareRoot(), ell.sigma))
    @test isapprox(PO.ucs_variance(ell, HU_PR.sigma, v),
                   dot(v, something(ell.val, HU_PR.sigma), v) +
                   ell.k * norm(G * vec(v * v')); rtol = 1e-10)
end
