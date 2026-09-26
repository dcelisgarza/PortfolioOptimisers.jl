# The second-moment measures of `src/16_RiskMeasures/02_Variance.jl`, checked against the forms
# their docstrings state: the functors against the closed forms, the models of the four encodings
# against the functors, the worst-case variance of every fitted set against an independent solve
# of the worst case over that set, and the factories, views and no-bounds copies of the measures.
using Clarabel, HiGHS, JuMP
import JuMP.MOI

# A Clarabel model with tolerances tight enough that a solve agrees with a closed form to 1e-8.
function tight_clarabel()
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    for (k, v) in ("tol_gap_abs" => 1e-12, "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                   "max_iter" => 500)
        set_attribute(m, k, v)
    end
    return m
end

@testset "Variance measures" begin
    PO = PortfolioOptimisers
    rng = StableRNG(817)
    N = 4
    X = randn(rng, 250, N) *
        [1.0 0.3 0.0 0.1; 0.0 1.0 0.4 0.0; 0.0 0.0 1.0 0.2; 0.0 0.0 0.0 1.0] .* 0.01
    S = cov(X)
    w = [0.4, 0.3, 0.2, 0.1]
    W = w * transpose(w)
    rd = ReturnsResult(; X = X, nx = string.("A", 1:N))
    pr = prior(EmpiricalPrior(), rd)
    lb = S .- 0.3 .* abs.(S)
    ub = S .+ 0.2 .* abs.(S)
    box = BoxUncertaintySet(; lb = lb, ub = ub)
    Om = let B = randn(rng, N^2, N^2)
        B * transpose(B) / N^2 + I
    end
    ell = EllipsoidalUncertaintySet(; sigma = Om, k = 0.05,
                                    class = SigmaUncertaintySetClass())

    @testset "The functors are the closed forms" begin
        @test Variance(; sigma = S)(w) ≈ dot(w, S * w)
        @test StandardDeviation(; sigma = S)(w) ≈ sqrt(dot(w, S * w))
        # An estimator defines no set, so the functor gives the nominal variance.
        @test UncertaintySetVariance(; sigma = S)(w) ≈ dot(w, S * w)
        @test UncertaintySetVariance(; ucs = nothing, sigma = S)(w) ≈ dot(w, S * w)
        @test UncertaintySetVariance(; ucs = box, sigma = S)(w) ≈ PO.ucs_variance(box, S, w)
    end

    @testset "Each encoding reports the units of its functor" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false, "max_iter" => 500,
                                     "tol_gap_abs" => 1e-10, "tol_gap_rel" => 1e-10,
                                     "tol_feas" => 1e-10),
                     check_sol = (; allow_local = true, allow_almost = true))
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        model_and_value(r) =
            let res = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt), rd)
                (value(res.model[:risk]), expected_risk(factory(r, pr), res.w, X), res.w)
            end
        for alg in (QuadRiskExpr(), SquaredSOCRiskExpr())
            m, v, wo = model_and_value(Variance(; alg = alg))
            @test isapprox(m, v; rtol = 1e-6)
            @test v ≈ dot(wo, pr.sigma * wo)
        end
        m, v, wo = model_and_value(StandardDeviation())
        @test isapprox(m, v; rtol = 1e-6)
        @test v ≈ sqrt(dot(wo, pr.sigma * wo))
        # The three quadratic encodings report the second moment, and the cone encoding its
        # square root, in the model and in the functor alike.
        vals = Dict{Any, Float64}()
        for alg in (QuadRiskExpr(), SquaredSOCRiskExpr(), RSOCRiskExpr(), SOCRiskExpr())
            m, v, _ = model_and_value(LowOrderMoment(; alg = SecondMoment(; alg2 = alg)))
            @test isapprox(m, v; rtol = 1e-6)
            vals[typeof(alg)] = v
        end
        @test isapprox(vals[SOCRiskExpr], sqrt(vals[QuadRiskExpr]); rtol = 1e-6)
        @test isapprox(vals[RSOCRiskExpr], vals[QuadRiskExpr]; rtol = 1e-6)
    end

    @testset "ucs_variance is the worst case over the set" begin
        # Box: a linear programme over the symmetric entries between the bounds. A weight vector
        # with a short position reads the lower bound on its negative entries.
        for v in (w, [0.7, -0.2, 0.4, 0.1])
            m = Model(HiGHS.Optimizer)
            set_silent(m)
            @variable(m, lb[i, j] <= Sg[i = 1:N, j = 1:N] <= ub[i, j])
            @constraint(m, [i = 1:N, j = 1:N; i < j], Sg[i, j] == Sg[j, i])
            @objective(m, Max, dot(v * transpose(v), Sg))
            optimize!(m)
            @test isapprox(PO.ucs_variance(box, S, v), objective_value(m); rtol = 1e-10)
        end
        # Ellipsoid: the value is the worst case without the condition that the matrix is
        # positive semidefinite, and it lies above the worst case with that condition.
        L = cholesky(Om).L
        function ell_primal(psd)
            m = tight_clarabel()
            @variable(m, u[1:(N ^ 2)])
            @constraint(m, [0.05; u] in SecondOrderCone())
            s = vec(S) .+ L * u
            psd && @constraint(m,
                               Symmetric(0.5 .* (reshape(s, N, N) .+ transpose(reshape(s, N, N)))) in
                               PSDCone())
            @objective(m, Max, dot(vec(W), s))
            optimize!(m)
            return objective_value(m)
        end
        @test isapprox(PO.ucs_variance(ell, S, w), ell_primal(false); rtol = 1e-7)
        @test PO.ucs_variance(ell, S, w) > ell_primal(true)
        # The set names its own centre, and `sigma` is the fallback.
        ellc = EllipsoidalUncertaintySet(; sigma = Om, k = 0.05,
                                         class = SigmaUncertaintySetClass(), val = 2 .* S)
        @test PO.ucs_variance(ellc, S, w) ≈ PO.ucs_variance(ell, 2 .* S, w)
        # Norm ball of orders 2, Inf and 1: the radius times the dual norm of the exposure.
        Lb = randn(rng, N^2, 3) .* 1e-3
        cones = Dict(2 => SecondOrderCone(), Inf => MOI.NormInfinityCone(4),
                     1 => MOI.NormOneCone(4))
        for p in (2, Inf, 1)
            nb = NormBallUncertaintySet(; kappa = 0.7, L = Lb, p = p,
                                        class = SigmaUncertaintySetClass())
            m = tight_clarabel()
            @variable(m, u[1:3])
            @constraint(m, [0.7; u] in cones[p])
            @objective(m, Max, dot(vec(W), vec(S) .+ Lb * u))
            optimize!(m)
            @test isapprox(PO.ucs_variance(nb, S, w), objective_value(m); rtol = 1e-7)
        end
        nb0 = NormBallUncertaintySet(; kappa = 0.7, L = zeros(N^2, 0), p = 2,
                                     class = SigmaUncertaintySetClass())
        @test PO.ucs_variance(nb0, S, w) ≈ dot(w, S * w)
        # Compact: the penalty is the squared norm of the part of `C w` outside the span of the
        # basis, for an orthonormal basis and for any other basis of the same span.
        Q = Matrix(qr(randn(rng, N, 2)).Q)[:, 1:2]
        C = [1.0, 0.5, 2.0, 1.5] .* 0.01
        P = I - Q * transpose(Q)
        closed = dot(w, S * w) + 3.0 * dot(C .* w, P * (C .* w))
        cpt = CompactCovarianceUncertaintySet(; kappa = 3.0, C = C, Q = Q)
        @test PO.ucs_variance(cpt, S, w) ≈ closed
        cpt2 = CompactCovarianceUncertaintySet(; kappa = 3.0, C = C,
                                               Q = Q * [2.0 1.0; 0.0 3.0])
        @test PO.ucs_variance(cpt2, S, w) ≈ closed
        cpt0 = CompactCovarianceUncertaintySet(; kappa = 3.0, C = C, Q = zeros(N, 0))
        @test PO.ucs_variance(cpt0, S, w) ≈ dot(w, S * w) + 3.0 * sum(abs2, C .* w)
    end

    @testset "The factories select a pair and keep a stated set" begin
        r = factory(Variance(), pr)
        @test r.sigma === pr.sigma && r.chol === pr.chol
        # A stated matrix is not paired with the factorisation of the prior.
        r = factory(StandardDeviation(; sigma = 2 .* S), pr)
        @test r.sigma == 2 .* S && isnothing(r.chol)
        # The set that the measure holds wins over the argument.
        usv = UncertaintySetVariance(; ucs = box)
        usv0 = UncertaintySetVariance(; ucs = nothing)
        @test factory(usv, pr, nothing, ell).ucs === box
        @test factory(usv, pr, ell).ucs === box
        @test factory(usv, ell, pr).ucs === box
        @test factory(usv0, pr, nothing, ell).ucs === ell
        @test factory(usv0, pr, ell).ucs === ell
        @test factory(usv0, pr).sigma === pr.sigma
        # Without a prior the uncertainty-set-first method keeps `sigma` as it stands.
        r = factory(usv0, ell)
        @test r.ucs === ell && isnothing(r.sigma)
        r = factory(usv0, ell, pr)
        @test r.ucs === ell && r.sigma === pr.sigma
        @test factory(UncertaintySetVariance(; ucs = nothing, sigma = S), ell, pr).sigma ===
              S
    end

    @testset "The views slice sigma on both axes and chol on its columns" begin
        G = Matrix(cholesky(S).U)
        for (T, i) in ((Variance, [1, 3]), (StandardDeviation, [2, 4]))
            rv = PO.port_opt_view(T(; sigma = S, chol = G), i)
            @test rv.sigma == S[i, i] && rv.chol == G[:, i]
        end
        rv = PO.port_opt_view(UncertaintySetVariance(; ucs = box, sigma = S), [1, 4])
        @test rv.sigma == S[[1, 4], [1, 4]] && rv.ucs.ub == ub[[1, 4], [1, 4]]
    end

    @testset "The no-bounds copies" begin
        usv = UncertaintySetVariance(; ucs = box, sigma = S,
                                     settings = RiskMeasureSettings(; ub = 1.0,
                                                                    scale = 2.0))
        # `flag` keeps the set, or drops it for the nominal Variance.
        for f in (nothing, Val(true))
            a = PO.no_bounds_risk_measure(usv, f)
            @test a isa UncertaintySetVariance && a.ucs === box && a.sigma === S
            @test isnothing(a.settings.ub) && a.settings.scale == 2 && a.settings.rke
            c = PO.no_bounds_no_risk_expr_risk_measure(usv, f)
            @test c isa UncertaintySetVariance && c.ucs === box
            @test isnothing(c.settings.ub) && isone(c.settings.scale) && !c.settings.rke
        end
        b = PO.no_bounds_risk_measure(usv, Val(false))
        @test b isa Variance && b.sigma === S
        @test isnothing(b.settings.ub) && b.settings.scale == 2 && b.settings.rke
        d = PO.no_bounds_no_risk_expr_risk_measure(usv, Val(false))
        @test d isa Variance && d.sigma === S && isnothing(d.rc)
        @test isnothing(d.settings.ub) && isone(d.settings.scale) && !d.settings.rke
        # The copy of an uncertainty-set variance matches the copy of every other measure.
        e = PO.no_bounds_no_risk_expr_risk_measure(Variance(; sigma = S,
                                                            settings = usv.settings))
        @test e.settings.scale == d.settings.scale && e.settings.rke == d.settings.rke
    end

    @testset "ucs_risk_measure fits the set on the returns" begin
        r = PO.ucs_risk_measure(UncertaintySetVariance(; sigma = S), rd)
        @test r.ucs isa BoxUncertaintySet && r.sigma === S
        # A set that reads a prior result has nothing to fit before the prior exists.
        est = NormalUncertaintySet(; pe = nothing)
        @test PO.ucs_risk_measure(UncertaintySetVariance(; ucs = est), rd).ucs === est
        rs = PO.ucs_risk_measure([UncertaintySetVariance(; ucs = box), Variance()], rd)
        @test rs[1].ucs === box && rs[2] isa Variance
    end

    @testset "Validation" begin
        for T in (Variance, StandardDeviation, UncertaintySetVariance)
            @test_throws DimensionMismatch T(; sigma = rand(3, 4))
            @test_throws IsEmptyError T(; sigma = zeros(0, 0))
        end
        @test_throws IsEmptyError Variance(; sigma = S, chol = zeros(0, 0))
        @test_throws ArgumentError Variance(; chol = Matrix(cholesky(S).U))
    end
end
