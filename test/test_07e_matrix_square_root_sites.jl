#=
The square root of a covariance at every site that took a bare Cholesky factor or the fixed eigen
factor `covariance_factor` (#1410, after #1396). Each estimator that owns such a site carries the
field `mtx_sqrt`, and every one defaults to the eigen square root (#1506). On a positive definite
matrix that is the plain Cholesky factor, so the default changes no value where the plain factor
succeeds.

THE MATRIX. Every case takes a matrix whose last row and column are zero, so its plain Cholesky
factor fails at the last pivot on exact zeros, and no rounding decides the case. The one
exception is the norm ball of a `NormalUncertaintySet` with `pdm = nothing`, whose asymptotic
covariance of the covariance has rank `N(N+1)/2` of `N^2` by construction.

THE CHECK. Under `nothing` the site raises `PosDefException`. Under the eigen and the
ridge algorithms the model solves, and the value of its cone is the risk of the returned weights
under the singular matrix itself.
=#
using Test, PortfolioOptimisers, JuMP, Clarabel, StableRNGs, LinearAlgebra, Statistics

const PO = PortfolioOptimisers

X_ms = 0.01 * randn(StableRNG(42), 150, 6) .+ 0.001 * (1:6)'
N_ms = size(X_ms, 2)
rd_ms = ReturnsResult(; X = X_ms, nx = string.("A", 1:N_ms))
pr_ms = prior(EmpiricalPrior(), rd_ms)
S_ms = pr_ms.sigma
slv_ms = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                settings = "verbose" => false,
                check_sol = (; allow_local = true, allow_almost = true))
# The weight ceiling keeps the minimum risk above zero, because the last asset is flat.
opt_ms = JuMPOptimiser(; pe = pr_ms, slv = slv_ms, wb = WeightBounds(; lb = 0.0, ub = 0.25))
key_ms(name, i = 1) = PO.state_key(Symbol(""), name, i)
value_ms(res, name) = JuMP.value(res.model[key_ms(name)])
algs_ms = (EigenFallbackSquareRoot(), RidgeCholeskySquareRoot())
function zero_last(A)
    B = copy(A)
    B[end, :] .= 0
    B[:, end] .= 0
    return B
end

@testset "The ellipsoid of an UncertaintySetVariance, and its norm ball" begin
    Om = zero_last(Matrix(1.0I, N_ms^2, N_ms^2))
    @test !issuccess(cholesky(Om; check = false))
    e0 = EllipsoidalUncertaintySet(; sigma = Om, k = 5e-3,
                                   class = SigmaUncertaintySetClass())
    # The compact centre took the eigen square root before, so the measure keeps it.
    @test UncertaintySetVariance().mtx_sqrt === EigenFallbackSquareRoot()
    w = fill(1 / N_ms, N_ms)
    W = w * transpose(w)
    @test_throws PosDefException PO.ucs_variance(e0, S_ms, w, nothing)
    for alg in algs_ms
        # The eigen square root is exact, and the ridge moves the value by its ridge alone.
        @test isapprox(PO.ucs_variance(e0, S_ms, w, alg),
                       dot(w, S_ms, w) + 5e-3 * sqrt(dot(vec(W), Om, vec(W))); rtol = 1e-10)
    end
    @test_throws PosDefException optimise(MeanRisk(;
                                                   r = UncertaintySetVariance(; ucs = e0,
                                                                              mtx_sqrt = nothing),
                                                   obj = MinimumRisk(), opt = opt_ms),
                                          rd_ms)
    for alg in algs_ms
        r = UncertaintySetVariance(; ucs = e0, mtx_sqrt = alg)
        res = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt_ms), rd_ms)
        @test isa(res.retcode, OptimisationSuccess)
        # The factor reproduces the singular matrix: `‖G vec(W + E)‖² = vec(W + E)' Ω vec(W + E)`.
        WpE = vec(JuMP.value.(res.model[:WpE]))
        x = JuMP.value.(res.model[key_ms(:x_eucs)])
        @test isapprox(sum(abs2, x), dot(WpE, Om, WpE); rtol = 1e-10)
        @test isapprox(value_ms(res, :t_eucs), norm(x); rtol = 1e-6)
        # The model keeps the dual matrix, so its value is at or below the value level.
        @test value_ms(res, :eucs_variance_risk_) <=
              PO.ucs_variance(e0, S_ms, res.w, alg) * (1 + 1e-6)
        # A variance rebuilt for a sub-problem keeps the algorithm.
        @test PO.no_bounds_risk_measure(r).mtx_sqrt === alg
    end
    # The converter reads the ellipsoid's matrix through the same algorithm, the eigen square
    # root by default.
    @test_throws PosDefException NormBallUncertaintySet(e0, nothing)
    nb = NormBallUncertaintySet(e0)
    @test maximum(abs, nb.L * nb.L' - Om) < 1e-15
end

@testset "The ellipsoid of an ArithmeticReturn" begin
    Smu = zero_last(S_ms ./ 150)
    em = EllipsoidalUncertaintySet(; sigma = Smu, k = 2.0, class = MuUncertaintySetClass())
    @test ArithmeticReturn().mtx_sqrt === EigenFallbackSquareRoot()
    solve(ret) = optimise(MeanRisk(; r = Variance(), obj = MaximumUtility(),
                                   opt = JuMPOptimiser(; pe = pr_ms, slv = slv_ms,
                                                       ret = ret)), rd_ms)
    @test_throws PosDefException solve(ArithmeticReturn(; ucs = em, mtx_sqrt = nothing))
    @test isa(solve(ArithmeticReturn(; ucs = em)).retcode, OptimisationSuccess)
    for alg in algs_ms
        ret = ArithmeticReturn(; ucs = em, mtx_sqrt = alg)
        res = solve(ret)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(value_ms(res, :ret_),
                       dot(pr_ms.mu, res.w) - 2.0 * sqrt(dot(res.w, Smu, res.w));
                       rtol = 1e-6)
        @test PO.no_bounds_returns_estimator(ret).mtx_sqrt === alg
        @test PO.factory(ret, pr_ms).mtx_sqrt === alg
    end
end

@testset "RelaxedRiskBudgeting on a prior without a factor" begin
    Ss = zero_last(S_ms)
    prs = PO.LowOrderPrior(; X = X_ms, mu = pr_ms.mu, sigma = Ss)
    rrb(alg) = RelaxedRiskBudgeting(; opt = JuMPOptimiser(; pe = prs, slv = slv_ms),
                                    mtx_sqrt = alg)
    @test RelaxedRiskBudgeting(; opt = JuMPOptimiser(; pe = prs, slv = slv_ms)).mtx_sqrt ===
          EigenFallbackSquareRoot()
    @test rrb(nothing).mtx_sqrt === nothing
    @test_throws PosDefException optimise(rrb(nothing), rd_ms)
    for alg in algs_ms
        res = optimise(rrb(alg), rd_ms)
        @test isa(res.retcode, OptimisationSuccess)
        # The cone on `psi` is tight, so it states the standard deviation.
        @test isapprox(JuMP.value(res.model[:psi]), sqrt(dot(res.w, Ss, res.w));
                       rtol = 1e-3)
        @test PO.port_opt_view(rrb(alg), 1:3, X_ms).mtx_sqrt === alg
    end
end

@testset "Kurtosis and NegativeSkewness with a stated singular matrix" begin
    Xc = X_ms .- mean(X_ms; dims = 1)
    Xc[:, end] .= 0
    T = size(Xc, 1)
    kt = sum(kron(Xc[t, :], Xc[t, :]) * transpose(kron(Xc[t, :], Xc[t, :])) for t in 1:T) /
         T
    solve(r) = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt_ms), rd_ms)
    @test Kurtosis().mtx_sqrt === EigenFallbackSquareRoot()
    @test_throws PosDefException solve(Kurtosis(; kt = kt, mtx_sqrt = nothing))
    @test isa(solve(Kurtosis(; kt = kt)).retcode, OptimisationSuccess)
    # A positive definite co-kurtosis: the default is the plain Cholesky factor.
    Xp = X_ms .- mean(X_ms; dims = 1)
    ktp = sum(kron(Xp[t, :], Xp[t, :]) * transpose(kron(Xp[t, :], Xp[t, :])) for t in 1:T) /
          T
    Sp = PO.dup_elim_sum_matrices(N_ms)[3]
    @test isposdef(Symmetric(Sp * ktp * transpose(Sp)))
    @test solve(Kurtosis(; kt = ktp)).w == solve(Kurtosis(; kt = ktp, mtx_sqrt = nothing)).w
    for alg in algs_ms
        res = solve(Kurtosis(; kt = kt, mtx_sqrt = alg))
        @test isa(res.retcode, OptimisationSuccess)
        ww = kron(res.w, res.w)
        @test isapprox(value_ms(res, :kurtosis_risk_), sqrt(dot(ww, kt, ww)); rtol = 1e-6)
    end
    # The measure took the eigen square root of `V` before, as `sqrt(V)`, so it keeps it.
    V = zero_last(S_ms)
    sk = zeros(N_ms, N_ms^2)
    @test NegativeSkewness().mtx_sqrt === EigenFallbackSquareRoot()
    @test_throws PosDefException solve(NegativeSkewness(; sk = sk, V = V,
                                                        mtx_sqrt = nothing))
    for alg in algs_ms
        res = solve(NegativeSkewness(; sk = sk, V = V, mtx_sqrt = alg))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(value_ms(res, :nskew_risk_), sqrt(dot(res.w, V, res.w)); rtol = 1e-6)
    end
end

@testset "The norm ball of a NormalUncertaintySet with the repair off" begin
    function ue(alg)
        return NormalUncertaintySet(; pdm = nothing, rng = StableRNG(1), n_sim = 100,
                                    alg = NormBallUncertaintySetAlgorithm(;
                                                                          diagonal = false,
                                                                          mtx_sqrt = alg))
    end
    @test NormBallUncertaintySetAlgorithm().mtx_sqrt === EigenFallbackSquareRoot()
    @test_throws PosDefException PO.ucs(ue(nothing), rd_ms)
    # The default reads the singular matrix, as the explicit eigen policy does.
    @test PO.ucs(NormalUncertaintySet(; pdm = nothing, rng = StableRNG(1), n_sim = 100,
                                      alg = NormBallUncertaintySetAlgorithm(;
                                                                            diagonal = false)),
                 rd_ms)[2].L == PO.ucs(ue(EigenFallbackSquareRoot()), rd_ms)[2].L
    # A normal radius solves with the map, so the eigen square root of a singular matrix
    # refuses it, as the field states. The ridge gives a map of full rank.
    function uk(alg)
        return NormalUncertaintySet(; pdm = nothing, rng = StableRNG(1), n_sim = 100,
                                    alg = NormBallUncertaintySetAlgorithm(;
                                                                          method = NormalKUncertaintyAlgorithm(),
                                                                          diagonal = false,
                                                                          mtx_sqrt = alg))
    end
    @test_throws SingularException PO.ucs(uk(EigenFallbackSquareRoot()), rd_ms)
    @test isfinite(PO.ucs(uk(RidgeCholeskySquareRoot()), rd_ms)[2].kappa)
    # With the repair on, the matrix is positive definite, and the default is the plain factor.
    function ur(alg)
        return NormalUncertaintySet(; rng = StableRNG(1), n_sim = 100,
                                    alg = NormBallUncertaintySetAlgorithm(;
                                                                          diagonal = false,
                                                                          mtx_sqrt = alg))
    end
    @test PO.ucs(ur(EigenFallbackSquareRoot()), rd_ms)[2].L ==
          PO.ucs(ur(nothing), rd_ms)[2].L
    T = size(X_ms, 1)
    Sss = PO.sigma_asymptotic_cov(nothing, PO.mu_asymptotic_cov(nothing, S_ms, T), S_ms, T)
    # The covariance of the covariance has rank N(N+1)/2 of N^2 before any repair.
    @test rank(Sss) == N_ms * (N_ms + 1) ÷ 2
    _, sigma_set = PO.ucs(ue(EigenFallbackSquareRoot()), rd_ms)
    @test maximum(abs, sigma_set.L * sigma_set.L' - Sss) <= 1e-13 * maximum(abs, Sss)
end

@testset "Variance, StandardDeviation and DistributionValueatRisk on a singular matrix" begin
    Ss = zero_last(S_ms)
    @test !issuccess(cholesky(Ss; check = false))
    for R in (Variance, StandardDeviation)
        @test R().mtx_sqrt === EigenFallbackSquareRoot()
        @test_throws PosDefException optimise(MeanRisk(;
                                                       r = R(; sigma = Ss,
                                                             mtx_sqrt = nothing),
                                                       obj = MinimumRisk(), opt = opt_ms),
                                              rd_ms)
    end
    for alg in algs_ms
        res = optimise(MeanRisk(; r = StandardDeviation(; sigma = Ss, mtx_sqrt = alg),
                                obj = MinimumRisk(), opt = opt_ms), rd_ms)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(value_ms(res, :sd_risk_), sqrt(dot(res.w, Ss, res.w)); rtol = 1e-6)
        r = factory(Variance(; sigma = Ss, mtx_sqrt = alg), pr_ms)
        @test r.mtx_sqrt === alg
        @test PO.port_opt_view(r, 1:3).mtx_sqrt === alg
    end
    # The prior's factor is shared, so the first measure that falls back to it sets the algorithm.
    prs = PO.LowOrderPrior(; X = X_ms, mu = pr_ms.mu, sigma = Ss)
    opts = JuMPOptimiser(; pe = prs, slv = slv_ms, wb = WeightBounds(; lb = 0.0, ub = 0.25))
    res = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(), opt = opts), rd_ms)
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(transpose(res.model[:G]) * res.model[:G], Ss; atol = 1e-15)
    @test_throws PosDefException optimise(MeanRisk(; r = Variance(; mtx_sqrt = nothing),
                                                   obj = MinimumRisk(), opt = opts), rd_ms)
    @test DistributionValueatRisk().mtx_sqrt === EigenFallbackSquareRoot()
    for alg in (nothing, algs_ms...)
        dvar = ValueatRisk(; alg = DistributionValueatRisk(; sigma = Ss, mtx_sqrt = alg))
        solve() = optimise(MeanRisk(; r = dvar, obj = MinimumRisk(), opt = opt_ms), rd_ms)
        if isnothing(alg)
            @test_throws PosDefException solve()
        else
            @test isa(solve().retcode, OptimisationSuccess)
        end
    end
end
