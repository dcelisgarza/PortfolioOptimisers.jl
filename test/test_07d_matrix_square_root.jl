#=
The square root of a covariance matrix (#1396, map #1375): the three policies that
`matrix_square_root` names, the helpers that route through them, and the field `mtx_sqrt` of
`FactorPrior` and `CrossSectionalFactorPrior`. Every `Parity_MatrixSquareRoot_*` and
`Parity_CrossSectionalFactorPrior_Mirror_*` file this test reads is an output of the oracle,
stored with the harness of #1376.

THE CASES.

| Case | Input | Path of the ridge |
| --- | --- | --- |
| Full | a positive definite 4 x 4 sample covariance | the plain factor |
| RankOne | `[1 1; 1 1]` | the first ridge |
| RankOneScaled | `1e-4 [1 1; 1 1]` | the first ridge, scaled |
| ZeroScale | the zero 2 x 2 matrix | the first ridge on the scale one |
| Mirror | the factor covariance of the prior below, with the repair off | the first ridge |
| Indefinite | `[0 1; 1 0]` | every ridge fails, and both sides refuse |

The prior case `Mirror` is the configuration grid's `Macro` case with a second observed factor on
the loading `macro_beta`, whose series is the negated `MACRO`. The two factor returns are opposite,
so the factor covariance is singular, with the null direction `e4 + e5`, and the plain Cholesky
factor of it fails. `f_mp` repairs nothing, and neither does the oracle's factor covariance.

THE MEASURE. Four unit cases are bit-equal, and "Mirror" is bit-equal off the pivot of its null
direction. That pivot is `sqrt(S55 - sum(L5k^2) + λ)`, where the sum cancels the variance `v` of
the factor to leave the ridge `λ = 1e-12 s`. A rounding of `eps v` in the sum moves the pivot by
`eps v / 2λ` of itself, 1.6e-4 at most here, so an implementation that orders one sum differently
moves the pivot by that much. Measured 4.2e-5 between the two sides, and 4.3e-6 of ours against a
512-bit factorisation of the same matrix (3.7e-5 of the oracle's).

In the prior, the oracle's own factor covariance differs from ours by 2.2e-15, and its plain
Cholesky factor succeeds on it by rounding, where ours fails. So its pivot of the null direction is
the root of a rounding, and ours is the root of the ridge. Every other output is at parity. The
column of the null direction is `macro_beta` times the pivot on each side, and the two square roots
reproduce each other's covariance to within the ridge `λ β β'`.

| Unit | Verdict |
| --- | --- |
| `RidgeCholeskySquareRoot`, every path | Parity |
| the pivot of a null direction | Parity to its condition, `eps v / 2λ` |
| the prior under `RidgeCholeskySquareRoot()` | Parity: `mu`, `sigma`, the factor moments, the square root off the null direction |
| the floor of the ridge at `ridge = 0` | Better: `eps s`, the oracle's, where `safe_regime_cholesky` read `eps(s) s` and refused a small block |
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

const PO = PortfolioOptimisers

@testset "matrix_square_root: the three policies on a matrix of rank one" begin
    sigma = [1.0 1.0; 1.0 1.0]
    @test_throws PosDefException matrix_square_root(nothing, sigma)
    L = matrix_square_root(RidgeCholeskySquareRoot(), sigma)
    @test L isa LowerTriangular
    # The first ridge is 1e-12 of the mean variance.
    @test maximum(abs, L * L' - sigma - 1e-12I) < 1e-15
    L = matrix_square_root(EigenFallbackSquareRoot(), sigma)
    @test maximum(abs, L * L' - sigma) < 1e-15
    # A positive definite matrix gives its plain lower Cholesky factor under every policy.
    A = [4.0 2.0; 2.0 3.0]
    for alg in (nothing, RidgeCholeskySquareRoot(), EigenFallbackSquareRoot())
        @test matrix_square_root(alg, A) == cholesky(A).L
    end
    # The ridge is relative to the mean variance, so the ridge of `c Σ` is `c` times the ridge.
    L = matrix_square_root(RidgeCholeskySquareRoot(), 1e-4 .* sigma)
    @test maximum(abs, L * L' - 1e-4 .* sigma - 1e-16I) < 1e-19
    # The refusals: an indefinite matrix survives no ridge, and the eigen policy refuses it and
    # a matrix that is not Hermitian.
    @test_throws PosDefException(1) matrix_square_root(RidgeCholeskySquareRoot(),
                                                       [0.0 1.0; 1.0 0.0])
    @test_throws PosDefException(1) matrix_square_root(EigenFallbackSquareRoot(),
                                                       [1.0 2.0; 2.0 1.0])
    @test_throws PosDefException(-1) matrix_square_root(EigenFallbackSquareRoot(),
                                                        [1.0 0.5; 0.4 1.0])
    # More tries reach a larger ridge.
    # Its smallest eigenvalue is about -2.5e-11, so the third ridge, 1e-10, is the first that
    # repairs it.
    B = [1.0 1.0; 1.0 1.0 - 5e-11]
    @test isnothing(PO.ridge_cholesky(RidgeCholeskySquareRoot(; tries = 2), B))
    @test !isnothing(PO.ridge_cholesky(RidgeCholeskySquareRoot(; tries = 3), B))
    @test RidgeCholeskySquareRoot(; ridge = 0) == RidgeCholeskySquareRoot(0, 3)
    @test_throws DomainError RidgeCholeskySquareRoot(; ridge = -1e-12)
    @test_throws DomainError RidgeCholeskySquareRoot(; ridge = Inf)
    @test_throws DomainError RidgeCholeskySquareRoot(; tries = 0)
end

@testset "The factor of a variance and safe_regime_cholesky route through the policies" begin
    A = let X = randn(StableRNG(1396), 30, 4)
        X' * X ./ 30
    end
    S1 = [1.0 1.0; 1.0 1.0]
    # The cone of a variance reads the transpose of the square root under its `mtx_sqrt`.
    model = PO.JuMP.Model()
    G = PO.chol_sigma_selector(model, nothing, Variance(; sigma = S1))
    @test G == transpose(matrix_square_root(EigenFallbackSquareRoot(), S1))
    G = PO.chol_sigma_selector(model, nothing,
                               Variance(; sigma = A, mtx_sqrt = RidgeCholeskySquareRoot()))
    @test G == transpose(matrix_square_root(RidgeCholeskySquareRoot(), A))
    @test_throws PosDefException PO.chol_sigma_selector(model, nothing,
                                                        StandardDeviation(; sigma = S1,
                                                                          mtx_sqrt = nothing))
    @test PO.safe_regime_cholesky(S1, 1e-12).L ==
          matrix_square_root(RidgeCholeskySquareRoot(), S1)
    # A negative scale gives the ridge of a zero one.
    @test PO.safe_regime_cholesky(S1, -1.0).L ==
          matrix_square_root(RidgeCholeskySquareRoot(; ridge = 0), S1)
    # The floor of the ridge is `eps s`, so a small block factorises at every scale. It was
    # `eps(s) s`, about `eps s²`, which refused this block at the scales 1e-8 and 1e-4.
    for c in (1e-8, 1e-4, 1.0, 1e4)
        chol = PO.safe_regime_cholesky(c .* S1, 0.0)
        @test !isnothing(chol)
        @test maximum(abs, chol.L * chol.L' - c .* S1) <= 1e-14 * c
    end
end

@testset "Parity: the ridge of the oracle on bare matrices" begin
    load(c) = parity_load("MatrixSquareRoot", c, "Chol")
    ridge(A) = Matrix(matrix_square_root(RidgeCholeskySquareRoot(), A))
    Full = let X = randn(StableRNG(1396), 30, 4)
        X' * X ./ 30
    end
    S1 = [1.0 1.0; 1.0 1.0]
    for (c, A) in (("Full", Full), ("RankOne", S1), ("RankOneScaled", 1e-4 .* S1),
                   ("ZeroScale", zeros(2, 2)))
        @test ridge(A) == load(c)
    end
    A = parity_load("MatrixSquareRoot", "Mirror", "Input")
    L = ridge(A)
    o = load("Mirror")
    off = [i != CartesianIndex(5, 5) for i in CartesianIndices(L)]
    @test L[off] == o[off]
    S = (A + A') / 2
    s = Statistics.mean(abs, diag(S))
    lam = 1e-12 * s
    # The pivot of the null direction, against a 512-bit factorisation of the same matrix.
    Lb = setprecision(512) do
        return Float64(cholesky(Symmetric(big.(S) + big(lam) * I, :L)).L[5, 5])
    end
    bound = eps() * maximum(diag(S)) / (2 * lam)
    @test abs(L[5, 5] - Lb) <= bound * Lb
    @test abs(o[5, 5] - Lb) <= bound * Lb
    @test_throws PosDefException matrix_square_root(RidgeCholeskySquareRoot(),
                                                    [0.0 1.0; 1.0 0.0])
end

@testset "Parity: the Cross-Sectional Factor Prior on a singular factor covariance" begin
    rd0 = grid_fixture(parity_small_panel())
    rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, ne = [rd0.ne; "MACRO2"],
                       E = hcat(rd0.E, -rd0.E[:, 4]), pnl = rd0.pnl)
    cfg = grid_config("Macro", rd)
    function mirror(mtx_sqrt)
        f = [cfg.factors;
             "macro2" =>
                 ObservedExposure(; xe = grid_pass("macro_beta"; family = "macro2"),
                                  series = "MACRO2", family = "macro2")]
        return CrossSectionalFactorPrior(; cfg..., factors = f,
                                         f_mp = MatrixProcessing(; pdm = nothing),
                                         mtx_sqrt = mtx_sqrt)
    end
    # The default keeps the Cholesky factor that refuses the singular factor covariance.
    @test_throws PosDefException prior(mirror(nothing), rd)
    pr = prior(mirror(RidgeCholeskySquareRoot()), rd)
    load(o) = parity_load("CrossSectionalFactorPrior", "Mirror", o)
    @test pr.fpr.sigma == parity_load("MatrixSquareRoot", "Mirror", "Input")
    # Measured maxrel 8.8e-15, 2.2e-15 and 1.6e-15, and maxscaled 3.0e-13 on `sigma`.
    @test parity_compare(pr.mu, vec(load("Mu")); name = "Mirror mu").ok
    @test parity_compare(pr.sigma, load("Sigma"); scale = :array, name = "Mirror sigma").ok
    @test parity_compare(pr.fpr.sigma, load("FactorCov"); name = "Mirror fcov").ok
    @test parity_compare(pr.fpr.mu, vec(load("FactorMu")); name = "Mirror fmu").ok
    K = size(pr.rr.L, 2)
    i = findall(PO.investable_mask(pr))
    C = Matrix(transpose(pr.chol[1:K, i]))
    O = load("InvSqrtSystematic")
    # The columns off the null direction. Column 4 is `macro_beta` times `L44 + L54`, a
    # cancellation, so it compares against the largest entry. Measured maxscaled 9.3e-13.
    @test parity_compare(C[:, 1:4], O[:, 1:4]; scale = :array, name = "Mirror sqrt").ok
    # The column of the null direction is `macro_beta` times the pivot of the ridge.
    Lf = matrix_square_root(RidgeCholeskySquareRoot(), pr.fpr.sigma)
    @test C[:, 5] == pr.rr.L[i, 5] .* Lf[5, 5]
    # The two square roots reproduce each other's covariance to within the ridge `λ β β'`.
    # Measured maxscaled 1.2e-12.
    @test parity_compare(C * C', O * O'; rtol = 2e-12, scale = :array,
                         name = "Mirror sqrt product").ok
    # Measured maxrel 3.3e-16.
    @test parity_compare(diag(pr.chol[(K + 1):end, i]), vec(load("InvSqrtDiagonal"));
                         name = "Mirror sqrt diagonal").ok
    # The eigen policy reproduces the covariance to rounding.
    pe = prior(mirror(EigenFallbackSquareRoot()), rd)
    @test isapprox(transpose(pe.chol[:, i]) * pe.chol[:, i], pe.sigma[i, i]; rtol = 1e-12)
end

@testset "FactorPrior on a singular factor covariance" begin
    rng = StableRNG(1396)
    F = 0.01 .* randn(rng, 120, 3)
    # The fourth factor is the first negated, so the factor covariance is singular.
    F = hcat(F, -F[:, 1])
    X = F[:, 1:3] * (0.5 .+ rand(rng, 3, 6)) .+ 0.005 .* randn(rng, 120, 6)
    mp = MatrixProcessing(; pdm = nothing)
    function fp(mtx_sqrt)
        return FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  mp = mp)),
                           mp = mp, mtx_sqrt = mtx_sqrt)
    end
    @test FactorPrior().mtx_sqrt === nothing
    @test_throws PosDefException prior(fp(nothing), X, F)
    pr = prior(fp(RidgeCholeskySquareRoot()), X, F)
    @test isapprox(pr.chol' * pr.chol, pr.sigma; rtol = 1e-11)
    pr = prior(fp(EigenFallbackSquareRoot()), X, F)
    @test isapprox(pr.chol' * pr.chol, pr.sigma; rtol = 1e-14)
end
