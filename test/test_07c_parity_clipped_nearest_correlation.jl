#=
Parity of map #1375 for the clipped nearest-correlation repair (#1412), the algorithm
`ClippedNearestCorrelation` that `Posdef` takes. Every `Parity_ClippedNearestCorrelation_*` file
this test reads is stored with the harness of #1376: `Input` is the matrix, `Sigma` the oracle's
repair of it. The conventions of the harness are in the comment that #1376 links; this file does
not repeat them.

THE CASES. The inputs are random matrices chosen so that each exit path of the repair runs.

| Case | Input | Exit path |
| --- | --- | --- |
| Indefinite | an indefinite 6 x 6 correlation matrix with unequal variances | the clip at `tau` |
| IndefiniteHigham | the same, with the alternating projections | the clip at `tau` |
| Borderline | 5 x 5, passes the Cholesky factorisation, smallest correlation eigenvalue 1.07e-15 | the clip at `tau` |
| Retry | 10 x 10 | the retry at `10 tau` |
| RetryHigham | 10 x 10, with the alternating projections | the retry at `10 tau` |
| Tolerance | 30 x 30 of low rank, variances over six orders | the last acceptance, `-n eps max abs(λ)` |
| Large | 80 x 80, near equicorrelation | the clip at `tau` |
| Frame | "Indefinite" inside a `NaN` frame of two assets | the block rule of `matrix_processing_block!` |
| HighamStop | 30 x 30, with the alternating projections | the oracle refuses it; see below |
| ZeroVariance | "Indefinite" with a constant asset | the oracle refuses it; see below |

THE MEASURE. Every case agreed on the `NaN` pattern. A covariance compares with `scale = :array`
at `rtol = 1e-12`, the harness rule for a covariance. The largest difference over the largest entry
is 8.8e-14 (Retry), and 6.1e-13 for HighamStop, whose 100 iterations carry the round-off of two
eigen solvers. Cell by cell, the differences reach 4.1e-12 on the Retry case: the oracle's eigen solver calls
LAPACK `syevd`, and Julia's `eigen` calls `syevr`. The library keeps the generic `eigen`, which the
Julia versions of its compat bound share.

| Unit | Verdict |
| --- | --- |
| `ClippedNearestCorrelation`, every exit path | Parity |
| a matrix the repair returned, repaired again | Parity: the clip and the retry are accepted unchanged, and the last acceptance moves by round-off on both sides |
| the alternating projections that have not converged after `iter` iterations | Better |
| a constant asset | Better, #1429 |
| a non-finite entry, a negative variance, an asymmetric input | the same refusals |
| the symmetry test | Deliberate difference: it reads the correlation matrix, so it does not move with the units |
| the last result that fails every acceptance | Deliberate difference: a warning, as every `posdef!` gives, where the oracle refuses |

Better: the alternating projections. The oracle stops after 100 iterations and refuses the input
when the last iterate is not positive semidefinite to round-off. The iterate has a unit diagonal
and a spectrum that is nearly positive semidefinite, so its clip is a valid correlation matrix.
The sequence of HighamStop converges after more iterations, and the oracle's own converged answer
is stored as `Converged`. Ours clips the 100th iterate: it is within 1.4e-11 of that answer in the
Frobenius norm of the correlation matrix, and both lie 0.5402 from the input, where the clip alone
lies 1.266 from it.

Better: a constant asset. The oracle refuses a variance that is not positive. `posdef!` gives the
constant asset a zero row and column and repairs the block of positive variances (#1429), which is
"Indefinite", so the block equals the oracle's repair of "Indefinite".
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

p1412_in(case) = parity_load("ClippedNearestCorrelation", case, "Input")
p1412_out(case) = parity_load("ClippedNearestCorrelation", case, "Sigma")
function p1412_pdm(higham::Bool = false)
    return Posdef(; alg = ClippedNearestCorrelation(; higham = higham))
end
p1412_cor(X) = X ./ (sqrt.(diag(X)) .* transpose(sqrt.(diag(X))))

const P1412_CASES = [("Indefinite", false), ("IndefiniteHigham", true),
                     ("Borderline", false), ("Retry", false), ("RetryHigham", true),
                     ("Tolerance", false), ("Large", false), ("HighamStop", true)]

@testset "Parity: the clipped nearest-correlation repair (#1412)" begin
    @testset "every exit path equals the oracle's repair: $case" for (case, higham) in
                                                                     P1412_CASES

        X = p1412_in(case)
        Y = posdef(p1412_pdm(higham), X)
        @test parity_compare(Y, p1412_out(case); scale = :array, name = case).ok
        @test isposdef(Y)
        @test diag(Y) == diag(X)
        # A matrix the clip or the retry returned passes the acceptance test, so a second repair
        # leaves it. A matrix the last acceptance returned fails the strict test again, and the
        # oracle repairs it again too, by round-off (2.2e-16 on its own Tolerance output).
        if case in ("Tolerance", "HighamStop")
            @test parity_compare(posdef(p1412_pdm(higham), Y), Y; scale = :array,
                                 name = "$(case), again").ok
        else
            @test posdef(p1412_pdm(higham), Y) == Y
        end
    end
    @testset "the borderline input passes isposdef, and only the clip repairs it" begin
        X = p1412_in("Borderline")
        @test isposdef(X)
        @test posdef(Posdef(), X) == X
        @test eigmin(Symmetric(p1412_cor(X))) < 5e-14
        @test eigmin(Symmetric(p1412_cor(posdef(p1412_pdm(), X)))) >= 5e-14
    end
    @testset "the NaN frame keeps its pattern, and the block equals the oracle's" begin
        X = p1412_in("Frame")
        mp = MatrixProcessing(; pdm = p1412_pdm())
        PortfolioOptimisers.matrix_processing_block!(mp, X, zeros(1, size(X, 2)))
        @test parity_compare(X, p1412_out("Frame"); scale = :array, name = "Frame").ok
    end
    @testset "Better: the alternating projections that stop go on to the clip" begin
        X = p1412_in("HighamStop")
        Y = posdef(p1412_pdm(true), X)
        Z = parity_load("ClippedNearestCorrelation", "HighamStop", "Converged")
        C = p1412_cor(X)
        @test norm(p1412_cor(Y) - p1412_cor(Z)) < 1e-10
        @test norm(p1412_cor(Y) - C) < norm(p1412_cor(posdef(p1412_pdm(), X)) - C) / 2
    end
    @testset "Better: a constant asset keeps a zero row, and the block is repaired" begin
        X = p1412_in("ZeroVariance")
        Y = posdef(p1412_pdm(), X)
        @test parity_compare(Y, p1412_out("ZeroVariance"); scale = :array,
                             name = "ZeroVariance").ok
        @test all(iszero, Y[4, :]) && all(iszero, Y[:, 4])
    end
end

@testset "ClippedNearestCorrelation: refusals and the Newton default (#1412)" begin
    pdm = p1412_pdm()
    X = p1412_in("Indefinite")
    @test_throws IsNonFiniteError posdef(pdm, [1.0 NaN; NaN 1.0])
    @test_throws DomainError posdef(pdm, [-1.0 0.5; 0.5 1.0])
    B = copy(X)
    B[1, 2] += 0.01
    @test_throws ArgumentError posdef(pdm, B)
    # The symmetry test reads the correlation matrix, so a change of units does not move it.
    @test posdef(pdm, 1e-6 .* X) ≈ 1e-6 .* posdef(pdm, X) rtol = 1e-12
    @test_throws DimensionMismatch posdef(pdm, rand(2, 3))
    @test_throws ArgumentError Posdef(; alg = ClippedNearestCorrelation(),
                                      kwargs = (; tau = 1))
    @test_throws DomainError ClippedNearestCorrelation(; tau = 0)
    @test_throws DomainError ClippedNearestCorrelation(; tau = 1)
    @test_throws DomainError ClippedNearestCorrelation(; iter = 0)
    # Newton accepts every matrix that passes isposdef, as it always did.
    @test PortfolioOptimisers.posdef_accepts(Posdef().alg, X) == isposdef(X)
    @test PortfolioOptimisers.posdef_accepts(Posdef().alg, p1412_in("Borderline"))
    @test !PortfolioOptimisers.posdef_accepts(pdm.alg, p1412_in("Borderline"))
    # A zero variance is left to the zero-row rule of `posdef!`.
    @test !PortfolioOptimisers.posdef_accepts(pdm.alg, p1412_in("ZeroVariance"))
    # The last result that fails every acceptance warns and is returned, as every `posdef!`
    # does. An eigenvalue floor this far below the round-off of an 80 x 80 rebuild cannot hold.
    tiny = Posdef(; alg = ClippedNearestCorrelation(; tau = 1e-300))
    Xl = p1412_in("Large")
    @test_logs (:warn, "Matrix could not be made positive definite.") match_mode = :any posdef(tiny,
                                                                                               Xl)
end
