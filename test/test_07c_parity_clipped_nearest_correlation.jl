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
| OverlayLarge | the oracle's idiosyncratic block of `test_12w` before its repair, `th = 0.1` on the large panel, 39 finite assets | the clip, on the block rule |
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
| the last result that fails every acceptance | Better: it is positive semidefinite to round-off, so it returns with no message where the oracle refuses it. A result that is not raises a `PosdefRepairError` (#1506) |

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
        # Measured maxscaled 7.7e-16 (maxrel 4.8e-15). The tolerance keeps a margin for the
        # eigen solver, whose round-off moves with the host and reaches 8.8e-14 on Retry.
        @test parity_compare(X, p1412_out("Frame"); rtol = 1e-13, scale = :array,
                             name = "Frame").ok
    end
    @testset "the idiosyncratic overlay of the prior, where the repair binds (#1383)" begin
        # The input is the oracle's own block before its repair, `th = 0.1` on the large panel
        # (`test_12w`). Its smallest correlation eigenvalue is -0.031, so the repair binds, and
        # the oracle leaves by the clip. The prior repairs this block with its field `pdm` over
        # the assets with a finite variance, which is the block rule below, and the clip is the
        # default of `pdm` (#1604). Newton differs from the oracle's repair by 7.8e-4 of the
        # largest entry.
        E = parity_load("CrossSectionalFactorPrior", "PitLargeOverlayRaw", "IdioCov")
        mp = MatrixProcessing(; pdm = p1412_pdm())
        PortfolioOptimisers.matrix_processing_block!(mp, E, zeros(1, size(E, 2)))
        # Measured maxscaled 1.6e-15. A covariance compares against its largest entry, the
        # harness rule that THE MEASURE above states.
        @test parity_compare(E, p1412_out("OverlayLarge"); scale = :array,
                             name = "OverlayLarge").ok
    end
    @testset "Better: the alternating projections that stop go on to the clip" begin
        X = p1412_in("HighamStop")
        Y = posdef(p1412_pdm(true), X)
        Z = parity_load("ClippedNearestCorrelation", "HighamStop", "Converged")
        C = p1412_cor(X)
        # Measured maxrel 8.2e-13 cell by cell. The two answers come from two different
        # iterates, so the tolerance is five times the measure.
        @test parity_compare(p1412_cor(Y), p1412_cor(Z); rtol = 5e-12,
                             name = "HighamStop clip vs converged").ok
        @test norm(p1412_cor(Y) - C) < norm(p1412_cor(posdef(p1412_pdm(), X)) - C) / 2
    end
    @testset "Better: a constant asset keeps a zero row, and the block is repaired" begin
        X = p1412_in("ZeroVariance")
        Y = posdef(p1412_pdm(), X)
        # Measured maxscaled 7.7e-16 (maxrel 4.8e-15), the value of the Frame case: both
        # repair the block of "Indefinite". The margin is the one of the Frame case.
        @test parity_compare(Y, p1412_out("ZeroVariance"); rtol = 1e-13, scale = :array,
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
    # ADR 0186: Newton stays the default repair, also in `f_mp` and `mp` of the cross-sectional
    # prior. Only the prior's idiosyncratic block takes the clip by default, in `pdm` (#1603).
    newton = PortfolioOptimisers.NearestCorrelationMatrix.Newton
    csp = CrossSectionalFactorPrior(; factors = ["market" => ConstantExposure()])
    @test Posdef().alg === newton
    @test csp.f_mp.pdm.alg === newton && csp.mp.pdm.alg === newton
    @test csp.pdm.alg isa ClippedNearestCorrelation
    # A zero variance is left to the zero-row rule of `posdef!`.
    @test !PortfolioOptimisers.posdef_accepts(pdm.alg, p1412_in("ZeroVariance"))
    # An eigenvalue floor this far below the round-off of an 80 x 80 rebuild cannot hold, so
    # the last result fails every acceptance. The clip makes it positive semidefinite to
    # round-off: measured smallest eigenvalue -1.7e-15 against the tolerance -3.9e-13, though
    # it has no Cholesky factor. So it returns with no message, where the oracle refuses it.
    tiny = Posdef(; alg = ClippedNearestCorrelation(; tau = 1e-300))
    Xl = p1412_in("Large")
    Yl = @test_logs min_level = Logging.Warn posdef(tiny, Xl)
    @test !issuccess(cholesky(Symmetric(Yl, :L); check = false))
    l = eigvals(Symmetric(Yl, :L))
    @test first(l) >= -length(l) * eps() * maximum(abs, l)
    # A PSD result with a zero-variance row returns with no message.
    @test_logs min_level = Logging.Warn posdef(pdm, p1412_in("ZeroVariance"))
end

@testset "ClippedNearestCorrelation: Equation 3.3 of Rousseeuw and Molenberghs (1993) (#1536)" begin
    # Section 3 of the original replaces the negative eigenvalues by a small positive number,
    # Delta = 0.05 in its example, and scales the diagonal back to one. Equation 3.3 prints the
    # result for the matrix of its Equation 2.5 to three decimals.
    R = [1.0 -0.9 -0.9; -0.9 1.0 0.3; -0.9 0.3 1.0]
    Y = posdef(Posdef(; alg = ClippedNearestCorrelation(; tau = 0.05)), R)
    @test round.(Y; digits = 3) == [1.0 -0.781 -0.781; -0.781 1.0 0.327; -0.781 0.327 1.0]
end
