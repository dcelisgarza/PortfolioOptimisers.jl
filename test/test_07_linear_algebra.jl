using PortfolioOptimisers, Test, NearestCorrelationMatrix, LinearAlgebra

# Card 7 of the 2026-08-17 maintainability review: the out-of-place verbs are documented
# as an optional part of the extension interface, so an estimator that declares only the
# in-place form must inherit a copying fallback. These estimators declare nothing else.
struct ShapeOnlyProbe <: PortfolioOptimisers.AbstractMatrixProcessingEstimator end
function PortfolioOptimisers.assert_shape_only_matrix_processing(::ShapeOnlyProbe)
    return nothing
end
struct Card7Posdef <: PortfolioOptimisers.AbstractPosdefEstimator end
function PortfolioOptimisers.posdef!(::Card7Posdef, X::AbstractMatrix)
    X .= (X .+ transpose(X)) ./ 2
    return X
end
struct Card7Denoise <: PortfolioOptimisers.AbstractDenoiseEstimator end
function PortfolioOptimisers.denoise!(::Card7Denoise, X::AbstractMatrix, q::Number)
    X .*= 2
    return X
end
struct Card7Detone <: PortfolioOptimisers.AbstractDetoneEstimator end
function PortfolioOptimisers.detone!(::Card7Detone, X::AbstractMatrix)
    X .-= 1
    return X
end

# Ticket 447: `posdef!` warns rather than throwing when the projection leaves the matrix
# indefinite, and returns the unrepaired matrix either way. Every algorithm
# NearestCorrelationMatrix ships repairs its input, and `nearest_cor!` passes
# `ensure_pd = true` on top of it, so the arm is reachable only through a `pdm.alg` that
# declines to repair and a `pdm.kwargs` that lifts that guard. Both are documented fields.
struct Ticket447Stub <: NearestCorrelationMatrix.NCMAlgorithm end
NearestCorrelationMatrix.modifies_in_place(::Ticket447Stub) = true
NearestCorrelationMatrix.supports_symmetric(::Ticket447Stub) = true
function NearestCorrelationMatrix.solve!(solver::NearestCorrelationMatrix.NCMSolver,
                                         alg::Ticket447Stub; kwargs...)
    return NearestCorrelationMatrix.build_ncm_solution(alg, solver.A, nothing, solver;
                                                       iters = 0)
end

# Ticket 447: an algorithm whose effect does not commute with detoning, so the order of
# `mp.order` is observable rather than merely stored.
struct Ticket447Bump <: PortfolioOptimisers.AbstractMatrixProcessingAlgorithm end
function PortfolioOptimisers.matrix_processing_algorithm!(::Ticket447Bump,
                                                          sigma::AbstractMatrix, args...;
                                                          kwargs...)
    sigma[LinearAlgebra.diagind(sigma)] .*= 4
    return sigma
end

@testset "Linear Algebra" begin
    using PortfolioOptimisers, Test, CSV, LinearAlgebra, DataFrames, TimeSeries, StableRNGs,
          Random, AverageShiftedHistograms
    @testset "Operators" begin
        rng = StableRNG(987654321)
        X1 = rand(rng, 10)
        X2 = rand(rng, 10)
        res1 = X1 * transpose(X2)
        res2 = PortfolioOptimisers.:⊗(X1, X2)
        res3 = PortfolioOptimisers.:⊗(transpose(X1), transpose(X2))
        res4 = PortfolioOptimisers.:⊗(X1, X2)
        res5 = PortfolioOptimisers.:⊗(transpose(X1), transpose(X2))
        res6 = PortfolioOptimisers.:⊗(X2, X1)
        res7 = PortfolioOptimisers.:⊗(X2, X1)
        @test isapprox(res1, res2)
        @test isapprox(res1, res3)
        @test isapprox(res1, res4)
        @test isapprox(res1, res5)
        @test isapprox(res1, res6')
        @test isapprox(res1, res7')
    end
    @testset "Non Positive Definite Matrices" begin
        rng = StableRNG(123456789)
        X = randn(rng, 10, 20)
        sigma1 = cov(X)
        sigma2 = copy(sigma1)
        sigma3 = copy(sigma1)
        sigma4 = posdef(Posdef(), sigma1)

        posdef!(Posdef(), sigma1)
        @test LinearAlgebra.isposdef(sigma1)
        @test sigma4 == sigma1

        posdef!(nothing, sigma2)
        @test !LinearAlgebra.isposdef(sigma2)
        @test isapprox(sigma2, sigma3)

        @test posdef(nothing, sigma2) === sigma2
    end
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    T, N = size(rd.X)
    q = T / N
    @testset "Denoise algorithms" begin
        dns = [nothing, Denoise(; alg = FixedDenoise()), Denoise(; alg = ShrunkDenoise()),
               Denoise(; alg = SpectralDenoise())]
        df = CSV.read(joinpath(@__DIR__, "./assets/Denoise.csv.gz"), DataFrame)
        for (i, dn) in pairs(dns)
            sigma1 = copy(pr.sigma)
            denoise!(dn, sigma1, q)
            success = isapprox(vec(sigma1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma1), df[!, i])
            end
            @test success
        end
        @test denoise(nothing, pr.sigma) === pr.sigma
        sigma1 = copy(pr.sigma)
        sigma2 = denoise(dns[2], sigma1, q)
        denoise!(dns[2], sigma1, q)
        @test sigma1 == sigma2
    end
    @testset "Detone" begin
        dts = [nothing, Detone(), Detone(; n = 3)]
        df = CSV.read(joinpath(@__DIR__, "./assets/Detone.csv.gz"), DataFrame)
        for (i, dt) in pairs(dts)
            sigma1 = copy(pr.sigma)
            detone!(dt, sigma1)
            success = isapprox(vec(sigma1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(mu, df[!, i])
            end
            @test success
        end
        @test denoise(nothing, pr.sigma) === pr.sigma
        sigma1 = copy(pr.sigma)
        sigma2 = detone(dts[2], sigma1)
        detone!(dts[2], sigma1)
        @test sigma1 == sigma2
        @test detone(nothing, pr.sigma) === pr.sigma
    end
    @testset "Matrix processing" begin
        @test matrix_processing!(nothing, pr.sigma) === pr.sigma
        @test matrix_processing(nothing, pr.sigma) === pr.sigma
        @test PortfolioOptimisers.matrix_processing_algorithm(nothing, pr.sigma) ===
              pr.sigma
        sigma1 = copy(pr.sigma)
        sigma2 = copy(pr.sigma)
        sigma3 = copy(pr.sigma)
        sigma4 = matrix_processing(MatrixProcessing(), sigma1, pr.X)
        sigma5 = matrix_processing(nothing, sigma1, pr.X)
        sigma4 == matrix_processing!(MatrixProcessing(), sigma2, pr.X)
        sigma5 == matrix_processing!(nothing, sigma3, pr.X)
    end
    @testset "Ticket 447: the covariance-to-correlation wrapper" begin
        # `posdef!`, `denoise!` and `detone!` all decide with `any(!isone, s)`, so it is the
        # value of the diagonal that names a covariance, never the type of the matrix. The
        # two routes must therefore agree up to the rescaling.
        rng = StableRNG(11223344)
        f = randn(rng, 200)
        Y = hcat([0.8 .* f .+ 0.6 .* randn(rng, 200) for _ in 1:10]...)
        C = cor(Y)
        qr = 200 / 10
        sd = collect(range(0.5, 2.0; length = 10))
        S = StatsBase.cor2cov(copy(C), sd)
        for alg in (SpectralDenoise(), FixedDenoise(), ShrunkDenoise(; alpha = 0.3))
            Xc = denoise(Denoise(; alg = alg), C, qr)
            Sc = denoise(Denoise(; alg = alg), S, qr)
            @test all(isone, LinearAlgebra.diag(Xc))
            @test isapprox(LinearAlgebra.diag(Sc), LinearAlgebra.diag(S))
            @test isapprox(StatsBase.cov2cor(copy(Sc), sqrt.(LinearAlgebra.diag(Sc))), Xc)
        end
        Xt = detone(Detone(), C)
        St = detone(Detone(), S)
        @test all(isone, LinearAlgebra.diag(Xt))
        @test isapprox(LinearAlgebra.diag(St), LinearAlgebra.diag(S))
        @test isapprox(StatsBase.cov2cor(copy(St), sqrt.(LinearAlgebra.diag(St))), Xt)
    end
    @testset "Ticket 447: ShrunkDenoise's alpha is the weight kept" begin
        # `alpha` scales the off-diagonal part of the noise block, so `alpha = 0` is total
        # shrinkage and `alpha = 1` is a no-op. A reading of `alpha` as a shrinkage
        # intensity has the polarity backwards.
        rng = StableRNG(987654321)
        C = cor(randn(rng, 24, 8))
        offd(M) = sum(abs, M - LinearAlgebra.Diagonal(M))
        c0 = denoise(Denoise(; alg = ShrunkDenoise(; alpha = 0.0)), C, 24 / 8)
        c5 = denoise(Denoise(; alg = ShrunkDenoise(; alpha = 0.5)), C, 24 / 8)
        c1 = denoise(Denoise(; alg = ShrunkDenoise(; alpha = 1.0)), C, 24 / 8)
        # Every eigenvalue of this sample is noise, so the whole off-diagonal is shrunk.
        @test isapprox(offd(C), 9.42045698910268)
        @test iszero(offd(c0))
        @test isapprox(offd(c5), offd(C) / 2)
        @test isapprox(c1, C; atol = 1e-14)
        # Each branch pins the diagonal, the last one by hand.
        @test all(isone, LinearAlgebra.diag(c0))
        @test all(isone, LinearAlgebra.diag(c5))
        @test all(isone, LinearAlgebra.diag(c1))
    end
    @testset "Ticket 476: SpectralDenoise on an all-noise spectrum" begin
        #=
        Every eigenvalue of this sample sits below the fitted edge. `SpectralDenoise`
        zeroes them all, so its reconstruction is the zero matrix and `cov2cor` divides
        zero by zero. It used to return `NaN` everywhere, and `posdef!` then raised an
        `ArgumentError` from LAPACK that named neither the cause nor the file. The
        identity is the answer that matches the claim of the algorithm: no signal
        survives, so no pair keeps a correlation.
        =#
        rng = StableRNG(987654321)
        C = cor(randn(rng, 24, 8))
        q = 24 / 8
        vals = LinearAlgebra.eigen(C).values
        max_val = PortfolioOptimisers.find_max_eval(copy(vals), q,
                                                    AverageShiftedHistograms.Kernels.gaussian,
                                                    10, 1000, (), (;))
        @test searchsortedlast(vals, max_val) == 8
        Id = Matrix(1.0 * LinearAlgebra.I, 8, 8)
        Xs = denoise(Denoise(; alg = SpectralDenoise()), C, q)
        @test all(isfinite, Xs)
        @test isapprox(Xs, Id)
        # `FixedDenoise` already answered the identity here, so the two tags agree.
        @test isapprox(denoise(Denoise(; alg = FixedDenoise()), C, q), Id)
        # The covariance route keeps each asset's own variance and drops every covariance.
        sd = collect(range(0.5, 2.0; length = 8))
        S = StatsBase.cor2cov(copy(C), sd)
        @test isapprox(denoise(Denoise(; alg = SpectralDenoise()), S, q),
                       Matrix(LinearAlgebra.Diagonal(sd .^ 2)))
    end
    @testset "Ticket 447: detone! removes the top n modes" begin
        # `detone!` decrements `n` and slices `(end - n):end`, so `dt.n` is the count of
        # modes removed and `dt.n = 1` removes the market mode alone.
        rng = StableRNG(11223344)
        f = randn(rng, 200)
        Y = hcat([0.8 .* f .+ 0.6 .* randn(rng, 200) for _ in 1:10]...)
        C = cor(Y)
        sd = collect(range(0.5, 2.0; length = 10))
        S = StatsBase.cor2cov(copy(C), sd)
        function detone_byhand(X, n)
            s = LinearAlgebra.diag(X)
            iscov = any(!isone, s)
            Xw = copy(X)
            if iscov
                s = sqrt.(s)
                StatsBase.cov2cor!(Xw, s)
            end
            v, V = LinearAlgebra.eigen(Xw)
            for k in (size(Xw, 1) - n + 1):size(Xw, 1)
                Xw .-= v[k] * V[:, k] * transpose(V[:, k])
            end
            Xw .= StatsBase.cov2cor(Xw)
            PortfolioOptimisers.posdef!(Posdef(), Xw)
            iscov && StatsBase.cor2cov!(Xw, s)
            return Xw
        end
        for n in 1:3
            @test isapprox(detone(Detone(; n = n), C), detone_byhand(C, n))
        end
        for n in 1:2
            @test isapprox(detone(Detone(; n = n), S), detone_byhand(S, n))
        end
        @test_throws DomainError detone(Detone(; n = 11), C)
    end
    @testset "Ticket 447: a detoned matrix is singular, never indefinite" begin
        # The subtraction sets the `n` removed eigenvalues to zero and keeps the others, and the
        # rescale to a unit diagonal is a congruence. So with no repair the result is positive
        # semidefinite with `n` zero eigenvalues, as Section 2.6 of `mlp1` states.
        rng = StableRNG(11223344)
        f = randn(rng, 200)
        Y = hcat([0.8 .* f .+ 0.6 .* randn(rng, 200) for _ in 1:10]...)
        C = cor(Y)
        for n in 1:3
            Xt = detone(Detone(; pdm = nothing, n = n), C)
            l = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Xt))
            tol = 10 * length(l) * eps(eltype(l)) * maximum(abs, l)
            @test count(x -> abs(x) <= tol, l) == n
            @test minimum(l) >= -tol
        end
    end
    @testset "Ticket 447: find_max_eval and its fallback" begin
        #=
        On data drawn from an identity covariance the fitted edge sits at the theoretical
        Marcenko-Pastur edge, because the noise variance of a correlation matrix is one.

        The tolerance is the accuracy of the average shifted histogram, not of the search.
        The estimate is built from `N` eigenvalues, so it sharpens as `N` rises: over five
        seeds the worst fitted variance is 0.9849 at `N = 50` and 0.9914 at `N = 100`.

        Ticket 475: the draws below also pin the *support* of that estimate. An estimate
        refitted over `[e_min, e_max]` at every trial variance renormalises over a shrinking
        window, which gives the objective a spurious local minimum. On the `N = 50` draw the
        search converged to it, at 0.415 of the true variance, so the second assertion of
        each pair reds when the support follows the search.
        =#
        for (T, N) in ((2000, 100), (1000, 50))
            Z = cor(randn(StableRNG(555), T, N))
            vals = LinearAlgebra.eigen(Z).values
            edge = (1 + sqrt(N / T))^2
            fitted = PortfolioOptimisers.find_max_eval(vals, T / N,
                                                       AverageShiftedHistograms.Kernels.gaussian,
                                                       10, 1000, (), (;))
            @test isapprox(fitted, edge; rtol = 5e-2)
            @test fitted > 0.9 * edge
        end
        #=
        Ticket 475: a spectrum whose eigenvalues are all equal has a range of zero, and the
        average shifted histogram needs a range it can bin. The fit carries no information
        there, but it must not raise: a matrix with such a spectrum is a multiple of the
        identity, and `denoise!` is asked for one by any caller who hands it one.
        =#
        @test isfinite(PortfolioOptimisers.find_max_eval(ones(10), 5.0,
                                                         AverageShiftedHistograms.Kernels.gaussian,
                                                         10, 1000, (), (;)))
        @test denoise(Denoise(), Matrix(1.0 * LinearAlgebra.I, 10, 10), 5.0) ==
              Matrix(1.0 * LinearAlgebra.I, 10, 10)
        # A search that does not converge substitutes a unit variance, and says so. Only
        # `dn.kwargs` can make the search fail; the defaults converge.
        vals = LinearAlgebra.eigen(cor(randn(StableRNG(555), 500, 50))).values
        q0 = 10.0
        @test isapprox(PortfolioOptimisers.find_max_eval(copy(vals), q0,
                                                         AverageShiftedHistograms.Kernels.gaussian,
                                                         10, 1000, (), (; iterations = 0)),
                       (1 + sqrt(inv(q0)))^2)
        @test_logs (:warn,
                    "Marčenko-Pastur fit did not converge, using a unit noise variance.") PortfolioOptimisers.find_max_eval(copy(vals),
                                                                                                                            q0,
                                                                                                                            AverageShiftedHistograms.Kernels.gaussian,
                                                                                                                            10,
                                                                                                                            1000,
                                                                                                                            (),
                                                                                                                            (;
                                                                                                                             iterations = 0))
    end
    @testset "Ticket 1506: a repair refuses a result that is not PSD, and returns one that is" begin
        # The stub leaves the indefinite matrix as it is, so the repair cannot return it.
        X = [1.0 2.0; 2.0 1.0]
        est = Posdef(; alg = Ticket447Stub(), kwargs = (; ensure_pd = false))
        @test_throws PosdefRepairError posdef!(est, copy(X))
        err = try
            posdef!(est, copy(X))
        catch e
            e
        end
        # The message names the algorithm, the tolerance and the smallest eigenvalue, -1.
        @test occursin("Ticket447Stub", err.msg)
        @test occursin("but it is -1.0", err.msg)
        @test_throws PosdefRepairError PortfolioOptimisers.assert_posdef_repair(Ticket447Stub(),
                                                                                [1.0 NaN;
                                                                                 NaN 1.0])
        # A singular PSD result is a valid covariance, though `isposdef` is false on it. The
        # stub returns its input, so the generic arm meets the singular result itself, as a
        # correlation and as a covariance, and returns it with no message.
        for S in ([1.0 1.0; 1.0 1.0], [4.0 2.0; 2.0 1.0])
            @test !LinearAlgebra.isposdef(S)
            r = @test_logs min_level = Logging.Warn posdef!(est, copy(S))
            @test r == S
        end
    end
    @testset "Ticket 1429: a zero variance keeps a zero row, and the rest is repaired" begin
        # A PSD matrix with a zero diagonal entry has a zero row and column (every 2x2 minor
        # is non-negative), and `posdef!` keeps the diagonal, so that row has no other answer.
        # The repair used to divide it by zero in `cov2cor!` and meet a LAPACK refusal.
        z = [2, 4]
        p = [1, 3, 5]
        X = [1.0 0.0 0.9 0.0 -0.8
             0.0 0.0 0.0 0.0 0.0
             0.9 0.0 1.0 0.0 0.9
             0.0 0.0 0.0 0.0 0.0
             -0.8 0.0 0.9 0.0 2.0]
        @test !LinearAlgebra.isposdef(X[p, p])
        # The zero rows are the correct answer, so no warning comes for them.
        r = @test_logs min_level = Logging.Warn posdef!(Posdef(), copy(X))
        @test all(iszero, r[z, :])
        @test all(iszero, r[:, z])
        @test r[p, p] == posdef!(Posdef(), X[p, p])
        @test LinearAlgebra.isposdef(r[p, p])
        @test LinearAlgebra.diag(r)[z] == [0.0, 0.0]
        # A covariance with a zero variance and a non-zero entry in its row is not PSD, and
        # the only PSD matrix with that diagonal sets the entry to zero.
        Y = copy(X)
        Y[2, 3] = Y[3, 2] = 0.1
        @test posdef!(Posdef(), Y) == r
        # A matrix of zeros is PSD already in the only way it can be.
        @test posdef!(Posdef(), zeros(3, 3)) == zeros(3, 3)
        # The test is `iszero`: a tiny variance is positive and stays in the block.
        @test PortfolioOptimisers.zero_variance_rows!([1e-34 0.0; 0.0 1.0], trues(2)) ==
              [true, true]
        # The refusal still comes when the positive block stays indefinite.
        est = Posdef(; alg = Ticket447Stub(), kwargs = (; ensure_pd = false))
        W = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 1.0]
        @test_throws PosdefRepairError posdef!(est, copy(W))
    end
    @testset "Ticket 1429: the block repair leaves a zero variance out of every step" begin
        # Denoise and Detone each convert to a correlation, so a zero variance broke them as it
        # broke the positive definite step. The block holds the finite, non-zero variances.
        rng = StableRNG(1429)
        Xd = randn(rng, 120, 6) ./ 100
        Xd[:, 3] .= 0.0
        S = cov(Xd)
        @test S[3, 3] == 0
        mp = MatrixProcessing(; pdm = Posdef(), dn = Denoise(), dt = Detone(; n = 1))
        p = [1, 2, 4, 5, 6]
        byhand = matrix_processing(mp, S[p, p], Xd[:, p])
        r = PortfolioOptimisers.matrix_processing_block!(mp, copy(S), Xd)
        @test r[p, p] == byhand
        @test all(iszero, r[3, :])
        @test all(iszero, r[:, 3])
        # The shape twin cuts the count of the block, not its columns.
        T = size(Xd, 1)
        byshape = copy(S[p, p])
        matrix_processing!(mp, byshape, T, length(p))
        rs = PortfolioOptimisers.matrix_processing_block!(mp, copy(S), T, size(S, 1))
        @test rs[p, p] == byshape
        @test all(iszero, rs[3, :])
        # Every variance zero leaves nothing to repair, on both methods.
        @test PortfolioOptimisers.matrix_processing_block!(mp, zeros(3, 3), zeros(4, 3)) ==
              zeros(3, 3)
        @test PortfolioOptimisers.matrix_processing_block!(mp, zeros(3, 3), 4, 3) ==
              zeros(3, 3)
        # A matrix with no finite variance is handed to the plain method, which has no step to
        # run, and comes back as it was.
        none = MatrixProcessing(; pdm = nothing)
        allnan = fill(NaN, 3, 3)
        @test all(isnan, PortfolioOptimisers.matrix_processing_block!(none, allnan, 4, 3))
        # A matrix processing estimator of a caller's, whose shape check passes, is returned as
        # given by the plain method.
        mine = ShapeOnlyProbe()
        S3 = [1.0 0.5; 0.5 2.0]
        @test matrix_processing!(mine, S3, 4, 2) === S3
        # A NaN frame stays as it is: the cell of a constant asset and an asset the fit left
        # out is still NaN, and the zero row is zero inside the finite rows alone.
        F = copy(S)
        F[6, :] .= NaN
        F[:, 6] .= NaN
        Fs = copy(F)
        rf = PortfolioOptimisers.matrix_processing_block!(mp, F, Xd)
        q = [1, 2, 4, 5]
        @test rf[q, q] == matrix_processing(mp, S[q, q], Xd[:, q])
        @test all(isnan, rf[6, :])
        @test all(iszero, rf[3, 1:5])
        rfs = PortfolioOptimisers.matrix_processing_block!(mp, Fs, T, size(S, 1))
        byq = copy(S[q, q])
        matrix_processing!(mp, byq, T, length(q))
        @test rfs[q, q] == byq
        @test all(isnan, rfs[:, 6])
        @test all(iszero, rfs[1:5, 3])
        # The default covariance of returns with a constant asset (cash at a zero return) is
        # finite and PSD, where it used to throw.
        sigma = cov(PortfolioOptimisersCovariance(), Xd)
        @test all(isfinite, sigma)
        @test all(iszero, sigma[3, :])
        @test LinearAlgebra.isposdef(sigma[p, p])
    end
    @testset "Ticket 1731: Denoise and Detone leave a zero variance out on the plain route" begin
        # `denoise!` and `detone!` each divide by the standard deviations, so a constant asset
        # gave `0 / 0` and `eigen` refused the matrix. Each verb now works on the block of the
        # positive variances, as `posdef!` does.
        Xd = randn(StableRNG(1429), 120, 6) ./ 100
        Xd[:, 3] .= 0.0
        S = cov(Xd)
        T, N = size(Xd)
        p = [1, 2, 4, 5, 6]
        # The Marchenko-Pastur ratio of the block reads the size of the block.
        r = denoise!(Denoise(), copy(S), T / N)
        @test r[p, p] == denoise!(Denoise(), S[p, p], T / length(p))
        # On this sample the two ratios keep the same count of noise eigenvalues, and the
        # default replaces them by their mean, so the ratio of the whole matrix gives the same
        # block. On the next one an eigenvalue lies between the two edges (R16 of #1630).
        Xq = randn(StableRNG(1), 120, 6) ./ 100
        Xq[:, 3] .= 0.0
        Sq = cov(Xq)
        rq = denoise!(Denoise(), copy(Sq), T / N)
        @test rq[p, p] == denoise!(Denoise(), Sq[p, p], T / length(p))
        @test rq[p, p] != denoise!(Denoise(), Sq[p, p], T / N)
        @test all(iszero, r[3, :])
        @test all(iszero, r[:, 3])
        r = detone!(Detone(), copy(S))
        @test r[p, p] == detone!(Detone(), S[p, p])
        @test all(iszero, r[3, :])
        @test all(iszero, r[:, 3])
        # The call on the block refuses more modes than the block has assets.
        @test_throws DomainError detone!(Detone(; n = 6), copy(S))
        # A matrix of zeros has no block, and each verb leaves it as it is.
        @test denoise!(Denoise(), zeros(3, 3), 2.0) == zeros(3, 3)
        @test detone!(Detone(), zeros(3, 3)) == zeros(3, 3)
        # `PortfolioOptimisersCovariance` takes the plain route on a finite sample.
        for mp in (MatrixProcessing(; dn = Denoise()), MatrixProcessing(; dt = Detone()))
            sigma = cov(PortfolioOptimisersCovariance(; mp = mp), Xd)
            @test all(isfinite, sigma)
            @test all(iszero, sigma[3, :])
            @test LinearAlgebra.isposdef(sigma[p, p])
        end
    end
    @testset "Ticket 447: matrix_processing! applies mp.order" begin
        rng = StableRNG(90210)
        Xd = randn(rng, 120, 6)
        S0 = cov(Xd)
        mp = MatrixProcessing(; pdm = Posdef(), dn = Denoise(; alg = FixedDenoise()),
                              dt = Detone(; n = 1), alg = nothing)
        T, N = size(Xd)
        byhand = copy(S0)
        posdef!(mp.pdm, byhand)
        denoise!(mp.dn, byhand, T / N)
        detone!(mp.dt, byhand)
        PortfolioOptimisers.matrix_processing_algorithm!(mp.alg, byhand, Xd)
        @test matrix_processing(mp, S0, Xd) == byhand
        # A `nothing` estimator skips its step rather than leaving the order.
        skipped = MatrixProcessing(; pdm = nothing, dn = nothing, dt = nothing,
                                   alg = nothing)
        @test matrix_processing(skipped, S0, Xd) == S0
        # The order is applied, not merely stored.
        b1 = matrix_processing(MatrixProcessing(; dt = Detone(; n = 1),
                                                alg = Ticket447Bump(), order = (:alg, :dt)),
                               S0, Xd)
        b2 = matrix_processing(MatrixProcessing(; dt = Detone(; n = 1),
                                                alg = Ticket447Bump(), order = (:dt, :alg)),
                               S0, Xd)
        @test !isapprox(b1, b2)
        @test isapprox(maximum(abs, b1 - b2), 1.2495649955451098)
        @test_throws ArgumentError MatrixProcessing(; order = (:pdm, :nope))
    end
    @testset "Inherited out-of-place fallback" begin
        X = [1.0 2.0; 0.0 1.0]
        Xc = copy(X)
        # Each verb inherits the copying wrapper from its abstract supertype.
        @test posdef(Card7Posdef(), X) == [1.0 1.0; 1.0 1.0]
        @test X == Xc
        @test denoise(Card7Denoise(), X, 2.0) == [2.0 4.0; 0.0 2.0]
        @test X == Xc
        @test detone(Card7Detone(), X) == [0.0 1.0; -1.0 0.0]
        @test X == Xc
        # The field bounds accept a user estimator, so the contract is usable end to end.
        mp = MatrixProcessing(; pdm = Card7Posdef(), dn = Card7Denoise(),
                              dt = Card7Detone())
        @test mp.pdm isa Card7Posdef
        @test mp.dn isa Card7Denoise
        @test mp.dt isa Card7Detone
        @test Denoise(; pdm = Card7Posdef()).pdm isa Card7Posdef
        @test Detone(; pdm = Card7Posdef()).pdm isa Card7Posdef
        sigma = [1.0 0.5; 0.25 1.0]
        sigmac = copy(sigma)
        Xr = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        # Order (:pdm, :dn, :dt, :alg): symmetrise, then double, then subtract one.
        @test matrix_processing(mp, sigma, Xr) == [1.0 -0.25; -0.25 1.0]
        @test sigma == sigmac
    end
end

#=
The three structure matrices and the block vectorisation of `src/10_Prior/04_HighOrderPrior.jl`.
Sweep ticket #534: no test in the suite named `duplication_matrix`, `elimination_matrix` or
`summation_matrix` on the identity each is defined by — they were reached only through the high
order prior, so every property they are supposed to have was untested. These check the identities
themselves, at `n = 3` and `n = 4`, for both `diag` settings.
=#
@testset "Duplication, elimination and summation matrices" begin
    PO = PortfolioOptimisers
    # The lower triangle read column by column, with and without the diagonal.
    vech(A) = [A[i, j] for j in axes(A, 2) for i in j:size(A, 1)]
    vech_strict(A) = [A[i, j] for j in axes(A, 2) for i in (j + 1):size(A, 1)]

    for n in (3, 4)
        m = div(n * (n + 1), 2)
        ms = div(n * (n - 1), 2)
        A = reshape(collect(1.0:(n ^ 2)), n, n)
        A = A + transpose(A)

        D = PO.duplication_matrix(n)
        L = PO.elimination_matrix(n)
        S = PO.summation_matrix(n)
        Ds = PO.duplication_matrix(n, false)
        Ls = PO.elimination_matrix(n, false)
        Ss = PO.summation_matrix(n, false)

        # `D * vech(A) == vec(A)`, the identity the duplication matrix is named for. Without
        # the diagonal it restores the hollow matrix instead.
        @test size(D) == (n^2, m)
        @test D * vech(A) == vec(A)
        @test size(Ds) == (n^2, ms)
        @test Ds * vech_strict(A) == vec(A - Diagonal(A))

        # `L * vec(A) == vech(A)`. Dropping the diagonal drops exactly the `n` rows that read
        # it, so the row count falls from `n(n+1)/2` to `n(n-1)/2` and the column count stays.
        @test size(L) == (m, n^2)
        @test L * vec(A) == vech(A)
        @test size(Ls) == (ms, n^2)
        @test Ls * vec(A) == vech_strict(A)
        @test size(L, 1) - size(Ls, 1) == n
        @test size(L, 2) == size(Ls, 2) == n^2

        # `S == transpose(D) * D * L`, which weights each row by the number of places its
        # entry occupies in `vec(A)`: one on the diagonal and two off it. The sum identity
        # follows. Without the diagonal every row is off it, so every weight is two and the
        # sum reaches the off-diagonal entries alone.
        @test size(S) == (m, n^2)
        @test S == transpose(D) * D * L
        @test sum(S * vec(A)) == sum(vec(A))
        @test size(Ss) == (ms, n^2)
        @test Ss == transpose(Ds) * Ds * Ls
        @test sum(Ss * vec(A)) == sum(vec(A)) - sum(diag(A))
        @test all(x -> x == 2, filter(!iszero, Array(Ss)))

        # The one-walk builder returns the same three matrices as three separate calls.
        D2, L2, S2 = PO.dup_elim_sum_matrices(n)
        @test D2 == D
        @test L2 == L
        @test S2 == S
    end

    # `n` must be positive. The docstring claimed the precondition and nothing checked it:
    # `n = 0` returned three empty matrices and `n = -1` surfaced an opaque `ArgumentError`
    # from `SparseArrays`. Sweep ticket #534.
    @test_throws DomainError PO.dup_elim_sum_matrices(0)
    @test_throws DomainError PO.dup_elim_sum_matrices(-1)
    @test size(PO.dup_elim_sum_matrices(1)[1]) == (1, 1)

    # `dup_elim_sum_view` is two methods. The matrix one reads the dimension from its second
    # argument and never from the matrix, which is what lets `port_opt_view` pass the full
    # cokurtosis and the subproblem's asset count. The varargs one builds nothing, and it
    # also takes any call whose argument count the matrix method does not.
    M = reshape(collect(1.0:24.0), 6, 4)
    @test PO.dup_elim_sum_view(M, 3) == PO.dup_elim_sum_matrices(3)
    @test size(PO.dup_elim_sum_view(M, 3)[1]) == (9, 6)
    @test PO.dup_elim_sum_view(nothing, 3) == (nothing, nothing, nothing)
    @test PO.dup_elim_sum_view(M, 3, :ignored) == (nothing, nothing, nothing)
    @test PO.dup_elim_sum_view() == (nothing, nothing, nothing)
end

#=
`block_vec_pq` on a non-square partition. The `jldoctest` gives a `4 × 4` case with `p == q` and
`m == n`, and a square partition cannot separate a block-column-major stacking from a
block-row-major one. Sweep ticket #534.
=#
@testset "Block vectorisation" begin
    PO = PortfolioOptimisers
    p, q, m, n = 2, 3, 3, 4
    A = reshape(collect(1:(m * p * n * q)), m * p, n * q)
    B = PO.block_vec_pq(A, p, q)
    @test size(A) == (6, 12)
    @test size(B) == (m * n, p * q)

    # Row `(j - 1) * m + i` holds the vectorisation of block `(i, j)`, so the block column
    # index runs slowest.
    H = Matrix{Int}(undef, m * n, p * q)
    for j in 0:(n - 1), i in 0:(m - 1)
        H[j * m + i + 1, :] = vec(A[(i * p + 1):((i + 1) * p), (j * q + 1):((j + 1) * q)])
    end
    @test B == H

    # The other stacking order is a different matrix, which is the property the square
    # example in the docstring cannot show.
    H2 = Matrix{Int}(undef, m * n, p * q)
    for i in 0:(m - 1), j in 0:(n - 1)
        H2[i * n + j + 1, :] = vec(A[(i * p + 1):((i + 1) * p), (j * q + 1):((j + 1) * q)])
    end
    @test B != H2

    # Both divisibility guards.
    @test_throws DomainError PO.block_vec_pq(A, 4, q)
    @test_throws DomainError PO.block_vec_pq(A, p, 5)
end

using PortfolioOptimisers, Test, LinearAlgebra, Statistics, StatsBase, StableRNGs
using CovarianceEstimation: CovarianceEstimation

const PO = PortfolioOptimisers
include(joinpath(@__DIR__, "parity_harness.jl"))

# The geodesic from its first form, S^(1/2) (S^(-1/2) T S^(-1/2))^a S^(1/2). The library
# computes the second form, from the target end, so this derivation shares no step with it.
function geodesic_first_form(S, T, a)
    Sh = sqrt(Symmetric(S))
    Si = inv(Sh)
    vals, vecs = eigen(Symmetric(Si * T * Si))
    R = Sh * (vecs * Diagonal(vals .^ a) * transpose(vecs)) * Sh
    return (R + transpose(R)) / 2
end
# The affine-invariant distance, from the generalised eigenvalues of the pair.
airm(A, B) = sqrt(sum(abs2, log.(eigvals(Symmetric(A), Symmetric(B)))))

# A fixed start and a fixed target. The answers below were produced by the oracle's own
# interpolation on these two matrices, printed to 17 digits.
const S3 = [4.0 1.2 -0.6; 1.2 2.0 0.3; -0.6 0.3 1.0]
const T3 = [2.0 0.5 0.0; 0.5 1.5 0.2; 0.0 0.2 1.0]
const REF3 = Dict((0.25, :scaled) =>
                      [3.438923625902696 0.8730560672708096 -0.4794913233793692;
                       0.8730560672708096 2.0119785426973906 0.2960423861817764;
                       -0.4794913233793692 0.2960423861817764 1.1970237477339962],
                  (0.25, :diagonal) =>
                      [3.878865540882063 0.9374303288196849 -0.47799293139738863;
                       0.9374303288196849 1.948710537428578 0.24827423268624055;
                       -0.47799293139738863 0.24827423268624055 0.9836330357018354],
                  (0.25, :custom) =>
                      [3.3394422526679675 0.9788009981501989 -0.41637694497517863;
                       0.9788009981501989 1.8513979548948831 0.2852875162156939;
                       -0.41637694497517863 0.2852875162156939 0.9819440525365716],
                  (0.75, :scaled) =>
                      [2.61467620399093 0.2854499498476929 -0.19280380315354267;
                       0.2854499498476929 2.171736554349288 0.16202243511913234;
                       -0.19280380315354267 0.16202243511913234 1.8319770835376343],
                  (0.75, :diagonal) =>
                      [3.865400271957605 0.3488455531264325 -0.18441448454452036;
                       0.3488455531264325 1.9426918439601069 0.10219895025356447;
                       -0.18441448454452036 0.10219895025356447 0.9813376299613578],
                  (0.75, :custom) =>
                      [2.3583159785076746 0.634617864619692 -0.12123085972978617;
                       0.634617864619692 1.603348184331043 0.23427292498098157;
                       -0.12123085972978617 0.23427292498098157 0.9826669317233228])
const TARGETS3 = Dict(:identity => IdentityTarget(), :scaled => ScaledIdentityTarget(),
                      :common => CommonCovarianceTarget(),
                      :constcor => ConstantCorrelationTarget(),
                      :diagonal => DiagonalTarget(), :custom => T3)
# A small returns matrix, and the oracle estimator's answer on it at intensity 0.4, with
# its default sample covariance and its default positive definite repair.
const X10 = [1.2e-05 0.002992 -0.001246; -0.008906 -0.008109 -0.01308;
             0.000601 0.013643 0.001839; -0.006205 0.002417 0.005398;
             0.001054 -0.008883 -0.004839; 0.006953 -0.010661 -0.010602;
             -0.019012 -0.0205 -0.026766; -0.002351 -0.013615 -0.00386;
             0.001568 -0.001242 -0.025946; -0.005387 -0.00264 0.000352]
const REFX10 = Dict(:scaled =>
                        [6.219377158902727e-05 1.8438924377515432e-05 1.379562879829589e-05;
                         1.8438924377515432e-05 8.750185832227732e-05 3.529617755463757e-05;
                         1.379562879829589e-05 3.529617755463757e-05 0.0001054049007256008],
                    :diagonal =>
                        [5.0243224895258666e-05 1.7159579139855345e-05 1.3565025437153587e-05;
                         1.7159579139855345e-05 8.929446129947326e-05 3.76565581356144e-05;
                         1.3565025437153587e-05 3.76565581356144e-05 0.0001187745858675302])

@testset "Geodesic shrinkage covariance" begin
    @testset "construction" begin
        ce = GeodesicShrinkageCovariance()
        @test ce.ce isa Covariance
        @test ce.pdm isa Posdef
        @test ce.tgt isa ScaledIdentityTarget
        @test ce.alpha == 0.1
        @test GeodesicShrinkageCovariance(; alpha = 1, tgt = T3).tgt === T3
        @test_throws DomainError GeodesicShrinkageCovariance(; alpha = -0.1)
        @test_throws DomainError GeodesicShrinkageCovariance(; alpha = 1.1)
        @test_throws DimensionMismatch GeodesicShrinkageCovariance(; tgt = ones(2, 3))
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = [1.0 NaN; NaN 1.0])
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = [1.0 0.5; 0.0 1.0])
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = [1.0 2.0; 2.0 1.0])
    end
    @testset "the targets" begin
        @test PO.shrinkage_target(IdentityTarget(), S3) == I(3)
        @test PO.shrinkage_target(ScaledIdentityTarget(), S3) == 7 / 3 * I(3)
        @test PO.shrinkage_target(DiagonalTarget(), S3) == Diagonal(diag(S3))
        cc = PO.shrinkage_target(CommonCovarianceTarget(), S3)
        @test diag(cc) ≈ fill(7 / 3, 3)
        @test cc[1, 2] ≈ cc[2, 3] ≈ (1.2 - 0.6 + 0.3) / 3
        d = sqrt.(diag(S3))
        r = (1.2 / (d[1] * d[2]) - 0.6 / (d[1] * d[3]) + 0.3 / (d[2] * d[3])) / 3
        kc = PO.shrinkage_target(ConstantCorrelationTarget(), S3)
        @test diag(kc) == diag(S3)
        @test kc ≈ [4.0 r*d[1]*d[2] r*d[1]*d[3]; r*d[1]*d[2] 2.0 r*d[2]*d[3];
                    r*d[1]*d[3] r*d[2]*d[3] 1.0]
        @test PO.shrinkage_target(CommonCovarianceTarget(), fill(2.0, 1, 1)) ==
              fill(2.0, 1, 1)
        @test PO.shrinkage_target(ConstantCorrelationTarget(), fill(2.0, 1, 1)) ==
              fill(2.0, 1, 1)
    end
    @testset "parity with the oracle interpolation" begin
        # Measured maxrel 5.1e-15 cell by cell.
        for a in (0.25, 0.75), key in (:scaled, :diagonal, :custom)
            got = PO.geodesic_point(TARGETS3[key], S3, a)
            @test parity_compare(got, REF3[(a, key)]; name = "geodesic $(a) $(key)").ok
        end
        for a in (0.25, 0.75), (key, tgt) in TARGETS3
            T = PO.shrinkage_target(tgt, S3)
            @test isapprox(PO.geodesic_point(tgt, S3, a), geodesic_first_form(S3, T, a);
                           rtol = 1e-12)
        end
    end
    @testset "parity with the oracle estimator" begin
        for key in (:scaled, :diagonal)
            ce = GeodesicShrinkageCovariance(; tgt = TARGETS3[key], alpha = 0.4)
            # Measured maxrel 3.1e-15 cell by cell, in both orientations.
            @test parity_compare(cov(ce, X10), REFX10[key]; name = "geodesic X10 $(key)").ok
            @test parity_compare(cov(ce, permutedims(X10); dims = 2), REFX10[key];
                                 name = "geodesic X10' $(key)").ok
        end
    end
    @testset "end points" begin
        S = cov(Covariance(), X10)
        for (key, tgt) in TARGETS3
            key == :custom && continue
            @test cov(GeodesicShrinkageCovariance(; tgt = tgt, alpha = 0), X10) ≈ S
            @test cov(GeodesicShrinkageCovariance(; tgt = tgt, alpha = 1), X10) ==
                  PO.shrinkage_target(tgt, S)
        end
    end
    @testset "geometry of the geodesic" begin
        v = tr(S3) / 3
        for a in (0.2, 0.5, 0.9)
            got = PO.geodesic_point(ScaledIdentityTarget(), S3, a)
            @test isapprox(cond(got), cond(S3)^(1 - a); rtol = 1e-10)
            @test isapprox(eigvals(Symmetric(got)),
                           eigvals(Symmetric(S3)) .^ (1 - a) .* v^a; rtol = 1e-12)
            # The geometric mean is below the arithmetic mean of linear shrinkage, so the
            # trace falls at an intermediate intensity.
            @test all(eigvals(Symmetric(got)) .<=
                      (1 - a) .* eigvals(Symmetric(S3)) .+ a * v .+ 1e-12)
            @test tr(got) < tr(S3)
            for (key, tgt) in TARGETS3
                T = PO.shrinkage_target(tgt, S3)
                got = PO.geodesic_point(tgt, S3, a)
                @test isposdef(got)
                @test isapprox(airm(S3, got), a * airm(S3, T); rtol = 1e-10)
                @test isapprox(airm(got, T), (1 - a) * airm(S3, T); rtol = 1e-10)
            end
        end
        # Towards the diagonal, the closed form is D^(1/2) R^(1 - a) D^(1/2).
        d = sqrt.(diag(S3))
        R = S3 ./ (d * transpose(d))
        vals, vecs = eigen(Symmetric(R))
        Ra = vecs * Diagonal(vals .^ 0.3) * transpose(vecs)
        @test isapprox(PO.geodesic_point(DiagonalTarget(), S3, 0.7),
                       d .* Ra .* transpose(d); rtol = 1e-12)
    end
    @testset "a singular start is answered, an indefinite one is refused" begin
        rng = StableRNG(123456789)
        Xs = randn(rng, 4, 6)
        for tgt in (ScaledIdentityTarget(), DiagonalTarget(), ConstantCorrelationTarget())
            ce = GeodesicShrinkageCovariance(; pdm = nothing, tgt = tgt, alpha = 0.5)
            got = cov(ce, Xs)
            vals = eigvals(Symmetric(got))
            # The rank of the start survives: its three zero eigenvalues stay at round-off.
            @test count(<(1e-12 * maximum(vals)), vals) == 3
            @test minimum(vals) > -1e-12 * maximum(vals)
            # It is the limit of the answers on the positive definite starts nearby.
            near = PO.geodesic_point(tgt, cov(Xs) + 1e-12 * I, 0.5)
            @test isapprox(got, near; atol = 1e-5 * maximum(vals))
            # The default repair makes the start positive definite, so the answer is too.
            @test isposdef(cov(GeodesicShrinkageCovariance(; tgt = tgt, alpha = 0.5), Xs))
        end
        bad = copy(S3)
        bad[1, 2] = bad[2, 1] = 4.0
        @test_throws DomainError PO.geodesic_point(ScaledIdentityTarget(), bad, 0.5)
        @test_throws DomainError PO.geodesic_point(DiagonalTarget(), [1.0 0.0; 0.0 0.0],
                                                   0.5)
        # A zero variance makes the constant correlation target `NaN` off the diagonal. Its
        # Cholesky factorisation reports success, so the finite check is what refuses it.
        Sz = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 2.0]
        @test any(isnan, PO.shrinkage_target(ConstantCorrelationTarget(), Sz))
        @test_throws DomainError PO.geodesic_point(ConstantCorrelationTarget(), Sz, 0.5)
        Xz = hcat(Xs[:, 1:2], ones(4))
        @test_throws DomainError cov(GeodesicShrinkageCovariance(; pdm = nothing,
                                                                 tgt = ConstantCorrelationTarget(),
                                                                 alpha = 0.5), Xz)
        @test_throws DimensionMismatch PO.geodesic_point(T3, [1.0 0.0; 0.0 1.0], 0.5)
    end
    @testset "element types and the correlation" begin
        S32 = Float32.(S3)
        @test eltype(PO.geodesic_point(ScaledIdentityTarget(), S32, 0.5f0)) == Float32
        @test isapprox(PO.geodesic_point(ScaledIdentityTarget(), S32, 0.5f0),
                       PO.geodesic_point(ScaledIdentityTarget(), S3, 0.5); rtol = 1e-5)
        @test eltype(cov(GeodesicShrinkageCovariance(; alpha = 0.5f0), Float32.(X10))) ==
              Float32
        ce = GeodesicShrinkageCovariance(; alpha = 0.3)
        sigma = cov(ce, X10)
        @test cor(ce, X10) ≈ StatsBase.cov2cor(sigma, sqrt.(diag(sigma)))
        @test all(isone, diag(cor(ce, X10)))
    end
    @testset "views, weights and gaps" begin
        rng = StableRNG(24680)
        X = randn(rng, 100, 4)
        tgt = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]) + fill(0.1, 4, 4))
        ce = GeodesicShrinkageCovariance(; tgt = tgt, alpha = 0.3)
        cev = PO.port_opt_view(ce, [1, 3])
        @test cev.tgt == tgt[[1, 3], [1, 3]]
        @test size(cov(cev, X[:, [1, 3]])) == (2, 2)
        w = StatsBase.pweights(fill(1.0, 100))
        cew = PO.factory(ce, w)
        @test cew.ce.w === w
        @test cov(cew, X) ≈ cov(ce, X)
        # A prior over a gapped sample: an asset with no observation keeps its `NaN` row
        # and column, and the rest is shrunk to a positive definite block.
        Xn = hcat(X[:, 1:2], fill(NaN, 100), X[:, 3:4])
        Xn[1:20, 5] .= NaN
        cvg = GeodesicShrinkageCovariance(; ce = Covariance(; cvg = CoveragePolicy()),
                                          alpha = 0.3)
        # The policy reads the gap, so both answer `NaN` (#1694).
        @test isnan(PO.gap_fill_value(cvg.ce))
        @test isequal(PO.gap_fill_value(cvg), PO.gap_fill_value(cvg.ce))
        pr = prior(EmpiricalPrior(; ce = cvg), Xn)
        @test isnan.(diag(pr.sigma)) == [false, false, true, false, false]
        @test isposdef(pr.sigma[[1, 2, 4, 5], [1, 2, 4, 5]])
        rho = cor(cvg, Xn, nothing)
        @test isnan.(diag(rho)) == [false, false, true, false, false]
        @test diag(rho)[[1, 2, 4, 5]] ≈ ones(4)
        # A frame of `NaN` around the finite block: the block is shrunk, the frame is kept.
        sigma = cov(Covariance(), X)
        framed = fill(NaN, 5, 5)
        framed[[1, 2, 4, 5], [1, 2, 4, 5]] = sigma
        PO.geodesic_shrinkage!(ce, framed)
        @test framed[[1, 2, 4, 5], [1, 2, 4, 5]] ≈ PO.geodesic_point(tgt, sigma, 0.3)
        @test all(isnan, framed[3, :]) && all(isnan, framed[:, 3])
        @test all(isnan, PO.geodesic_shrinkage!(ce, fill(NaN, 2, 2)))
    end
    @testset "an incremental fit shrinks the folded matrix" begin
        rng = StableRNG(13579)
        X = randn(rng, 60, 4)
        ce = GeodesicShrinkageCovariance(; tgt = ConstantCorrelationTarget(), alpha = 0.3)
        @test PO.supports_partial_fit(ce)
        @test !PO.supports_partial_fit(GeodesicShrinkageCovariance(;
                                                                   ce = StatsBase.SimpleCovariance()))
        @test isapprox(cov(partial_fit!(ce, X)), cov(ce, X); rtol = 1e-12)
        @test isapprox(cor(partial_fit!(ce, X)), cor(ce, X); rtol = 1e-12)
        folded = foldl(partial_fit!, eachrow(X); init = ce)
        @test isapprox(cov(folded), cov(ce, X); rtol = 1e-10)
    end
    # Each property below is stated in a paper the docstrings cite, and each is checked against
    # the implementation. `phi(T, S, a)` is the geodesic from `S` to `T` at `a`, computed from
    # the target end, and `geodesic_first_form` is the geodesic from the start end.
    phi(T, S, a) = PO.geodesic_point(T, S, a)
    @testset "parity with Musolas, Smith and Marzouk (2021), section 2" begin
        for a in (0.1, 0.35, 0.8)
            got = phi(T3, S3, a)
            # Equation 2.3: A1^(1/2) U Λ^t Uᵀ A1^(1/2), with U Λ Uᵀ the eigendecomposition of
            # the whitened A1^(-1/2) A2 A1^(-1/2).
            @test isapprox(got, geodesic_first_form(S3, T3, a); rtol = 1e-12)
            # The two further forms after equation 2.3: A1 (A1⁻¹ A2)^t = A2 (A2⁻¹ A1)^(1 - t).
            @test isapprox(got, S3 * real(exp(a * log(S3 \ T3))); rtol = 1e-10)
            @test isapprox(got, T3 * real(exp((1 - a) * log(T3 \ S3))); rtol = 1e-10)
            # Equations 2.4 and 2.5: d(A1, φ(t)) = |t| d(A1, A2), with d from the generalised
            # eigenvalues of the pencil.
            @test isapprox(airm(S3, got), a * airm(S3, T3); rtol = 1e-10)
            # Equation 2.1: the distance is invariant under inversion.
            @test isapprox(airm(inv(S3), inv(got)), airm(S3, got); rtol = 1e-10)
            # Equation 2.2: the distance is invariant under congruence, and so is the geodesic.
            Z = [1.0 0.3 -0.2; 0.0 2.0 0.5; 0.4 0.0 1.5]
            @test isapprox(airm(Z * S3 * Z', Z * got * Z'), airm(S3, got); rtol = 1e-10)
            @test isapprox(phi(Z * T3 * Z', Z * S3 * Z', a), Z * got * Z'; rtol = 1e-10)
            # Section 2.2, property 1: φ⁻¹_{A1→A2}(t) = φ_{A1⁻¹→A2⁻¹}(t).
            @test isapprox(inv(got), phi(inv(T3), inv(S3), a); rtol = 1e-10)
            # Section 2.2, property 2: φ_{A1→A2}(t) = φ_{A2→A1}(1 - t).
            @test isapprox(got, phi(S3, T3, 1 - a); rtol = 1e-10)
            # Remark 2.2: φ_{A1→c A1}(t) = c^t A1.
            @test isapprox(phi(2.5 * S3, S3, a), 2.5^a * S3; rtol = 1e-12)
        end
        # The two anchors are the two ends of the curve.
        @test phi(T3, S3, 1) == T3
        @test PO.geodesic_shrinkage!(GeodesicShrinkageCovariance(; pdm = nothing, tgt = T3,
                                                                 alpha = 0), copy(S3)) == S3
    end
    @testset "parity with Bhatia (2007), the geometric mean of chapters 4 and 6" begin
        for a in (0.2, 0.5, 0.7)
            got = phi(T3, S3, a)
            # The weighted geometric mean has det(A #_t B) = det(A)^(1 - t) det(B)^t.
            @test isapprox(det(got), det(S3)^(1 - a) * det(T3)^a; rtol = 1e-12)
            # A pair that commutes: A #_t B = A^(1 - t) B^t. Here B = A², so the answer is A^(1 + t).
            vals, vecs = eigen(Symmetric(S3))
            @test isapprox(phi(S3^2, S3, a), vecs * Diagonal(vals .^ (1 + a)) * vecs';
                           rtol = 1e-10)
        end
        # The midpoint A # B is the unique positive definite solution of X A⁻¹ X = B.
        mid = phi(T3, S3, 0.5)
        @test isposdef(mid)
        @test isapprox(mid * (S3 \ mid), T3; rtol = 1e-12)
    end
    @testset "parity with Schäfer and Strimmer (2005), table 2" begin
        # CovarianceEstimation.jl builds the targets of the same table. A linear shrinkage at
        # intensity one returns its target, so it gives the target built from the same
        # corrected sample covariance.
        CE = CovarianceEstimation
        rng = StableRNG(97531)
        X = randn(rng, 80, 5) * [1.0 0.4 0.2 0.0 0.1; 0 1 0.3 0.2 0; 0 0 1 0.5 0.1;
                                 0 0 0 1 0.3; 0 0 0 0 1]
        S = cov(X)
        for (tgt, cet) in ((IdentityTarget(), CE.DiagonalUnitVariance()),
                           (ScaledIdentityTarget(), CE.DiagonalCommonVariance()),
                           (CommonCovarianceTarget(), CE.CommonCovariance()),
                           (DiagonalTarget(), CE.DiagonalUnequalVariance()),
                           (ConstantCorrelationTarget(), CE.ConstantCorrelation()))
            ref = cov(CE.LinearShrinkage(cet, 1.0; corrected = true), X)
            @test isapprox(PO.shrinkage_target(tgt, S), ref; rtol = 1e-12)
        end
        # The positive definite conditions the docstrings state. Target C has the eigenvalue
        # v - c with multiplicity N - 1 and v + (N - 1) c once; target F has the correlation
        # eigenvalues 1 - r with multiplicity N - 1 and 1 + (N - 1) r once.
        n = 5
        C = PO.shrinkage_target(CommonCovarianceTarget(), S)
        v, c = C[1, 1], C[1, 2]
        @test isapprox(eigvals(Symmetric(C)), sort([fill(v - c, n - 1); v + (n - 1) * c]);
                       rtol = 1e-12)
        F = PO.shrinkage_target(ConstantCorrelationTarget(), S)
        d = sqrt.(diag(F))
        r = F[1, 2] / (d[1] * d[2])
        @test isapprox(eigvals(Symmetric(F ./ (d * d'))),
                       sort([fill(1 - r, n - 1); 1 + (n - 1) * r]); rtol = 1e-12)
        # A positive definite start meets both bounds of target C. A singular start fails
        # the lower bound when the equally weighted portfolio has zero variance, although no
        # pair is perfectly correlated, and fails the upper bound when every variance is
        # equal and every correlation is one. Perfect correlation with unequal variances
        # fails neither.
        @test isposdef(C) && isposdef(F)
        x2 = randn(StableRNG(5), 50, 2)
        S0 = cov(hcat(x2, -sum(x2; dims = 2)))
        @test all(<(0.99), abs.(cor(hcat(x2, -sum(x2; dims = 2)))[[2, 3, 6]]))
        C0 = PO.shrinkage_target(CommonCovarianceTarget(), S0)
        @test isapprox(C0[1, 2], -C0[1, 1] / 2; rtol = 1e-12)
        singular(M) = minimum(eigvals(Symmetric(M))) <
                      1e-12 * maximum(eigvals(Symmetric(M)))
        @test singular(C0)
        @test singular(PO.shrinkage_target(CommonCovarianceTarget(), [1.0 1.0; 1.0 1.0]))
        @test isposdef(PO.shrinkage_target(CommonCovarianceTarget(), [1.0 2.0; 2.0 4.0]))
        # Target E, perfect positive correlation, has rank one, so no geodesic reaches it.
        E = sqrt.(diag(S)) * sqrt.(diag(S))'
        @test rank(E) == 1
        @test_throws DomainError GeodesicShrinkageCovariance(; tgt = E)
    end
end

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
include(joinpath(@__DIR__, "parity_grid.jl"))

const PO = PortfolioOptimisers

@testset "matrix_square_root: the three policies on a matrix of rank one" begin
    sigma = [1.0 1.0; 1.0 1.0]
    # `nothing` takes the plain Cholesky factor, which refuses the matrix.
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
    # The ridge takes the type of the plain factor (#1442). A `Float64` ridge widened a `Float32`
    # matrix to a `Float64` factor, and `eps` of a `Rational` threw a MethodError.
    L = matrix_square_root(RidgeCholeskySquareRoot(), Float32.(sigma))
    @test L isa LowerTriangular{Float32}
    # The first ridge, 1e-12, is below the floor `eps s` of `Float32`.
    @test maximum(abs, L * L' - sigma - eps(Float32) * I) <= 2 * eps(Float32)
    @test matrix_square_root(RidgeCholeskySquareRoot(), Rational.(sigma)) ==
          matrix_square_root(RidgeCholeskySquareRoot(), sigma)
    @test PO.safe_regime_cholesky(Float32.(sigma), 0.0f0) isa Cholesky{Float32}
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
    # Bit-equal on the development host. Another host's BLAS moves the last bits, so the
    # check reads the parity tolerance and not `==`. The check is `:array`, because a factor
    # entry near zero carries the round-off of the largest entry.
    @test parity_compare(L[off], o[off]; scale = :array, name = "Mirror chol").ok
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
    function mirror(; kwargs...)
        f = [cfg.factors;
             "macro2" =>
                 ObservedExposure(; xe = grid_pass("macro_beta"; family = "macro2"),
                                  series = "MACRO2", family = "macro2")]
        return CrossSectionalFactorPrior(; lambda = 1, cfg..., factors = f,
                                         f_mp = MatrixProcessing(; pdm = nothing),
                                         kwargs...)
    end
    # The plain Cholesky factor refuses the singular factor covariance.
    @test_throws PosDefException prior(mirror(; mtx_sqrt = nothing), rd)
    # The oracle's ridge is one keyword away.
    pr = prior(mirror(; mtx_sqrt = RidgeCholeskySquareRoot()), rd)
    load(o) = parity_load("CrossSectionalFactorPrior", "Mirror", o)
    # Measured 0. The check is `:array`, because the mirrored factor makes covariances that
    # cancel to near zero.
    @test parity_compare(pr.fpr.sigma, parity_load("MatrixSquareRoot", "Mirror", "Input");
                         scale = :array, name = "Mirror factor sigma").ok
    # Measured maxrel 8.8e-15, 2.2e-15 and 1.6e-15, and maxscaled 7.9e-16 on `sigma`. The
    # cell maxrel of `sigma` is 2.0e-14: an entry of `L F L'` near zero cancels, so that
    # check is `:array`. They were 3.0e-13 and 9.8e-12 when the lift repaired the systematic
    # block `L F L'`, which the oracle does not repair (#1576).
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
    # The default is the eigen policy, and it reproduces the covariance to rounding.
    @test mirror().mtx_sqrt === EigenFallbackSquareRoot()
    pe = prior(mirror(), rd)
    @test isapprox(transpose(pe.chol[:, i]) * pe.chol[:, i], pe.sigma[i, i]; rtol = 1e-12)
end

@testset "FactorPrior on a singular factor covariance" begin
    rng = StableRNG(1396)
    F = 0.01 .* randn(rng, 120, 3)
    # The fourth factor is the first negated, so the factor covariance is singular.
    F = hcat(F, -F[:, 1])
    X = F[:, 1:3] * (0.5 .+ rand(rng, 3, 6)) .+ 0.005 .* randn(rng, 120, 6)
    mp = MatrixProcessing(; pdm = nothing)
    function fp(; kwargs...)
        return FactorPrior(;
                           pe = EmpiricalPrior(;
                                               ce = PortfolioOptimisersCovariance(;
                                                                                  mp = mp)),
                           mp = mp, kwargs...)
    end
    @test FactorPrior().mtx_sqrt === EigenFallbackSquareRoot()
    @test_throws PosDefException prior(fp(; mtx_sqrt = nothing), X, F)
    pr = prior(fp(; mtx_sqrt = RidgeCholeskySquareRoot()), X, F)
    @test isapprox(pr.chol' * pr.chol, pr.sigma; rtol = 1e-11)
    pr = prior(fp(), X, F)
    @test isapprox(pr.chol' * pr.chol, pr.sigma; rtol = 1e-14)
    # On a positive definite factor covariance the default is the plain Cholesky factor.
    pp = prior(fp(), X, F[:, 1:3])
    @test pp.chol == prior(fp(; mtx_sqrt = nothing), X, F[:, 1:3]).chol
end

@testset "FactorBlackLittermanPrior on a singular factor covariance (#1732)" begin
    # The prior lifts its posterior through `factor_lift`, as `FactorPrior` does, so it holds
    # the same field. It took the keyword default of the lift before, which a caller could not
    # change. A view leaves the null direction of the factor covariance in the posterior.
    rng = StableRNG(1396)
    F = 0.01 .* randn(rng, 120, 3)
    F = hcat(F, -F[:, 1])
    X = F[:, 1:3] * (0.5 .+ rand(rng, 3, 6)) .+ 0.005 .* randn(rng, 120, 6)
    mp = MatrixProcessing(; pdm = nothing)
    nx = string.("A", 1:6)
    views = LinearConstraintEstimator(; val = ["F2 == 0.001"])
    function fbl(K; kwargs...)
        sets = UniverseSets(; dict = Dict("nx" => nx, "nf" => string.("F", 1:K)))
        return FactorBlackLittermanPrior(;
                                         pe = EmpiricalPrior(;
                                                             ce = PortfolioOptimisersCovariance(;
                                                                                                mp = mp)),
                                         f_mp = mp, mp = mp, views = views, sets = sets,
                                         kwargs...)
    end
    @test fbl(4).mtx_sqrt === EigenFallbackSquareRoot()
    @test_throws PosDefException prior(fbl(4; mtx_sqrt = nothing), X, F)
    pr = prior(fbl(4; mtx_sqrt = RidgeCholeskySquareRoot()), X, F)
    @test isapprox(pr.chol' * pr.chol, pr.sigma; rtol = 1e-11)
    pr = prior(fbl(4), X, F)
    @test isapprox(pr.chol' * pr.chol, pr.sigma; rtol = 1e-14)
    # On a positive definite posterior the default is the plain Cholesky factor.
    @test prior(fbl(3), X, F[:, 1:3]).chol ==
          prior(fbl(3; mtx_sqrt = nothing), X, F[:, 1:3]).chol
end

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
    # A normal radius applies the pseudo-inverse of the map, so it reads the eigen square
    # root of a singular matrix (#1526). The ridge adds directions that no sampled error
    # fills, so its radius is the same to the size of the ridge.
    function uk(alg)
        return NormalUncertaintySet(; pdm = nothing, rng = StableRNG(1), n_sim = 100,
                                    alg = NormBallUncertaintySetAlgorithm(;
                                                                          method = NormalKUncertaintyAlgorithm(),
                                                                          diagonal = false,
                                                                          mtx_sqrt = alg))
    end
    k_eig = PO.ucs(uk(EigenFallbackSquareRoot()), rd_ms)[2].kappa
    k_rdg = PO.ucs(uk(RidgeCholeskySquareRoot()), rd_ms)[2].kappa
    @test isfinite(k_eig)
    @test isapprox(k_eig, k_rdg; rtol = 1e-9)
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
