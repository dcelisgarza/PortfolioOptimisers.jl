using PortfolioOptimisers, Test, NearestCorrelationMatrix, LinearAlgebra

# Card 7 of the 2026-08-17 maintainability review: the out-of-place verbs are documented
# as an optional part of the extension interface, so an estimator that declares only the
# in-place form must inherit a copying fallback. These estimators declare nothing else.
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
    @testset "Ticket 447: posdef! warns and returns the unrepaired matrix" begin
        X = [1.0 2.0; 2.0 1.0]
        est = Posdef(; alg = Ticket447Stub(), kwargs = (; ensure_pd = false))
        r = @test_logs (:warn, "Matrix could not be made positive definite.") posdef!(est,
                                                                                      copy(X))
        @test r == X
        @test !LinearAlgebra.isposdef(r)
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
