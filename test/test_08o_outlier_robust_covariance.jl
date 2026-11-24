#=
The robust covariance estimators of #2025: the minimum covariance determinant
(`MCDCovariance`, with the searches `DetMCD` and `FastMCD`) and Tyler's M-estimator
(`TylerCovariance`), with the location adapter `RobustExpectedReturns`.

The reference values come from R 4.5.3 with `robustbase` 0.99-7 (`covMcd` with
`nsamp = "deterministic"`, `Qn`) and `ICSNP` 1.1-3 (`tyler.shape` and `HR.Mest`, with a tight
tolerance). The other tests check the properties that define each estimator: the resistance
to outliers, the agreement with the sample covariance on clean data, the fixed points of the
C-step and of the Tyler equations, the closed forms of the factors, the determinism of the
searches and the refusals.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, Random

const POR = PortfolioOptimisers
const StatsAPI = PortfolioOptimisers.StatsAPI

function robust_test_sample(rng, n, p; nbad = 0, shift = 8)
    A = randn(rng, p, p)
    L = cholesky(A * A' + I).L
    X = Matrix((L * randn(rng, p, n))')
    clean = copy(X)
    bad = sort(randperm(rng, n)[1:nbad])
    X[bad, :] .= shift .+ randn(rng, nbad, p)
    return X, clean, bad
end
relerr(A, B) = norm(A - B) / norm(B)
# The samples that R read. The outliers are the first rows.
function r_parity_sample(seed, n, p, nbad)
    rng = StableRNG(seed)
    A = randn(rng, p, p)
    X = Matrix((cholesky(A * A' + I).L * randn(rng, p, n))')
    X[1:nbad, :] .= 8 .+ 0.5 .* randn(rng, nbad, p)
    return X
end

@testset "Robust covariance estimators (#2025)" begin
    @testset "Parity with R" begin
        X = r_parity_sample(11, 120, 3, 12)
        raw = StatsAPI.fit(MCDCovariance(; rw = nothing), X)
        @test raw.mu ≈ [0.352054507387890514, -0.075191857287023711, 0.329141313421352999]
        @test raw.sigma ≈ [5.4742125785071618 2.2349347443650411 4.7046195841581495
                           2.2349347443650411 3.6824108285441408 3.3253952751535536
                           4.7046195841581504 3.3253952751535540 6.8925701452664683]
        @test cov(MCDCovariance(; rw = nothing, ssc = nothing), X) ≈
              [5.0484860790926884 2.0611250993260497 4.3387439083570065
               2.0611250993260497 3.3960317650789866 3.0667811147866533
               4.3387439083570074 3.0667811147866542 6.3565387584997506]
        r = StatsAPI.fit(MCDCovariance(), X)
        @test r.mu ≈ [0.1802475948173479781, -0.0044548433328605924, 0.1743586070336486660]
        @test r.sigma ≈ [5.1717619062118718 2.0737125297779788 5.0771151244300698
                         2.0737125297779788 3.0438061177501177 3.2061450946704912
                         5.0771151244300698 3.2061450946704912 7.6587564976802840]
        @test cov(MCDCovariance(; ssc = nothing), X) ≈
              [5.1596263219182488 2.0717872727430215 4.9784124396987455
               2.0717872727430215 3.0618886024095451 3.1836323356261209
               4.9784124396987455 3.1836323356261209 7.3582957839725003]
        # A subset fraction between the two that the small-sample factors were simulated at.
        Y = r_parity_sample(16, 60, 4, 6)
        raw = StatsAPI.fit(MCDCovariance(; alpha = 0.75, rw = nothing), Y)
        @test raw.sigma ≈
              [3.16671953242966975 -2.70165077070260606 -1.38588523212224768 -1.11958771161179449
               -2.70165077070260606 8.14000619909545087 4.66531443347527386 0.88349719114699732
               -1.38588523212224835 4.66531443347527564 12.25661501136657705 -0.91103865362331127
               -1.11958771161179449 0.88349719114699732 -0.91103865362331127 2.35829958805300066]
        r = StatsAPI.fit(MCDCovariance(; alpha = 0.75), Y)
        @test r.mu ≈ [-0.121367381516236980, -0.177936158979843495, 0.069576080185424369,
                      -0.016113701108301088]
        @test r.sigma ≈
              [2.79514864506473648 -2.22131959020888514 -0.37693572791331581 -1.04759875659915047
               -2.22131959020888514 5.98272024931919066 2.47381824547427254 0.85774833668370942
               -0.37693572791331581 2.47381824547427254 10.63364148587304570 -1.10694445130068897
               -1.04759875659915047 0.85774833668370942 -1.10694445130068897 2.33570923377717943]
        # The shape of `ICSNP` has the determinant one. The stop rule of the library is
        # `sqrt(eps)` on a step, so the fit is near to the limit of the iteration by 1e-7.
        shape(V) = V / det(V)^(1 / size(V, 1))
        V = last(POR.tyler_fixed_point(X, nothing, vec(median(X; dims = 1)), nothing))
        @test isapprox(shape(V),
                       [1.88859915647789878 0.94601098862414912 1.74109641251885061
                        0.94601098862414912 1.24327631827079554 1.26499011224387026
                        1.74109641251885106 1.26499011224387026 2.49388847143658277];
                       rtol = 1e-6)
        mu, V = POR.tyler_fixed_point(X, nothing, nothing, nothing)
        @test isapprox(mu, [0.64721712353539251, 0.33552574391256379, 0.60611848121963507];
                       rtol = 1e-6)
        @test isapprox(shape(V),
                       [2.1813803005688044 1.1462099349900512 2.0445149617704170
                        1.1462099349900507 1.4050267122619411 1.4201641036756165
                        2.0445149617704166 1.4201641036756167 2.6363271826510792];
                       rtol = 1e-6)
        x7 = [0.918977371608218241, 0.782136300731067102, 0.074564983365190601,
              -1.989351695863372793, 0.619825747894710233, -0.056128739529000785,
              -0.155795506705329295]
        @test POR.qn_scale(x7) ≈ 0.570102480471806450
        x14 = [-0.54252003099165036, 1.20786780598317223, 1.16040261569495162,
               0.70021364951499820, 1.58683345454084557, 0.55848642556530392,
               -1.27659220845803656, -0.57326541423688626, -1.22461261489835582,
               -0.47340063643931157, -0.62036667722412420, 0.04211587314423524,
               -0.91092164855244551, 0.15802877240407498]
        @test POR.qn_scale(x14) ≈ 1.07265984690971816
    end
    @testset "Qn scale" begin
        rng = StableRNG(2025)
        function brute_qn(x)
            n = length(x)
            d = sort([abs(x[i] - x[j]) for i in 1:n for j in (i + 1):n])
            h = div(n, 2) + 1
            return d[div(h * (h - 1), 2)] * 2.21914 * POR.qn_finite_factor(n)
        end
        for _ in 1:300
            n = rand(rng, 2:70)
            x = rand(rng, Bool) ? randn(rng, n) : round.(randn(rng, n); digits = 1)
            @test POR.qn_scale(x) ≈ brute_qn(x)
        end
        @test iszero(POR.qn_scale([1.0]))
        # More than half of the values are equal: the quantile rule gives a positive scale.
        z = vcat(zeros(30), randn(rng, 10))
        @test iszero(POR.qn_scale(z))
        @test POR.robust_column_scale(z) > 0
        @test POR.robust_column_scale(zeros(10)) == 1
        # The weighted high median against a sort, with tied values and a longer buffer.
        function sorted_high_median(a)
            s = sort(a; by = first)
            c = cumsum(last.(s))
            return first(s[findfirst(x -> 2 * x >= last(c), c)])
        end
        for _ in 1:300
            m = rand(rng, 1:40)
            v = rand(rng, Bool) ? randn(rng, m) : Float64.(rand(rng, 0:5, m))
            a = collect(zip(v, rand(rng, 1:9, m)))
            @test POR.weighted_high_median!([a; (-9.0, 99)], similar(a), m) ==
                  sorted_high_median(a)
        end
        # A trial allocates nothing, because it reads and writes the buffers alone.
        y = sort(randn(rng, 200))
        a = Vector{Tuple{Float64, Int}}(undef, 200)
        b = similar(a)
        L, R = zeros(Int, 200), collect(199:-1:0)
        trial_bytes(a, b, y, L, R) = @allocated POR.pairwise_trial_value!(a, b, y, L, R)
        trial_bytes(a, b, y, L, R)
        @test iszero(trial_bytes(a, b, y, L, R))
    end
    @testset "Factors" begin
        # Croux-Haesbroeck at p = 2, a = 1/2 has a closed form: q = 2 log 2 and
        # F_{chi2_4}(q) = 1 - e^{-q/2} (1 + q/2).
        q = 2 * log(2)
        @test POR.mcd_consistency(2, 0.5) ≈ 0.5 / (1 - exp(-q / 2) * (1 + q / 2))
        @test POR.mcd_consistency(4, 1.0) == 1
        @test POR.mcd_subset_size(nothing, 0.5, 100, 5) == 53
        @test POR.mcd_subset_size(nothing, 1, 100, 5) == 100
        for p in (1, 2, 5), kind in (:raw, :rew)
            f = POR.mcd_correction(MCDSmallSampleCorrection(), p, 60, 0.5, kind)
            @test f > 1
            @test POR.mcd_correction(MCDSmallSampleCorrection(), p, 60, 0.7, kind) > 1
            @test POR.mcd_correction(MCDSmallSampleCorrection(), p, 60, 1.0, kind) ≈ 1
            @test POR.mcd_correction(MCDSmallSampleCorrection(), p, 10^6, 0.5, kind) ≈ 1 atol = 1e-2
        end
        @test POR.mcd_correction(nothing, 3, 60, 0.5, :raw) == 1
    end
    @testset "Outliers" begin
        rng = StableRNG(1)
        X, clean, bad = robust_test_sample(rng, 400, 5; nbad = 20)
        S = cov(clean[setdiff(1:400, bad), :])
        @test relerr(cov(X), S) > 0.5
        for ce in (MCDCovariance(), MCDCovariance(; alg = FastMCD(; seed = 3)),
                   MCDCovariance(; rw = nothing), TylerCovariance())
            r = StatsAPI.fit(ce, X)
            @test relerr(r.sigma, S) < min(0.3, relerr(cov(X), S) / 3)
            @test bad ⊆ findall(r.outlier)
            @test cov(ce, X) == r.sigma
            @test cor(ce, X) ≈ cov2cor(r.sigma)
            @test vec(var(ce, X)) ≈ diag(r.sigma)
        end
        r = StatsAPI.fit(MCDCovariance(), X)
        @test all(iszero, r.w[bad])
        @test sum(r.w) > 0.9 * 380
        raw = StatsAPI.fit(MCDCovariance(; rw = nothing), X)
        @test sum(raw.w) == POR.mcd_subset_size(nothing, 0.5, 400, 5)
        rt = StatsAPI.fit(TylerCovariance(), X)
        @test rt.w ≈ 5 ./ rt.d2
        @test all(rt.w[bad] .< 0.2)
    end
    @testset "Clean data" begin
        X, = robust_test_sample(StableRNG(5), 600, 4)
        for ce in (MCDCovariance(), TylerCovariance(), MCDCovariance(; ssc = nothing))
            @test relerr(cov(ce, X), cov(X)) < 0.1
            @test norm(mean(RobustExpectedReturns(; ce = ce), X) - mean(X; dims = 1)) < 0.15
        end
    end
    @testset "Searches" begin
        X, = robust_test_sample(StableRNG(7), 150, 3; nbad = 20, shift = 5)
        Z = first(POR.mcd_standardise(X, nothing))
        h = POR.mcd_subset_size(nothing, 0.5, 150, 3)
        idx = first(POR.mcd_search(DetMCD(), POR.mcd_sample(nothing, X, nothing), h, 1.0))
        # The best subset of the search is a fixed point of the C-step.
        @test first(POR.mcd_csteps(Z, idx, h, false, nothing, 1)) == idx
        # A C-step never raises the determinant.
        start = collect(1:h)
        _, o0 = POR.mcd_csteps(Z, start, h, false, nothing, 0)
        _, o1 = POR.mcd_csteps(Z, start, h, false, nothing, 1)
        _, o9 = POR.mcd_csteps(Z, start, h, false, nothing, 200)
        @test o9 <= o1 <= o0
        # A seed makes FastMCD reproducible, and no seed draws from the generator.
        fm = MCDCovariance(; alg = FastMCD(; n_starts = 50, seed = 11))
        @test cov(fm, X) == cov(fm, X)
        free = MCDCovariance(; alg = FastMCD(; n_starts = 50, rng = StableRNG(3)))
        @test isapprox(cov(free, X), cov(fm, X); rtol = 0.2)
        # A start that is singular grows until it is not: 40 % of the rows are one point.
        Y = randn(StableRNG(9), 100, 2)
        Y[1:40, :] .= 0.3
        @test all(isfinite, cov(MCDCovariance(; alg = FastMCD(; seed = 1)), Y))
        # The exact search of one variable, with a free and a fixed centre.
        x = X[:, 1:1]
        r1 = StatsAPI.fit(MCDCovariance(; rw = nothing, ssc = nothing), x)
        hx = POR.mcd_subset_size(nothing, 0.5, 150, 1)
        sx = sort(vec(x))
        best = minimum(var(sx[k:(k + hx - 1)]) for k in 1:(150 - hx + 1))
        @test r1.sigma[1] ≈ best * POR.mcd_consistency(1, hx / 150)
        rf = StatsAPI.fit(MCDCovariance(; me = MedianExpectedReturns(), rw = nothing), x)
        @test rf.mu == vec(median(x; dims = 1))
        # alpha = 1 is the classical estimate.
        ra = StatsAPI.fit(MCDCovariance(; alpha = 1, rw = nothing, ssc = nothing), X)
        @test ra.sigma ≈ cov(X)
        @test ra.mu ≈ vec(mean(X; dims = 1))
        # Data on a hyperplane have no robust scatter.
        H = randn(StableRNG(4), 60, 3)
        H[:, 3] .= H[:, 1] .+ H[:, 2]
        @test_throws DomainError cov(MCDCovariance(), H)
        # A random start can end at a subset whose covariance is singular in the arithmetic. Its
        # determinant is then zero, and the fit refuses it.
        for s in 1:40
            H = randn(StableRNG(s), 60, 4)
            H[:, 4] .= H[:, 1] .+ 0.3 .* H[:, 2] .- 1.7 .* H[:, 3]
            @test_throws "hyperplane" cov(MCDCovariance(;
                                                        alg = FastMCD(; seed = 1,
                                                                      n_starts = 10)), H)
        end
        @test isnothing(POR.scatter_cholesky([1.0 1.0; 1.0 1.0]))
        C = randn(StableRNG(4), 60, 3)
        C[:, 2] .= 0.7
        for ce in (MCDCovariance(), MCDCovariance(; alg = FastMCD(; seed = 1)),
                   MCDCovariance(; me = MedianExpectedReturns()))
            @test_throws "column 2 of the data is constant" cov(ce, C)
        end
        # The scale of a column with more than half of its values equal keeps the element type.
        F = Float32.(C)
        F[1:40, 2] .= 0.0f0
        F[41:60, 2] .= randn(StableRNG(6), Float32, 20)
        @test eltype(first(POR.mcd_standardise(F, nothing))) == Float32
    end
    @testset "Outliers of any size" begin
        # A bad tick can be many orders of magnitude larger than a return.
        X = r_parity_sample(5, 300, 4, 0)
        S = cov(X[16:end, :])
        for m in (1e4, 1e8, 1e12)
            Y = copy(X)
            Y[1:15, :] .= m .+ randn(StableRNG(1), 15, 4)
            Y[1, 1] = -3 * m
            for ce in
                (MCDCovariance(), MCDCovariance(; alg = FastMCD(; seed = 1, n_starts = 50)),
                 TylerCovariance())
                @test relerr(cov(ce, Y), S) < 0.2
            end
            # The subset of one variable does not depend on the size of an outlier.
            y = Y[:, 1:1]
            h = POR.mcd_subset_size(nothing, 0.5, 300, 1)
            sy = sort(vec(y))
            best = minimum(var(sy[k:(k + h - 1)]) for k in 1:(300 - h + 1))
            @test cov(MCDCovariance(; rw = nothing, ssc = nothing), y)[1] ≈
                  best * POR.mcd_consistency(1, h / 300)
        end
    end
    @testset "Tyler" begin
        X, = robust_test_sample(StableRNG(5), 300, 4)
        X .+= [0.1 -0.2 0.3 0.0]
        mu, V = POR.tyler_fixed_point(X, nothing, nothing, nothing)
        D = X .- mu'
        d2 = POR.squared_distances(X, mu, V)
        Vfp = 4 * (D' * (D ./ d2)) / 300
        @test isapprox(Vfp, V; rtol = 1e-6)
        @test norm(vec(sum(D ./ sqrt.(d2); dims = 1))) < 1e-5
        # Integer weights are the rows repeated by their weights.
        Y = X[1:60, :]
        k = rand(StableRNG(2), 1:3, 60)
        Yr = reduce(vcat, [repeat(Y[i:i, :], k[i]) for i in 1:60])
        @test cov(TylerCovariance(; w = fweights(k)), Y) ≈ cov(TylerCovariance(), Yr)
        # Unit weights hold no vector of values, and they are the unweighted fit.
        @test cov(TylerCovariance(; w = uweights(60)), Y) ≈ cov(TylerCovariance(), Y)
        # A fixed centre, a variance estimator, and an observation at the centre.
        m0 = vec(median(X; dims = 1))
        @test StatsAPI.fit(TylerCovariance(; me = MedianExpectedReturns()), X).mu == m0
        @test StatsAPI.fit(TylerCovariance(), X; mean = m0').mu == m0
        tv = cov(TylerCovariance(; ve = SimpleVariance()), X)
        @test sqrt.(diag(tv)) ≈ vec(std(X; dims = 1))
        # An estimator of another package has no `std`, so the library reads its diagonal.
        @test cov(TylerCovariance(; ve = StatsBase.SimpleCovariance(; corrected = true)),
                  X) ≈ tv
        Xc = vcat(X, reshape(StatsAPI.fit(TylerCovariance(), X).mu, 1, :))
        @test all(isfinite, cov(TylerCovariance(), Xc))
        Xm = vcat(X, repeat(reshape(m0, 1, :), 3))
        rm = StatsAPI.fit(TylerCovariance(; me = MedianExpectedReturns()), Xm)
        # An observation at the centre leaves the fixed point, so its weight is zero.
        @test findall(iszero, rm.w) == 301:303
        @test all(isfinite, rm.w)
        # The shape is exactly symmetric, as `cholesky` needs it.
        @test issymmetric(cov(TylerCovariance(), X))
        @test isposdef(cov(TylerCovariance(; ve = SimpleVariance()), X))
        # A denoiser reads the count of the weights.
        ew = eweights(1:300, 0.05)
        count_of = POR.effective_observation_count
        @test count_of(TylerCovariance(), X) == 300
        @test count_of(TylerCovariance(; w = ew), X) ≈ sum(ew)^2 / sum(abs2, ew)
        @test count_of(RobustExpectedReturns(; ce = TylerCovariance(; w = ew)), X) ≈
              sum(ew)^2 / sum(abs2, ew)
        for ce in (MCDCovariance(; me = MedianExpectedReturns(; w = ew)),
                   TylerCovariance(; me = MedianExpectedReturns(; w = ew)))
            @test count_of(RobustExpectedReturns(; ce = ce), X) ≈ sum(ew)^2 / sum(abs2, ew)
            @test count_of(ce, X) == 300
        end
        @test POR.observation_median([3.0, 1.0, 2.0], nothing) == 2.0
        # Half of the rows on a line through the centre: the shape does not exist, and the
        # iteration warns.
        Yb = vcat(hcat(randn(StableRNG(1), 10), zeros(10)), randn(StableRNG(2), 10, 2))
        @test_logs (:warn, r"did not converge") POR.tyler_fixed_point(Yb, nothing, zeros(2),
                                                                      nothing)
    end
    @testset "Element type, orientation and composition" begin
        X, = robust_test_sample(StableRNG(8), 200, 3)
        X32 = Float32.(X)
        for ce in (MCDCovariance(), TylerCovariance(),
                   MCDCovariance(; me = SimpleExpectedReturns()))
            @test eltype(cov(ce, X32)) == Float32
            @test cov(ce, Matrix(X'); dims = 2) ≈ cov(ce, X)
        end
        @test eltype(StatsAPI.fit(TylerCovariance(), X32).w) == Float32
        @test size(mean(RobustExpectedReturns(), X; dims = 1)) == (1, 3)
        @test size(mean(RobustExpectedReturns(), Matrix(X'); dims = 2)) == (3, 1)
        pr = prior(EmpiricalPrior(; me = RobustExpectedReturns(), ce = MCDCovariance()), X)
        @test pr.mu ≈ vec(mean(RobustExpectedReturns(), X))
        @test pr.sigma ≈ cov(MCDCovariance(), X)
        for ce in (MCDCovariance(), TylerCovariance())
            on = partial_fit!(POR.update_online_estimator(Online(ce)), X[1:120, :])
            on = partial_fit!(on, X[121:end, :])
            @test cov(on) ≈ cov(ce, X)
            @test !POR.supports_partial_fit(ce)
        end
        # The weights of a factory call reach `me` of MCD and `w` of Tyler.
        @test POR.show_fields(MCDCovariance()) == (:me, :alg, :alpha, :shr, :ssc, :rw, :q)
        @test POR.show_fields(TylerCovariance()) == (:me, :ve, :shr, :w, :q)
        w = eweights(1:200, 0.05)
        @test POR.factory(MCDCovariance(; me = MedianExpectedReturns()), w).me.w === w
        @test POR.factory(TylerCovariance(), w).w === w
        @test POR.factory(RobustExpectedReturns(; ce = TylerCovariance()), w).ce.w === w
        v = POR.port_opt_view(TylerCovariance(; me = MedianExpectedReturns()), 1:2)
        @test v isa TylerCovariance
    end
    @testset "Refusals" begin
        rng = StableRNG(3)
        @test_throws DomainError cov(MCDCovariance(), randn(rng, 6, 8))
        @test_throws DomainError cov(TylerCovariance(), randn(rng, 8, 8))
        w = fweights(vcat(ones(8), zeros(20)))
        @test_throws DomainError cov(TylerCovariance(; w = w), randn(rng, 28, 8))
        @test_throws DimensionMismatch cov(TylerCovariance(; w = w), randn(rng, 30, 8))
        # The fitted small-sample factor has no positive value for a very small sample.
        Xs = randn(rng, 10, 5)
        @test_throws "ssc = nothing" cov(MCDCovariance(), Xs)
        @test isposdef(cov(MCDCovariance(; ssc = nothing), Xs))
        @test_throws DomainError MCDCovariance(; alpha = 0.4)
        @test_throws DomainError MCDCovariance(; q = 1.0)
        @test_throws DomainError MCDReweighting(; q = 0.0)
        @test_throws DomainError TylerCovariance(; q = 1.5)
        @test_throws DomainError FastMCD(; n_starts = 0)
        @test_throws DomainError FastMCD(; n_starts = 5, n_best = 10)
        @test_throws DimensionMismatch RobustCovarianceResult(; mu = [0.0],
                                                              sigma = ones(2, 2),
                                                              d2 = [1.0], w = [1.0],
                                                              outlier = BitVector([0]))
        @test_throws DimensionMismatch RobustCovarianceResult(; mu = [0.0],
                                                              sigma = ones(1, 1),
                                                              d2 = [1.0, 2.0], w = [1.0],
                                                              outlier = BitVector([0]))
    end
end
