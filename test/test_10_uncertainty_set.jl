@testset "Uncertainty set" begin
    using PortfolioOptimisers, Test, DataFrames, CSV, TimeSeries, StableRNGs, Random,
          Statistics, LinearAlgebra
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    @testset "L1 Uncertainty sets" begin
        # Build returns whose sample mean and standard deviation are exactly known, so the
        # activation ladders of Zhou & Palomar's closed forms can be written out by hand.
        rng = StableRNG(123456789)
        N = 20
        mu_t = round.(sort(randn(rng, N) .* 0.02 .+ 0.05; rev = true); digits = 4)
        sd_t = round.(0.1 .+ 0.15 .* rand(rng, N); digits = 4)
        X = randn(rng, 2000, N)
        X .= (X .- mean(X; dims = 1)) ./ std(X; dims = 1) .* sd_t' .+ mu_t'
        rdl1 = ReturnsResult(; X = X, nx = string.("A", 1:N))
        pr = prior(EmpiricalPrior(), rdl1.X)
        sd_hat = sqrt.(diag(pr.sigma))
        # Lemma 2 / Lemma 9 ladders: the radius at which the k-th asset activates.
        g(k) = sum(mu_t[i] - mu_t[k] for i in 1:k)
        gs(k) = sum((mu_t[i] - mu_t[k]) / sd_hat[i] for i in 1:k)
        @testset "Result construction and validation" begin
            @test isnothing(L1UncertaintySet(; eps = 0.1).sd)
            @test L1UncertaintySet(; eps = 0.1).eps == 0.1
            @test_throws DomainError L1UncertaintySet(; eps = -0.1)
            @test_throws DomainError L1UncertaintySet(; eps = Inf)
            @test_throws DomainError L1UncertaintySet(; eps = 0.1, sd = [1.0, -1.0])
            @test_throws Exception L1UncertaintySet(; eps = 0.1, sd = Float64[])
            @test SignedL1UncertaintySet(; ep = 0.1, en = 0.2).ep == 0.1
            @test SignedL1UncertaintySet(; ep = 0.1, en = 0.2).en == 0.2
            @test_throws DomainError SignedL1UncertaintySet(; ep = -0.1, en = 0.2)
            @test_throws DomainError SignedL1UncertaintySet(; ep = 0.1, en = -0.2)
            @test_throws DomainError L1UncertaintySetAlgorithm(; method = -1.0)
            @test_throws DomainError SignedL1UncertaintySetAlgorithm(; mp = -1.0)
            @test_throws DomainError SignedL1UncertaintySetAlgorithm(; mm = -1.0)
            @test_throws DomainError ActiveAssetsUncertaintyAlgorithm(; active = 0)
            @test_throws DomainError ActiveAssetsUncertaintyAlgorithm(; active = 0.0)
            @test_throws DomainError ActiveAssetsUncertaintyAlgorithm(; active = 1.0)
            # alg must be one of the two L1 shapes.
            @test_throws ArgumentError CharacteristicUncertaintySet(;
                                                                    alg = BoxUncertaintySetAlgorithm())
        end
        @testset "Mean-only: ucs and sigma_ucs throw" begin
            # The l1 ball bounds a characteristic vector; the paper defines no covariance
            # analogue, so these exist only to say so.
            ue = CharacteristicUncertaintySet()
            @test_throws ArgumentError ucs(ue, rdl1)
            @test_throws ArgumentError sigma_ucs(ue, rdl1)
            @test_throws ArgumentError ucs(ue, rdl1.X)
            @test_throws ArgumentError sigma_ucs(ue, rdl1.X)
            @test isa(mu_ucs(ue, rdl1), L1UncertaintySet)
        end
        @testset "Explicit radius passes through" begin
            ue = CharacteristicUncertaintySet(;
                                              alg = L1UncertaintySetAlgorithm(;
                                                                              method = 0.0123))
            s = mu_ucs(ue, rdl1)
            @test s.eps == 0.0123
            @test isnothing(s.sd)
            ues = CharacteristicUncertaintySet(;
                                               alg = SignedL1UncertaintySetAlgorithm(;
                                                                                     mp = 0.02,
                                                                                     mm = 0.03,
                                                                                     scaled = false))
            ss = mu_ucs(ues, rdl1)
            @test (ss.ep, ss.en) == (0.02, 0.03)
        end
        @testset "scaled toggles the sigma scaling (S vs A1)" begin
            unscaled = mu_ucs(CharacteristicUncertaintySet(;
                                                           alg = L1UncertaintySetAlgorithm(;
                                                                                           method = 0.01,
                                                                                           scaled = false)),
                              rdl1)
            scaled = mu_ucs(CharacteristicUncertaintySet(;
                                                         alg = L1UncertaintySetAlgorithm(;
                                                                                         method = 0.01,
                                                                                         scaled = true)),
                            rdl1)
            @test isnothing(unscaled.sd)
            @test isapprox(scaled.sd, sd_hat)
        end
        @testset "Corollary 4: radius calibrated to a target active count" begin
            # active as a count and as a fraction of the universe must agree.
            for q in 1:(N - 1)
                ue = CharacteristicUncertaintySet(;
                                                  alg = L1UncertaintySetAlgorithm(;
                                                                                  method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                            active = q)))
                @test isapprox(mu_ucs(ue, rdl1).eps, (g(q) + g(q + 1)) / 2)
            end
            frac = mu_ucs(CharacteristicUncertaintySet(;
                                                       alg = L1UncertaintySetAlgorithm(;
                                                                                       method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                                 active = 0.2))),
                          rdl1)
            @test isapprox(frac.eps, (g(4) + g(5)) / 2)
            # Top of the ladder: any radius above g(N) activates everything, so the
            # calibration continues the final increment rather than sitting on the boundary.
            top = mu_ucs(CharacteristicUncertaintySet(;
                                                      alg = L1UncertaintySetAlgorithm(;
                                                                                      method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                                active = N))),
                         rdl1)
            @test top.eps > g(N)
        end
        @testset "Corollary 11: scaled calibration uses the volatility-adjusted ladder" begin
            ue = CharacteristicUncertaintySet(;
                                              alg = L1UncertaintySetAlgorithm(;
                                                                              method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                        active = 4),
                                                                              scaled = true))
            s = mu_ucs(ue, rdl1)
            @test isapprox(s.eps, (gs(4) + gs(5)) / 2)
            # The scaled ladder is a genuinely different one.
            @test !isapprox(s.eps, (g(4) + g(5)) / 2)
        end
        @testset "Corollary 7: paired calibration activates long/short pairs" begin
            f(m) = sum(mu_t[i] - mu_t[m] for i in 1:m) +
                   sum(mu_t[N - m + 1] - mu_t[N - j + 1] for j in 1:m)
            # active counts assets; each paired ladder entry activates two.
            ue = CharacteristicUncertaintySet(;
                                              alg = L1UncertaintySetAlgorithm(;
                                                                              method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                        active = 8),
                                                                              paired = true))
            @test isapprox(mu_ucs(ue, rdl1).eps, (f(4) + f(5)) / 2)
            # The paired ladder differs from the long-only one.
            unpaired = CharacteristicUncertaintySet(;
                                                    alg = L1UncertaintySetAlgorithm(;
                                                                                    method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                              active = 8),
                                                                                    paired = false))
            @test !isapprox(mu_ucs(ue, rdl1).eps, mu_ucs(unpaired, rdl1).eps)
        end
        @testset "Ties across the cut are rejected, not silently mis-calibrated" begin
            # Assumption 1 of the paper excludes ties; a tied cut makes the closed form's
            # open interval empty, so no radius can deliver the requested count.
            Xt = copy(X)
            Xt[:, 2] .= Xt[:, 1]                      # assets 1 and 2 now identical
            rdt = ReturnsResult(; X = Xt, nx = string.("A", 1:N))
            ue = CharacteristicUncertaintySet(;
                                              alg = L1UncertaintySetAlgorithm(;
                                                                              method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                        active = 1)))
            @test_throws DomainError mu_ucs(ue, rdt)
        end
    end
    @testset "Box Uncertainty sets" begin
        rng = StableRNG(123456789)
        ues = [DeltaUncertaintySet(;),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                    alg = BoxUncertaintySetAlgorithm(), seed = 987654321),
               ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), rng = rng,
                                  bootstrap = StationaryBootstrap(), seed = 987654321),
               ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), rng = rng,
                                  bootstrap = MovingBootstrap(), seed = 987654321),
               ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), rng = rng,
                                  bootstrap = CircularBootstrap(), seed = 987654321)]
        df = CSV.read(joinpath(@__DIR__, "assets/BoxUncertaintySet.csv.gz"), DataFrame)
        for (i, ue) in pairs(ues)
            mu_set1, sigma_set1 = ucs(ue, rd.X)
            mu1 = [mu_set1.lb; mu_set1.ub]
            sigma1 = [vec(sigma_set1.lb); vec(sigma_set1.ub)]

            mu_set2 = mu_ucs(ue, rd.X)
            mu2 = [mu_set2.lb; mu_set2.ub]

            sigma_set2 = sigma_ucs(ue, rd.X)
            sigma2 = [vec(sigma_set2.lb); vec(sigma_set2.ub)]

            success = isapprox(mu1, mu2)
            if !success
                println("Counter: $i")
                find_tol(mu1, mu2)
            end
            @test success

            success = isapprox(sigma1, sigma2)
            if !success
                println("Counter: $i")
                find_tol(sigma1, sigma2)
            end
            @test success

            success = isapprox([mu1; sigma1], df[!, i])
            if !success
                println("Dataframe iteration $i failed")
                find_tol([mu1; sigma1], df[!, i]; name1 = :sigma1, name2 = :sigma2)
            end
            @test success
        end
    end

    @testset "Ellipsoidal Uncertainty sets" begin
        rng = StableRNG(123456789)
        ues = [NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = true,
                                                                             method = NormalKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = false,
                                                                             method = GeneralKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = true,
                                                                             method = ChiSqKUncertaintyAlgorithm())),
               NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng, seed = 987654321,
                                    alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                             diagonal = true,
                                                                             method = 10)),
               ARCHUncertaintySet(;
                                  alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                           diagonal = true,
                                                                           method = NormalKUncertaintyAlgorithm()),
                                  rng = rng, seed = 987654321,
                                  bootstrap = StationaryBootstrap()),
               ARCHUncertaintySet(;
                                  alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                           diagonal = false,
                                                                           method = GeneralKUncertaintyAlgorithm()),
                                  rng = rng, seed = 987654321,
                                  bootstrap = MovingBootstrap()),
               ARCHUncertaintySet(;
                                  alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                           diagonal = true,
                                                                           method = ChiSqKUncertaintyAlgorithm()),
                                  rng = rng, seed = 987654321,
                                  bootstrap = CircularBootstrap())]
        df = CSV.read(joinpath(@__DIR__, "assets/EllipsoidalUncertaintySet.csv.gz"),
                      DataFrame)
        for (i, ue) in pairs(ues)
            mu_set1, sigma_set1 = ucs(ue, rd.X)
            mu1 = [vec(mu_set1.sigma); mu_set1.k]
            sigma1 = [vec(sigma_set1.sigma); sigma_set1.k]

            mu_set2 = mu_ucs(ue, rd.X)
            mu2 = [vec(mu_set2.sigma); mu_set2.k]
            sigma_set2 = sigma_ucs(ue, rd.X)
            sigma2 = [vec(sigma_set2.sigma); sigma_set2.k]

            success = isapprox(mu1, mu2; rtol = 1e-6)
            if !success
                println("Mu iteration $i failed")
                find_tol(mu1, mu2)
            end
            @test success

            res = [mu1; sigma1; mu2; sigma2]
            success = isapprox(res, df[!, i]; rtol = 1e-6)
            if !success
                println("Data iteration $i failed")
                find_tol(res, df[!, i])
            end
            @test success
        end
    end
    @testset "Extraction primitives" begin
        # Shared post-sampling kernels used by every ucs / mu_ucs / sigma_ucs body.
        q = 0.25
        # box_quantile_bounds: symmetric element-wise quantile fill via an accessor.
        samples = Dict((1, 1) => [1.0, 2.0, 3.0, 4.0], (2, 2) => [10.0, 20.0, 30.0, 40.0],
                       (1, 2) => [5.0, 6.0, 7.0, 8.0])
        get_ij = (i, j) -> samples[(min(i, j), max(i, j))]
        lb, ub = PortfolioOptimisers.box_quantile_bounds(Float64, get_ij, 2, q, (;))
        @test lb[1, 1] == quantile([1.0, 2.0, 3.0, 4.0], q)
        @test ub[2, 2] == quantile([10.0, 20.0, 30.0, 40.0], 1 - q)
        @test lb[1, 2] == lb[2, 1] == quantile([5.0, 6.0, 7.0, 8.0], q)          # symmetric
        @test ub[1, 2] == ub[2, 1] == quantile([5.0, 6.0, 7.0, 8.0], 1 - q)
        @test eltype(lb) === Float64

        # vec_quantile_bounds: per-component quantile bounds over an N × M sample matrix.
        mus = [1.0 2.0 3.0 4.0; 10.0 20.0 30.0 40.0]
        vlb, vub = PortfolioOptimisers.vec_quantile_bounds(mus, q, (;))
        @test vlb ==
              [quantile([1.0, 2.0, 3.0, 4.0], q), quantile([10.0, 20.0, 30.0, 40.0], q)]
        @test vub == [quantile([1.0, 2.0, 3.0, 4.0], 1 - q),
                      quantile([10.0, 20.0, 30.0, 40.0], 1 - q)]

        # ellipsoidal_set: optional diagonalisation + k fit + class tag, one k-method at a time.
        cov = [2.0 0.5; 0.5 3.0]
        Xs = randn(StableRNG(1), 50, 2)
        # Number method returns k == the number verbatim; cov passed through untouched.
        s = PortfolioOptimisers.ellipsoidal_set(false, 5, q, nothing, cov,
                                                MuEllipsoidalUncertaintySet())
        @test s.k == 5
        @test s.sigma == cov
        @test s.class isa MuEllipsoidalUncertaintySet
        # diagonal = true restricts cov to its diagonal before fitting k.
        sd = PortfolioOptimisers.ellipsoidal_set(true, 5, q, nothing, cov,
                                                 SigmaEllipsoidalUncertaintySet())
        @test sd.sigma == LinearAlgebra.Diagonal(cov)
        @test sd.class isa SigmaEllipsoidalUncertaintySet
        # General / ChiSq / Normal k-methods match k_ucs on the (possibly diagonalised) cov.
        for (method, samp) in ((GeneralKUncertaintyAlgorithm(), nothing),
                               (ChiSqKUncertaintyAlgorithm(), nothing), (NormalKUncertaintyAlgorithm(), Xs))
            for diag in (false, true)
                e = PortfolioOptimisers.ellipsoidal_set(diag, method, q, samp, cov,
                                                        MuEllipsoidalUncertaintySet())
                cov_ref = diag ? LinearAlgebra.Diagonal(cov) : cov
                @test e.sigma == cov_ref
                @test e.k == PortfolioOptimisers.k_ucs(method, q, samp, cov_ref)
            end
        end
    end
    @testset "The ellipsoid radius is the radius of THIS ellipsoid" begin
        # Both radius routes once read a quantity that has nothing to do with the ellipsoid
        # they size. ChiSq took its degrees of freedom from the number of simulations, and
        # the normal mean route drew its errors from the return law N(mu, sigma) rather than
        # from the estimator law N(mu, sigma / T). Both made k far too large.
        rng = StableRNG(191919)
        T, N = 300, 6
        X = randn(rng, T, N) * 0.01 .+ 0.001
        q = 0.05
        pr = prior(EmpiricalPrior(), X)

        # ChiSq: the degrees of freedom is the DIMENSION of the ellipsoid, so the two axes
        # must disagree, and n_sim must not enter at all.
        chisq = EllipsoidalUncertaintySetAlgorithm(; method = ChiSqKUncertaintyAlgorithm(),
                                                   diagonal = true)
        mu_set, sigma_set = ucs(NormalUncertaintySet(; alg = chisq, q = q, seed = 1), X)
        @test isapprox(mu_set.k,
                       sqrt(PortfolioOptimisers.Distributions.cquantile(PortfolioOptimisers.Distributions.Chisq(N),
                                                                        q)))
        @test isapprox(sigma_set.k,
                       sqrt(PortfolioOptimisers.Distributions.cquantile(PortfolioOptimisers.Distributions.Chisq(N^2),
                                                                        q)))
        @test mu_set.k != sigma_set.k
        for n_sim in (10, 5000)
            other = ucs(NormalUncertaintySet(; alg = chisq, q = q, seed = 1, n_sim = n_sim),
                        X)
            @test other[1].k == mu_set.k
            @test other[2].k == sigma_set.k
        end

        # NormalK on the mean axis: the empirical radius of a normal sample must land on the
        # chi-squared closed form, which is the same quantity computed two ways.
        normalk = EllipsoidalUncertaintySetAlgorithm(;
                                                     method = NormalKUncertaintyAlgorithm(),
                                                     diagonal = false)
        k_emp = mu_ucs(NormalUncertaintySet(; alg = normalk, q = q, seed = 1,
                                            n_sim = 20_000), X).k
        @test isapprox(k_emp,
                       sqrt(PortfolioOptimisers.Distributions.cquantile(PortfolioOptimisers.Distributions.Chisq(N),
                                                                        q)); rtol = 0.05)
        # It is a Mahalanobis radius against sigma / T, so it must not carry a factor of
        # sqrt(T): drawing from the return law instead would put one there.
        @test k_emp < sqrt(T) / 2
    end
    @testset "A set carries the quantity it bounds (ADR 0050)" begin
        pr_ref = prior(EmpiricalPrior(), rd.X)
        @testset "The field is optional and validated" begin
            @test isnothing(BoxUncertaintySet(; lb = [0.1, 0.2], ub = [0.3, 0.4]).val)
            @test BoxUncertaintySet(; lb = [0.1, 0.2], ub = [0.3, 0.4], val = [1.0, 2.0]).val ==
                  [1.0, 2.0]
            @test_throws DimensionMismatch BoxUncertaintySet(; lb = [0.1, 0.2],
                                                             ub = [0.3, 0.4],
                                                             val = [1.0, 2.0, 3.0])
            @test isnothing(EllipsoidalUncertaintySet(; sigma = [1.0 0.0; 0.0 1.0], k = 2.0,
                                                      class = MuEllipsoidalUncertaintySet()).val)
            @test_throws DimensionMismatch EllipsoidalUncertaintySet(;
                                                                     sigma = [1.0 0.0;
                                                                              0.0 1.0],
                                                                     k = 2.0,
                                                                     class = MuEllipsoidalUncertaintySet(),
                                                                     val = [1.0, 2.0, 3.0])
            @test isnothing(L1UncertaintySet(; eps = 0.1).mu)
            @test L1UncertaintySet(; eps = 0.1, mu = [1.0, 2.0]).mu == [1.0, 2.0]
            @test_throws DimensionMismatch L1UncertaintySet(; eps = 0.1, sd = [1.0, 2.0],
                                                            mu = [1.0, 2.0, 3.0])
            @test_throws Exception L1UncertaintySet(; eps = 0.1, mu = [1.0, Inf])
            @test isnothing(SignedL1UncertaintySet(; ep = 0.1, en = 0.2).mu)
            @test_throws DimensionMismatch SignedL1UncertaintySet(; ep = 0.1, en = 0.2,
                                                                  sd = [1.0, 2.0],
                                                                  mu = [1.0, 2.0, 3.0])
            # The pre-fix positional arities still construct, and carry nothing.
            @test isnothing(BoxUncertaintySet([0.1, 0.2], [0.3, 0.4]).val)
            @test isnothing(EllipsoidalUncertaintySet([1.0 0.0; 0.0 1.0], 2.0,
                                                      MuEllipsoidalUncertaintySet()).val)
            @test isnothing(L1UncertaintySet(0.1, nothing).mu)
            @test isnothing(SignedL1UncertaintySet(0.1, 0.2, nothing).mu)
        end
        @testset "Every producer populates it" begin
            rng = StableRNG(123456789)
            ues = [DeltaUncertaintySet(;),
                   NormalUncertaintySet(; rng = rng, alg = BoxUncertaintySetAlgorithm(),
                                        seed = 987654321),
                   NormalUncertaintySet(; rng = rng, seed = 987654321,
                                        alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                                 method = ChiSqKUncertaintyAlgorithm())),
                   ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), rng = rng,
                                      bootstrap = StationaryBootstrap(), seed = 987654321),
                   ARCHUncertaintySet(;
                                      alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                               method = ChiSqKUncertaintyAlgorithm()),
                                      rng = rng, bootstrap = CircularBootstrap(),
                                      seed = 987654321)]
            for ue in ues
                mu_set, sigma_set = ucs(ue, rd.X)
                @test mu_set.val == pr_ref.mu
                @test sigma_set.val == pr_ref.sigma
                @test mu_ucs(ue, rd.X).val == pr_ref.mu
                @test sigma_ucs(ue, rd.X).val == pr_ref.sigma
            end
            # The l1 family is mean-only, so the field is the characteristic vector itself.
            for alg in (L1UncertaintySetAlgorithm(), SignedL1UncertaintySetAlgorithm())
                @test mu_ucs(CharacteristicUncertaintySet(; alg = alg), rd.X).mu ==
                      pr_ref.mu
            end
        end
        @testset "The carried quantity wins over the fallback" begin
            # The value-level twin of the JuMP worst-case variance. It is the same
            # resolution, so a scalar evaluation cannot disagree with the model.
            w = fill(1 / size(rd.X, 2), size(rd.X, 2))
            A = pr_ref.sigma
            B = A * 2
            shape = LinearAlgebra.Diagonal(ones(size(A, 1)^2))
            carried = UncertaintySetVariance(;
                                             ucs = EllipsoidalUncertaintySet(;
                                                                             sigma = shape,
                                                                             k = 1.5,
                                                                             class = SigmaEllipsoidalUncertaintySet(),
                                                                             val = A),
                                             sigma = B)
            bare_a = UncertaintySetVariance(;
                                            ucs = EllipsoidalUncertaintySet(; sigma = shape,
                                                                            k = 1.5,
                                                                            class = SigmaEllipsoidalUncertaintySet()),
                                            sigma = A)
            bare_b = UncertaintySetVariance(;
                                            ucs = EllipsoidalUncertaintySet(; sigma = shape,
                                                                            k = 1.5,
                                                                            class = SigmaEllipsoidalUncertaintySet()),
                                            sigma = B)
            @test carried(w) == bare_a(w)
            @test carried(w) != bare_b(w)
            # A box covariance set names no centre: its worst case is built from the
            # bounds alone, so the field is inert on that route.
            box_carried = UncertaintySetVariance(;
                                                 ucs = BoxUncertaintySet(; lb = A - abs.(A),
                                                                         ub = A + abs.(A),
                                                                         val = A),
                                                 sigma = B)
            box_bare = UncertaintySetVariance(;
                                              ucs = BoxUncertaintySet(; lb = A - abs.(A),
                                                                      ub = A + abs.(A)),
                                              sigma = B)
            @test box_carried(w) == box_bare(w)
        end
        @testset "A view carries the field" begin
            i = [1, 3, 5]
            lbv, ubv, valv = pr_ref.mu .- 0.1, pr_ref.mu .+ 0.1, pr_ref.mu
            bv = PortfolioOptimisers.port_opt_view(BoxUncertaintySet(; lb = lbv, ub = ubv,
                                                                     val = valv), i)
            @test bv.val == valv[i]
            bm = PortfolioOptimisers.port_opt_view(BoxUncertaintySet(;
                                                                     lb = pr_ref.sigma .-
                                                                          0.1,
                                                                     ub = pr_ref.sigma .+
                                                                          0.1,
                                                                     val = pr_ref.sigma), i)
            @test bm.val == pr_ref.sigma[i, i]
            ev = PortfolioOptimisers.port_opt_view(EllipsoidalUncertaintySet(;
                                                                             sigma = pr_ref.sigma,
                                                                             k = 1.5,
                                                                             class = MuEllipsoidalUncertaintySet(),
                                                                             val = pr_ref.mu),
                                                   i)
            @test ev.val == pr_ref.mu[i]
            # On the covariance ellipsoid the carried value takes the asset index, while
            # the shape matrix takes the fourth-moment index.
            N = size(pr_ref.sigma, 1)
            em = PortfolioOptimisers.port_opt_view(EllipsoidalUncertaintySet(;
                                                                             sigma = LinearAlgebra.Diagonal(ones(N^2)),
                                                                             k = 1.5,
                                                                             class = SigmaEllipsoidalUncertaintySet(),
                                                                             val = pr_ref.sigma),
                                                   i)
            @test em.val == pr_ref.sigma[i, i]
            @test size(em.sigma, 1) == length(i)^2
            l1v = PortfolioOptimisers.port_opt_view(L1UncertaintySet(; eps = 0.1,
                                                                     sd = sqrt.(LinearAlgebra.diag(pr_ref.sigma)),
                                                                     mu = pr_ref.mu), i)
            @test l1v.eps == 0.1
            @test l1v.mu == pr_ref.mu[i]
            @test l1v.sd == sqrt.(LinearAlgebra.diag(pr_ref.sigma))[i]
            sl1v = PortfolioOptimisers.port_opt_view(SignedL1UncertaintySet(; ep = 0.1,
                                                                            en = 0.2,
                                                                            mu = pr_ref.mu),
                                                     i)
            @test sl1v.ep == 0.1
            @test sl1v.en == 0.2
            @test sl1v.mu == pr_ref.mu[i]
        end
    end
end
