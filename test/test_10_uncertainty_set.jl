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
            @test SignedL1UncertaintySet(; ep = 0.1, em = 0.2).ep == 0.1
            @test SignedL1UncertaintySet(; ep = 0.1, em = 0.2).em == 0.2
            @test_throws DomainError SignedL1UncertaintySet(; ep = -0.1, em = 0.2)
            @test_throws DomainError SignedL1UncertaintySet(; ep = 0.1, em = -0.2)
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
            @test (ss.ep, ss.em) == (0.02, 0.03)
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
end
