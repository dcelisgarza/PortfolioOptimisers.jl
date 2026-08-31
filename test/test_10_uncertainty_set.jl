@testset "Uncertainty set" begin
    using PortfolioOptimisers, Test, DataFrames, CSV, TimeSeries, StableRNGs, Random,
          Clarabel, Statistics, LinearAlgebra, Distributions
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
        @testset "The ladder starts at zero and never falls (#579)" begin
            # Lemma 2 reads the ladder as a bracket, so it must be non-decreasing, and its
            # first rung must be zero or no positive radius activates the first asset.
            for sds in (nothing, sd_hat)
                ladder = PortfolioOptimisers.l1_activation_ladder(mu_t, sds)
                @test iszero(ladder[1])
                @test issorted(ladder)
            end
        end
        @testset "A calibrated radius activates exactly the assets it targets (#579)" begin
            # The claim the whole family rests on: a radius from the open interval of
            # Lemma 2 activates exactly `q` assets, equally weighted (Corollary 4) or
            # inverse-volatility weighted (Corollary 11).
            slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                         check_sol = (; allow_local = true, allow_almost = true),
                         settings = Dict("verbose" => false))
            function solve_long_only(q, scaled)
                ue = CharacteristicUncertaintySet(;
                                                  alg = L1UncertaintySetAlgorithm(;
                                                                                  method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                            active = q),
                                                                                  scaled = scaled))
                opt = JuMPOptimiser(; slv = slv, bgt = 1.0,
                                    wb = WeightBounds(; lb = 0.0, ub = 1.0),
                                    ret = ArithmeticReturn(; ucs = ue))
                return optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(), opt = opt),
                                rdl1).w
            end
            for q in 1:N
                @test count(x -> abs(x) > 1e-6, solve_long_only(q, false)) == q
            end
            # The active weights are equal unscaled, and inverse-volatility scaled.
            w = solve_long_only(4, false)
            @test isapprox(w[w .> 1e-6], fill(0.25, 4); rtol = 1e-5)
            ws = solve_long_only(4, true)
            act = findall(x -> abs(x) > 1e-6, ws)
            @test isapprox(ws[act], inv.(sd_hat[act]) ./ sum(inv.(sd_hat[act]));
                           rtol = 1e-5)
            # The paired ladder activates the same count, split into two equal legs.
            function solve_paired(active)
                ue = CharacteristicUncertaintySet(;
                                                  alg = L1UncertaintySetAlgorithm(;
                                                                                  method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                            active = active),
                                                                                  paired = true))
                opt = JuMPOptimiser(; slv = slv, bgt = 0.0, sbgt = 0.5,
                                    wb = WeightBounds(; lb = -1.0, ub = 1.0),
                                    ret = ArithmeticReturn(; ucs = ue))
                w = optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(), opt = opt),
                             rdl1).w
                return (count(x -> x > 1e-6, w), count(x -> x < -1e-6, w))
            end
            for a in 2:2:(N - 2)
                @test solve_paired(a) == (a ÷ 2, a ÷ 2)
            end
        end
        @testset "The three edges of the ladder resolution (#579)" begin
            alg = ActiveAssetsUncertaintyAlgorithm(; active = 1)
            # A one-rung ladder has no increment to continue, so the radius is the inert
            # `one`: a one-asset universe has its single weight pinned by the budget.
            @test isone(PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0], 1))
            @test isone(mu_ucs(CharacteristicUncertaintySet(),
                               ReturnsResult(; X = X[:, 1:1], nx = ["A1"])).eps)
            # An empty ladder cannot be bracketed at all.
            @test_throws Exception PortfolioOptimisers.l1_eps_from_ladder(alg, Float64[], 1)
            # A paired calibration needs a pair.
            @test_throws ArgumentError mu_ucs(CharacteristicUncertaintySet(;
                                                                           alg = L1UncertaintySetAlgorithm(;
                                                                                                           paired = true)),
                                              ReturnsResult(; X = X[:, 1:1], nx = ["A1"]))
            # A number is a passthrough on both resolution routes.
            @test PortfolioOptimisers.l1_eps_from_ladder(0.5) == 0.5
            @test PortfolioOptimisers.l1_resolve_eps(0.25) == 0.25
        end
        @testset "An odd universe leaves its middle asset unpaired (#579)" begin
            # Lemma 5 forces w[i] == -w[N+1-i], so with an odd N the middle asset pairs
            # with itself and its weight solves w == -w. The ladder stops at N ÷ 2, and
            # every target at or above that rung resolves to the same radius.
            No = N + 1
            rdo = ReturnsResult(; X = hcat(X, X[:, 1] .+ 0.001), nx = string.("B", 1:No))
            eps_of(a) = mu_ucs(CharacteristicUncertaintySet(;
                                                            alg = L1UncertaintySetAlgorithm(;
                                                                                            method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                                      active = a),
                                                                                            paired = true)),
                               rdo).eps
            @test eps_of(2 * (No ÷ 2)) == eps_of(No) == eps_of(2 * No)
        end
        @testset "The paired count halves an integer and rounds a fraction (#579)" begin
            # Both targets name pairs, so an even count and the matching fraction agree.
            # An odd count truncates while the fraction rounds to nearest, so the two can
            # name a different number of pairs.
            paired_eps(a) = mu_ucs(CharacteristicUncertaintySet(;
                                                                alg = L1UncertaintySetAlgorithm(;
                                                                                                method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                                          active = a),
                                                                                                paired = true)),
                                   rdl1).eps
            @test paired_eps(4) == paired_eps(4 / N)
            @test paired_eps(8) == paired_eps(8 / N)
            @test paired_eps(7) != paired_eps(7 / N)
            @test paired_eps(7) == paired_eps(6)
            @test paired_eps(7 / N) == paired_eps(8)
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

    @testset "The delta box conventions" begin
        # #576. The two axes of the delta method are not in the same convention, and the
        # numbers below are the ones the docstrings of `mu_delta_box_set` and
        # `sigma_delta_box_set` state.
        rng = StableRNG(987654321)
        X = randn(rng, 60, 4)
        X[:, 2] .-= 0.5                      # force a negative mean on asset 2
        pr = prior(EmpiricalPrior(), X)
        dmu, dsigma = 0.1, 0.2
        ue = DeltaUncertaintySet(; dmu = dmu, dsigma = dsigma)
        # The mean axis writes a width. `lb` is zero everywhere, and the half-width the
        # model reads is `dmu * abs(mu)`, so `abs` fixes the width and never the centre.
        mu_set = mu_ucs(ue, X)
        @test all(iszero, mu_set.lb)
        @test (mu_set.ub .- mu_set.lb) ./ 2 ≈ dmu .* abs.(pr.mu)
        @test mu_set.val == pr.mu
        @test pr.mu[2] < 0
        half = (mu_set.ub .- mu_set.lb) ./ 2
        @test mu_set.val[2] - half[2] < mu_set.val[2] < mu_set.val[2] + half[2]
        # The covariance axis writes absolute bounds. They order entry by entry, on a
        # negative entry as much as on a positive one.
        sigma_set = sigma_ucs(ue, X)
        @test sigma_set.lb ≈ pr.sigma .- dsigma .* abs.(pr.sigma)
        @test sigma_set.ub ≈ pr.sigma .+ dsigma .* abs.(pr.sigma)
        @test sigma_set.val == pr.sigma
        @test all(sigma_set.lb .<= pr.sigma .<= sigma_set.ub)
        @test any(pr.sigma .< 0)
        @test diag(sigma_set.ub) ≈ (1 + dsigma) .* diag(pr.sigma)
        @test diag(sigma_set.lb) ≈ (1 - dsigma) .* diag(pr.sigma)
        # A zero fraction is admitted on either axis and gives a degenerate box.
        ue0 = DeltaUncertaintySet(; dmu = 0.0, dsigma = 0.0)
        mu_set0, sigma_set0 = ucs(ue0, X)
        @test all(iszero, mu_set0.lb)
        @test all(iszero, mu_set0.ub)
        @test sigma_set0.lb == pr.sigma
        @test sigma_set0.ub == pr.sigma
        # The builder applies no `posdef!`, so a large `dsigma` leaves the lower bound
        # indefinite. The consumer reads both bounds entry by entry, so it needs neither
        # to be definite: the worst case stays finite, and for a long-only `w` it is
        # `w' * ub * w`, which is what `tr(Au * ub) - tr(Al * lb)` reduces to.
        big = sigma_ucs(DeltaUncertaintySet(; dsigma = 1.0), X)
        @test minimum(eigvals(Symmetric(Matrix(big.lb)))) < 0
        w = fill(1 / size(X, 2), size(X, 2))
        wc = UncertaintySetVariance(; ucs = big, sigma = pr.sigma)(w)
        @test isfinite(wc)
        @test wc ≈ dot(w, big.ub, w)
        @test wc > dot(w, pr.sigma, w)
        # The three verbs fit the prior separately, and agree on their common axes.
        mu_pair, sigma_pair = ucs(ue, X)
        @test mu_pair.lb == mu_set.lb
        @test mu_pair.ub == mu_set.ub
        @test sigma_pair.lb == sigma_set.lb
        @test sigma_pair.ub == sigma_set.ub
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
    @testset "The map's vocabulary (#575)" begin
        @testset "The three passthrough routes return their own argument" begin
            # `ucs` on an already-built pair hands the pair back, so a consumer never has
            # to ask whether its slot holds an estimator or a result.
            b1 = BoxUncertaintySet(; lb = [0.1, 0.2], ub = [0.3, 0.4])
            b2 = BoxUncertaintySet(; lb = [0.0 0.0; 0.0 0.0], ub = [1.0 0.5; 0.5 1.0])
            @test isnothing(ucs(nothing))
            @test ucs((b1, b2)) === (b1, b2)
            @test ucs((nothing, b2)) === (nothing, b2)
            # The trailing arguments are absorbed, so the same call shape serves the
            # estimator method.
            @test ucs((b1, b2), rd.X, nothing; dims = 1) === (b1, b2)
            @test mu_ucs(b1) === b1
            @test isnothing(mu_ucs(nothing))
            @test sigma_ucs(b2) === b2
            @test isnothing(sigma_ucs(nothing))
            # An estimator carries no asset axis, so a view hands it back unchanged.
            ue = NormalUncertaintySet()
            @test PortfolioOptimisers.port_opt_view(ue, [1, 2]) === ue
            @test isnothing(PortfolioOptimisers.port_opt_view(nothing, [1, 2]))
        end
        @testset "The ReturnsResult forwards check both preconditions" begin
            # `rd.X` is nothing: every one of the three raises.
            rd_no_X = ReturnsResult()
            ue = NormalUncertaintySet()
            @test_throws PortfolioOptimisers.IsNothingError ucs(ue, rd_no_X)
            @test_throws PortfolioOptimisers.IsNothingError mu_ucs(ue, rd_no_X)
            @test_throws PortfolioOptimisers.IsNothingError sigma_ucs(ue, rd_no_X)
            # A factor prior with no factor returns: the second raise, on all three.
            uf = NormalUncertaintySet(; pe = FactorPrior(), n_sim = 20, seed = 7)
            rd_no_F = ReturnsResult(; X = rd.X, nx = rd.nx)
            @test isa(uf.pe, PortfolioOptimisers.AbstractHiLoOrderPriorEstimator_F)
            @test_throws PortfolioOptimisers.IsNothingError ucs(uf, rd_no_F)
            @test_throws PortfolioOptimisers.IsNothingError mu_ucs(uf, rd_no_F)
            @test_throws PortfolioOptimisers.IsNothingError sigma_ucs(uf, rd_no_F)
        end
        @testset "ucs_selector takes the branches its table names" begin
            b = BoxUncertaintySet(; lb = [0.1], ub = [0.3])
            ue = NormalUncertaintySet()
            @test isnothing(PortfolioOptimisers.ucs_selector(nothing, nothing))
            @test PortfolioOptimisers.ucs_selector(b, nothing) === b
            @test PortfolioOptimisers.ucs_selector(b, ue) === b
            @test PortfolioOptimisers.ucs_selector(nothing, ue) === ue
        end
        @testset "The general radius is Cantelli's bound at k" begin
            # k = sqrt((1 - q) / q) inverts to 1 / (1 + k^2) == q exactly, which is the
            # one-sided Chebyshev tail at k standard deviations.
            for q in (0.01, 0.05, 0.1, 0.25)
                k = PortfolioOptimisers.k_ucs(GeneralKUncertaintyAlgorithm(), q)
                @test k ≈ sqrt((1 - q) / q)
                @test 1 / (1 + k^2) ≈ q
            end
            # It reads neither the sample nor the shape matrix.
            @test PortfolioOptimisers.k_ucs(GeneralKUncertaintyAlgorithm(), 0.05) ==
                  PortfolioOptimisers.k_ucs(GeneralKUncertaintyAlgorithm(), 0.05,
                                            randn(StableRNG(3), 10, 4),
                                            LinearAlgebra.Diagonal(ones(4)))
        end
        @testset "The chi-squared radius reads size(sigma_X, 1)" begin
            # The degrees of freedom is the first dimension of the shape matrix, so the
            # same algorithm gives N on the mean axis and N^2 on the covariance axis.
            for p in (4, 16)
                k = PortfolioOptimisers.k_ucs(ChiSqKUncertaintyAlgorithm(), 0.05, nothing,
                                              LinearAlgebra.Diagonal(ones(p)))
                @test k ≈ sqrt(quantile(Distributions.Chisq(p), 0.95))
            end
            # A symmetric N x N matrix carries N(N+1)/2 free entries, so the N^2 the
            # covariance axis passes overstates the dimension of the ellipsoid, and the
            # radius is the conservative one.
            N = 4
            k_sq = PortfolioOptimisers.k_ucs(ChiSqKUncertaintyAlgorithm(), 0.05, nothing,
                                             LinearAlgebra.Diagonal(ones(N^2)))
            k_free = sqrt(quantile(Distributions.Chisq(N * (N + 1) ÷ 2), 0.95))
            @test k_sq > k_free
        end
        @testset "The empirical radius measures against the shape it is given" begin
            # `ellipsoidal_set` takes the diagonal BEFORE it fits k, so on the default the
            # quantile is one of distances against the diagonal shape, not the full one.
            rng = StableRNG(20250828)
            Xd = randn(rng, 252, 5) * 0.01
            cv = Statistics.cov(Xd)
            km = NormalKUncertaintyAlgorithm()
            e_full = PortfolioOptimisers.ellipsoidal_set(false, km, 0.05, Xd, cv,
                                                         MuEllipsoidalUncertaintySet())
            e_diag = PortfolioOptimisers.ellipsoidal_set(true, km, 0.05, Xd, cv,
                                                         MuEllipsoidalUncertaintySet())
            d2_diag = [LinearAlgebra.dot(Xd[t, :], LinearAlgebra.Diagonal(cv) \ Xd[t, :])
                       for t in axes(Xd, 1)]
            @test e_diag.k ≈ sqrt(quantile(d2_diag, 0.95))
            @test e_diag.sigma == LinearAlgebra.Diagonal(cv)
            @test !isapprox(e_full.k, e_diag.k)
            # The full-shape radius is the hand computation against the full shape.
            d2_full = [LinearAlgebra.dot(Xd[t, :], cv \ Xd[t, :]) for t in axes(Xd, 1)]
            @test e_full.k ≈ sqrt(quantile(d2_full, 0.95))
            # The docstrings of `NormalKUncertaintyAlgorithm` and `ellipsoidal_set` quote
            # this pair as the illustration of the difference, so pin it here: the prose
            # and the suite cannot then drift apart.
            @test round(e_full.k; digits = 4) == 3.4463
            @test round(e_diag.k; digits = 4) == 3.4407
            # Those same two docstrings say that neither shape is reliably the larger, so
            # a reader must not read an ordering rule out of the single pair above.
            n_diag_larger = count(1:60) do s
                Xs = randn(StableRNG(s), 252, 5) * 0.01
                cvs = Statistics.cov(Xs)
                kf = PortfolioOptimisers.ellipsoidal_set(false, km, 0.05, Xs, cvs,
                                                         MuEllipsoidalUncertaintySet()).k
                kd = PortfolioOptimisers.ellipsoidal_set(true, km, 0.05, Xs, cvs,
                                                         MuEllipsoidalUncertaintySet()).k
                return kd > kf
            end
            @test n_diag_larger == 34
        end
        @testset "vec_quantile_bounds reads rows, so its input is N x n_sim" begin
            # A rectangular sample settles the axis: three components, four hundred draws.
            mus = randn(StableRNG(99), 3, 400) .+ [10.0, 20.0, 30.0]
            lb, ub = PortfolioOptimisers.vec_quantile_bounds(mus, 0.025, (;))
            @test length(lb) == 3
            @test length(ub) == 3
            @test all(lb .<= ub)
            @test all(lb .<= vec(Statistics.mean(mus; dims = 2)) .<= ub)
        end
        @testset "box_quantile_bounds is symmetric and brackets the centre" begin
            rng = StableRNG(5150)
            N = 4
            base = Statistics.cov(randn(rng, 200, N))
            sims = map(1:300) do _
                s = base .+ 1e-4 * randn(rng, N, N)
                return (s + s') / 2
            end
            get_ij = (i, j) -> [s[i, j] for s in sims]
            lb, ub = PortfolioOptimisers.box_quantile_bounds(Float64, get_ij, N, 0.025, (;))
            @test LinearAlgebra.issymmetric(lb)
            @test LinearAlgebra.issymmetric(ub)
            @test all(lb .<= ub)
            @test all(lb .<= sum(sims) / length(sims) .<= ub)
        end
        @testset "A view carries k through, so it is not the subset's own fit" begin
            # The restricted shape matrix equals the one fitted on the subset alone, but
            # the radius does not: only the general route, which reads neither the data nor
            # the shape, is invariant under the restriction.
            rng = StableRNG(778899)
            Nf = 4
            i = [1, 3]
            Xf = randn(rng, 400, Nf) * 0.01
            Xs = Xf[:, i]
            for (method, invariant) in ((GeneralKUncertaintyAlgorithm(), true),
                                        (ChiSqKUncertaintyAlgorithm(), false),
                                        (NormalKUncertaintyAlgorithm(), false))
                alg = EllipsoidalUncertaintySetAlgorithm(; method = method, diagonal = true)
                ue = NormalUncertaintySet(; alg = alg, n_sim = 200, seed = 42)
                mu_full, sigma_full = ucs(ue, Xf)
                mu_sub, sigma_sub = ucs(ue, Xs)
                mv = PortfolioOptimisers.port_opt_view(mu_full, i)
                sv = PortfolioOptimisers.port_opt_view(sigma_full, i)
                # The shapes agree on both axes, whatever the radius route.
                @test Matrix(mv.sigma) ≈ Matrix(mu_sub.sigma)
                @test Matrix(sv.sigma) ≈ Matrix(sigma_sub.sigma)
                @test size(sv.sigma, 1) == length(i)^2
                @test isapprox(mv.k, mu_sub.k) == invariant
                @test isapprox(sv.k, sigma_sub.k) == invariant
                # The view never shrinks the radius, so it is the conservative choice.
                @test mv.k >= mu_sub.k
                @test sv.k >= sigma_sub.k
            end
        end
    end
    @testset "The normal law's arithmetic (#577)" begin
        @testset "The chi-squared route lands on the generic method" begin
            # The three chi-squared specialisations were deleted: their bodies were
            # identical, line for line, to the `<:Any` fallback below them, and the
            # fallback already catches `ChiSqKUncertaintyAlgorithm`. Dispatch must
            # still land on one unambiguous method, and give the same radii.
            rng = StableRNG(1357)
            Xd = randn(rng, 300, 4) * 0.01
            alg = EllipsoidalUncertaintySetAlgorithm(;
                                                     method = ChiSqKUncertaintyAlgorithm())
            ue = NormalUncertaintySet(; alg = alg, seed = 11, n_sim = 500)
            for f in (ucs, mu_ucs, sigma_ucs)
                ms = methods(f, (typeof(ue), Matrix{Float64}))
                @test length(ms) == 1
                # The one method that answers is the generic ellipsoidal fallback, not
                # a specialisation on the radius algorithm.
                # With the specialisation gone the algorithm parameter is
                # unconstrained, so the signature names no radius algorithm at all.
                sig = string(only(ms).sig)
                @test occursin("EllipsoidalUncertaintySetAlgorithm", sig)
                @test !occursin("ChiSqKUncertaintyAlgorithm", sig)
                @test !occursin("NormalKUncertaintyAlgorithm", sig)
            end
            m, s = ucs(ue, Xd)
            @test mu_ucs(ue, Xd).k == m.k
            @test sigma_ucs(ue, Xd).k == s.k
            # The radii the deleted specialisations produced, recorded before the cut.
            @test m.k ≈ 3.0802157451680485
            @test s.k ≈ 5.127984750841625
            # A plain number and the general algorithm take the same route.
            for method in (GeneralKUncertaintyAlgorithm(), 2.5)
                uf = NormalUncertaintySet(; seed = 11,
                                          alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                                   method = method))
                mf, sf = ucs(uf, Xd)
                @test mf.k == mu_ucs(uf, Xd).k
                @test sf.k == sigma_ucs(uf, Xd).k
            end
        end
        @testset "The commutation matrix transposes the vectorisation" begin
            # A SYMMETRIC matrix cannot separate K from the identity, so the check is
            # run on non-symmetric ones.
            A = reshape(collect(1.0:6.0), 2, 3)
            K23 = PortfolioOptimisers.commutation_matrix(A)
            @test size(K23) == (6, 6)
            @test K23 * vec(A) == vec(transpose(A))
            B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
            K3 = PortfolioOptimisers.commutation_matrix(B)
            @test !issymmetric(B)
            @test K3 * vec(B) == vec(transpose(B))
            # It is a permutation, so it is its own inverse and holds one entry per row.
            @test K3 * K3 == I(9)
            @test count(!iszero, K3) == 9
            # The symmetric case is the one that proves nothing.
            S = B + transpose(B)
            @test K3 * vec(S) == vec(S)
        end
        @testset "The two asymptotic covariances are the normal sampling law" begin
            rng = StableRNG(20250828)
            N, T = 4, 500
            L = tril(randn(rng, N, N)) + 3 * I
            sigma = L * transpose(L) / 50
            sigma_mu = PortfolioOptimisers.mu_asymptotic_cov(Posdef(), sigma, T)
            @test sigma_mu ≈ sigma / T
            ss = PortfolioOptimisers.sigma_asymptotic_cov(Posdef(), sigma_mu, sigma, T)
            # Cajas (2025) Equations 11.17 and 11.24, Magnus and Neudecker's form of the
            # asymptotic covariance of vec(S): T (I + K) kron(S/T, S/T) == (I + K) kron(S, S) / T.
            K = Matrix(PortfolioOptimisers.commutation_matrix(sigma))
            @test Matrix(ss) ≈ (I + K) * kron(sigma, sigma) / T
            # Its (i,j) diagonal entry is the variance of the (i,j) entry of a
            # Wishart(T, sigma / T) draw, so the sample `ucs` builds from those draws and
            # the shape it measures them against are on one scale.
            hand = [(sigma[i, i] * sigma[j, j] + sigma[i, j]^2) / T for j in 1:N
                    for i in 1:N]
            @test diag(Matrix(ss)) ≈ hand
            ws = rand(StableRNG(5), Distributions.Wishart(T, sigma_mu), 20_000)
            @test isapprox(sum(ws) / length(ws), sigma; rtol = 5e-3)
            emp = [var(getindex.(ws, i, j)) for j in 1:N for i in 1:N]
            @test isapprox(emp, hand; rtol = 5e-2)
        end
        @testset "The empirical route measures an error, not a level" begin
            # Drawing from N(mu, sigma) in place of N(mu, sigma / T) multiplies every
            # deviation, and therefore the radius, by sqrt(T).
            rng = StableRNG(777)
            N, T, q = 4, 250, 0.05
            L = tril(randn(rng, N, N)) + 3 * I
            sigma = L * transpose(L) / 50
            sigma_mu = sigma / T
            Z = randn(StableRNG(11), N, 5_000)
            Xa = transpose(cholesky(Symmetric(sigma_mu)).L * Z)
            Xb = transpose(cholesky(Symmetric(sigma)).L * Z)
            ka = PortfolioOptimisers.k_ucs(NormalKUncertaintyAlgorithm(), q, Xa, sigma_mu)
            kb = PortfolioOptimisers.k_ucs(NormalKUncertaintyAlgorithm(), q, Xb, sigma_mu)
            @test kb / ka ≈ sqrt(T)
        end
        @testset "The box halves the tail and the ellipsoid does not" begin
            rng = StableRNG(4242)
            N = 3
            Xd = randn(rng, 400, N) * 0.01
            ue = NormalUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), q = 0.05,
                                      n_sim = 4_000, seed = 1)
            pr_b, Tb, sigma_mu, q = PortfolioOptimisers.normal_box_preamble(ue, Xd)
            @test q == ue.q * 0.5
            @test Tb == size(Xd, 1)
            mset, sset = ucs(ue, Xd)
            # The mean box writes a WIDTH: the lower bound is zero, and the half-width
            # the consumer reads is z_{1 - q/2} times the standard error.
            z = Distributions.cquantile(Distributions.Normal(), ue.q * 0.5)
            @test all(iszero, mset.lb)
            @test mset.ub ≈ 2 * z * sqrt.(diag(sigma_mu))
            @test (mset.ub - mset.lb) * 0.5 ≈ z * sqrt.(diag(sigma_mu))
            @test mset.val == pr_b.mu
            # The covariance bounds bracket the point estimate, and the two `posdef!`
            # calls move a bound that is already positive definite by nothing.
            @test all(sset.lb .<= pr_b.sigma .<= sset.ub)
            @test isposdef(sset.lb)
            @test isposdef(sset.ub)
            lb1 = copy(sset.lb)
            PortfolioOptimisers.posdef!(ue.pdm, lb1)
            @test lb1 == sset.lb
            # The ellipsoidal route passes `ue.q` UNDIVIDED, because a radius cuts only
            # the upper tail of a distance that cannot be negative. Both routes therefore
            # cover 1 - q: the box one entry at a time, the ellipsoid jointly.
            sigma_mu_e = sigma_mu
            k = PortfolioOptimisers.k_ucs(ChiSqKUncertaintyAlgorithm(), ue.q, nothing,
                                          sigma_mu_e)
            E = cholesky(Symmetric(sigma_mu_e)).L * randn(StableRNG(9), N, 100_000)
            d2 = vec(sum(E .* (sigma_mu_e \ E); dims = 1))
            @test isapprox(mean(d2 .<= k^2), 1 - ue.q; atol = 5e-3)
            sd = sqrt.(diag(sigma_mu_e))
            for i in 1:N
                @test isapprox(mean(abs.(E[i, :]) .<= z * sd[i]), 1 - ue.q; atol = 5e-3)
            end
            # The per-entry box is not a joint region: over N entries it covers less.
            @test mean(vec(all(abs.(E) .<= z .* sd; dims = 1))) < 1 - ue.q
        end
        @testset "The scaling parameter reads three sources in order" begin
            rng = StableRNG(2468)
            N, T = 3, 400
            Xd = randn(rng, T, N) * 0.01
            base = NormalUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), q = 0.05,
                                        n_sim = 2_000, seed = 3)
            pr_s = prior(base.pe, Xd)
            @test isnothing(pr_s.ens)
            @test PortfolioOptimisers.choose_scaling_parameter(base, pr_s) == size(Xd, 1)
            # The estimator's own `ens` wins over the row count, and a prior that reports
            # fewer effective scenarios widens the set: the width scales as T^(-1/2).
            quarter = NormalUncertaintySet(; alg = BoxUncertaintySetAlgorithm(), q = 0.05,
                                           n_sim = 2_000, seed = 3, ens = T / 4)
            @test PortfolioOptimisers.choose_scaling_parameter(quarter, pr_s) == T / 4
            m1, _ = ucs(base, Xd)
            m2, _ = ucs(quarter, Xd)
            @test m2.ub ≈ 2 * m1.ub
            # The prior's own `ens` wins over the row count when the estimator states none.
            pr_ens = PortfolioOptimisers.LowOrderPrior(; X = pr_s.X, mu = pr_s.mu,
                                                       sigma = pr_s.sigma, ens = T / 9)
            @test PortfolioOptimisers.choose_scaling_parameter(base, pr_ens) == T / 9
            @test PortfolioOptimisers.choose_scaling_parameter(quarter, pr_ens) == T / 4
        end
        @testset "Both verbs read the asset count from the prior" begin
            # `ucs` and `sigma_ucs` both take N from `size(pr.X, 2)`, so a factor prior
            # that changes the asset count moves the two together.
            rng = StableRNG(31337)
            Xd = randn(rng, 260, 5) * 0.01
            Fd = randn(rng, 260, 2) * 0.01
            alg = EllipsoidalUncertaintySetAlgorithm(;
                                                     method = NormalKUncertaintyAlgorithm(),
                                                     diagonal = false)
            ue = NormalUncertaintySet(; pe = FactorPrior(), alg = alg, n_sim = 200,
                                      seed = 4242)
            m_pair, s_pair = ucs(ue, Xd, Fd)
            m_alone = mu_ucs(ue, Xd, Fd)
            s_alone = sigma_ucs(ue, Xd, Fd)
            @test size(s_pair.sigma, 1) == size(Xd, 2)^2
            @test Matrix(s_alone.sigma) == Matrix(s_pair.sigma)
            @test Matrix(m_alone.sigma) == Matrix(m_pair.sigma)
        end
        @testset "One generator serves two draws, so the pair splits the stream (#590)" begin
            # `ucs` on the empirical route resolves ONE generator and draws the mean
            # sample off it first, so its Wishart draws come from a stream that has
            # already advanced. `sigma_ucs` draws the Wishart first. The mean radius
            # therefore agrees between the two and the covariance radius does not.
            rng = StableRNG(31337)
            Xd = randn(rng, 260, 5) * 0.01
            Fd = randn(rng, 260, 2) * 0.01
            alg = EllipsoidalUncertaintySetAlgorithm(;
                                                     method = NormalKUncertaintyAlgorithm(),
                                                     diagonal = false)
            ue = NormalUncertaintySet(; pe = FactorPrior(), alg = alg, n_sim = 200,
                                      seed = 4242)
            m_pair, s_pair = ucs(ue, Xd, Fd)
            @test m_pair.k == mu_ucs(ue, Xd, Fd).k
            @test s_pair.k != sigma_ucs(ue, Xd, Fd).k
            # The box route resolves its own generator inside `sigma_normal_box_set`,
            # so there the two entry points agree entry for entry.
            ub = NormalUncertaintySet(; pe = FactorPrior(),
                                      alg = BoxUncertaintySetAlgorithm(), n_sim = 200,
                                      seed = 4242)
            _, b_pair = ucs(ub, Xd, Fd)
            b_alone = sigma_ucs(ub, Xd, Fd)
            @test b_alone.lb == b_pair.lb
            @test b_alone.ub == b_pair.ub
        end
    end
    @testset "The three block bootstraps (#578)" begin
        # Every claim the docstrings of `04_BootstrapUncertaintySets.jl` make about the
        # three schemes is measured here, not read off the bodies.
        Xb = rd.X[:, 1:5]
        Tb = size(Xb, 1)
        @testset "Only the moving scheme needs a block_size guard" begin
            # A block that wraps is taken through `mod1`, which cannot leave `1:T`. A block
            # that does not wrap is drawn from `1:(T - block_size + 1)`, which is empty as
            # soon as `block_size` passes `T`, so only that scheme raises.
            T = 10
            for alg in (StationaryBootstrap(), CircularBootstrap(), MovingBootstrap())
                idx = PortfolioOptimisers.bootstrap_indices(alg, StableRNG(11), T, 3)
                @test length(idx) == T
                @test all(i -> 1 <= i <= T, idx)
            end
            @test_throws DomainError PortfolioOptimisers.bootstrap_indices(MovingBootstrap(),
                                                                           StableRNG(11), T,
                                                                           T + 1)
            # The two that wrap take the same input without a raise, and stay in range.
            for alg in (StationaryBootstrap(), CircularBootstrap())
                idx = PortfolioOptimisers.bootstrap_indices(alg, StableRNG(11), T, T + 1)
                @test all(i -> 1 <= i <= T, idx)
            end
            # Without the guard the moving scheme would fail on an empty range, and the
            # message would name neither `block_size` nor `T`.
            @test_throws ArgumentError rand(StableRNG(11), 1:(T - (T + 1) + 1))
        end
        @testset "A circular block_size at or above T collapses the set" begin
            # The first block already fills the vector, so every resample is a cyclic shift
            # of the series. A cyclic shift is a permutation, and neither the mean nor the
            # covariance changes under a permutation of the rows.
            T = 10
            for bs in (T, T + 1, 2 * T)
                idx = PortfolioOptimisers.bootstrap_indices(CircularBootstrap(),
                                                            StableRNG(11), T, bs)
                @test sort(idx) == collect(1:T)
            end
            ue = ARCHUncertaintySet(; bootstrap = CircularBootstrap(), block_size = Tb + 1,
                                    n_sim = 20, seed = 987654321)
            mus, sigmas = PortfolioOptimisers.bootstrap_generator(ue, Xb)
            @test maximum(abs, mus .- mus[:, 1]) < 1e-15
            @test maximum(abs, sigmas .- sigmas[:, :, 1]) < 1e-15
            mset, sset = ucs(ue, Xb)
            @test maximum(mset.ub .- mset.lb) < 1e-15
            @test maximum(sset.ub .- sset.lb) < 1e-15
            # Nothing raises on the ellipsoidal route either: the shape matrix goes to
            # zero and the radius stays finite, so the set is empty rather than an error.
            uee = ARCHUncertaintySet(; bootstrap = CircularBootstrap(), block_size = Tb + 1,
                                     n_sim = 20, seed = 987654321,
                                     alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                              diagonal = false))
            emu, _ = ucs(uee, Xb)
            @test maximum(abs, emu.sigma) < 1e-30
            @test isfinite(emu.k)
            # The stationary scheme does not collapse there: a restart still fires with
            # probability `1 / block_size`, so the spread only narrows.
            big = ARCHUncertaintySet(; bootstrap = StationaryBootstrap(),
                                     block_size = Tb + 1, n_sim = 100, seed = 987654321)
            small = ARCHUncertaintySet(; bootstrap = StationaryBootstrap(), block_size = 3,
                                       n_sim = 100, seed = 987654321)
            sd_big = maximum(std(PortfolioOptimisers.mu_bootstrap_generator(big, Xb);
                                 dims = 2))
            sd_small = maximum(std(PortfolioOptimisers.mu_bootstrap_generator(small, Xb);
                                   dims = 2))
            @test 0 < sd_big < sd_small
        end
        @testset "The stationary block length is geometric with mean block_size" begin
            # The helper's accumulator must not share a name with the one below it. A
            # nested function inside a `@testset` rebinds the enclosing local of the same
            # name rather than making its own, which silently resets the tally.
            function block_lengths(idx, T)
                out = Int[]
                run = 1
                for t in 2:length(idx)
                    if idx[t] == mod1(idx[t - 1] + 1, T)
                        run += 1
                    else
                        push!(out, run)
                        run = 1
                    end
                end
                push!(out, run)
                return out
            end
            T, bs = 500, 10
            rng = StableRNG(20260828)
            lens = Int[]
            for _ in 1:500
                append!(lens,
                        block_lengths(PortfolioOptimisers.bootstrap_indices(StationaryBootstrap(),
                                                                            rng, T, bs), T))
            end
            # The truncation of the last block of each vector biases the mean slightly low.
            @test length(lens) > 20_000
            @test isapprox(mean(lens), bs; rtol = 0.05)
            for k in 1:5
                @test isapprox(count(==(k), lens) / length(lens), (1 - 1 / bs)^(k - 1) / bs;
                               rtol = 0.1)
            end
        end
        @testset "Only the moving scheme draws the ends less often" begin
            # Observation `j` of the first `block_size` lies inside only `j` of the start
            # positions, so it is drawn about `j / block_size` as often as a middle one.
            function coverage(alg, T, bs, n)
                c = zeros(Int, T)
                rng = StableRNG(20260828)
                for _ in 1:n
                    for i in PortfolioOptimisers.bootstrap_indices(alg, rng, T, bs)
                        c[i] += 1
                    end
                end
                return c ./ (n * one(eltype(c)))
            end
            T, bs, n = 100, 5, 5000
            cm = coverage(MovingBootstrap(), T, bs, n)
            mid = mean(cm[10:90])
            for j in 1:bs
                @test isapprox(cm[j] / mid, j / bs; atol = 0.05)
                @test isapprox(cm[end - j + 1] / mid, j / bs; atol = 0.05)
            end
            # The two schemes that wrap are flat to Monte-Carlo noise.
            for alg in (CircularBootstrap(), StationaryBootstrap())
                c = coverage(alg, T, bs, n)
                @test isapprox(minimum(c) / mean(c), 1; atol = 0.05)
                @test isapprox(maximum(c) / mean(c), 1; atol = 0.05)
            end
        end
        @testset "The three verbs agree with a seed and not without one" begin
            # `resolve_rng` hands back a private reseeded copy, so each of the three calls
            # restarts one index stream at the same place. With no seed the three walk
            # different parts of the shared generator.
            ue = ARCHUncertaintySet(; bootstrap = StationaryBootstrap(), n_sim = 100,
                                    rng = StableRNG(123456789), seed = 987654321)
            m1, s1 = ucs(ue, Xb)
            @test m1.lb == mu_ucs(ue, Xb).lb
            @test m1.ub == mu_ucs(ue, Xb).ub
            @test s1.lb == sigma_ucs(ue, Xb).lb
            @test s1.ub == sigma_ucs(ue, Xb).ub
            un = ARCHUncertaintySet(; bootstrap = StationaryBootstrap(), n_sim = 100,
                                    rng = StableRNG(123456789))
            n1, t1 = ucs(un, Xb)
            @test n1.lb != mu_ucs(un, Xb).lb
            @test t1.lb != sigma_ucs(un, Xb).lb
        end
        @testset "The centre comes from pe and the width from me, so they can disagree" begin
            # `val` is the prior's point estimate and the bounds are quantiles of the
            # refits, so a box need not contain its own centre.
            ue = ARCHUncertaintySet(; pe = EmpiricalPrior(), me = MedianExpectedReturns(),
                                    n_sim = 250, seed = 987654321, block_size = 3)
            mset = mu_ucs(ue, Xb)
            @test any(.!(mset.lb .<= mset.val .<= mset.ub))
            # The prior's own mean estimator agrees with the refits, and then it does not.
            agree = ARCHUncertaintySet(; pe = EmpiricalPrior(),
                                       me = SimpleExpectedReturns(), n_sim = 250,
                                       seed = 987654321, block_size = 3)
            aset = mu_ucs(agree, Xb)
            @test all(aset.lb .<= aset.val .<= aset.ub)
        end
        @testset "ce fits the resamples and the deviations, so it enters twice" begin
            # Dropping the bias correction divides by `n` instead of `n - 1`, so the move
            # is exact and names which fit each factor came from.
            n_sim = 50
            corrected = PortfolioOptimisersCovariance()
            uncorrected = PortfolioOptimisersCovariance(;
                                                        ce = Covariance(;
                                                                        ce = GeneralCovariance(;
                                                                                               ce = StatsBase.SimpleCovariance(;
                                                                                                                               corrected = false))))
            alg = EllipsoidalUncertaintySetAlgorithm(; diagonal = false)
            uA = ARCHUncertaintySet(; ce = corrected, n_sim = n_sim, seed = 987654321,
                                    alg = alg)
            uB = ARCHUncertaintySet(; ce = uncorrected, n_sim = n_sim, seed = 987654321,
                                    alg = alg)
            # Inside every resample: the resampled covariances move by `(T - 1) / T`.
            sigA = PortfolioOptimisers.sigma_bootstrap_generator(uA, Xb)
            sigB = PortfolioOptimisers.sigma_bootstrap_generator(uB, Xb)
            @test isapprox(sigB, sigA .* ((Tb - 1) / Tb))
            # Over the deviations: the mean-axis shape moves by `(n_sim - 1) / n_sim`, and
            # `mu_bootstrap_generator` fits no covariance, so that is the only `ce` there.
            mA = mu_ucs(uA, Xb)
            mB = mu_ucs(uB, Xb)
            @test isapprox(Matrix(mB.sigma), Matrix(mA.sigma) .* ((n_sim - 1) / n_sim))
            # The covariance axis compounds the two, so it moves by more than either.
            sA = sigma_ucs(uA, Xb)
            sB = sigma_ucs(uB, Xb)
            rel(x, y) = maximum(abs, x .- y) / maximum(abs, x)
            @test rel(Matrix(sA.sigma), Matrix(sB.sigma)) >
                  rel(Matrix(mA.sigma), Matrix(mB.sigma))
        end
    end
end
