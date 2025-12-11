@safetestset "OWA" begin
    using PortfolioOptimisers, CSV, DataFrames, Clarabel, Test, Random, Logging
    Logging.disable_logging(Logging.Warn)
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    @testset "OWA l-moments" begin
        slv = Solver(; solver = Clarabel.Optimizer,
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        owa_t = CSV.read(joinpath(@__DIR__, "./assets/OWA_l_moment_weights.csv.gz"),
                         DataFrame)
        owas = [NormalisedConstantRelativeRiskAversion(; g = 0.75),
                NormalisedConstantRelativeRiskAversion(),
                NormalisedConstantRelativeRiskAversion(; g = 0.25),
                OWAJuMP(; alg = MaximumEntropy(), max_phi = 0.75, slv = slv),
                OWAJuMP(; alg = MaximumEntropy(), slv = slv),
                OWAJuMP(; alg = MaximumEntropy(), max_phi = 0.25, slv = slv),
                OWAJuMP(; alg = MinimumSumSquares(), max_phi = 0.75, slv = slv),
                OWAJuMP(; alg = MinimumSumSquares(), slv = slv),
                OWAJuMP(; alg = MinimumSumSquares(), max_phi = 0.25, slv = slv),
                OWAJuMP(; alg = MinimumSquaredDistance(), max_phi = 0.75, slv = slv),
                OWAJuMP(; alg = MinimumSquaredDistance(), slv = slv),
                OWAJuMP(; alg = MinimumSquaredDistance(), max_phi = 0.25, slv = slv)]
        for i in eachindex(owas)
            owa = owa_l_moment_crm(200, owas[i]; k = 5)
            rtol = if i == 4
                0.05
            elseif i == 5
                0.005
            elseif i == 7 || Sys.iswindows() && i == 8 || Sys.isapple() && i == 11
                1e-6
            elseif i == 8
                1e-7
            elseif i == 11
                5e-8
            else
                1e-9
            end
            res = isapprox(owa, owa_t[!, i]; rtol = rtol)
            if !res
                println("Fails on OWA l-moments iteration $i")
                find_tol(owa, owa_t[!, i]; name1 = :owa, name2 = :owa_t)
            end
            @test res
        end
        @test_throws IsEmptyError OWAJuMP(slv = Solver[])
    end
    @testset "OWA weight vectors" begin
        owa_t = CSV.read(joinpath(@__DIR__, "./assets/OWA_weights.csv.gz"), DataFrame)
        @test isapprox(owa_gmd(100), owa_t[!, 1])
        @test isapprox(owa_cvar(100), owa_t[!, 2])
        @test isapprox(owa_tg(100), owa_t[!, 3])
        @test isapprox(owa_wr(100), owa_t[!, 4])
        @test isapprox(owa_rg(100), owa_t[!, 5])
        @test isapprox(owa_cvarrg(100), owa_t[!, 6])
        @test isapprox(owa_tgrg(100), owa_t[!, 7])
        @test isapprox(owa_l_moment(100), owa_t[!, 8])
        @test isapprox(owa_l_moment(100, 3), owa_t[!, 9])
        @test isapprox(owa_l_moment(100, 4), owa_t[!, 10])
        @test isapprox(owa_l_moment(100, 10), owa_t[!, 11])

        w1 = owa_tgrg(100)
        alpha_i = 0.0001
        alpha = 0.05
        a_sim = 100
        alphas = range(; start = alpha_i, stop = alpha, length = a_sim)
        n = length(alphas)
        w = Vector{typeof(alpha)}(undef, n)

        w[1] = alphas[2] * alphas[1] / alphas[n]^2
        for i in 2:(n - 1)
            w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
        end
        w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]
        w2 = owa_wcvarrg(100, alphas, w)
        @test isapprox(w1, w2)

        w = owa_tg(100)
        r = OrderedWeightsArrayRange(; w1 = w, w2 = w)
        @test r.w1 == reverse(r.w2)

        r = OrderedWeightsArrayRange(; w1 = w, w2 = w, rev = true)
        @test r.w1 == r.w2
    end
end
