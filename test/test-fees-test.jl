@safetestset "Fees" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "Fees" begin
        rng = StableRNG(987654321)
        w1 = randn(rng, 20)
        w2 = randn(rng, 20)
        lw = rand(rng, 20)
        sw = rand(rng, 20)
        flw = rand(rng, 20)
        slw = rand(rng, 20)
        tv = rand(rng, 20)
        P = randn(rng, 20)
        fees_estimator = [Fees(; long = 0.01, short = 0.02, fixed_long = 0.05,
                               fixed_short = 0.07,
                               turnover = Turnover(; val = 0.11, w = w2)), Fees(;),
                          Fees(; long = 0.01, short = 0.02, fixed_long = 0.05,
                               fixed_short = 0.07),
                          Fees(; long = lw, short = sw, fixed_long = flw, fixed_short = slw,
                               turnover = Turnover(; val = tv, w = w2)), Fees(;), Fees(;),
                          Fees(;)]
        f1_t = CSV.read(joinpath(@__DIR__, "assets/Fees.csv"), DataFrame)
        f2_t = CSV.read(joinpath(@__DIR__, "assets/Asset-Fees.csv"), DataFrame)

        for (i, fe) ∈ enumerate(fees_estimator)
            f1 = calc_fees(w1, fe)
            res1 = isapprox(f1, f1_t[(i - 1) * 2 + 1, 1])
            if !res1
                println("Fee $i failed: $f1 != $(f1_t[(cntr+1),1])")
                find_tol(f1, f1_t[(i - 1) * 2 + 1, 1]; name1 = :f1, name2 = :f1_t)
            end
            @test res1

            f2 = calc_fees(w1, P, fe)
            res2 = isapprox(f2, f1_t[(i - 1) * 2 + 2, 1])
            if !res2
                println("Fee $i failed: $f2 != $(f1_t[(cntr+2),1])")
                find_tol(f2, f1_t[(i - 1) * 2 + 2, 1]; name1 = :f3, name2 = :f1_t)
            end
            @test res2

            f3 = calc_asset_fees(w1, fe)
            res3 = isapprox(f3, f2_t[1:20, i])
            if !res3
                println("Asset fee $i failed")
                find_tol(f3, f2_t[1:20, i]; name1 = :f2, name2 = :f2_t)
            end
            @test res3
            @test isapprox(f1, sum(f3))

            f4 = calc_asset_fees(w1, P, fe)
            res4 = isapprox(f4, f2_t[21:end, i])
            if !res4
                println("Asset fee with prices $i failed")
                find_tol(f4, f2_t[21:end, i]; name1 = :f2, name2 = :f2_t)
            end
            @test res4
            @test isapprox(f2, sum(f4))
        end
    end
end
