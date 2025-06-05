@safetestset "Fees" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames, FLoops
    import PortfolioOptimisers: fees_view
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
        fes = [Fees(; l = 0.01, s = 0.02, fl = 0.05, fs = 0.07,
                    tn = Turnover(; val = 0.11, w = w2)), Fees(;),
               Fees(; l = 0.01, s = 0.02, fl = 0.05, fs = 0.07),
               Fees(; l = lw, s = sw, fl = flw, fs = slw,
                    tn = Turnover(; val = tv, w = w2)), Fees(;), Fees(;), Fees(;)]
        f1_t = CSV.read(joinpath(@__DIR__, "assets/Fees.csv"), DataFrame)
        f2_t = CSV.read(joinpath(@__DIR__, "assets/Asset-Fees.csv"), DataFrame)

        Threads.@threads for i ∈ eachindex(fes)
            fe = fes[i]
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
        X = randn(rng, 1000, 20)
        Xr = calc_net_returns(w1, X, nothing)
        f1 = calc_fees(w1, fes[4])
        @test Xr == X * w1
        @test Xr .- f1 == calc_net_returns(w1, X, fes[4])

        Xar = calc_net_asset_returns(w1, X, nothing)
        fa1 = calc_asset_fees(w1, fes[4])
        @test Xar == X .* transpose(w1)
        @test Xar .- transpose(fa1) == calc_net_asset_returns(w1, X, fes[4])

        @test isnothing(fees_view(nothing, 3))
        fer1 = fees_view(fes[1], [3, 7])
        @test fer1.l == 0.01
        @test fer1.fl == 0.05
        @test fer1.s == 0.02
        @test fer1.fs == 0.07
        @test fer1.tn.val == 0.11
        @test fer1.tn.w == view(w2, [3, 7])
        fer2 = fees_view(fer1, [1])
        @test fer2.l == 0.01
        @test fer2.fl == 0.05
        @test fer2.s == 0.02
        @test fer2.fs == 0.07
        @test fer2.tn.val == 0.11
        @test fer2.tn.w == view(w2, [3])

        fer3 = fees_view(fes[3], [9, 8])
        @test fer3.l == 0.01
        @test fer3.fl == 0.05
        @test fer3.s == 0.02
        @test fer3.fs == 0.07
        @test isnothing(fer3.tn)

        fer1 = fees_view(fes[4], [5, 2])
        @test fer1.l == view(lw, [5, 2])
        @test fer1.fl == view(flw, [5, 2])
        @test fer1.s == view(sw, [5, 2])
        @test fer1.fs == view(slw, [5, 2])
        @test fer1.tn.val == view(tv, [5, 2])
        @test fer1.tn.w == view(w2, [5, 2])
        fer2 = fees_view(fer1, [2])
        @test fer2.l == view(lw, [2])
        @test fer2.fl == view(flw, [2])
        @test fer2.s == view(sw, [2])
        @test fer2.fs == view(slw, [2])
        @test fer2.tn.val == view(tv, [2])
        @test fer2.tn.w == view(w2, [2])
    end
end
