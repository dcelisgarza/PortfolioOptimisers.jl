@safetestset "OWA" begin
    using PortfolioOptimisers, CSV, DataFrames, Clarabel, Test, Random, StableRNGs
    @testset "OWA weight vectors" begin
        owa_t = CSV.read(joinpath(@__DIR__, "./assets/OWA_weights.csv"), DataFrame)
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
    end
end
