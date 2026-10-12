include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk VSK" begin
    mr_block5()
end
