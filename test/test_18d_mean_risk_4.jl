include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk block1 25:32" begin
    mr_block1(25:32)
end
