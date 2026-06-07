include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk block1 33:40" begin
    mr_block1(33:40)
end
