include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk block1 9:16" begin
    mr_block1(9:16)
end
