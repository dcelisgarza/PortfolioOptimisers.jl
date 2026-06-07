include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk block1 1:8" begin
    mr_block1(1:8)
end
