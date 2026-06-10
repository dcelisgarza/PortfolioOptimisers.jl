include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk DependentTracking" begin
    mr_block2(1:47)
end
