include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk IndependentTracking" begin
    mr_block3(1:47)
end
