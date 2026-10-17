include(joinpath(@__DIR__, "test18_setup.jl"))

# The five VSK solves are split over three files, so that each file of the suite runs in
# about the same time. This file solves the case 4.
@testset "Mean Risk VSK, case 4" begin
    mr_block5(4:4)
end
