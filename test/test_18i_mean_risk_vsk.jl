include(joinpath(@__DIR__, "test18_setup.jl"))

# The five VSK solves are split over three files, so that each file of the suite runs in about
# the same time. This file solves the cases 1 to 3.
@testset "Mean Risk VSK, cases 1 to 3" begin
    mr_block5(1:3)
end
