include(joinpath(@__DIR__, "test18_setup.jl"))

# The five VSK solves are split over three files, so that each file of the suite runs in
# about the same time. This file solves the cases 5 and 6.
@testset "Mean Risk VSK, cases 5 and 6" begin
    mr_block5(5:6)
end
