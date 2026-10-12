include(joinpath(@__DIR__, "test18_setup.jl"))

# The five VSK solves are split over two files, so that each file of the suite runs in about
# the same time. This file solves the cases 4 to 6.
@testset "Mean Risk VSK, cases 4 to 6" begin
    mr_block5(4:6)
end
