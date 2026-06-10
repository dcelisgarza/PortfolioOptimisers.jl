include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk fallback/BDV/VSK" begin
    mr_inline()
    mr_block4()
    mr_block5()
end
