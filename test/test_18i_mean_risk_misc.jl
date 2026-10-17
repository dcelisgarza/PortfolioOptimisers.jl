include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Mean Risk fallback/BDV" begin
    mr_inline()
    mr_block4()
end
