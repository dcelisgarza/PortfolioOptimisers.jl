@safetestset "Docs completeness" begin
    using PortfolioOptimisers, Test
    all_names = Base.undocumented_names(PortfolioOptimisers; private = true)
    public_names = Base.undocumented_names(PortfolioOptimisers; private = false)
    private_names = setdiff(all_names, public_names)
    @test length(public_names) == 166
    @test length(private_names) == 395
end
