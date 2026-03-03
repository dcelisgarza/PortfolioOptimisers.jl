@safetestset "Docs completeness" begin
    using PortfolioOptimisers, Test
    all_names = Base.undocumented_names(PortfolioOptimisers; private = true)
    public_names = Base.undocumented_names(PortfolioOptimisers; private = false)
    private_names = setdiff(all_names, public_names)
    @test length(public_names) == 163
    @test length(private_names) == 403
end
