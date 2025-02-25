module PortfolioOptimisers

# Turn readme into PortfolioOptimiser's docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers

end
