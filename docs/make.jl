using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations

DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup, :(using PortfolioOptimisers);
                    recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [file
                        for file in readdir(joinpath(@__DIR__, "src"))
                        if file != "index.md" && splitext(file)[2] == ".md"]

makedocs(; #modules = [PortfolioOptimisers],
         authors = "Daniel Celis Garza <daniel.celis.garza@gmail.com>",
         repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimisers.jl",
         format = Documenter.HTML(;
                                  canonical = "https://dcelisgarza.github.io/PortfolioOptimisers.jl",),
         pages = ["index.md"; numbered_pages[7:end];
                  "API" => numbered_pages[1:6]],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "References.bib");
                                         style = :numeric)])

deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimisers.jl")
