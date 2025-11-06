using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations, Literate, StatsPlots, GraphRecipes,
      Handcalcs, Statistics, StatsBase, DocumenterVitepress

DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup, :(using PortfolioOptimisers);
                    recursive = true)

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
function pre_process_content_md(content)
    return replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```",
                   r"^#note # (.*)$"m => s"""
           # !!! note
           #     \1""", r"^#warning # (.*)$"m => s"""
                       # !!! warning
                       #     \1""", r"^#tip # (.*)$"m => s"""
                            # !!! tip
                            #     \1""", r"^#info # (.*)$"m => s"""
                                 # !!! info
                                 #     \1""")
end
function pre_process_content_nb(content)
    return replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```",
                   r"^#note # (.*)$"m => s"""
           # > *note*
           # > \1""", r"^#warning # (.*)$"m => s"""
                # > *warning*
                # > \1""", r"^#tip # (.*)$"m => s"""
                   # > *tip*
                   # > \1""", r"^#info # (.*)$"m => s"""
                      # > *info*
                      # > \1""")
end

fix_suffix_md(filename) = replace(filename, ".jl" => ".md")
function postprocess(cont)
    return """
           The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
           """ * cont
end

example_path = joinpath(@__DIR__, "../examples/")
build_path_md = joinpath(@__DIR__, "src", "examples/")
files = readdir(example_path)
code_files = filter(x -> endswith(x, ".jl"), files)
data_files = filter(x -> (endswith(x, ".csv") || endswith(x, ".csv.gz")), files)
example_pages = fix_suffix_md.("examples/" .* code_files)

for file in data_files
    cp(joinpath(@__DIR__, "../examples/" * file),
       joinpath(@__DIR__, "src/examples/" * file); force = true)
end

for file in code_files
    Literate.markdown(example_path * file, build_path_md;
                      preprocess = pre_process_content_md, postprocess = postprocess,
                      documenter = true, credit = true)
    Literate.notebook(example_path * file, example_path;
                      preprocess = pre_process_content_nb, documenter = true, credit = true)
end

root_pages = [file
              for file in readdir(joinpath(@__DIR__, "src")) if splitext(file)[2] == ".md"]
api_pages = [item for item in walkdir(joinpath(@__DIR__, "src/api"))]
contribute = [joinpath("contribute", file)
              for file in readdir(joinpath(@__DIR__, "src/contribute"))
              if splitext(file)[2] == ".md"]
idx1 = findfirst("api", api_pages[1][1])[1]

makedocs(; #modules = [PortfolioOptimisers],
         authors = "Daniel Celis Garza <daniel.celis.garza@gmail.com>",
         repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimisers.jl",
         format = DocumenterVitepress.MarkdownVitepress(;
                                                        repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl"),
         pages = ["Home" => root_pages[1];
                  "Examples" => example_pages;
                  "API" => [joinpath.(api_pages[1][1][idx1:end], api_pages[1][3]);
                            "Moments" => joinpath.(api_pages[2][1][idx1:end],
                                                   api_pages[2][3])
                            "Distance" => joinpath.(api_pages[3][1][idx1:end],
                                                    api_pages[3][3])
                            "Phylogeny" => joinpath.(api_pages[4][1][idx1:end],
                                                     api_pages[4][3])
                            "Constraint Generation" => joinpath.(api_pages[5][1][idx1:end],
                                                                 api_pages[5][3])
                            "Prior" => joinpath.(api_pages[6][1][idx1:end], api_pages[6][3]);
                            "Uncertainty Sets" => joinpath.(api_pages[7][1][idx1:end],
                                                            api_pages[7][3])
                            "Risk Measures" => joinpath.(api_pages[8][1][idx1:end],
                                                         api_pages[8][3])
                            "Optimisation" => [joinpath.(api_pages[9][1][idx1:end],
                                                         api_pages[9][3])
                                               "Constraints" => joinpath.(api_pages[10][1][idx1:end],
                                                                          api_pages[10][3])
                                               "Risk Constraints" => joinpath.(api_pages[11][1][idx1:end],
                                                                               api_pages[11][3])]];
                  "Contribute" => contribute;
                  "References" => root_pages[2]],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "References.bib");
                                         style = :numeric)])

DocumenterVitepress.deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimisers.jl",
                               target = "build", devbranch = "main", branch = "gh-pages",
                               push_preview = true)

# ~/docs $ npm run docs:dev

# allpages = String[]
# for page in api_pages
#     append!(allpages, joinpath.(page[1][idx1:end], page[3]))
# end
