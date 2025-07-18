using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations, Literate, StatsPlots, GraphRecipes

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
examples_nav = fix_suffix_md.("./examples/" .* code_files)

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
         pages = ["index.md"; "Examples" => examples_nav;
                  numbered_pages[24:end];
                  "API" => [numbered_pages[1:6]; "Moments" => numbered_pages[7:23]]],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "References.bib");
                                         style = :numeric)])

deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimisers.jl")
