using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations, Literate, StatsPlots, GraphRecipes,
      Handcalcs, StatsBase, DocumenterVitepress

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
examples_nav = fix_suffix_md.("examples/" .* code_files)

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

page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
numbered_pages = [file
                  for file in readdir(joinpath(@__DIR__, "src"))
                  if file != "index.md" && splitext(file)[2] == ".md"]

makedocs(; #modules = [PortfolioOptimisers],
         authors = "Daniel Celis Garza <daniel.celis.garza@gmail.com>",
         repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimisers.jl",
         format = DocumenterVitepress.MarkdownVitepress(;
                                                        repo = "https://dcelisgarza.github.io/PortfolioOptimisers.jl"),
         pages = ["index.md";
                  "Examples" => examples_nav;
                  "API" => [numbered_pages[1:6];
                            "Moments" => numbered_pages[7:27];
                            "Distance" => numbered_pages[28:32];
                            "JuMP Model Optimisation" => numbered_pages[33];
                            "Ordered Weights Array" => numbered_pages[34];
                            "Phylogeny" => numbered_pages[35:39];
                            "Constraint Generation" => numbered_pages[40:44];
                            "Prior" => numbered_pages[45];
                            "Optimisation" => numbered_pages[46]];
                  numbered_pages[end];
                  "Contribute" => numbered_pages[47:(end - 1)]],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "References.bib");
                                         style = :numeric)])

DocumenterVitepress.deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimisers.jl",
                               target = "build", devbranch = "main", branch = "gh-pages",
                               push_preview = true)

# ~/docs $ npm run docs:dev
