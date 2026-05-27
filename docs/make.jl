using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations, Literate, StatsPlots, GraphRecipes,
      Handcalcs, StatsBase, DocumenterVitepress, Dates, JuMP, StatsAPI, Random

exported_symbols = names(PortfolioOptimisers)
all_symbols = names(PortfolioOptimisers; all = true)
filter!(x -> !contains(string(x), r"#|^eval$|^include$"), all_symbols)
private_symbols = setdiff(all_symbols, exported_symbols)
for sym in private_symbols
    eval(quote
             import PortfolioOptimisers: $(sym)
         end)
end

DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup,
                    :(using PortfolioOptimisers, StatsBase, Statistics, LinearAlgebra,
                            Dates, Distributions, StableRNGs, TimeSeries); recursive = true)

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

function generate_files(source::String, build::String, diff_flag::Bool)
    function postprocess(cont)
        return """
               The source files can be found in [$(build)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/$(build)).
               """ * cont
    end

    src_path = joinpath(@__DIR__, source)
    build_path = joinpath(@__DIR__, "src", build)
    files = readdir(src_path)
    code_files = filter(x -> endswith(x, ".jl"), files)
    data_files = filter(x -> (endswith(x, ".csv") || endswith(x, ".csv.gz")), files)
    pages = fix_suffix_md.(build .* code_files)

    for file in data_files
        if isempty(String(read(Cmd(`git diff $(joinpath(@__DIR__, source, file))`))))
            continue
        end
        cp(joinpath(@__DIR__, source, file), joinpath(@__DIR__, "src", build, file);
           force = true)
    end

    for file in code_files
        if diff_flag &&
           isempty(String(read(Cmd(`git diff $(joinpath(@__DIR__, source, file))`))))
            continue
        end
        Literate.markdown(src_path * file, build_path; preprocess = pre_process_content_md,
                          postprocess = postprocess, documenter = true, credit = true)
        Literate.notebook(src_path * file, src_path; preprocess = pre_process_content_nb,
                          documenter = true, credit = true)
    end

    return pages
end

diff_flag = isempty(String(read(Cmd(`git diff $(@__DIR__) $(joinpath(@__DIR__, "../src/")) $(joinpath(@__DIR__, "../ext/")) $(joinpath(@__DIR__, "../test/"))`))))

examples = generate_files("../examples/", "examples/", diff_flag)
user_guide = generate_files("../user_guide/", "user_guide/", diff_flag)

root_pages = [file
              for file in readdir(joinpath(@__DIR__, "src")) if splitext(file)[2] == ".md"]
api_pages = [item for item in walkdir(joinpath(@__DIR__, "src/api"))]
contribute = [joinpath("contribute", file)
              for file in readdir(joinpath(@__DIR__, "src/contribute"))
              if splitext(file)[2] == ".md"]
idx1 = findfirst("api", api_pages[1][1])[1]

makedocs(; modules = [PortfolioOptimisers], doctest = false,
         authors = "Daniel Celis Garza <daniel.celis.garza@gmail.com>",
         repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimisers.jl",
         format = DocumenterVitepress.MarkdownVitepress(;
                                                        repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl"),
         pages = ["Home" => root_pages[1];
                  "User Guide" => user_guide;
                  "Examples" => examples;
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
                                               joinpath.(api_pages[10][1][idx1:end],
                                                         api_pages[10][3])
                                               joinpath.(api_pages[11][1][idx1:end],
                                                         api_pages[11][3])
                                               joinpath.(api_pages[12][1][idx1:end],
                                                         api_pages[12][3])]];
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
