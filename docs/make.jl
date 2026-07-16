using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations, Literate, StatsPlots, GraphRecipes,
      Handcalcs, StatsBase, DocumenterVitepress, Dates, JuMP, StatsAPI, Random

f = x -> !contains(string(x), r"#|^eval$|^include$")
exported_symbols = filter!(f, names(PortfolioOptimisers))
all_symbols = filter!(f, names(PortfolioOptimisers; all = true))
private_symbols = setdiff(all_symbols, exported_symbols)
public_symbols = exported_symbols[findall(x->!Base.isexported(PortfolioOptimisers, x),
                                          exported_symbols)]
for sym in [private_symbols; public_symbols]
    eval(quote
             import PortfolioOptimisers: $(sym)
         end)
end

# Keep rendered docs and @example output fully expanded (no large-struct collapsing).
PortfolioOptimisers.set_compact_show!(false)

DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup,
                    :(using PortfolioOptimisers, StatsBase, Statistics, LinearAlgebra,
                            Dates, Distributions, StableRNGs, TimeSeries;
                    PortfolioOptimisers.set_compact_show!(false)); recursive = true)

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

# Display labels for the pipeline-stage example groups (see ADR 0014). Subdirectories not
# listed here fall back to a derived title-case label.
const GROUP_LABELS = Dict("1_foundations" => "Foundations",
                          "2_moments_priors" => "Moments & Priors",
                          "3_optimisers" => "Optimisers",
                          "4_constraints_costs" => "Constraints & Costs",
                          "5_validation_tuning" => "Validation & Tuning",
                          "6_post_processing" => "Post-processing",
                          "7_putting_it_together" => "Putting It Together")

# Regenerate Literate outputs against the previous commit: resolve HEAD's parent and only
# rebuild the examples/guide whose sources changed since then. A shared-code change
# (src/ext/test) can alter every example's rendered output, so it forces a full rebuild.
# If the parent can't be resolved (initial commit or a shallow CI checkout), rebuild
# everything.
const DIFF_REF = try
    strip(String(read(`git rev-parse HEAD"~"1`)))
catch
    ""
end

function group_label(dir)
    return get(GROUP_LABELS, dir,
               titlecase(replace(replace(dir, r"^\d+_" => ""), "_" => " ")))
end

# Build the Documenter page list for a narrative section (user guide or examples).
# Top-level `.jl` files become flat pages (e.g. the section intro); each subdirectory
# becomes a nested group of pages, in directory-name order. Shared data files live in the
# section root and are copied into the build root so `@example` blocks in any subdirectory
# resolve `../<data>` at build time (ADR 0014).
function generate_files(source::String, build::String, diff_flag::Bool)
    function postprocess(cont)
        return """
               The source files can be found in [$(build)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/$(build)).
               """ * cont
    end

    src_path = joinpath(@__DIR__, source)
    build_path = joinpath(@__DIR__, "src", build)
    mkpath(build_path)

    for file in readdir(src_path)
        if !(endswith(file, ".csv") || endswith(file, ".csv.gz"))
            continue
        end
        cp(joinpath(src_path, file), joinpath(build_path, file); force = true)
    end

    # Render one Literate source: markdown into the build dir, notebook next to the source.
    # Returns the page path relative to docs/src.
    function process(jl, src_dir_abs, out_build_abs, rel_build)
        if !(diff_flag &&
             isempty(String(read(Cmd(`git diff $DIFF_REF -- $(joinpath(src_dir_abs, jl))`)))))
            Literate.markdown(joinpath(src_dir_abs, jl), out_build_abs;
                              preprocess = pre_process_content_md,
                              postprocess = postprocess, documenter = true, credit = true)
            Literate.notebook(joinpath(src_dir_abs, jl), src_dir_abs;
                              preprocess = pre_process_content_nb, documenter = true,
                              credit = true)
        end
        return joinpath(rel_build, fix_suffix_md(jl))
    end

    code_files(dir) = sort(filter(x -> endswith(x, ".jl"), readdir(dir)))

    pages = []
    for jl in code_files(src_path)
        push!(pages, process(jl, src_path, build_path, build))
    end
    for d in sort(filter(x -> isdir(joinpath(src_path, x)), readdir(src_path)))
        group_src = joinpath(src_path, d)
        group_build = joinpath(build_path, d)
        mkpath(group_build)
        group_pages = [process(jl, group_src, group_build, joinpath(build, d))
                       for jl in code_files(group_src)]
        if isempty(group_pages)
            continue
        end
        push!(pages, group_label(d) => group_pages)
    end

    return pages
end

# `diff_flag == true` enables selective per-file rebuilding (see `process`); we only enable
# it when the shared code is unchanged relative to the previous commit.
diff_flag = !isempty(DIFF_REF) &&
            isempty(String(read(Cmd(`git diff $DIFF_REF -- $(joinpath(@__DIR__, "../src/")) $(joinpath(@__DIR__, "../ext/")) $(joinpath(@__DIR__, "../test/"))`))))

examples = generate_files("../examples/", "examples/", diff_flag)
user_guide = generate_files("../user_guide/", "user_guide/", diff_flag)

include(joinpath(@__DIR__, "generate_type_hierarchy.jl"))
generate_type_hierarchy()

root_pages = [file
              for file in readdir(joinpath(@__DIR__, "src")) if splitext(file)[2] == ".md"]
home = popat!(root_pages, findfirst(x->contains(x, "index"), root_pages))
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
         pages = ["Home" => home;
                  "User Guide" => user_guide;
                  "Examples" => examples;
                  "API" => [joinpath.(api_pages[1][1][idx1:end], api_pages[1][3]);
                            "Moments" => joinpath.(api_pages[2][1][idx1:end], api_pages[2][3])
                            "Distance" => joinpath.(api_pages[3][1][idx1:end], api_pages[3][3])
                            "Phylogeny" => joinpath.(api_pages[4][1][idx1:end], api_pages[4][3])
                            "Constraint Generation" =>
                                joinpath.(api_pages[5][1][idx1:end], api_pages[5][3])
                            "Prior" => joinpath.(api_pages[6][1][idx1:end], api_pages[6][3]);
                            "Uncertainty Sets" =>
                                joinpath.(api_pages[7][1][idx1:end], api_pages[7][3])
                            "Risk Measures" =>
                                joinpath.(api_pages[8][1][idx1:end], api_pages[8][3])
                            "Optimisation" =>
                                [joinpath.(api_pages[9][1][idx1:end], api_pages[9][3])
                                 joinpath.(api_pages[10][1][idx1:end], api_pages[10][3])
                                 joinpath.(api_pages[11][1][idx1:end], api_pages[11][3])
                                 joinpath.(api_pages[12][1][idx1:end], api_pages[12][3])]
                            "Pipeline" =>
                                joinpath.(api_pages[13][1][idx1:end], api_pages[13][3])];
                  "Contribute" => contribute;
                  "References" => root_pages[end]],
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
