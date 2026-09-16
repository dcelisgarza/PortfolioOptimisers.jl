using PortfolioOptimisers
using Documenter, DocumenterTools, DocumenterCitations, DocumenterCodeBlocks,
      DocumenterLandingPage, Literate, StatsPlots, GraphRecipes, Handcalcs, StatsBase,
      Dates, JuMP, StatsAPI, Random

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

# Keep rendered docs and @example output fully expanded (no large-struct collapsing), and
# render every field, including one that holds `nothing`, so the docs show the complete type.
PortfolioOptimisers.set_compact_show!(false)
PortfolioOptimisers.set_show_nothing_fields!(true)

# `@example` output is captured into a plain `IOBuffer`, whose `displaysize` falls back to
# `Base.displaysize()` — i.e. `ENV["LINES"]`/`ENV["COLUMNS"]`, defaulting to 24×80. PrettyTables
# v3 fits tables to that size, so every `pretty_table` call in the guide and examples silently
# loses rows and columns. Widening the environment here fixes all of them at once, with no
# call-site options to keep in sync.
ENV["LINES"] = 100_000
ENV["COLUMNS"] = 100_000

DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup,
                    :(using PortfolioOptimisers, StatsBase, Statistics, LinearAlgebra,
                            Dates, Distributions, StableRNGs, TimeSeries;
                    PortfolioOptimisers.set_compact_show!(false);
                    PortfolioOptimisers.set_show_nothing_fields!(true)); recursive = true)

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
fix_suffix_nb(filename) = replace(filename, ".jl" => ".ipynb")

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
        jlp = joinpath(src_dir_abs, jl)
        md_out = joinpath(out_build_abs, fix_suffix_md(jl))
        nb_out = joinpath(src_dir_abs, fix_suffix_nb(jl))
        # The selective path asks only whether the SOURCE changed since `DIFF_REF`. A
        # missing output therefore has to force a rebuild on its own: otherwise a deleted
        # (or never generated) `.md` stays missing and `makedocs` fails on the absent page,
        # and a deleted `.ipynb` is never restored.
        unchanged = diff_flag && isempty(String(read(Cmd(`git diff $DIFF_REF -- $jlp`))))
        if !unchanged || !isfile(md_out) || !isfile(nb_out)
            Literate.markdown(jlp, out_build_abs; preprocess = pre_process_content_md,
                              postprocess = postprocess, documenter = true, credit = true)
            Literate.notebook(jlp, src_dir_abs; preprocess = pre_process_content_nb,
                              documenter = true, credit = true)
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

include(joinpath(@__DIR__, "generate_capability_catalogue.jl"))
generate_capability_catalogue()

# Root-level pages are named explicitly rather than picked positionally out of
# `readdir`. The previous form took `root_pages[end]` as the references page,
# which held only while `99_references.md` happened to sort last -- dropping any
# new root page silently repointed the "References" nav entry at it.
const HOME_PAGE = "index.md"
const REFERENCES_PAGE = "99_references.md"
const CATALOGUE_PAGE = "capability_catalogue.md"
const MIGRATION_PAGE = "migration.md"
for page in (HOME_PAGE, REFERENCES_PAGE, CATALOGUE_PAGE, MIGRATION_PAGE)
    if !(isfile(joinpath(@__DIR__, "src", page)))
        error("docs/make.jl: expected root page `$page` is missing.")
    else
        true
    end
end
# Display labels for the API groups whose directory name does not read as a title once its
# numeric prefix is dropped and its CamelCase is split on the lower-to-upper boundaries.
const API_GROUP_LABELS = Dict("InputData" => "Input Data",
                              "TimeSeriesRegression" => "Time-series Regression",
                              "CrossSectionalFactorModel" => "Cross-sectional Factor Model",
                              "DBHT" => "DBHT", "BlackLitterman" => "Black-Litterman",
                              "XatRisk" => "X-at-Risk", "JuMP" => "JuMP",
                              "JuMPConstraints" => "JuMP Constraints")

function api_group_label(dir)
    name = replace(dir, r"^\d+_" => "")
    return get(API_GROUP_LABELS, name, replace(name, r"(?<=[a-z])(?=[A-Z])" => " "))
end

# The API page tree mirrors `src/`: one page per source file at the same path, and a
# directory of pages for every directory of files (ADR 0150). Files and directories share
# one numbering at each level, so the listing sorted by name is the load order of `src/`,
# and a directory becomes a nested group labelled by its name. Nothing here is positional:
# a directory added under `docs/src/api/` appears where its number puts it.
function api_pages(dir::String, rel::String)
    pages = Any[]
    for entry in sort(readdir(dir))
        path = joinpath(dir, entry)
        if isdir(path)
            group = api_pages(path, joinpath(rel, entry))
            isempty(group) || push!(pages, api_group_label(entry) => group)
        elseif endswith(entry, ".md")
            push!(pages, joinpath(rel, entry))
        end
    end
    return pages
end
api = api_pages(joinpath(@__DIR__, "src/api"), "api")
contribute = [joinpath("contribute", file)
              for file in readdir(joinpath(@__DIR__, "src/contribute"))
              if splitext(file)[2] == ".md"]

# The base URL of the deployed site. `Documenter.HTML` writes it into every page's
# `<link rel="canonical">`, and `generate_sitemap` roots every sitemap entry at it.
const CANONICAL_URL = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable"

# The site-wide `<meta name="description">`, which Documenter writes into every page that
# states no `Description` of its own (#561). It is the landing page's own line, read off
# `index.md` so the two cannot drift, and the generated search page is the one page that
# always takes it. `test/test_64_docs_page_metadata_census.jl` checks that every other page
# states its own line, so a page that carries this one has slipped the gate.
include(joinpath(@__DIR__, "page_metadata.jl"))
const SITE_DESCRIPTION = meta_description(read(joinpath(@__DIR__, "src", HOME_PAGE),
                                               String))
if isnothing(SITE_DESCRIPTION)
    error("docs/make.jl: `$HOME_PAGE` states no `Description` in its `@meta` block.")
end

makedocs(; modules = [PortfolioOptimisers], doctest = false,
         authors = "Daniel Celis Garza <daniel.celis.garza@gmail.com>",
         repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimisers.jl",
         # Bare Documenter HTML. `DocumenterCodeBlocks` and `DocumenterLandingPage` both
         # need the stock HTML writer: the first rewrites rendered code blocks, the second
         # injects its stylesheet into `HTML.assets`. Neither sees a Markdown build.
         format = Documenter.HTML(;
                                  # Always on, so the deployed and the local build agree. A
                                  # local build therefore needs a server, not `file://`:
                                  # `using LiveServer; serve(; dir = "docs/build")`.
                                  prettyurls = true,
                                  # The base URL of the deployed site.
                                  # `generate_sitemap` roots every sitemap
                                  # entry at the same constant, so the sitemap
                                  # and the canonical tags cannot disagree.
                                  canonical = CANONICAL_URL,
                                  # The fallback description; see `SITE_DESCRIPTION`.
                                  description = SITE_DESCRIPTION,
                                  # `repo` above is a String, so Documenter cannot
                                  # derive the navbar link. Name the remote explicitly.
                                  repolink = "https://github.com/dcelisgarza/PortfolioOptimisers.jl",
                                  # Landing-page customisation. Documenter emits
                                  # this BEFORE the plugin's own stylesheet, so read
                                  # the file's header before adding an override.
                                  # `generated-pages.css` styles the two generated
                                  # pages (capability catalogue, type hierarchy),
                                  # whose markup no stock theme covers.
                                  assets = ["assets/landing-overrides.css",
                                            "assets/generated-pages.css"],
                                  edit_link = "main",
                                  # The API pages and the capability catalogue go
                                  # far past the 200 KiB default, so the size gate is
                                  # off and only the warning survives.
                                  size_threshold = nothing,
                                  size_threshold_warn = 400 * 2^10),
         # The landing page's label is its `<title>` and its sidebar entry, so it carries
         # the words a search ranks the site on, not "Home" (#561).
         pages = ["Portfolio optimisation library in Julia" => HOME_PAGE;
                  "Capability Catalogue" => CATALOGUE_PAGE;
                  "User Guide" => user_guide;
                  "Examples" => examples;
                  "API" => api;
                  "Contribute" => contribute;
                  "Migration guide" => MIGRATION_PAGE;
                  "References" => REFERENCES_PAGE],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "References.bib");
                                         style = :numeric), CodeBlocks(), LandingPage()])

# The sitemap is written from the pages `makedocs` actually built, so it must run after
# `makedocs`, and before `deploydocs` copies `docs/build` into the deployed tree.
include(joinpath(@__DIR__, "generate_sitemap.jl"))
generate_sitemap(joinpath(@__DIR__, "build"), CANONICAL_URL)

deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimisers.jl", target = "build",
           devbranch = "main", branch = "gh-pages", push_preview = true)

# ---------------------------------------------------------------------------------------
# Viewing the docs locally
# ---------------------------------------------------------------------------------------
#
# 1. Build, from the repository root:
#
#        julia --project=docs docs/make.jl
#
#    The first run precompiles the docs environment. Expect about 35 minutes in total:
#    Literate executes all 62 notebooks, and Documenter then runs every `@example` block.
#    `deploydocs` is inert outside CI, so a local build never touches `gh-pages`.
#
# 2. Serve `docs/build`. `prettyurls` is always on, so every page is `<name>/index.html`
#    and a `file://` open shows nothing. A server is required. Either of these works:
#
#        julia --project=docs -e 'using LiveServer; serve(; dir = "docs/build")'
#        python3 -m http.server 8000 --directory docs/build
#
#    Then open http://localhost:8000.
#
# A rebuild is only needed for a content change. To tune CSS, such as the landing-page
# glow, edit `docs/build/assets/landing-overrides.css` and reload the page. Copy the final
# value back into `docs/src/assets/landing-overrides.css`, because the next build
# overwrites everything under `docs/build`.
#
# Selective rebuilds: `process` re-renders a page when its source changed since `HEAD~1`,
# or when its `.md` or `.ipynb` output is missing. Delete an output to force one page.
