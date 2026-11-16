#=
The placement census (ADR 0128 § Decision, § Consequences,
`docs/adr/0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md`).
Two mirrored top-level trees, `docs/src/public_api/` and `docs/src/private_api/`, replace
`docs/src/api/` outright: every `@docs` entry on a page under either tree must classify onto
the side that holds it — `Base.isexported`/`Base.ispublic`, or the foreign-owned-generic
rule, both in `docs/api_classification.jl`.

The gate is absolute, not a ratchet: a page enters either tree only when someone
deliberately migrates it, so there is no legacy debt to grandfather the way
`test_26_docs.jl`'s docstring ratchets do. It checks placement, not completeness --
issue #554 already measured that every exported and `public` name is documented somewhere,
and that is not a per-page migration risk the way placement is. It also checks that every page
sits at the path of a source file under `src/` (ADR 0150), because the docs build walks the
pages it finds and so builds a page that a rename left behind.

The census reads the `.md` sources directly, for the reason
`test_64_docs_page_metadata_census.jl` gives: a docs build is slow and run by hand, and
everything the placement check needs -- the name, and which tree it sits under -- is in the
source itself, parsed by `docs_block_names` in `docs/page_metadata.jl`. Classification does
need the live module, because the source declares what is public and the docs must only
ever derive it.

`mirror_tree_pages` answers no pages for a directory that does not exist. Both trees hold pages
since the migration, so the path check asserts that each tree is not empty.
=#
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "docs", "page_metadata.jl"))
include(joinpath(REPO_ROOT, "docs", "api_classification.jl"))

# Every `.md` page under a mirror tree's root, walked recursively so a nested directory
# (ADR 0150) is covered the same as a top-level file.
function mirror_tree_pages(repo::AbstractString, side::Symbol)
    root = joinpath(repo, "docs", "src", MIRROR_TREES[side])
    pages = String[]
    if !(isdir(root))
        return pages
    end
    for (dir, _, files) in walkdir(root)
        for file in files
            if !(endswith(file, ".md"))
                continue
            end
            push!(pages, joinpath(dir, file))
        end
    end
    return pages
end

@testset "Docs public/private placement census" begin
    using PortfolioOptimisers, Test

    for side in (:public, :private)
        @testset "$(MIRROR_TREES[side]): every entry is on its declared side" begin
            mismatches = Tuple{String, String, Symbol}[]
            for page in mirror_tree_pages(REPO_ROOT, side)
                for name in docs_block_names(read(page, String))
                    classification = classify_binding(name)
                    classification === side ||
                        push!(mismatches, (relpath(page, REPO_ROOT), name, classification))
                end
            end
            if !isempty(mismatches)
                rows = ["$page: `$name` classifies $c" for (page, name, c) in mismatches]
                @warn """$(length(mismatches)) entr$(length(mismatches) == 1 ? "y" : "ies") on
                         the wrong side of $(MIRROR_TREES[side]):\n  $(join(rows, "\n  "))"""
            end
            @test isempty(mismatches)
        end
        @testset "$(MIRROR_TREES[side]): every page has a source file at its path" begin
            # The trees mirror `src/`, one page per source file at the same path (ADR 0150).
            # The docs build walks the pages that it finds, so a page that a rename of its
            # source file leaves behind builds green: #2061 renamed
            # `13_CrossSectionalLeverage.jl` and left both of its pages at the old path.
            root = joinpath(REPO_ROOT, "docs", "src", MIRROR_TREES[side])
            pages = mirror_tree_pages(REPO_ROOT, side)
            @test !isempty(pages)
            orphans = String[]
            for page in pages
                source = joinpath(REPO_ROOT, "src",
                                  splitext(relpath(page, root))[1] * ".jl")
                isfile(source) || push!(orphans, relpath(page, REPO_ROOT))
            end
            if !isempty(orphans)
                @warn """Pages of $(MIRROR_TREES[side]) with no source file at the same path under
                         `src/`. Rename each page with its source file:\n  $(join(orphans, "\n  "))"""
            end
            @test isempty(orphans)
        end
    end
end
