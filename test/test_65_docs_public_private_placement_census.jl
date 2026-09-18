#=
The placement census (ADR 0128 § Decision, § Consequences,
`docs/adr/0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md`).
Two mirrored top-level trees, `docs/src/public_api/` and `docs/src/private_api/`, replace
`docs/src/api/` outright: every `@docs` entry on a page under either tree must classify onto
the side that holds it — `Base.isexported`/`Base.ispublic`, or the foreign-owned-generic
rule, both in `docs/api_classification.jl`.

The gate is absolute, not a ratchet: a page enters either tree only when someone
deliberately migrates it, so there is no legacy debt to grandfather the way
`test_26_docs.jl`'s docstring ratchets do. It checks placement only, not completeness --
issue #554 already measured that every exported and `public` name is documented somewhere,
and that is not a per-page migration risk the way placement is.

The census reads the `.md` sources directly, for the reason
`test_64_docs_page_metadata_census.jl` gives: a docs build is slow and run by hand, and
everything the placement check needs -- the name, and which tree it sits under -- is in the
source itself, parsed by `docs_block_names` in `docs/page_metadata.jl`. Classification does
need the live module, because the source declares what is public and the docs must only
ever derive it.

Both trees pass vacuously until a migration ticket creates their first page: `mirror_tree_pages`
answers no pages for a directory that does not exist yet, so this file needs no edit when
the first page lands.
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
    end
end
