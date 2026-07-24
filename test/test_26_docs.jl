@testset "Docs completeness" begin
    using PortfolioOptimisers, Test
    all_names = Base.undocumented_names(PortfolioOptimisers; private = true)
    public_names = Base.undocumented_names(PortfolioOptimisers; private = false)
    private_names = setdiff(all_names, public_names)
    @test length(public_names) == 0
    @test length(private_names) == 0
end

#=
The Capability Catalogue (`docs/capability_catalogue.jl`, ADR 0040) is the
user-facing inventory of everything this package can do. It curates only the
*grouping*; each entry's description is the first sentence of the corresponding
docstring, resolved when the docs are built.

The tests below are what stop the page falling behind the code, and they live
here rather than in `docs/make.jl` deliberately: a full docs build is slow and
run by hand, so a gap found only there is a gap found months late. That is
exactly how the hand-maintained page this replaced came to be missing 96
estimators.

The catalogue is plain data with no dependencies precisely so it can be
`include`d from this environment as well as from `docs/`. It is included at
top level because it defines structs, which `@testset` -- whose body becomes a
function -- cannot host.
=#
@testset "Capability catalogue" begin
    using Test, InteractiveUtils
    # The generator includes the catalogue itself, and brings `render_catalogue` /
    # `assert_refs_survive` with it -- the rendering checks below need the real
    # rendered page, not just the declaration.
    include(joinpath(@__DIR__, "..", "docs", "generate_capability_catalogue.jl"))

    # Every name the page reaches, whether as a `Cap`'s target or as an `@ref`
    # written into prose or a label. A prose link is just as reachable by a reader
    # as a bullet is, so it counts as catalogued.
    function catalogued_names(entries)
        acc = Set{Symbol}()
        scan_text(t::AbstractString) =
            for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", t)
                push!(acc, Symbol(m.captures[1]))
            end
        scan(c::Cap) = (foreach(n -> push!(acc, n isa Symbol ? n : Symbol(n)), c.names);
                        if !(isnothing(c.label))
                            scan_text(c.label)
                        end)
        scan(n::Prose) = scan_text(n.text)
        scan(n::Note) = (scan_text(n.text); foreach(scan, n.children))
        scan(n::Section) = foreach(scan, n.children)
        scan(n::Group) = (n.head isa Cap ? scan(n.head) : scan_text(n.head);
                          foreach(scan, n.children))
        foreach(scan, entries)
        return acc
    end

    # Visit every `Cap`, including the ones heading a `Group`.
    function each_cap(f, entries)
        visit(c::Cap) = f(c)
        visit(::Prose) = nothing
        visit(n::Note) = foreach(visit, n.children)
        visit(n::Section) = foreach(visit, n.children)
        visit(n::Group) = (if n.head isa Cap
                               f(n.head)
                           end; foreach(visit, n.children))
        foreach(visit, entries)
        return nothing
    end

    # A leaf is a non-abstract type with no subtypes. Note `!isabstracttype` rather
    # than `isconcretetype`: nearly every struct here is `@concrete`, so the bare
    # name is a `UnionAll` and `isconcretetype` is false for every one of them.
    function leaf_types(T, acc = Set{Type}())
        subs = subtypes(T)
        if isempty(subs)
            if !isabstracttype(T)
                push!(acc, T)
            else
                false
            end
        else
            foreach(S -> leaf_types(S, acc), subs)
        end
        return acc
    end

    @testset "Capability catalogue" begin
        PO = PortfolioOptimisers
        catalogued = catalogued_names(CATALOGUE)

        @testset "every estimator and algorithm is catalogued" begin
            # Estimators and Algorithms are the user's choice surface (CONTEXT.md);
            # Results are outputs nobody constructs, so they are not required here.
            required = Set(nameof.(collect(union(leaf_types(PO.AbstractEstimator),
                                                 leaf_types(PO.AbstractAlgorithm)))))
            filter!(n -> !contains(string(n), "_test"), required)
            uncatalogued = sort(collect(setdiff(required, catalogued)))
            if !isempty(uncatalogued)
                @warn """$(length(uncatalogued)) estimator(s)/algorithm(s) are missing from the
                         Capability Catalogue. Add each to `docs/capability_catalogue.jl` under
                         the group it belongs to:\n  $(join(uncatalogued, "\n  "))"""
            end
            @test isempty(uncatalogued)
        end

        @testset "every exported function is accounted for" begin
            exported = filter(n -> Base.isexported(PO, n) && !contains(string(n), "#"),
                              names(PO))
            fns = [n for n in exported if getfield(PO, n) isa Function]
            unaccounted = sort([n
                                for n in fns
                                if !(n in catalogued) && !haskey(NOT_A_FEATURE, n)])
            if !isempty(unaccounted)
                @warn """$(length(unaccounted)) exported function(s) are neither catalogued nor
                         listed in `NOT_A_FEATURE`. Add a `Cap` for each, or list it with a
                         reason (`:alias`, `:base_overload`, `:trait`, `:internal`):
                         \n  $(join(unaccounted, "\n  "))"""
            end
            @test isempty(unaccounted)

            # The other direction: an entry naming something no longer exported is a
            # stale exemption, and would silently widen the hole it was cut for.
            stale = sort([n for n in keys(NOT_A_FEATURE) if !(n in exported)])
            if !isempty(stale)
                @warn "Stale `NOT_A_FEATURE` entries (no longer exported): $(join(stale, ", "))"
            end
            @test isempty(stale)
        end

        @testset "every catalogued name resolves" begin
            # Symbols are resolved against the module, so a typo fails here rather
            # than rendering as dead text -- the failure mode of the markdown page
            # this replaced, which shipped four broken links nothing warned about.
            # String names are `@ref` targets that are not bare identifiers (method
            # signatures); only Documenter can resolve those.
            unresolved = Symbol[]
            each_cap(CATALOGUE) do c
                for n in c.names
                    if n isa Symbol && !isdefined(PO, n)
                        push!(unresolved, n)
                    else
                        false
                    end
                end
            end
            if !isempty(unresolved)
                @warn "Catalogued names not defined in PortfolioOptimisers: $(join(unresolved, ", "))"
            end
            @test isempty(unresolved)
        end

        @testset "every @ref survives markdown parsing" begin
            # Writing `[`X`](@ref)` does not guarantee a link: a bare `_` in a
            # neighbouring description pairs with the one inside a `snake_case`
            # link and eats both, and a paragraph that breaks out of a list turns
            # the bullets below it into a code block. Documenter resolves `@ref`
            # only in real links, so either way the text survives verbatim and the
            # site builder reports one anonymous dead `./@ref` -- no page, no name.
            # Rendering here costs a few seconds and names the offenders instead.
            page = String(take!(render_catalogue(IOBuffer())))
            @test (assert_refs_survive(page); true)
        end

        @testset "verbatim labels agree with their names" begin
            # A label that spells its own links is rendered as-is, so its links and
            # the `Cap`'s names are two statements of one fact and can drift apart.
            mismatched = Tuple{Symbol, Vector{Symbol}}[]
            each_cap(CATALOGUE) do c
                if (isnothing(c.label) || !occursin("](@ref)", c.label))
                    return nothing
                end
                inline = Set(Symbol(m.captures[1])
                             for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", c.label))
                named = Set(n isa Symbol ? n : Symbol(n) for n in c.names)
                extra = sort(collect(setdiff(inline, named)))
                if !(isempty(extra))
                    push!(mismatched, (Symbol(first(c.names)), extra))
                end
                return nothing
            end
            if !isempty(mismatched)
                @warn "Verbatim labels linking names absent from `names`: $mismatched"
            end
            @test isempty(mismatched)
        end
    end
end
