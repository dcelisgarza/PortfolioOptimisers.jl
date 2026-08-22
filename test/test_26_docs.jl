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
    #
    # The runner gives each test file its own module, but not its own process. An
    # estimator that another file declares therefore stays in the worker, and
    # `subtypes` finds it here. The catalogue is about the shipped universe, so a
    # leaf from another module is not one of its members: keep only what
    # `PortfolioOptimisers` itself declares. Which files share a worker changes
    # from run to run, so without this filter the test is a scheduling flake.
    function leaf_types(T, acc = Set{Type}())
        subs = subtypes(T)
        if isempty(subs)
            if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers
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
            # A type the library constructs for itself is not a choice, so it is
            # exempt by name and with a reason -- see `NOT_A_CHOICE`.
            uncatalogued = sort(collect(setdiff(required, catalogued, keys(NOT_A_CHOICE))))
            if !isempty(uncatalogued)
                @warn """$(length(uncatalogued)) estimator(s)/algorithm(s) are missing from the
                         Capability Catalogue. Add each to `docs/capability_catalogue.jl` under
                         the group it belongs to, or list it in `NOT_A_CHOICE` with a reason
                         (`:internal`):\n  $(join(uncatalogued, "\n  "))"""
            end
            @test isempty(uncatalogued)

            # The other direction, as for `NOT_A_FEATURE`: an exemption for a type
            # that is no longer a leaf estimator or algorithm is stale, and would
            # silently exempt whatever later takes that name.
            stale = sort([n for n in keys(NOT_A_CHOICE) if !(n in required)])
            if !isempty(stale)
                @warn "Stale `NOT_A_CHOICE` entries (no longer a leaf estimator/algorithm): $(join(stale, ", "))"
            end
            @test isempty(stale)

            # An exempt type is still catalogued nowhere, so nothing else states
            # that it is absent on purpose. A name in both places is a contradiction.
            contradictory = sort([n for n in keys(NOT_A_CHOICE) if n in catalogued])
            if !isempty(contradictory)
                @warn "Names both catalogued and listed in `NOT_A_CHOICE`: $(join(contradictory, ", "))"
            end
            @test isempty(contradictory)
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

#=
References (issue #351). The bibliography is the one part of a docstring whose text is
long, repeated, and easy to let drift: before `ref_dict` existed, `gerber2025squeezing`
was pasted 31 times and `mlp1` 13 times, and only 16 of the 193 files in `src/` cited
anything at all.

The four checks below are what a ticked box on #351 means. They run here rather than in
`docs/make.jl` for the reason the catalogue tests give: a full docs build is slow and run
by hand, so a gap found only there is a gap found months late. Check 4 is the one that
would otherwise be a *build error* -- a `Pages = [@__FILE__]` block on a page that cites
nothing raises `@error "File exists but no references were collected"` in
`DocumenterCitations`, so it fails the build rather than warning.
=#
@testset "References" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    SRC = joinpath(@__DIR__, "..", "src")
    BIB = joinpath(@__DIR__, "..", "docs", "src", "References.bib")
    API = joinpath(@__DIR__, "..", "docs", "src", "api")

    function files_under(dir, ext)
        acc = String[]
        for (root, _, files) in walkdir(dir), f in files
            endswith(f, ext) && push!(acc, joinpath(root, f))
        end
        return sort!(acc)
    end

    src_files = files_under(SRC, ".jl")
    # `ref_dict` itself names every key it defines, so it is not evidence that anything
    # cites the work. Exclude the file that holds the table when looking for users.
    dict_file = joinpath(SRC, "01_Base.jl")

    bib_keys = Set(m.captures[1]
                   for m in eachmatch(r"^@\w+\{([A-Za-z0-9_]+),"m, read(BIB, String)))

    # A citation is `[key](@cite)` or `[key1,key2](@cite)`.
    function cited_keys(text)
        acc = Set{String}()
        for m in eachmatch(r"\[([A-Za-z0-9_][A-Za-z0-9_,\s]*)\]\(@cite\)", text)
            foreach(k -> push!(acc, strip(k)), split(m.captures[1], ','))
        end
        return acc
    end

    @testset "every citation resolves in References.bib" begin
        for f in src_files
            for key in cited_keys(read(f, String))
                @test key in bib_keys
            end
        end
    end

    @testset "every ref_dict entry has a user" begin
        for key in keys(PO.ref_dict)
            @test string(key) in bib_keys
        end
        users = Set{Symbol}()
        for f in src_files
            f == dict_file && continue
            for m in eachmatch(r"ref_dict\[:([A-Za-z0-9_]+)\]", read(f, String))
                push!(users, Symbol(m.captures[1]))
            end
        end
        for key in keys(PO.ref_dict)
            @test key in users
        end
    end

    # A bullet under `# References` must be one interpolation of `ref_dict`, optionally
    # followed by a locator such as `Chapter 2.`. Anything else is a pasted copy of the
    # reference prose, which is what this table exists to stop.
    @testset "no # References bullet carries inline reference prose" begin
        bullet = r"^\s*- \$\(ref_dict\[:[A-Za-z0-9_]+\]\)"
        for f in src_files
            lines = split(read(f, String), '\n')
            i = 1
            while i <= length(lines)
                if strip(lines[i]) == "# References"
                    j = i + 1
                    while j <= length(lines) && isempty(strip(lines[j]))
                        j += 1
                    end
                    while j <= length(lines) && startswith(strip(lines[j]), "- ")
                        @test occursin(bullet, lines[j])
                        j += 1
                    end
                    i = j
                else
                    i += 1
                end
            end
        end
    end

    # A page cites either in its own prose or through a docstring it pulls in with a
    # `@docs` block, and it needs a non-canonical bibliography block exactly then.
    @testset "an API page carries a bibliography block iff it cites" begin
        # A `@docs` entry can wrap over several lines, so accumulate lines until the
        # text parses. Treating each line as its own entry leaves a wrapped signature as
        # two fragments, neither of which resolves to anything.
        function docs_block_names(text)
            acc, inside, buf = String[], false, ""
            for line in split(text, '\n')
                if startswith(line, "```@docs")
                    inside = true
                elseif inside && startswith(line, "```")
                    inside = false
                    if !isempty(buf)
                        push!(acc, buf)
                        buf = ""
                    end
                elseif inside && !isempty(strip(line))
                    buf = if isempty(buf)
                        String(strip(line))
                    else
                        buf * " " * String(strip(line))
                    end
                    ex = Meta.parse(buf; raise = false)
                    if !(isa(ex, Expr) && ex.head === :incomplete)
                        push!(acc, buf)
                        buf = ""
                    end
                end
            end
            return acc
        end
        # The bare name an entry documents: `cov`, `PortfolioOptimisers.densify` and
        # `cov(ce::DistanceCovariance, X::MatNum)` all reduce to one symbol.
        function leaf_name(x)
            if isa(x, Symbol)
                return x
            elseif isa(x, QuoteNode)
                return leaf_name(x.value)
            elseif isa(x, Expr)
                return leaf_name(x.head === :. ? x.args[2] : x.args[1])
            else
                return nothing
            end
        end
        # An entry that carries a signature splices THAT method's docstring, so the
        # signature has to take part in the lookup. Asking for the binding alone returns
        # every method's docstring concatenated, and then a page that lists a shared
        # generic function -- `cov`, `cor`, `ucs`, `factory` -- collects the citations of
        # methods it never renders. That is what let a stray block through: the page
        # looked like it cited, so check 4 passed while the docs build raised the error
        # this testset exists to pre-empt.
        function docstring_text(name)
            ex = Meta.parse(name; raise = false)
            (isa(ex, Expr) && ex.head === :incomplete) && return ""
            iscall = isa(ex, Expr) && ex.head === :call
            sym = leaf_name(iscall ? ex.args[1] : ex)
            isa(sym, Symbol) || return ""
            isdefined(PO, sym) || return ""
            # `Documenter` looks for an exact signature match and falls back to `<:`,
            # which is what `Base.Docs.doc(binding, sig)` does.
            sig = if iscall
                try
                    Core.eval(PO, Base.Docs.signature(ex))
                catch
                    Union{}
                end
            else
                Union{}
            end
            return try
                string(Base.Docs.doc(Base.Docs.Binding(PO, sym), sig))
            catch
                ""
            end
        end
        # Collect rather than assert per page: a bare `false == true` does not say which
        # page is wrong, and the fix is always "add (or remove) the block on that page".
        missing_block, stray_block = String[], String[]
        for p in files_under(API, ".md")
            text = read(p, String)
            has_block = occursin("```@bibliography", text)
            cites = occursin("(@cite)", text) ||
                    any(n -> occursin("(@cite)", docstring_text(n)), docs_block_names(text))
            if cites && !has_block
                push!(missing_block, relpath(p, API))
            elseif !cites && has_block
                push!(stray_block, relpath(p, API))
            end
        end
        @test missing_block == String[]
        @test stray_block == String[]
    end
end

#=
The function template in `.github/instructions/julia-docstrings.instructions.md` puts
`# Validation` before `# Returns`: a call states what it refuses before it states what it
produces. Documenter renders the sections in whatever order it is given, so the docs build
is blind to the ordering and this check is what holds it across the tree.
=#
@testset "Docstring section order" begin
    using Test
    ROOT = joinpath(@__DIR__, "..")

    # A section heading is a markdown `# Name` at column 0 inside a triple-quoted block.
    # Every triple-quoted literal toggles the same way, so a plain string literal is
    # scanned too and contributes no headings.
    function misordered_docstrings(file)
        acc, names, start, indoc = String[], String[], 0, false
        for (i, ln) in enumerate(readlines(file))
            q = length(findall("\"\"\"", ln))
            if !indoc
                if q == 1
                    indoc, start = true, i
                    empty!(names)
                end
            elseif q >= 1
                indoc = false
                v = findfirst(==("Validation"), names)
                r = findfirst(==("Returns"), names)
                if !isnothing(v) && !isnothing(r) && v > r
                    push!(acc, "$(relpath(file, ROOT)):$(start)")
                end
            elseif startswith(ln, "# ")
                push!(names, strip(ln[3:end]))
            end
        end
        return acc
    end

    @testset "# Validation precedes # Returns" begin
        offenders = String[]
        for dir in ("src", "ext"), (root, _, files) in walkdir(joinpath(ROOT, dir)),
            f in files

            endswith(f, ".jl") || continue
            append!(offenders, misordered_docstrings(joinpath(root, f)))
        end
        @test offenders == String[]
    end
end
