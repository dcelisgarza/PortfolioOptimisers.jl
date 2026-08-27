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
    EXT = joinpath(@__DIR__, "..", "ext")
    BIB = joinpath(@__DIR__, "..", "docs", "src", "References.bib")
    API = joinpath(@__DIR__, "..", "docs", "src", "api")

    function files_under(dir, ext)
        acc = String[]
        for (root, _, files) in walkdir(dir), f in files
            endswith(f, ext) && push!(acc, joinpath(root, f))
        end
        return sort!(acc)
    end

    #=
    Issue #404 widened the sweep's scope from `src/` to `src/` and `ext/`. The three checks
    below scan source text, and each of them is a CONFORMANCE check: if a file cites, the key
    must resolve, and if a file writes a `# References` bullet, that bullet must interpolate
    `ref_dict`. Such a check is vacuous on a file that does neither, so widening it cannot
    red a file that no child map has swept. It takes effect on the day the first `ext/`
    docstring cites. `Extension docs completeness` at the foot of this file holds the other
    kind of check, which cannot widen this way.
    =#
    source_files = vcat(files_under(SRC, ".jl"), files_under(EXT, ".jl"))
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
        for f in source_files
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
        for f in source_files
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
        for f in source_files
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

#=
`ext/` and the sweep manifest (issue #409, under the map of maps #404).

#404 widened the sweep's scope from `src/` to `src/` and `ext/`, for all three of its
concerns. The checks in this file stopped at `src/`. Widening them is not one act, because
they are of two kinds and only one kind can red a file that no child map has swept.

  A CONFORMANCE check says "if the file states X, X must be well formed". It is vacuous on a
  file that states no X, so it widens unconditionally and takes effect on the day the first
  `ext/` docstring states one. The three text checks under `References` are of this kind,
  and so is `# Validation precedes # Returns`, which #406 already widened.

  A PRESENCE check says "the file must state X". It reds a file the moment its scope reaches
  it. `ext/` carries ZERO docstrings over 2145 lines, so a presence check widened
  unconditionally would red the build across both files on the day it landed. It reads the
  sweep manifest instead: a file marked `swept = true` must satisfy it, and an unswept file
  is exempt. Both `ext/` rows are `swept = false` today, so the testset below asserts the
  exemption and nothing else. Child map 13 sweeps the two files and flips the flags.

--------------------------------------------------- what an extension file has to document

An extension is a module of its own, so `Base.undocumented_names` -- the instrument the
first testset in this file already uses -- answers for it directly. It also answers the
right question rather than a widened one, and what the two files hold is why:

  - `ext/PortfolioOptimisersPlotsExt.jl` defines 171 methods of 34 functions, and every one
    of the 34 is declared as a bare `function ... end` stub in `src/24_Plotting.jl`, which
    carries its docstring. Beyond those methods it declares four module-local `const`s
    holding error-message text, and nothing else.
  - `ext/PortfolioOptimisersImputeExt.jl` defines one method of
    `PortfolioOptimisers.apply_impute_method`, a seam declared in `src/03_Preprocessing.jl`.

A method binds in the module that declares the FUNCTION, so none of those 172 methods
reaches `names(ext; all = true)` and the census asks for a docstring on none of them. That
is the right demand: 172 docstrings restating 35 would be 137 copies free to drift, which is
what `ref_dict` and `field_dict` exist to stop. What the census does ask for is the four
`const`s and the extension module's own docstring -- the names the extension declares
itself. The main module carries a docstring under exactly that rule, so this is one rule and
not two.

#409's three questions are answered by that one rule.

 1. A NAME THE EXTENSION FILE DECLARES ITSELF must be documented, and a method of a function
    declared in `src/` is documented by that declaration. The tree holds no `@recipe` block
    today, and the rule settles one if it ever appears: `@recipe` declares no binding a
    caller can reach, so it never enters `names`, and the reachable name is the `plot_*`
    function it serves, which `src/24_Plotting.jl` declares.
 2. `ext/` NAMES DO NOT ENTER THE CAPABILITY CATALOGUE on their own account. The 34 `plot_*`
    functions are exported by `src/24_Plotting.jl` and are catalogued there already, which
    is why `every exported function is accounted for` is green today. ADR 0040 owns the
    catalogue and needs no amendment. An extension that declared a user-facing name of its
    own would be the smell, not the case to cater for: the declaration belongs in `src/` as
    a stub, which is the pattern both extensions already follow.
 3. AN `ext/` DOCSTRING MAY CITE, by the same rule as a `src/` one, and the API page that
    renders that docstring carries the bibliography block. Under rule 1 the page is always
    the page of the `src/` declaration -- `docs/src/api/24_Plotting.md` for the plotting
    family -- so an extension needs neither a page nor a block of its own. This matters:
    `an API page carries a bibliography block iff it cites` resolves each `@docs` entry
    against `PortfolioOptimisers`, so a name declared inside an extension resolves to
    nothing there, the page reads as citing nothing, and a block that page needs is reported
    as stray. Rule 1 keeps that case out of the tree.
=#
@testset "Extension docs completeness" begin
    using PortfolioOptimisers, Test, TOML
    ROOT = normpath(joinpath(@__DIR__, ".."))

    extensions = get(TOML.parsefile(joinpath(ROOT, "Project.toml")), "extensions",
                     Dict{String, Any}())
    rows = TOML.parsefile(joinpath(ROOT, "sweep", "manifest.toml"))["file"]
    ext_rows = sort([f for f in keys(rows) if startswith(f, "ext/")])

    # Julia loads an extension from `ext/<Name>.jl`, or from `ext/<Name>/` when it needs
    # more than one file. The manifest is per file and `[extensions]` is per module, so the
    # join is derived here rather than written down: a second file added to an extension
    # then needs no edit in this file.
    belongs_to(name, f) = f == string("ext/", name, ".jl") ||
                          startswith(f, string("ext/", name, "/"))

    @testset "every ext/ row belongs to a declared extension" begin
        # A `.jl` file under `ext/` that no `[extensions]` entry claims is loaded by
        # nothing, so no `using` reaches it and no gate here can see it.
        orphans = filter(f -> !any(n -> belongs_to(n, f), keys(extensions)), ext_rows)
        if !isempty(orphans)
            @warn """$(length(orphans)) file(s) under `ext/` that no `[extensions]` entry in
                     `Project.toml` claims:\n  $(join(orphans, "\n  "))"""
        end
        @test isempty(orphans)
    end

    @testset "a swept extension declares no undocumented name" begin
        # Loading is deferred to a swept extension, and to that extension's own triggers.
        # `StatsPlots` costs about a minute to load, and this file must not pay that to
        # assert an exemption.
        #
        # A trigger is required to be a declared dependency of `test/Project.toml`, and not
        # merely loadable. `Base.require` searches the whole `LOAD_PATH`, so in a hand-run
        # REPL it reaches the user's own environment and returns a version that no
        # `[compat]` bound here sanctions. `Pkg.test` puts the test environment and the
        # stdlib on `LOAD_PATH` and nothing else, so the same call answers differently in
        # CI. Reading the declaration makes this file answer the same in both.
        test_deps = keys(get(TOML.parsefile(joinpath(@__DIR__, "Project.toml")), "deps",
                             Dict{String, Any}()))
        function load_extension(name, triggers)
            all(t -> t in test_deps, triggers) || return nothing
            for t in triggers
                try
                    Base.require(Main, Symbol(t))
                catch
                    return nothing
                end
            end
            return Base.get_extension(PortfolioOptimisers, Symbol(name))
        end

        undocumented, unloadable = String[], String[]
        for (name, triggers) in extensions
            files = filter(f -> belongs_to(name, f), ext_rows)
            any(f -> rows[f]["swept"], files) || continue
            trigger_list = isa(triggers, AbstractString) ? [triggers] : triggers
            mod = load_extension(name, trigger_list)
            if isnothing(mod)
                # Refuse to be vacuously green. `Impute` is a weak dependency (ADR 0042)
                # and is absent from `test/Project.toml`, so on the day child map 13 sweeps
                # that extension this environment must gain it, or the gate is blind to the
                # file it is meant to hold.
                push!(unloadable, string(name, "  triggers: ", join(trigger_list, ", ")))
            else
                for n in Base.undocumented_names(mod; private = true)
                    push!(undocumented, string(name, ".", n))
                end
            end
        end

        if !isempty(unloadable)
            @warn """$(length(unloadable)) extension(s) marked `swept = true` in
                     `sweep/manifest.toml` cannot be loaded in this environment, so nothing
                     here can see them. Add each trigger package to `test/Project.toml`:
                     \n  $(join(sort(unloadable), "\n  "))"""
        end
        @test isempty(unloadable)

        if !isempty(undocumented)
            @warn """$(length(undocumented)) name(s) declared by a swept extension carry no
                     docstring. A method of a function declared in `src/` is documented by
                     that declaration and never appears here, so each name below is one the
                     extension declares itself:\n  $(join(sort(undocumented), "\n  "))"""
        end
        @test isempty(undocumented)
    end
end

#=
The two new documentation sections and the sweep manifest (issue #437, under the map of maps
#404).

#405 added `# Algorithm` and `# JuMP formulation` to the docstring standard, and its
resolution said the child maps would apply them "file by file, gated by the manifest's
`swept` flag". Nothing gated them. `test_26_docs.jl` read the manifest in exactly one
testset, `Extension docs completeness`, and that testset asks a question about an extension.
So the first child map to set `swept = true` on a `src/` row would write both sections by
hand, and nothing would hold them there afterwards.

Both are PRESENCE checks in the sense the block above defines, so both read the `swept` flag.
No file is swept today, so this testset asserts the exemption and nothing else, exactly as
`Extension docs completeness` did on the day it landed.

------------------------------------------- why the two sections get two different checks

The sections are not gated the same way, and the reason is the trigger, not the section.

  `# JuMP formulation` HAS A MECHANICAL TRIGGER IN THE CODE. A body that calls a JuMP
  model-building macro builds part of the model, and the parser sees the call. The trigger
  needs no judgement and no exemption list, so this section gets the strongest check
  available: a PER-UNIT PRESENCE RULE, on the section AND on each subsection the body's
  macros demand. #443 widened the trigger from `@constraint` alone; the block above the
  trigger table records what it measured.

  `# Algorithm` HAS NO SUCH TRIGGER. The standard says a procedure carries it, a closed form
  stays in `# Mathematical definition`, and a SELECTOR TAG carries neither. `AbstractAlgorithm`
  holds 277 of the library's 720 types and most of them are selector tags, so "every
  documented unit carries `# Algorithm`" would red the build on sight, and the exemption
  cannot be written as a parser rule without first defining a selector tag mechanically.
  What is left is a RATCHET: a swept file's count of docstrings carrying the section may not
  FALL. It needs no judgement about which unit deserves the section, and it catches the
  deletion this ticket exists to stop.

The third shape considered was a per-unit presence rule for `# Algorithm` with a written
exemption list. It is the most exact and the most expensive, it needs a definition of a
selector tag that a parser can apply, and it would carry an exemption row per marker type.
It was set aside for that cost. It stays available: the ratchet is a floor, and a later
ticket can raise it to a presence rule without moving the manifest key.

------------------------------------------------ what a model-building docstring is

A docstring does not always sit on the method that registers the row. Two shapes exist in
the tree and the attribution must read both.

  - `src/20_Optimisation/09_JuMPConstraints/04_WeightConstraints.jl` documents
    `set_weight_constraints!(args...)`, a dispatch-error stub that registers nothing. The
    methods that register `w_lb` and `w_ub` follow it and carry no docstring of their own.
  - `src/20_Optimisation/20_RiskMeasureConstraints/06_XatRiskConstraints.jl` documents five
    separate methods of `set_risk_constraints!`, each immediately above the method it
    describes, and each of the five registers its own rows.

So a docstring speaks for its own definition AND for every later definition of the same name
until the next docstring of that name -- which is how the file reads. Attributing per name
rather than per definition is what makes the first shape visible: read per definition, four
of the thirteen files under `09_JuMPConstraints/` register rows under no documented
definition at all, `04_WeightConstraints.jl` among them.

A function this file documents nowhere is skipped rather than reported, because the docstring
the rule addresses is in another file and that file has a `swept` flag of its own. No such
case exists in the tree today. It is the accepted blind spot, in the manner of the two
`test_45_sweep_census.jl` names.

--------------------------------------------------------------- what is NOT gated here

The CONTENT of a subsection. Nothing compares the model keys a docstring names with the keys
the body registers, and nothing reads a `## Relaxation` at all -- an inexact encoding is a
fact about the mathematics, not a token. ADR 0081 records the key census as the largest build
of this area and leaves it in the map's *Not yet specified*. `## Relaxation` holds by review,
in the sense of `STANDARDS.md`.
=#
@testset "Swept file section completeness" begin
    using Test, TOML

    ROOT = normpath(joinpath(@__DIR__, ".."))
    rows = TOML.parsefile(joinpath(ROOT, "sweep", "manifest.toml"))["file"]
    swept = sort([f for (f, r) in rows if r["swept"]])

    # The same instrument `test_45_sweep_census.jl` counts units with: one parse per file,
    # no package loaded, and a count that cannot move under a reformat.
    doc_macro = GlobalRef(Core, Symbol("@doc"))
    isdocstring(x) = Meta.isexpr(x, :macrocall) &&
                     !isempty(x.args) &&
                     (x.args[1] === doc_macro || x.args[1] === Symbol("@doc"))

    # A definition, not a local assignment. `Au = ...` inside a body whose right-hand side
    # holds a `JuMP.@constraint` is a local, and reading it as a definition attributes the
    # row to a name no docstring can ever carry.
    function isdefinition(x)
        Meta.isexpr(x, :function) && return true
        # A `macro` is a definition and it binds a name, so its body reaches
        # `demanded_subsections` like any other. `test_45_sweep_census.jl` counts a
        # documented macro as a unit, and these two checks read the same units it does.
        Meta.isexpr(x, :macro) && return true
        Meta.isexpr(x, :(=)) || return false
        lhs = x.args[1]
        while Meta.isexpr(lhs, :where)
            lhs = lhs.args[1]
        end
        return Meta.isexpr(lhs, :call)
    end

    # The name a definition binds. A definition reaches here through five declaration forms:
    # bare, `@concrete struct`, a short form, a macro-prefixed one, and a qualified one. A
    # `macro` declaration binds its name through its `:call` node, exactly as a `function`
    # does, and it is named here without its `@`.
    function bound_name(x)
        x isa Symbol && return x
        x isa Expr || return nothing
        if Meta.isexpr(x, :macrocall)
            for a in x.args[2:end]
                n = bound_name(a)
                isnothing(n) || return n
            end
            return nothing
        end
        # A method of a function another module owns is declared `Statistics.mean(...)`, so
        # the callee of its `:call` is an `Expr(:., :Statistics, QuoteNode(:mean))` and not a
        # Symbol. Reading only the Symbol forms dropped every such docstring, and the shape is
        # not rare: four files under `src/` carried a `# Details` section that no count below
        # could see, and four more carried one the count read short. The `# Algorithm` floor
        # and the `# JuMP formulation` rule read the same units, so both were blind to the
        # same docstrings. The name is taken bare, exactly as a `macro` is.
        Meta.isexpr(x, :.) && x.args[end] isa QuoteNode && return x.args[end].value
        x.head === :struct && return bound_name(x.args[2])
        x.head in
        (:function, :macro, :(=), :call, :where, :(::), :const, :abstract, :curly, :(<:)) &&
            return bound_name(x.args[1])
        return nothing
    end

    # A docstring that interpolates parses to an `Expr(:string, ...)`, and a section heading
    # is a literal line inside it, so the literal pieces alone carry every heading.
    function docstring_text(x)
        for a in x.args[2:end]
            a isa AbstractString && return String(a)
            Meta.isexpr(a, :string) &&
                return join(p isa AbstractString ? p : " " for p in a.args)
        end
        return ""
    end

    # Julia strips the indentation of a `"""` block, so a section heading sits at column 0.
    # The count, not the flag, is the primitive: one string block can document several
    # methods, separated by horizontal rules, and then carries one heading per method that
    # holds the section. `port_opt_view` in `src/03_Preprocessing.jl` is such a block.
    count_section(text, name) = count(==(string("# ", name)), rstrip.(split(text, '\n')))
    has_section(text, name) = count_section(text, name) > 0
    has_subsection(text, name) = count(==(string("## ", name)),
                                       rstrip.(split(text, '\n'))) > 0

    #=
    THE TRIGGER IS THE MACRO FAMILY, AND EACH FAMILY OWNS ONE SUBSECTION.

    #437 built this check on `JuMP.@constraint` alone, because the standard's trigger
    sentence read "any code that ADDS ROWS to a `JuMP.Model`". #443 widened both, and the
    reason is the standard's OWN justification for the section: an entry carries a name, and
    a caller reads the entry back by that name. THAT IS NOT A PROPERTY OF A ROW.
    `model[:sc]`, `model[:w]`, `model[:ret]` and `model[:risk]` are each a variable or an
    expression, `src/` reads a model key back by name in 51 places over 16 distinct keys, and
    `08_Base_JuMPOptimisation.jl` wraps nine of those keys in an accessor that raises a named
    `ArgumentError` when its builder has not run. A row name is public, and so is every one
    of those.

    Measured over `src/` and `ext/` at the tip that widened it: 137 documented units call one
    of the macros below, 96 register a row, and 41 touch the model without one. 31 of the 41
    register only an expression, 3 only a variable, 4 both, 2 only the objective and 1 an
    expression and the objective. `set_model_scales!` is one of the 41, and its swapped
    `sc`/`so` is the defect #404's charter names when it says the JuMP layer is undocumented
    as a model. Two of the four `## Relaxation` cases ADR 0081 cites -- the
    `BrownianDistanceVariance` bound and the scalarisers -- sat on the exempt side of the
    trigger that same decision wrote.

    `@objective` is IN. A formulation is variables, constraints and an objective, and the
    third was missing. `owa_l_moment_crm_sumsq_obj` in `src/19_RiskMeasures/` is two methods
    that differ in `Min so * t` against `Min so * t^2` AND IN NOTHING ELSE, so the objective
    line is the only text that can tell them apart.

    The widening costs no judgement and no exemption list, which is the property that made
    the narrow rule worth building: the parser sees a macro call, and each family maps to the
    subsection that documents what that macro registers. It is measured to red nothing today
    -- no file marked `swept = true` calls a JuMP macro.
    =#
    subsection_of = Dict(Symbol("@variable") => "Variables",
                         Symbol("@variables") => "Variables",
                         Symbol("@expression") => "Expressions",
                         Symbol("@expressions") => "Expressions",
                         Symbol("@constraint") => "Constraints",
                         Symbol("@constraints") => "Constraints",
                         Symbol("@objective") => "Objective")

    # Every subsection the macros under `node` demand of the docstring that speaks for it.
    function demanded_subsections(node, acc = Set{String}())
        node isa Expr || return acc
        if Meta.isexpr(node, :macrocall) && !isempty(node.args)
            m = node.args[1]
            s = if m isa GlobalRef
                m.name
            elseif m isa Symbol
                m
            elseif Meta.isexpr(m, :.) && m.args[end] isa QuoteNode
                m.args[end].value
            else
                nothing
            end
            if !isnothing(s) && haskey(subsection_of, s)
                push!(acc, subsection_of[s])
            end
        end
        for a in node.args
            demanded_subsections(a, acc)
        end
        return acc
    end

    #=
    Returns the file's documented units in source order -- the name each one documents and
    its text -- and, for each unit that owns a model-building definition, the subsections
    that definition demands. A docstring takes charge of its own name when it is read, and
    holds it until the next docstring of that name, which is the attribution the two shapes
    above need. A unit whose definitions call no such macro carries no entry.
    =#
    function scan(path)
        names, texts = Symbol[], String[]
        demanded = Dict{Int, Set{String}}()
        in_force = Dict{Symbol, Int}()
        function walk(node)
            node isa Expr || return nothing
            if isdocstring(node)
                d = length(node.args) >= 4 ? node.args[4] : nothing
                nm = isnothing(d) ? nothing : bound_name(d)
                if !isnothing(nm)
                    push!(names, nm)
                    push!(texts, docstring_text(node))
                    in_force[nm] = length(names)
                end
            elseif isdefinition(node)
                nm = bound_name(node)
                if !isnothing(nm) && haskey(in_force, nm)
                    subs = demanded_subsections(node)
                    isempty(subs) ||
                        union!(get!(demanded, in_force[nm], Set{String}()), subs)
                end
            end
            foreach(walk, node.args)
            return nothing
        end
        walk(Meta.parseall(read(path, String)))
        return names, texts, demanded
    end

    @testset "a swept file's model-building docstring carries # JuMP formulation" begin
        offenders = String[]
        for f in swept
            names, texts, demanded = scan(joinpath(ROOT, f))
            for u in sort(collect(keys(demanded)))
                text = texts[u]
                if !has_section(text, "JuMP formulation")
                    push!(offenders, string(f, "  ", names[u], "  # JuMP formulation"))
                    continue
                end
                # The section is there, so each family the body calls owes its subsection.
                for sub in sort(collect(demanded[u]))
                    has_subsection(text, sub) ||
                        push!(offenders, string(f, "  ", names[u], "  ## ", sub))
                end
            end
        end
        if !isempty(offenders)
            @warn """$(length(offenders)) docstring(s) in a file marked `swept = true` in
                     `sweep/manifest.toml` document a function that builds part of a
                     `JuMP.Model` and carry neither the `# JuMP formulation` section nor a
                     subsection the body's macros demand. `@variable` owes `## Variables`,
                     `@expression` owes `## Expressions`, `@constraint` owes
                     `## Constraints` and `@objective` owes `## Objective`, each naming its
                     entry by the model key a caller reads it back
                     by:\n  $(join(offenders, "\n  "))"""
        end
        @test isempty(offenders)
    end

    @testset "a swept file's # Algorithm count does not fall" begin
        # The key is demanded only of a swept row, so an unswept row is untouched and the
        # session that flips the flag writes the count in the same edit.
        no_key = filter(f -> !haskey(rows[f], "algorithm"), swept)
        if !isempty(no_key)
            @warn """$(length(no_key)) row(s) marked `swept = true` in
                     `sweep/manifest.toml` carry no `algorithm` key. A swept row records the
                     count of its docstrings that carry a `# Algorithm` section, and the
                     count may not fall afterwards:\n  $(join(no_key, "\n  "))"""
        end
        @test isempty(no_key)

        fallen = Tuple{String, Int, Int}[]
        for f in swept
            haskey(rows[f], "algorithm") || continue
            _, texts, _ = scan(joinpath(ROOT, f))
            measured = count(t -> has_section(t, "Algorithm"), texts)
            measured < rows[f]["algorithm"] &&
                push!(fallen, (f, rows[f]["algorithm"], measured))
        end

        @test isempty(fallen)
        if !isempty(fallen)
            println("Files whose count of `# Algorithm` sections has fallen below the ",
                    "count their `sweep/manifest.toml` row records. Restore the section, ",
                    "or lower the count deliberately in the same commit that removes it:")
            for (f, was, now) in fallen
                row = rows[f]
                println("  ", f, "  ", was, " -> ", now)
                println("    \"", f, "\" = { map = ", row["map"], ", units = ",
                        row["units"], ", algorithm = ", now, ", swept = ", row["swept"],
                        " }")
            end
        end
    end

    #=
    `# Details` is abolished (issue #480, under the standards-hardening map #478).

    ADR 0085 records the decision. The section held facts that five other sections already
    own, and it held them because nothing said where they belonged: the Authority mentioned
    it four times, every mention sat inside a template, and the only text that described it
    was the placeholder `Additional implementation notes.` 299 sections over 84 files carried
    it on the day the rule was written.

    The two checks below INVERT the `# Algorithm` floor above. That floor may not FALL,
    because a swept file's steps must stay written. These counts may not RISE, because the
    section they count must reach zero.

      1. A file whose manifest row reads `swept = true` carries ZERO `# Details` sections.
         #485 migrated the 29 sections the seven swept files carried when the rule was
         written, so the debt list that held them is spent and gone, and the check now reads
         exactly as the Authority states it.
      2. The library-wide count may not rise above `DETAILS_TOTAL`. Each #404 sweep ticket
         lowers it as its file migrates, and the check retires when it reaches zero.

    Both numbers count SECTION HEADINGS and not docstrings that carry one. `port_opt_view` in
    `src/03_Preprocessing.jl` documents four methods under one string block and carries the
    heading twice, so a docstring count would read 298 and would not fall when the first of
    those two headings moved.

    The manifest gains NO key for either number. A swept row states that a file passed three
    conditions, and this decision changes one of them without re-opening the row, which #478
    settles explicitly. So the debt is written here, in the census idiom of
    `test_43_exported_abstract_type_census.jl`, where deleting the last entry is a visible
    edit and not a silent one.
    =#
    @testset "# Details is abolished" begin
        DETAILS_TOTAL = 117

        @testset "a swept file carries no # Details section" begin
            offenders = Tuple{String, Int}[]
            for f in swept
                _, texts, _ = scan(joinpath(ROOT, f))
                measured = sum(t -> count_section(t, "Details"), texts; init = 0)
                measured > 0 && push!(offenders, (f, measured))
            end
            if !isempty(offenders)
                @warn """$(length(offenders)) file(s) marked `swept = true` in
                         `sweep/manifest.toml` carry a `# Details` section. The section is
                         abolished: move each fact by its subject, under
                         `## What each section holds` in
                         `.github/instructions/julia-docstrings.instructions.md`. The columns
                         are the file and what it
                         carries:\n  $(join(string.(offenders), "\n  "))"""
            end
            @test isempty(offenders)
        end

        @testset "the library-wide # Details count does not rise" begin
            # `sweep/manifest.toml` carries one row per file under `src/` and `ext/`, and
            # `test_45_sweep_census.jl` reds when a file has no row. So its keys are the
            # scope of this count, already gated, and no second file walk is needed.
            per = Tuple{String, Int}[]
            for f in sort(collect(keys(rows)))
                _, texts, _ = scan(joinpath(ROOT, f))
                n = sum(t -> count_section(t, "Details"), texts; init = 0)
                n > 0 && push!(per, (f, n))
            end
            measured = sum(last, per; init = 0)

            if measured > DETAILS_TOTAL
                @warn """The library carries $measured `# Details` sections over
                         $(length(per)) file(s), above the $DETAILS_TOTAL this check records.
                         The section is abolished, so the count may only fall. Move the new
                         fact by its subject rather than raising the number."""
            end
            @test measured <= DETAILS_TOTAL

            if measured < DETAILS_TOTAL
                println("The count of `# Details` sections has fallen to ", measured,
                        " over ", length(per), " file(s). Lower the number in the same ",
                        "commit that removed them, and retire this testset at zero:")
                println("        DETAILS_TOTAL = ", measured)
            end
        end
    end
    #=
    ---------------------------------------------------------------- alias docstrings

    An alias is a second name for something another docstring already documents. The
    Authority gave a section structure for a type and one for a function, and none for an
    alias (issue #436, under the standards-hardening map #478). Read literally, a factory
    alias needed `# Arguments`, `# Returns` and `# Related`, and an acronym alias needed
    `# Related`. Nothing in the tree did either, so the standard and the code disagreed at
    381 units.

    ADR 0086 settles it: an alias LINKS its canonical unit and RESTATES nothing. Three
    kinds, and three section sets.

      - An ACRONYM alias, `const HRP = HierarchicalRiskParity`, carries NO section. It and
        its target are the same object.
      - A FACTORY alias, `MAD(; kwargs...)::LowOrderMoment`, carries `# Validation` alone,
        and only when its own body raises.
      - A DISPATCH alias, a `const` bound to a type EXPRESSION, carries `# Related` and
        `# References`. A union, a container such as `AbstractVector{<:LinearConstraint}`
        and a parametrised form such as `const RMCVaR{T} = Union{...}` are ONE kind,
        because a caller meets all three as the type a signature dispatches on. Reading
        only the unions would have missed 66 of the 249.

    The kind is read from the parse, not from a name. An acronym and a factory are scoped to
    `src/25_Aliases.jl`, which is where both live and is itself part of the rule. Without
    that scope `const PROP_TAG_MACRO_NAMES = ...` in `src/02_Tools.jl` reads as an acronym
    alias, and it is a computed constant.

    Three checks, and the split between them is the one ADR 0081 drew and ADR 0085 reused. A
    check that DEMANDS a section reads the `swept` flag, because it may not red a file that
    no child map of #404 has swept. A check that FORBIDS one does not, because a file passes
    it by changing nothing.
    =#
    @testset "an alias docstring carries only the sections its kind allows" begin
        ALLOWED = Dict(:acronym => String[], :factory => ["# Validation"],
                       :dispatch => ["# Related", "# References"])
        # The count of dispatch aliases carrying no `# Related`. Each file's own #404 prose
        # ticket pays its share. Lower the number in the commit that pays it, and retire the
        # ratchet at zero.
        NO_RELATED_TOTAL = 20

        # A `const` bound to a bare name is an acronym; to a type expression, a dispatch
        # alias. `Expr(:curly, ...)` is a type expression and `Expr(:call, ...)` is a value,
        # which is what keeps `const allowed_functions = Dict{Symbol, Function}(...)` out.
        function alias_kind(d, path)
            in_aliases = endswith(path, "25_Aliases.jl")
            if Meta.isexpr(d, :function) ||
               (Meta.isexpr(d, :(=)) && Meta.isexpr(d.args[1], :call))
                return in_aliases ? :factory : nothing
            end
            e = Meta.isexpr(d, :const) ? d.args[1] : d
            Meta.isexpr(e, :(=)) || return nothing
            lhs, rhs = e.args[1], e.args[2]
            lhs isa Symbol ||
                (Meta.isexpr(lhs, :curly) && lhs.args[1] isa Symbol) ||
                return nothing
            (rhs isa Symbol || Meta.isexpr(rhs, :.)) &&
                return in_aliases ? :acronym : nothing
            Meta.isexpr(rhs, :curly) && return :dispatch
            return nothing
        end

        # Every alias of a file: its name, its kind, and the sections its docstring carries.
        function scan_aliases(path)
            found = Tuple{Symbol, Symbol, Vector{String}}[]
            function walk(node)
                node isa Expr || return nothing
                if isdocstring(node) && length(node.args) >= 4
                    d = node.args[4]
                    k = alias_kind(d, path)
                    if !isnothing(k)
                        nm = bound_name(d)
                        secs = [String(rstrip(l))
                                for l in split(docstring_text(node), '\n')
                                if startswith(l, "# ")]
                        isnothing(nm) || push!(found, (nm, k, secs))
                    end
                end
                foreach(walk, node.args)
                return nothing
            end
            walk(Meta.parseall(read(path, String)))
            return found
        end

        # `sweep/manifest.toml` holds one row per file under `src/` and `ext/`, and
        # `test_45_sweep_census.jl` reds when a file has no row. So its keys are the scope,
        # already gated, and one walk serves all three checks.
        measured = Dict(f => scan_aliases(joinpath(ROOT, f))
                        for f in sort(collect(keys(rows))))

        @testset "no alias carries a section outside its kind" begin
            offenders = String[]
            for f in sort(collect(keys(measured))), (nm, k, secs) in measured[f]
                extra = setdiff(secs, ALLOWED[k])
                isempty(extra) || push!(offenders,
                                        string(f, "  ", nm, "  (", k, ")  ", join(extra, ", ")))
            end
            if !isempty(offenders)
                @warn """$(length(offenders)) alias docstring(s) carry a section their kind
                         does not allow. An alias links its canonical unit and restates
                         nothing: see `## Section Structure for Aliases` in
                         `.github/instructions/julia-docstrings.instructions.md`, and
                         ADR 0086. An acronym alias carries no section, a factory alias
                         carries `# Validation` alone and only when its body raises, and a
                         dispatch alias carries `# Related` and
                         `# References`:\n  $(join(offenders, "\n  "))"""
            end
            @test isempty(offenders)
        end

        @testset "a dispatch alias in a swept file carries # Related" begin
            offenders = String[]
            for f in swept, (nm, k, secs) in measured[f]
                k === :dispatch &&
                    "# Related" ∉ secs &&
                    push!(offenders, string(f, "  ", nm))
            end
            if !isempty(offenders)
                @warn """$(length(offenders)) dispatch alias(es) in a file marked
                         `swept = true` in `sweep/manifest.toml` carry no `# Related`
                         section. The section lists what the alias groups, one bullet per
                         member, and the summary paragraph states why the group
                         exists:\n  $(join(offenders, "\n  "))"""
            end
            @test isempty(offenders)
        end

        @testset "the count of dispatch aliases with no # Related does not rise" begin
            per = Tuple{String, Int}[]
            for f in sort(collect(keys(measured)))
                n = count(t -> t[2] === :dispatch && "# Related" ∉ t[3], measured[f])
                n > 0 && push!(per, (f, n))
            end
            total = sum(last, per; init = 0)

            if total > NO_RELATED_TOTAL
                @warn """$total dispatch alias(es) over $(length(per)) file(s) carry no
                         `# Related` section, above the $NO_RELATED_TOTAL this check records.
                         A new dispatch alias carries the section, so the count may only
                         fall:\n  $(join(string.(per), "\n  "))"""
            end
            @test total <= NO_RELATED_TOTAL

            if total < NO_RELATED_TOTAL
                println("The count of dispatch aliases carrying no `# Related` has fallen ",
                        "to ", total, " over ", length(per),
                        " file(s). Lower the number in ",
                        "the same commit that paid it, and retire this testset at zero:")
                println("        NO_RELATED_TOTAL = ", total)
            end
        end
    end

    #=
    The notation contract (issue #481, under the standards-hardening map #478).

    ADR 0085 records the decision and
    `.github/instructions/julia-docstrings.instructions.md` is the Authority, in its section
    "Notation is fixed by symbol and by family". A symbol that appears in the docstrings of
    two or more units gets a `math_dict` key in `src/01_Base.jl`, and every site
    interpolates it. A new description takes a NEW key, because editing a value already in
    the table moves every docstring that interpolates it.

    #478 opened on the drift that follows when a symbol stays inline. Three subtypes of
    `AbstractDenoiseAlgorithm` state one noise condition three ways -- in a parenthesis, in
    set notation, and in the `Where:` list -- and the symbol every sibling of that family
    needs is written by hand at each of its sites and has no key at all.

    ------------------------------------------------------------ what this check matches

    A WHOLE BULLET AGAINST A WHOLE VALUE, never a symbol against a symbol.

    A glyph is not owned by a key. `\boldsymbol{w}` is `math_dict[:w_port]`, the portfolio
    weights vector, inside a risk measure; it is the observation weights in
    `src/02_Tools.jl` and the OWA weight vector in
    `src/19_RiskMeasures/10_OWARiskMeasures.jl`. Matching on the symbol alone reported 149
    sites, and the great majority of them define a different quantity that the key would
    state wrongly -- `src/02_Tools.jl` among them, the one such site inside a swept file.
    Matching the whole bullet against the whole value reports only a COPY of the dictionary
    text. That copy is the drift the rule exists to stop, and the match cannot fire on a
    glyph that two families share.

    An interpolation leaves no text behind, which is what makes the match cheap.
    `docstring_text` above joins the literal pieces of an interpolating docstring, so
    `math_dict[:T]` contributes nothing to the text it returns. A bullet that still reads as
    the value is therefore a hand-written copy of it.

    The cost of the trade is stated plainly: a copy that drifts by one word stops matching
    and stops being reported. That is the trade the exact-name resolution of
    `test_46_standards_citation_census.jl` already makes, and a drifted copy is what the
    per-file sweep ticket reads by hand.

    The FAMILY half of the rule -- siblings of one leaf abstract supertype state a shared
    quantity in the same form -- is not gated here or anywhere. An equation's form is not a
    token. It holds by review, in the sense of `STANDARDS.md`.

    ---------------------------------------------------------------------- the two checks

      1. A file whose manifest row reads `swept = true` carries ZERO copies. No swept file
         carries one today, so this check needs no debt list of its own.
      2. The library-wide count may not rise above `MATH_COPY_TOTAL`. Each #404 sweep ticket
         lowers it as its file migrates, and the check retires when it reaches zero.

    Both mirror the `# Details` pair above, for the same reason: ADR 0085 settles that the
    migration is per file, inside each file's own sweep ticket, and never in one
    library-wide pass.
    =#
    @testset "a math_dict value is interpolated, never copied" begin
        MATH_COPY_TOTAL = 8

        #=
        `math_dict` is read from source with the same instrument the rest of this testset
        uses, so this check loads no package either. The table is built by a bare `Dict`
        call, so each of its entries parses to `Expr(:call, :(=>), QuoteNode(key), value)`.
        =#
        function math_dict_values(path)
            acc = Dict{String, Symbol}()
            function walk(node)
                node isa Expr || return nothing
                if Meta.isexpr(node, :(=)) &&
                   node.args[1] === :math_dict &&
                   node.args[2] isa Expr
                    for p in node.args[2].args
                        if Meta.isexpr(p, :call) &&
                           length(p.args) == 3 &&
                           p.args[1] === :(=>) &&
                           p.args[2] isa QuoteNode &&
                           p.args[3] isa AbstractString
                            acc[strip(p.args[3])] = p.args[2].value
                        end
                    end
                    return nothing
                end
                foreach(walk, node.args)
                return nothing
            end
            walk(Meta.parseall(read(path, String)))
            return acc
        end

        mvals = math_dict_values(joinpath(ROOT, "src", "01_Base.jl"))
        @test !isempty(mvals)

        # Every bullet of the docstring, stripped of its marker. A `math_dict` value is one
        # line, so a bullet that wraps onto a second line cannot be a copy of one.
        function copied_keys(text, mvals)
            acc = Symbol[]
            for line in split(text, '\n')
                m = match(r"^\s*[-*]\s+(.*?)\s*$", line)
                isnothing(m) && continue
                k = get(mvals, m.captures[1], nothing)
                isnothing(k) || push!(acc, k)
            end
            return acc
        end

        @testset "a swept file copies no math_dict value" begin
            offenders = String[]
            for f in swept
                names, texts, _ = scan(joinpath(ROOT, f))
                for (nm, t) in zip(names, texts), k in copied_keys(t, mvals)
                    push!(offenders, string(f, "  ", nm, "  math_dict[:", k, "]"))
                end
            end
            if !isempty(offenders)
                @warn """$(length(offenders)) `Where:` bullet(s) in a file marked
                         `swept = true` in `sweep/manifest.toml` write out a `math_dict`
                         value instead of interpolating it. Replace the bullet with
                         `\$(math_dict[:key])`, under `Notation is fixed by symbol and by
                         family` in
                         `.github/instructions/julia-docstrings.instructions.md`. The
                         columns are the file, the documented name, and the key it
                         copies:\n  $(join(offenders, "\n  "))"""
            end
            @test isempty(offenders)
        end

        @testset "the library-wide count of copied values does not rise" begin
            # The manifest keys are the scope, exactly as they are for `# Details` above:
            # `test_45_sweep_census.jl` reds when a file under `src/` or `ext/` has no row,
            # so no second file walk is needed.
            per = Tuple{String, Int}[]
            for f in sort(collect(keys(rows)))
                _, texts, _ = scan(joinpath(ROOT, f))
                n = sum(t -> length(copied_keys(t, mvals)), texts; init = 0)
                n > 0 && push!(per, (f, n))
            end
            measured = sum(last, per; init = 0)

            if measured > MATH_COPY_TOTAL
                @warn """$measured `Where:` bullet(s) over $(length(per)) file(s) copy a
                         `math_dict` value, above the $MATH_COPY_TOTAL this check records. A
                         shared symbol is interpolated, so the count may only
                         fall:\n  $(join(string.(per), "\n  "))"""
            end
            @test measured <= MATH_COPY_TOTAL

            if measured < MATH_COPY_TOTAL
                println("The count of copied `math_dict` values has fallen to ", measured,
                        " over ", length(per), " file(s). Lower the number in the same ",
                        "commit that paid it, and retire this testset at zero:")
                println("        MATH_COPY_TOTAL = ", measured)
            end
        end
    end
end
