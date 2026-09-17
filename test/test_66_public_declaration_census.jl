#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_61_interfaces_section_census.jl` loads it the
same way.
=#
module PublicDeclarationCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Public declaration census: an `# Interfaces` section and a `public`/`export` declaration gate each other (ADR 0154)" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1120, decided by [ADR 0154](../docs/adr/0154-a-private-abstract-type-earns-a-public-declaration-when-its-interfaces-section-names-the-seam.md):
    a `# Interfaces` docstring section is the seam signal, not a new marker. An abstract type
    carrying one is a genuine extension point, and documenting the contract already is the
    public promise -- the type, and every verb its section names under a `` ## `verb` ``
    heading or a call-shaped bullet, must carry a `public` or `export` declaration. The
    converse holds too: a `public`/`export`-declared abstract type's docstring must carry a
    `# Interfaces` section, unless the name is foreign-owned (ADR 0128) and therefore always
    public regardless of what its own docstring says.

    Neither direction was gated before this census (issue #1125, the shared infrastructure ADR
    0154 asks for). The ADR's own decision does not run the rule: measured 2026-09-17 against
    `origin/dev` at `56f8dace74`, 93 abstract types already carry a `# Interfaces` section and
    are not yet promoted, naming 44 verbs that are not yet declared either. One promotion
    ticket per top-level `src`/`ext` directory (#1126-#1139) does the promoting, mirroring the
    shape the mirror-tree migration used (#1101 -> #1102-#1118): each ticket removes, from the
    two debt lists below, exactly the type and verb entries it promotes. The debt lists may
    only shrink -- a hand-added entry, or a type that gains a concrete section without ever
    being promoted, is a regression this census must fail on.

    The reverse direction found five pre-existing violations the same day:
    `HierarchicalRiskMeasure` and `RiskMeasure` stated only prose ("subtype this to implement
    concrete risk measures"), and `AbstractCentralityAlgorithm`, `CustomJuMPConstraint` and
    `CustomJuMPObjective` named their contract method under `# Related` rather than under a
    `# Interfaces` heading. They sat on a second, symmetric debt list until issue #1142 wrote
    the five contracts and emptied it; the list stays, empty, so the reverse direction has a
    named place to record a future exemption rather than an edit no one remembers to make.

    `interfaces_section`, `interfaces_verbs` and `declared_types` are read from
    `code_health/CodeHealth.jl`, shared with `test_61_interfaces_section_census.jl` so the two
    censuses can never disagree about where an `# Interfaces` section starts and ends, or about
    which names the package itself declares.
    =#
    CH = PublicDeclarationCensusHealth.CodeHealth
    PO = PortfolioOptimisers
    root = normpath(joinpath(@__DIR__, ".."))

    # ------------------------------------------------------------- the doc side

    # The name an `abstract type` declaration binds, or `nothing` for any other unit.
    function abstract_binding(def)
        Meta.isexpr(def, :abstract) || return nothing
        n = def.args[1]
        while isa(n, Expr)
            n = n.args[1]
        end
        return isa(n, Symbol) ? n : nothing
    end

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)

    # Every abstract type's `# Interfaces` section, keyed by the name its declaration binds.
    # `test_47_alias_and_module_census.jl` already proves no two docstrings share a binding.
    sections = Dict{Symbol, Vector{String}}()
    for f in files
        CH.walk_ast(CH.parse_file(f; root)) do node
            if CH.isdocstring(node)
                b = abstract_binding(node.args[end])
                if b !== nothing
                    s = CH.interfaces_section(CH.docstring_text(node))
                    s === nothing || (sections[b] = s)
                end
            end
            return nothing
        end
    end

    # ------------------------------------------------------------- declaration status

    declared = CH.declared_types(PO)
    is_declared_abstract(n) = haskey(declared, n) && isabstracttype(declared[n])

    status(n) =
        if !isdefined(PO, n)
            :undefined
        elseif Base.isexported(PO, n)
            :exported
        elseif n in names(PO)
            :public
        else
            :neither
        end

    # ------------------------------------------------------------- direction A: a section
    # obliges a declaration

    #=
    Every abstract type this measurement found carrying a `# Interfaces` section on
    2026-09-17, that is neither exported nor `public`-declared. A per-directory promotion
    ticket removes the entries it promotes in the same commit that adds the `public`
    declaration; it never adds an entry.
    =#
    promotion_debt = Set([:AbstractExpectedReturnsAlgorithm, :AbstractMomentAlgorithm])

    #=
    The verbs the still-private types above name that are themselves neither exported nor
    `public`-declared, measured the same day. A verb already public through another route
    (`prior`, `port_opt_view`, `mu_ucs`, ...) is not here -- only a name a promotion ticket
    must still declare alongside its type.
    =#
    verb_debt = Set{Symbol}()

    undeclared_types = Symbol[]
    undeclared_verbs = Tuple{Symbol, Symbol}[]
    for (binding, section) in sections
        is_declared_abstract(binding) || continue
        if status(binding) === :neither && binding ∉ promotion_debt
            push!(undeclared_types, binding)
        end
        for v in CH.interfaces_verbs(section)
            isdefined(PO, v) || continue
            if status(v) === :neither && v ∉ verb_debt
                push!(undeclared_verbs, (binding, v))
            end
        end
    end
    sort!(undeclared_types)
    sort!(undeclared_verbs)
    @test ("'# Interfaces' types with no public/export declaration, off the promotion debt list",
           undeclared_types) ==
          ("'# Interfaces' types with no public/export declaration, off the promotion debt list",
           Symbol[])
    @test ("'# Interfaces' verbs with no public/export declaration, off the verb debt list",
           undeclared_verbs) ==
          ("'# Interfaces' verbs with no public/export declaration, off the verb debt list",
           Tuple{Symbol, Symbol}[])

    # Both directions of debt-list drift are named, so a promotion ticket edits this file in
    # the same commit that shrinks it.
    paid_types = sort!(collect(filter(t -> status(t) !== :neither, promotion_debt)))
    childless_types = sort!(collect(filter(t -> !haskey(sections, t), promotion_debt)))
    paid_verbs = sort!(collect(filter(v -> status(v) !== :neither, verb_debt)))
    @test ("promotion debt entries that are now public/exported", paid_types) ==
          ("promotion debt entries that are now public/exported", Symbol[])
    @test ("promotion debt entries with no '# Interfaces' section any more",
           childless_types) ==
          ("promotion debt entries with no '# Interfaces' section any more", Symbol[])
    @test ("verb debt entries that are now public/exported", paid_verbs) ==
          ("verb debt entries that are now public/exported", Symbol[])

    # ------------------------------------------------------------- direction B: a
    # declaration obliges a section

    #=
    A name naming a method the package adds to a function it does not own classifies always
    public per ADR 0128 -- the same foreign-owned-generic rule
    `test_65_docs_public_private_placement_census.jl` reads a page's placement against. No
    abstract type below is foreign-owned today (the rule concerns functions the package
    extends, not types it declares), so the set is empty; it is named here rather than
    hard-coded away, so a future foreign-owned abstract type is exempted the way ADR 0154
    states rather than by an edit no one remembers to make.
    =#
    foreign_owned_types = Set{Symbol}()

    #=
    Five abstract types were already `public`/`export`-declared on 2026-09-17 with a
    docstring that documents no `# Interfaces` contract; issue #1142 wrote the five contracts
    and emptied this list. May only shrink, so it stays empty: a public abstract type that
    gains no section is a failure below, never an addition here.
    =#
    section_debt = Set{Symbol}()

    missing_section = Symbol[]
    for (n, T) in declared
        isabstracttype(T) || continue
        st = status(n)
        st === :neither && continue
        haskey(sections, n) && continue
        n in foreign_owned_types && continue
        n in section_debt && continue
        push!(missing_section, n)
    end
    sort!(missing_section)
    @test ("public/exported abstract types with no '# Interfaces' section, off the debt list",
           missing_section) ==
          ("public/exported abstract types with no '# Interfaces' section, off the debt list",
           Symbol[])

    paid_sections = sort!(collect(filter(n -> haskey(sections, n), section_debt)))
    childless_sections = sort!(collect(filter(n -> status(n) === :neither, section_debt)))
    @test ("section debt entries that now carry a '# Interfaces' section", paid_sections) ==
          ("section debt entries that now carry a '# Interfaces' section", Symbol[])
    @test ("section debt entries no longer public/exported", childless_sections) ==
          ("section debt entries no longer public/exported", Symbol[])

    # ------------------------------------------------------------- floors

    # A section reader or a declaration classifier that quietly stopped matching would
    # satisfy every assertion above with an empty set on each side, so the shape is proven
    # alive. The promotion debt floor only ever falls, in the same commit that shrinks the
    # debt list below it: the map ends when it, too, reaches zero. `verb_debt` paid its last
    # entry, `regenerate_decay`, in #1144, so it carries no floor any more -- like
    # `section_debt` above, an empty debt list proves nothing about the classifier's health.
    @test length(sections) > 80
    @test length(promotion_debt) > 0
end
