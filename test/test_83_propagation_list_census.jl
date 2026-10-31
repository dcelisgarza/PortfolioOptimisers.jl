#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with:
`walk_ast`, `isdocstring` and `docstring_text`. The load sits OUTSIDE the `@testset` on
purpose: `include` defines methods, and a method defined inside one top-level statement is
not visible to a call in that same statement. The module wrapper keeps that module's own
names out of the worker module. `test_61_interfaces_section_census.jl` loads it the same way.
=#
module PropagationListCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Propagation list census: the two lists name every tagged field (#1798)" begin
    using Test

    #=
    Issue #1798. Thirteen estimators declare `@fprop @vprop cache`, and their docstrings
    left `cache` out of `## Propagated parameters` and `## View parameters`. Seven of them
    carry a `## The incremental fit` section that says that `cache` travels the propagation
    channels, so the prose and the two lists contradicted each other. Nothing read the lists,
    so the omission passed every gate.

    The rule is in `.github/instructions/julia-docstrings.instructions.md`: a subsection
    documents a channel, it lists each field the channel reads, and it lists them in the order
    of the struct body. This census holds the rule for the two channels below. For each
    docstring of a struct under `src/` and `ext/`:

      - the `factory` channel reads every `@fprop` and every `@wprop` field, and the
        `## Propagated parameters` list names exactly those fields, in body order;
      - the `view` channel reads every `@vprop` field, and the `## View parameters` list
        names exactly those fields, in body order.

    A list item is a line `  - \`field\`: …`. A struct with no tag for a channel needs no
    subsection, and the census does not read one.

    The `## Observation weight parameters` list is out of scope. `obs_weights_view` drops a
    `cache`, so that list omits a `@fprop` field on purpose, and a census of it needs a rule
    for the dropped field first.
    =#
    CH = PropagationListCensusHealth.CodeHealth
    root = normpath(joinpath(@__DIR__, ".."))

    tags = (Symbol("@fprop"), Symbol("@vprop"), Symbol("@wprop"), Symbol("@pprop"),
            Symbol("@cprop"))

    # The struct a docstring documents, under any number of macros, or `nothing`.
    function documented_struct(ex)
        Meta.isexpr(ex, :struct) && return ex
        if Meta.isexpr(ex, :macrocall)
            for a in ex.args
                s = documented_struct(a)
                isnothing(s) || return s
            end
        end
        return nothing
    end

    # The name a struct declares, through `Name{T} <: Super`.
    function struct_name(s)
        n = s.args[2]
        while n isa Expr
            n = n.args[1]
        end
        return n
    end

    # The field name under a `::T` annotation or a `= default`, or `nothing`.
    function field_name(ex)
        ex isa Symbol && return ex
        (Meta.isexpr(ex, :(::)) || Meta.isexpr(ex, :(=))) && return field_name(ex.args[1])
        return nothing
    end

    # The fields each channel reads, in body order. A field docstring is a bare string in
    # the body and has no field name, so it is skipped.
    function channel_fields(s)
        f = Symbol[]
        v = Symbol[]
        for ex in s.args[3].args
            seen = Symbol[]
            while Meta.isexpr(ex, :macrocall) && ex.args[1] in tags
                push!(seen, ex.args[1])
                ex = ex.args[end]
            end
            n = field_name(ex)
            isnothing(n) && continue
            any(in(seen), (Symbol("@fprop"), Symbol("@wprop"))) && push!(f, n)
            Symbol("@vprop") in seen && push!(v, n)
        end
        return f, v
    end

    # The field names a `## heading` subsection lists, in order, or `nothing` when the
    # docstring has no such subsection. The subsection ends at the next heading.
    function listed_fields(text, heading)
        m = findfirst("\n## " * heading * "\n", text)
        isnothing(m) && return nothing
        rest = text[(last(m) + 1):end]
        n = findfirst(r"\n#{1,2} ", rest)
        sec = isnothing(n) ? rest : rest[1:first(n)]
        return [Symbol(c[1]) for c in eachmatch(r"\n  - `([A-Za-z_][A-Za-z_0-9]*)`: ", sec)]
    end

    @testset "the readers" begin
        s = documented_struct(:(@propagatable @concrete struct T <: S
                                    @fprop @vprop pe::Int
                                    """
            doc
            """
                                    @vprop sets
                                    @wprop w
                                    @pprop alpha = 1
                                    plain
                                end))
        @test struct_name(s) === :T
        @test channel_fields(s) == ([:pe, :w], [:pe, :sets])
        text = "x\n## Propagated parameters\n\nintro:\n\n  - `pe`: a.\n  - `w`: b.\n\n## View parameters\n\n  - `sets`: c.\n\n# Examples\n\n  - `no`: d.\n"
        @test listed_fields(text, "Propagated parameters") == [:pe, :w]
        @test listed_fields(text, "View parameters") == [:sets]
        @test isnothing(listed_fields(text, "Observation weight parameters"))
    end

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)

    # Every list that does not equal its channel, as (type, heading, fields, listed).
    bad = Tuple{Symbol, String, Vector{Symbol}, Union{Nothing, Vector{Symbol}}}[]
    checked = 0
    function check_docstring(node)
        CH.isdocstring(node) || return nothing
        s = documented_struct(node.args[end])
        isnothing(s) && return CH.PRUNE
        text = CH.docstring_text(node)
        f, v = channel_fields(s)
        for (heading, fs) in (("Propagated parameters", f), ("View parameters", v))
            isempty(fs) && continue
            checked += 1
            l = listed_fields(text, heading)
            l == fs || push!(bad, (struct_name(s), heading, fs, l))
        end
        return CH.PRUNE
    end

    #=
    A struct that a macro writes is not in the source text. `@windowed_estimator` writes the five
    windowed estimators, with their tags and their docstrings, so the census expands each call
    one level and reads the expansion. No other macro of the library writes a struct:
    `@propagatable` rewrites a struct that the source holds.
    =#
    struct_macros = (Symbol("@windowed_estimator"),)
    expanded = Symbol[]
    for file in files
        CH.walk_ast(CH.parse_file(file; root)) do node
            if Meta.isexpr(node, :macrocall) && node.args[1] in struct_macros
                ex = macroexpand(PortfolioOptimisers, node; recursive = false)
                CH.walk_ast(check_docstring, ex)
                CH.walk_ast(ex) do n
                    s = documented_struct(n)
                    isnothing(s) || push!(expanded, struct_name(s))
                    return isnothing(s) ? nothing : CH.PRUNE
                end
                return CH.PRUNE
            end
            return check_docstring(node)
        end
    end

    # A floor on the walk: the library tags far more lists than this.
    @test checked > 150
    @test sort(expanded) == [:WindowedCokurtosis, :WindowedCoskewness, :WindowedCovariance,
                             :WindowedExpectedReturns, :WindowedVariance]

    # The label names the list, so a failure says which type and which list moved.
    sort!(bad; by = b -> (b[1], b[2]))
    @test ("lists that do not name their channel's fields in body order", bad) ==
          ("lists that do not name their channel's fields in body order", empty(bad))
end
