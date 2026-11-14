#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_79_type_decided_early_return_census.jl` loads it
the same way.
=#
module AnyContainerCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Any container census: a container takes the element type of its values (#1979)" begin
    using Test

    #=
    Issue #1979. The element type of a container comes from its values, and `Any` is the last
    resort. `.github/instructions/julia-source-code.instructions.md` § *A container takes the
    element type of its values* is the Authority. A container of `Any` makes each read a dynamic
    dispatch and boxes each value, and inference gives `Any` to every value that leaves it.

    The census reads the parse tree of every file under `src/` and `ext/`, so a docstring, a
    doctest and a comment are out of its reach: they are text, and a reader copies code. It reads
    these spellings of a container of `Any`:

    - a container type with `Any` as a parameter, such as `Vector{Any}`, `Dict{Symbol, Any}` or
      `Tuple{String, Any}`, and `Vararg{Any}`, so `Tuple{Vararg{Any}}`. A bound `<:Any` is not
      `Any`, so a method signature passes;
    - a typed literal or comprehension, `Any[…]` and `Any[x for x in v]`;
    - an empty untyped literal `[]`, and a constructor with no type parameter and no argument,
      `Dict()`, `Set()`, `IdDict()` and `Vector()`;
    - `collect(Any, v)`, `similar(v, Any)` and `fill`, `zeros` or `Array` with an `Any` argument.

    The grep of the issue found 71 lines in 28 files. The parse tree found five sites that the
    grep missed: two `[]`, two `Dict()` and one `Dict{String, Any}` that a line break hid. The
    commit that added this file narrowed every site that has a narrower type.

    `ALLOWED` holds each definition that keeps `Any`, with the count of its sites and the reason.
    A comment beside each site states the same reason. A definition is named as a Coverage
    Exemption names it, so a site in a closure counts for the definition that holds it. A count
    that falls is a stale entry, so the census asserts each count exactly.

    The census does not read a struct whose type parameter is `Any` at run time, or a literal of
    mixed values that Julia types as `Any`, such as `Dict(:a => 1, :b => "x")`. Neither spelling
    names `Any`, and only a run of the code shows the type.
    =#
    CH = AnyContainerCensusHealth.CodeHealth
    root = normpath(joinpath(@__DIR__, ".."))

    CONTAINERS = (:Vector, :Matrix, :Array, :AbstractVector, :AbstractMatrix,
                  :AbstractArray, :Dict, :AbstractDict, :IdDict, :Set, :Ref, :Pair, :Tuple,
                  :NTuple, :Vararg)
    EMPTY_CONSTRUCTORS = (:Dict, :Set, :IdDict, :Vector)
    ANY_ARGUMENT = (:collect, :similar, :fill, :zeros, :Array)

    # The bare name of a callee or of a type, also when a module qualifies it.
    bare(x::Symbol) = x
    bare(x::Expr) = x.head === :. && x.args[end] isa QuoteNode ? x.args[end].value : nothing
    bare(x) = nothing

    # The spelling of a container of `Any` that `ex` is, or `nothing`.
    function any_container(ex::Expr)
        h = ex.head
        if h === :curly && bare(ex.args[1]) in CONTAINERS && any(==(:Any), ex.args[2:end])
            return "$(ex.args[1]){…Any…}"
        elseif h in (:ref, :typed_comprehension, :typed_vcat, :typed_hcat) &&
               ex.args[1] === :Any
            return "Any[…]"
        elseif h === :vect && isempty(ex.args)
            return "[]"
        elseif h === :call && bare(ex.args[1]) in EMPTY_CONSTRUCTORS && length(ex.args) == 1
            return "$(ex.args[1])()"
        elseif h === :call &&
               bare(ex.args[1]) in ANY_ARGUMENT &&
               any(==(:Any), ex.args[2:end])
            return "$(ex.args[1])(…Any…)"
        end
        return nothing
    end

    # Every site of one file, as `(definition, line, spelling)`.
    function sites(tree)
        out = Tuple{String, Int, String}[]
        line = 0
        name = "<toplevel>"
        function visit(ex)
            if ex isa LineNumberNode
                line = ex.line
            elseif ex isa Expr
                if CH.isdocstring(ex)
                    visit(ex.args[end])
                    return nothing
                end
                s = any_container(ex)
                if !isnothing(s)
                    push!(out, (name, line, s))
                end
                foreach(visit, ex.args)
            end
            return nothing
        end
        for top in tree.args
            n = CH.definition_name(top)
            name = isempty(n) ? "<toplevel>" : n
            visit(top)
        end
        return out
    end

    #=
    Each definition that keeps `Any`, keyed by `file => definition`, with the count of its sites
    and the reason. Two reasons exist today.

    - An `Expr` argument list. Julia defines `Expr.args` as a `Vector{Any}`, so code generation
      that builds one keeps `Any[]`.
    - A value of any type: a value that `throw` takes, a TOML preference, a solver trial that
      holds values of unrelated types, and the value that a `PipelineStep` of a function fits.
    =#
    EXPR = "the list becomes the arguments of an `Expr`, which Julia defines as a `Vector{Any}`"
    ALLOWED = Dict(("src/01_Base/05_Preferences.jl", "__init__") => (1,
                                                                     "a preference is a TOML value of the type the user wrote, and `apply_preferences!` must receive a value of the wrong type to refuse it"),
                   ("src/02_Tools/05_Propagatable.jl", "prop_channel_pairs") => (1, EXPR),
                   ("src/02_Tools/05_Propagatable.jl", "propagatable_parse_body") =>
                       (1, EXPR),
                   ("src/02_Tools/05_Propagatable.jl", "propagatable") => (1, EXPR),
                   ("src/02_Tools/06_ForwardProperties.jl", "forward_walk_expr") =>
                       (1, EXPR),
                   ("src/02_Tools/06_ForwardProperties.jl", "forward_properties") =>
                       (4, EXPR),
                   ("src/05_Moments/01_Base_Moments.jl", "windowed_parse_forward") =>
                       (1, EXPR),
                   ("src/05_Moments/01_Base_Moments.jl", "windowed_method_doc") =>
                       (1, EXPR),
                   ("src/05_Moments/01_Base_Moments.jl", "windowed_method_def") =>
                       (10, EXPR),
                   ("src/05_Moments/01_Base_Moments.jl", "windowed_type_doc") => (3, EXPR),
                   ("src/05_Moments/01_Base_Moments.jl", "windowed_estimator") => (1, EXPR),
                   ("src/05_Moments/32_CrossSectionalFactorModel/01_CrossSectionalRegression_a.jl",
                    "cross_sectional_foreach") => (1,
                                                   "`throw` takes a value of any type, so the slot of a caught value holds any type"),
                   ("src/09_ConstraintGeneration/02_LinearConstraintGeneration_a.jl",
                    "collect_terms!") => (1, EXPR),
                   ("src/17_Optimisation/05_JuMP/01_Base_JuMPOptimisation/05_JuMPModelSetup.jl",
                    "optimise_JuMP_model!") => (1,
                                                "a solver trial holds values of unrelated types: the settings, an exception, a termination status and a solution summary"),
                   ("src/21_Pipeline/03_Pipeline.jl", "fit") => (1,
                                                                 "a `PipelineStep` that wraps a function fits to the value that the function returns, which can have any type"),
                   ("src/21_Pipeline/06_OnlinePipeline.jl", "fit_pipeline_from_state") =>
                       (1,
                        "a `PipelineStep` that wraps a function fits to the value that the function returns, which can have any type"))

    found = Dict{Tuple{String, String}, Vector{String}}()
    for f in CH.source_files(; root)
        for (name, line, s) in sites(CH.parse_file(f; root))
            push!(get!(found, (f, name), String[]), "$f:$line ($name): $s")
        end
    end
    @test !isempty(found)

    # A site outside the allow-list, or a definition whose count moved, fails the census.
    unlisted = String[]
    for (key, hits) in found
        if get(ALLOWED, key, (0, ""))[1] != length(hits)
            append!(unlisted, hits)
        end
    end
    if !isempty(unlisted)
        @info "Containers of `Any`. Derive the element type from the values: a comprehension, `map` or a broadcast takes it, a copy of two kinds takes `promote_type` or `typejoin`, and a container filled out of order is built in parts. Keep `Any` only where no narrower type exists, with a comment beside it, and add the definition to `ALLOWED` with its reason:\n" *
              join(sort!(unlisted), "\n")
    end
    @test isempty(unlisted)
    # An entry whose sites are gone is stale.
    stale = sort!([k for k in keys(ALLOWED) if !haskey(found, k)])
    if !isempty(stale)
        @info "Entries of `ALLOWED` that no longer name a site:\n" *
              join(string.(stale), "\n")
    end
    @test isempty(stale)
    # Every entry states a reason.
    @test all(v -> !isempty(v[2]), values(ALLOWED))

    # The matcher reads each spelling, and it lets a bound and a narrow type past.
    # `spell` walks the parse of one line, as the census walks a file, and gives the first site.
    function spell(s)
        found = sites(Meta.parseall(s))
        return isempty(found) ? nothing : found[1][3]
    end
    @test spell("Vector{Any}(undef, n)") == "Vector{…Any…}"
    @test spell("Dict{Symbol, Any}()") == "Dict{…Any…}"
    @test spell("Base.Dict{Symbol, Any}") == "Base.Dict{…Any…}"
    @test spell("Tuple{String, Any, Any}") == "Tuple{…Any…}"
    # `Tuple{Vararg{Any}}` is the plain `Tuple`, a tuple of any values.
    @test spell("Tuple{Vararg{Any}}") == "Vararg{…Any…}"
    @test spell("Any[1, 2]") == "Any[…]"
    @test spell("Any[x for x in v]") == "Any[…]"
    @test spell("[]") == "[]"
    @test spell("Dict()") == "Dict()"
    @test spell("Vector()") == "Vector()"
    @test spell("collect(Any, v)") == "collect(…Any…)"
    @test spell("similar(v, Any)") == "similar(…Any…)"
    @test isnothing(spell("AbstractDict{<:AbstractString, <:Any}"))
    @test isnothing(spell("Vector{Type}()"))
    @test isnothing(spell("Union{Nothing, Any}"))
    @test isnothing(spell("Dict(:a => 1)"))
    @test isnothing(spell("[1, 2]"))
    # A site inside a docstring is text, and a site in a closure counts for its definition.
    tree = Meta.parseall("\"\"\"\n    f() -> Vector{Any}\n\"\"\"\nfunction f()\n    g() = Any[]\n    return g()\nend\n")
    @test [(n, s) for (n, _, s) in sites(tree)] == [("f", "Any[…]")]
end
