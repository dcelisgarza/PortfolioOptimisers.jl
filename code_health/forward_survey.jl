#!/usr/bin/env julia
#
# The forward survey: every method under `src/` and `ext/` that the wrapper standard reads, with
# the verdict that the rulings of map #2072 give it. Issue #2086 builds it, and issue #2084 rules
# what it is for.
#
#     julia --project=code_health code_health/forward_survey.jl
#     julia --project=code_health code_health/forward_survey.jl src/13_Fees.jl
#     julia --project=code_health code_health/forward_survey.jl src/05_Moments
#
# With no argument the survey reports the tree. With a path it reports the findings in that file,
# or under that directory. It always reads the whole tree, because a verdict counts the callers of
# a callee and reads the methods of other files.
#
# THE DETECTORS. Each one reads a shape that a ruling of #2072 settles, and gives the verdict of
# that ruling. `.github/instructions/julia-source-code.instructions.md` § *A wrapper is for
# convenience or for a dispatch problem*, paragraph *The rulings of #2072*, states the rules.
#
#   interface, per-form  a method that only forwards, and the count of callers of its callee
#                        (#2078, rule 3). A shared callee, a public callee, a hook or a type
#                        keeps it. A callee with this method as its only caller moves its body in.
#                        One private method in front of a type goes to the constructor of the
#                        type as a method (#2112).
#   nothing pair         the value arm of a `Nothing`/value pair, and whether the callee has a
#                        `::Nothing` method (#2077). Case 1 goes. Cases 2 and 3 need a review.
#   default              an untyped or `::Any` fallback that returns its input, `false` or
#                        `nothing`, and a step function that copies one in front of an owner
#                        (#2074, rule 2). A fallback with no owner takes the verdict of #2108,
#                        which `fallback_verdict` states: an `args...` fallback goes, a check
#                        and a query keep theirs, and a contract verb takes a `::Nothing`
#                        method. A step function in front of an owner takes the verdict of
#                        #2109, which `step_verdict` states: a walk keeps its leaf, `nothing`
#                        goes to its owner, and an identity in front of a constructor stays as
#                        (b). `RULED_DEFAULTS` holds the verdicts that the parse cannot read.
#   rebuild              a constructor call that passes `x.f` for every field but one or two
#                        (#2082, rule 6). It goes to `rebuild_struct`, by the rulings of #2110
#                        that `rebuild_verdict` states: a root of another type is no rebuild, a
#                        Partial Fit State keeps the positional constructor as (b), and a type
#                        with no keyword constructor takes one first. `RULED_REBUILDS` holds the
#                        verdicts that the parse cannot read.
#   self-forward         a method that calls its own function with fewer arguments, or with
#                        converted ones (#2081, rule 5).
#   worker               a method of a public verb that sends one form to a private function
#                        with one caller, which is not a hook that `# Interfaces` names (#2080,
#                        rule 4). Its body moves into the verb.
#   preset               a public method that builds one type (#2079). Reason (a).
#   keyword form         a public method that passes its keywords as positional arguments.
#                        `src/CLAUDE.md` keeps a keyword wrapper on a verb of a workflow that a
#                        user calls. Reason (a).
#
# THE VERDICTS. `go`: the rulings remove the method. `keep (a)` and `keep (b)`: the two reasons of
# the standard. `keep (interface)`: the method is not a wrapper, by rule 3 or rule 4.
# `keep (default)`: a query trait owns `false`, and a traversal owns the identity (rule 2).
# `review`: the rulings do not settle the finding alone, and a person checks it.
#
# THE RULINGS. A finding that `kept`, `kept_forwards` or `interface_functions` of
# `test/test_70_trivial_unit_census.jl` holds counts as ruled. The survey reads the three lists
# from the parse of that file. A pair of `pending_forwards` is the debt of #2072, and it is not
# ruled. A candidate is a `go` or a `review` finding that no list holds. The sixth condition of
# #404 is that the survey reports no candidate for the file. The survey exits with 1 when it
# reports a candidate, and with 0 when it reports none.
#
# The survey parses with `Meta.parseall` and loads no package. The forward shape is the one of the
# forward census of `test_70`, through the one reader `CodeHealth.forward_methods`.
# `test/test_89_forward_survey_census.jl` drives each detector over a small fixture tree.

# `code_health/triage.jl` includes the measuring scripts into a module of their own, so the include
# happens once and always into `Main`. CodeHealth must be ONE module.
if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

const CH = Main.CodeHealth

"""
The census file whose lists rule a finding, relative to the root of the repository.
"""
const CENSUS = joinpath("test", "test_70_trivial_unit_census.jl")

"""
    Rulings

The lists of the census that rule a finding: `kept` (the presets and the private names of the
function census), `kept_forwards` (a pair with reason (a) or (b)), `interface_functions` (a
function whose forward methods are interface methods), and `pending_forwards` (the debt of #2072,
which rules nothing).
"""
struct Rulings
    kept::Dict{Symbol, String}
    kept_forwards::Dict{Tuple{Symbol, Symbol}, String}
    interface_functions::Dict{Symbol, String}
    pending::Set{Tuple{Symbol, Symbol}}
end

function Rulings()
    return Rulings(Dict{Symbol, String}(), Dict{Tuple{Symbol, Symbol}, String}(),
                   Dict{Symbol, String}(), Set{Tuple{Symbol, Symbol}}())
end

# The value of a literal of the census lists: a quoted name, a string, a tuple, a pair.
literal(e::QuoteNode) = e.value
literal(e::AbstractString) = String(e)
function literal(e::Expr)
    if e.head === :tuple
        return Tuple(literal(a) for a in e.args)
    elseif e.head === :call && e.args[1] === :(=>)
        return literal(e.args[2]) => literal(e.args[3])
    elseif e.head === :string
        return join(a isa AbstractString ? a : string(a) for a in e.args)
    end
    return error("not a literal of a census list: ", e)
end

# The entries of `Dict{…}(pairs...)` or of `Set{…}([items...])`.
function list_entries(rhs)
    if !(rhs isa Expr && rhs.head === :call)
        return Any[]
    end
    args = filter(a -> !(a isa LineNumberNode), rhs.args[2:end])
    if length(args) == 1 && args[1] isa Expr && args[1].head === :vect
        args = args[1].args
    end
    return Any[literal(a) for a in args]
end

"""
    read_rulings(path) -> Rulings

The four lists of the census file `path`, read from its parse. The file is not run, so the survey
needs no package. A list that the file does not hold is empty.
"""
function read_rulings(path::AbstractString)
    lists = Dict{Symbol, Vector{Any}}()
    wanted = (:kept, :kept_forwards, :interface_functions, :pending_forwards)
    CH.walk_ast(Meta.parseall(read(path, String); filename = path)) do x
        if x.head === :(=) && x.args[1] in wanted
            lists[x.args[1]] = list_entries(x.args[2])
            return CH.PRUNE
        end
    end
    kept = Dict{Symbol, String}(p for p in get(lists, :kept, Any[]))
    kept_forwards = Dict{Tuple{Symbol, Symbol}, String}(p
                                                        for p in get(lists, :kept_forwards,
                                                                     Any[]))
    interfaces = Dict{Symbol, String}(p for p in get(lists, :interface_functions, Any[]))
    pending = Set{Tuple{Symbol, Symbol}}(get(lists, :pending_forwards, Any[]))
    return Rulings(kept, kept_forwards, interfaces, pending)
end

# --- what the tree holds ------------------------------------------------------

"""
    Tree

What the survey reads from the parse of the source files: every method, the fields of every
struct, the supertype of every struct and abstract type, the public names (an `export` or a
`public` line), the names that an `# Interfaces` section names, and the forward methods of the
census.
"""
struct Tree
    defs::Vector{CH.MethodDef}
    by_name::Dict{Symbol, Vector{CH.MethodDef}}
    fields::Dict{Symbol, Vector{Symbol}}
    supers::Dict{Symbol, Symbol}
    public::Set{Symbol}
    hooks::Set{Symbol}
    forwards::Dict{Tuple{Symbol, Symbol}, Vector{Tuple{String, Int}}}
    callers::Dict{Symbol, Set{Symbol}}
end

field_name(a::Symbol) = a
function field_name(a::Expr)
    if a.head === :(::) && length(a.args) == 2 && a.args[1] isa Symbol
        return a.args[1]
    elseif a.head === :(=) || a.head === :const
        # A field with a default of `@kwdef`, and a `const` field of a mutable struct.
        return field_name(a.args[1])
    end
    return nothing
end
field_name(a) = nothing

function struct_fields!(out, e)
    e = CH.unwrap(e)
    if !(e isa Expr)
        return nothing
    end
    if e.head === :struct
        name = CH.defname(e.args[2])
        if isempty(name)
            return nothing
        end
        fs = Symbol[]
        for a in e.args[3].args
            f = field_name(a)
            f === nothing || push!(fs, f)
        end
        out[Symbol(name)] = fs
    elseif e.head === :macrocall
        struct_fields!(out, e.args[end])
    elseif e.head === :module || e.head === :block || e.head === :toplevel
        foreach(a -> struct_fields!(out, a), e.head === :module ? e.args[end].args : e.args)
    end
    return nothing
end

# The supertype of a struct or an abstract type, from its header `T{…} <: S{…}`.
function supertype!(out, header)
    if header isa Expr && header.head === :(<:) && length(header.args) == 2
        name = CH.defname(header.args[1])
        super = type_head(header.args[2])
        if !isempty(name) && super isa Symbol
            out[Symbol(name)] = super
        end
    end
    return nothing
end

# Every name in an expression, a field name too. A use that this misses would turn a shared
# callee into a callee with one caller, so the count errs on the side of a use, as
# `CodeHealth.reference_counts` does.
function names_in!(out, e)
    if e isa Symbol
        push!(out, e)
    elseif e isa QuoteNode
        names_in!(out, e.value)
    elseif e isa Expr
        foreach(a -> names_in!(out, a), e.args)
    end
    return out
end
names_in(e) = names_in!(Set{Symbol}(), e)

"""
    read_tree(files; root) -> Tree

Parse each of `files`, relative to `root`, and read the [`Tree`](@ref).
"""
function read_tree(files; root = CH.REPO_ROOT)
    defs = CH.MethodDef[]
    fields = Dict{Symbol, Vector{Symbol}}()
    supers = Dict{Symbol, Symbol}()
    public = Set{Symbol}()
    hooks = Set{Symbol}()
    for f in files
        tree = CH.parse_file(f; root)
        append!(defs, CH.method_defs(f, tree))
        struct_fields!(fields, tree)
        CH.walk_ast(tree) do x
            if x.head === :struct
                supertype!(supers, x.args[2])
            elseif x.head === :abstract
                supertype!(supers, x.args[1])
            elseif x.head === :export || x.head === :public
                foreach(a -> a isa Symbol && push!(public, a), x.args)
                return CH.PRUNE
            elseif CH.isdocstring(x)
                section = CH.interfaces_section(CH.docstring_text(x))
                if section !== nothing
                    union!(hooks, CH.interfaces_verbs(section),
                           CH.interfaces_links(section))
                end
            end
        end
    end
    by_name = Dict{Symbol, Vector{CH.MethodDef}}()
    for d in defs
        d.qualified || push!(get!(by_name, d.name, CH.MethodDef[]), d)
    end
    structs = Set(keys(fields))
    forwards = CH.forward_methods(defs; owned = Set(keys(by_name)), structs)
    # The functions whose methods name a function, other than that function. The rulings count
    # a caller by function: `worst_case_distribution` is the one caller of `ERM_worst_case`,
    # with four methods (#2078).
    callers = Dict{Symbol, Set{Symbol}}()
    for d in defs, n in names_in(d.body)
        n === d.name || push!(get!(callers, n, Set{Symbol}()), d.name)
    end
    return Tree(defs, by_name, fields, supers, public, hooks, forwards, callers)
end

# --- the findings -------------------------------------------------------------

"""
    SurveyFinding

One finding: the file and line of the method, the shape that a detector read, the name and the
callee, the verdict, the reason for the verdict, and the ruling of the census, or `""`. The callee
of a default is `nothing` when no owner stands behind it.
"""
struct SurveyFinding
    file::String
    line::Int
    shape::String
    name::Symbol
    callee::Union{Nothing, Symbol}
    verdict::String
    why::String
    ruling::String
end

"""
    is_candidate(f::SurveyFinding) -> Bool

Whether the finding asks for work: its verdict is `go` or `review`, and no list of the census rules
it. A pair of `pending_forwards` is a candidate, because it is the debt that #2072 removes.
"""
function is_candidate(f::SurveyFinding)
    return f.verdict in ("go", "review") && !startswith(f.ruling, "ruled")
end

function ruling_of(r::Rulings, name, callee)
    if haskey(r.kept, name)
        return "ruled by `kept`: " * r.kept[name]
    end
    if haskey(r.interface_functions, name)
        return "ruled by `interface_functions`: " * r.interface_functions[name]
    end
    if callee === nothing
        return ""
    end
    if haskey(r.kept_forwards, (name, callee))
        return "ruled by `kept_forwards`: " * r.kept_forwards[(name, callee)]
    end
    if (name, callee) in r.pending
        return "pending"
    end
    return ""
end

method_count(t::Tree, name) = length(get(t.by_name, name, CH.MethodDef[]))
function has_nothing_arm(t::Tree, name)
    return any(d -> CH.forward_signature(d.sig)[3], get(t.by_name, name, CH.MethodDef[]))
end
is_struct(t::Tree, name) = haskey(t.fields, name)

# The positional arguments of a signature, as `(name, type)`, with `nothing` for a missing part.
function positional_arguments(sig)
    sig = CH.call_signature(sig)
    if !(sig isa Expr && sig.head === :call)
        return Tuple{Any, Any}[]
    end
    out = Tuple{Any, Any}[]
    for a in sig.args[2:end]
        if a isa Expr && a.head === :parameters
            continue
        end
        push!(out, argument_parts(a))
    end
    return out
end
argument_parts(a::Symbol) = (a, nothing)
function argument_parts(a::Expr)
    if a.head === :(::)
        return length(a.args) == 2 ? (a.args[1], a.args[2]) : (nothing, a.args[1])
    elseif a.head === :kw || a.head === :(...)
        return argument_parts(a.args[1])
    end
    return (nothing, nothing)
end
argument_parts(a) = (nothing, nothing)

# The name at the head of a type expression: `T` of `T{A}`, of `<:T` and of `Type{T}` alike.
type_head(t::Symbol) = t
function type_head(t::Expr)
    if t.head === :curly
        return type_head(t.args[1])
    end
    if t.head === :(<:)
        return type_head(t.args[end])
    end
    if t.head === :. && t.args[end] isa QuoteNode
        return t.args[end].value
    end
    return nothing
end
type_head(t) = nothing

# --- rebuild (#2082, rule 6) ---------------------------------------------------

# `x.f` as `(x, f)`, or `nothing`.
function field_access(e)
    if e isa Expr &&
       e.head === :. &&
       length(e.args) == 2 &&
       e.args[1] isa Symbol &&
       e.args[2] isa QuoteNode
        return (e.args[1], e.args[2].value)
    end
    return nothing
end

# The root, the count of copied fields and the count of fields that change, when `call` builds
# the struct `T` from the fields of one value with one or two fields changed. `nothing` otherwise.
function rebuild_shape(call, fs)
    args = Any[]
    kws = Any[]
    for a in call.args[2:end]
        if a isa Expr && a.head === :parameters
            append!(kws, a.args)
        elseif a isa Expr && a.head === :kw
            push!(kws, a)
        else
            push!(args, a)
        end
    end
    if !isempty(kws) && isempty(args)
        # The keyword form, `T(; a = x.a, b = v)`.
        if !(length(kws) == length(fs))
            return nothing
        end
        pairs = [(k.args[1], field_access(k.args[2]))
                 for k in kws if k isa Expr && k.head === :kw && k.args[1] isa Symbol]
        if !(length(pairs) == length(fs))
            return nothing
        end
        copies = [p[2] for p in pairs if p[2] !== nothing && p[1] === p[2][2]]
    elseif isempty(kws) && length(args) == length(fs)
        copies = [field_access(a)
                  for (a, f) in zip(args, fs)
                  if field_access(a) !== nothing && field_access(a)[2] === f]
    else
        return nothing
    end
    if isempty(copies)
        return nothing
    end
    roots = first.(copies)
    root = argmax(r -> count(==(r), roots), unique(roots))
    copied = count(==(root), roots)
    changed = length(fs) - copied
    # With one copy, the call by hand is as short as the rebuild.
    if !(1 <= changed <= 2 && copied >= 2)
        return nothing
    end
    return (root, copied, changed)
end

"""
    has_keyword_constructor(t::Tree, T) -> Bool

Whether the type `T` has a method that takes keywords alone. `rebuild_struct` calls that
constructor, so a rebuild of a type without one cannot go to it.
"""
function has_keyword_constructor(t::Tree, T)
    return any(d -> CH.keywords_alone(d.sig), get(t.by_name, T, CH.MethodDef[]))
end

"""
    is_partial_fit_state(t::Tree, T) -> Bool

Whether the struct `T` is a subtype of `AbstractPartialFitState`, through the supertypes that the
tree declares. A rebuild of such a state keeps the positional constructor, called by name, as the
fold exception of `.github/instructions/julia-source-code.instructions.md` § *Immutability* states
(#1807, #2110).
"""
function is_partial_fit_state(t::Tree, T)
    seen = Set{Symbol}()
    S = get(t.supers, T, nothing)
    while S isa Symbol && !(S in seen)
        if S === :AbstractPartialFitState
            return true
        end
        push!(seen, S)
        S = get(t.supers, S, nothing)
    end
    return false
end

"""
The verdict "no rebuild" of [`RULED_REBUILDS`](@ref): the call builds the struct from another type,
so the survey reports nothing.
"""
const NO_REBUILD = "no rebuild"

"""
The verdicts of #2110 that the parse cannot read, keyed by the method name and the struct it
builds. Each root is a local, whose type the parse does not know. An entry is a verdict and its
reason. The verdict [`NO_REBUILD`](@ref) reports nothing. An entry whose pair is no longer a
rebuild shape is stale, and `test/test_89_forward_survey_census.jl` fails on it, so a build ticket
that moves the site to `rebuild_struct` deletes the entry.
"""
const RULED_REBUILDS = Dict{Tuple{Symbol, Symbol}, Tuple{String, String}}(#
                                                                          (:fit,
                                                                           :ElasticNetFit) =>
                                                                              (NO_REBUILD,
                                                                               "`cnt` is the `NamedTuple` of `elastic_net_counts` (#2110)"),
                                                                          (:exposure_ic_summary,
                                                                           :ExposureICSummaryResult) =>
                                                                              (NO_REBUILD,
                                                                               "`s` is the `NamedTuple` of the matrix method (#2110)"),
                                                                          (:cross_sectional_factor_sets,
                                                                           :UniverseSets) =>
                                                                              ("go",
                                                                               "`us` is a `UniverseSets`, and the call changes `dict`: `rebuild_struct(us, (; dict = dict))` (rule 6, #2110)"),
                                                                          (:cross_sectional_benchmark_returns,
                                                                           :AssetPanel) =>
                                                                              ("go",
                                                                               "`pnl` is the `AssetPanel` of `rd`, and the call changes `pf`: `rebuild_struct(pnl, (; pf = pf))` (rule 6, #2110)"),
                                                                          (:unit_scale_risk_measure,
                                                                           :MaxRiskMeasureSettings) =>
                                                                              ("go",
                                                                               "`settings` is the `MaxRiskMeasureSettings` of `r`: `rebuild_struct(settings, (; scale = one(scale)))`, and `r` takes `rebuild_struct` too (rule 6, #2110)"),
                                                                          (:unit_scale_risk_measure,
                                                                           :RiskMeasureSettings) =>
                                                                              ("go",
                                                                               "`settings` is the `RiskMeasureSettings` of `r`: `rebuild_struct(settings, (; scale = one(scale)))` (rule 6, #2110)"),
                                                                          (:no_risk_expr_risk_measure,
                                                                           :RiskMeasureSettings) =>
                                                                              ("go",
                                                                               "`settings` is the `RiskMeasureSettings` of `r`: `rebuild_struct(settings, (; rke = false))` (rule 6, #2110)"),
                                                                          (:bounds_risk_measure,
                                                                           :RiskMeasureSettings) =>
                                                                              ("go",
                                                                               "`settings` is the `RiskMeasureSettings` of `r`: `rebuild_struct(settings, (; ub = ub))` (rule 6, #2110)"),
                                                                          (:unit_scale_returns_estimator,
                                                                           :JuMPReturnsSettings) =>
                                                                              ("go",
                                                                               "`settings` is the `JuMPReturnsSettings` of `r`: `rebuild_struct(settings, (; scale = one(scale)))` (rule 6, #2110)"),
                                                                          (:no_return_expr_returns_estimator,
                                                                           :JuMPReturnsSettings) =>
                                                                              ("go",
                                                                               "`settings` is the `JuMPReturnsSettings` of `r`: `rebuild_struct(settings, (; rte = false))` (rule 6, #2110)"),
                                                                          (:forecast_evaluation_align,
                                                                           :ForecastEvaluationResult) =>
                                                                              ("go",
                                                                               "each `fe` is a `ForecastEvaluationResult`, which has no keyword constructor: add the outer keyword constructor of § *Constructor Pattern*, then `rebuild_struct(fe, (; dates = dates))` (rule 6, #2110)"),
                                                                          (:fit_data_step_from_state,
                                                                           :PriceGapFillResult) =>
                                                                              ("go",
                                                                               "`res` is the `PriceGapFillResult` of `fit_preprocessing`, which has no keyword constructor: add the outer keyword constructor of § *Constructor Pattern*, then `rebuild_struct(res, (; nx = res.nx[keep], v = res.v[keep]))` (rule 6, #2110)"),
                                                                          (:schedule_update!,
                                                                           :WindowedBestRateState) =>
                                                                              ("keep (b)",
                                                                               "the schedule state that the fold of a `MirrorDescentState` steps: the fold exception keeps the positional constructor (#1807, #2110)"))

"""
    rebuild_sites(t::Tree) -> Vector

Every call in a method of the tree that builds a struct `T` from the fields of one value `root`,
with one or two fields changed, as `(d, T, root, copied, changed, nfields)`. A method keeps only
the first call of each `T`. A constructor method of `T` owns the build of `T`, so it is skipped.
"""
function rebuild_sites(t::Tree)
    out = []
    for d in t.defs
        if is_struct(t, d.name)
            continue
        end
        seen = Set{Symbol}()
        CH.walk_ast(d.body) do x
            if !(x.head === :call)
                return nothing
            end
            T = if x.args[1] isa Expr && x.args[1].head === :curly
                x.args[1].args[1]
            else
                x.args[1]
            end
            if !(T isa Symbol && is_struct(t, T))
                return nothing
            end
            fs = t.fields[T]
            if !(length(fs) >= 2)
                return nothing
            end
            if T in seen
                return nothing
            end
            shape = rebuild_shape(x, fs)
            if shape === nothing
                return nothing
            end
            push!(seen, T)
            push!(out, (d, T, shape..., length(fs)))
            return nothing
        end
    end
    return out
end

"""
    rebuild_verdict(t::Tree, d, T, root, copied, changed, n) -> Option{Tuple{String, String}}

The verdict and the reason for a call of method `d` that copies `copied` of the `n` fields of
`root` into a `T` and changes `changed`, by the rulings of #2082 and #2110. `nothing` when the
call is no rebuild.

 1. A pair of [`RULED_REBUILDS`](@ref): its verdict.
 2. A root whose declared type is another struct of the tree, or a `NamedTuple`: the call builds
    `T` from another type, so it is no rebuild.
 3. A Partial Fit State keeps the positional constructor by the fold exception: `keep (b)`.
 4. A root of type `T` goes to `rebuild_struct`. A `T` with no keyword constructor first takes
    the outer keyword constructor of § *Constructor Pattern*.
 5. Any other root needs a review: the parse does not know its type.
"""
function rebuild_verdict(t::Tree, d, T, root, copied, changed, n)
    if haskey(RULED_REBUILDS, (d.name, T))
        verdict, why = RULED_REBUILDS[(d.name, T)]
        return verdict == NO_REBUILD ? nothing : (verdict, why)
    end
    args = positional_arguments(d.sig)
    i = findfirst(a -> a[1] === root, args)
    head = i === nothing ? nothing : type_head(args[i][2])
    if head !== T && (head === :NamedTuple || (head isa Symbol && is_struct(t, head)))
        return nothing
    end
    if is_partial_fit_state(t, T)
        return ("keep (b)",
                "rebuilds the Partial Fit State `$T`: the fold exception keeps the positional constructor, called by name (#1807, #2110)")
    end
    if head === T && has_keyword_constructor(t, T)
        return ("go",
                "copies $copied of the $n fields of `$root::$T` and changes $changed: `rebuild_struct($root, (; …))` (rule 6). A measured loss keeps it by hand as (b)")
    elseif head === T
        return ("go",
                "copies $copied of the $n fields of `$root::$T` and changes $changed. `$T` has no keyword constructor: add the outer keyword constructor of § *Constructor Pattern*, then `rebuild_struct($root, (; …))` (rule 6, #2110)")
    end
    return ("review",
            "copies $copied of the $n fields of `$root` into a `$T` and changes $changed. It is a rebuild when `$root` is a `$T`, and then it goes to `rebuild_struct` (rule 6). Rule it, and add it to `RULED_REBUILDS` (#2110)")
end

function rebuild_findings!(out, t::Tree, rulings)
    for (d, T, root, copied, changed, n) in rebuild_sites(t)
        v = rebuild_verdict(t, d, T, root, copied, changed, n)
        if v === nothing
            continue
        end
        push!(out,
              SurveyFinding(d.file, d.line, "rebuild", d.name, T, v[1], v[2],
                            ruling_of(rulings, d.name, T)))
    end
    return out
end

# --- self-forward (#2081, rule 5) ----------------------------------------------

function mentions_field(e)
    if !(e isa Expr)
        return false
    end
    if e.head === :. && length(e.args) == 2 && e.args[2] isa QuoteNode
        return true
    end
    return any(mentions_field, e.args)
end

# `go` or `keep (a)` for a self-call, and the reason.
function self_verdict(t::Tree, d, call)
    names = CH.forward_signature(d.sig)[1]
    passed = Set{Symbol}()
    converted = false
    for a in call.args[2:end]
        if a isa Expr && a.head === :parameters
            (names_in!(passed, a); continue)
        end
        names_in!(passed, a)
        if !CH.forward_leaf(a)
            converted = true
        elseif !(CH.forward_root(a) in names)
            # A literal or a global fills a slot: the canonical form of the call.
            converted = true
        end
    end
    dropped = any(p -> p[1] === nothing || !(p[1] in passed), positional_arguments(d.sig))
    if slots_keywords(d, call) && !dropped
        # `src/CLAUDE.md` keeps a keyword wrapper on a verb of a workflow that a user calls.
        return if d.name in t.public
            ("keep (a)", "a public verb takes a keyword and passes it as a slot (rule 5)")
        else
            ("go",
             "a private function moves a keyword to a slot: the callers pass the slot")
        end
    elseif positional_count(call) > length(positional_arguments(d.sig))
        return ("go",
                "fills a slot: one method takes it as a defaulted positional argument (rule 5)")
    elseif converted
        return if d.name in t.public
            ("keep (a)",
             "a public verb converts its arguments to the canonical form (rule 5)")
        else
            ("go",
             "a private function converts its arguments: the caller passes the canonical form (rule 5)")
        end
    elseif dropped
        return ("go",
                "drops an argument: one method takes the slot as an ignored or defaulted positional argument (rule 5)")
    end
    return ("go", "reorders the arguments: the callers call the other method")
end

# Whether `call` passes a keyword of the method `d` as a positional argument.
function slots_keywords(d, call)
    keywords = keyword_names(d.sig)
    return any(a -> !(a isa Expr && a.head in (:parameters, :kw)) &&
                    CH.forward_root(a) in keywords, call.args[2:end])
end

function positional_count(call)
    return count(a -> !(a isa Expr && a.head in (:parameters, :kw)), call.args[2:end])
end

# The names of the keyword arguments of a signature.
function keyword_names(sig)
    sig = CH.call_signature(sig)
    out = Set{Symbol}()
    if !(sig isa Expr && sig.head === :call)
        return out
    end
    for a in sig.args[2:end], k in (a isa Expr && a.head === :parameters ? a.args : ())
        n = argument_parts(k)[1]
        n isa Symbol && push!(out, n)
    end
    return out
end

function self_findings!(out, t::Tree, rulings, done)
    for d in t.defs
        if (d.qualified || is_struct(t, d.name))
            continue
        end
        if (d.file, d.line, d.name) in done
            continue
        end
        call = CH.single_call(d.body)
        if (call === nothing || call.args[1] !== d.name)
            continue
        end
        if CH.keywords_alone(d.sig)
            continue
        end
        names, typed, nothing_arm = CH.forward_signature(d.sig)
        if !(typed && !nothing_arm)
            continue
        end
        if any(mentions_field, call.args[2:end])
            continue
        end
        if !(any(n -> n in names, names_in(Expr(:tuple, call.args[2:end]...))))
            continue
        end
        verdict, why = self_verdict(t, d, call)
        push!(out,
              SurveyFinding(d.file, d.line, "self-forward", d.name, d.name, verdict, why,
                            ruling_of(rulings, d.name, d.name)))
    end
    return out
end

# --- the forward methods (#2077, #2078, #2079, #2080) ---------------------------

# Whether a method of `name` holds logic of its own, so that it can be a driver.
function has_driver(t::Tree, name)
    return any(m -> CH.forward_call(m.body) === nothing,
               get(t.by_name, name, CH.MethodDef[]))
end

# The count of the methods of `name` that are no `Nothing` arm: its value arms.
function value_arm_count(t::Tree, name)
    return count(d -> !CH.forward_signature(d.sig)[3], get(t.by_name, name, CH.MethodDef[]))
end

function forward_verdict(t::Tree, name, callee, d)
    others = setdiff(get(t.callers, callee, Set{Symbol}()), (name,))
    n_values = value_arm_count(t, name)
    if name in t.public && slots_keywords(d, CH.forward_call(d.body))
        # `src/CLAUDE.md` keeps a keyword wrapper on a verb of a workflow that a user calls.
        return ("keyword form", "keep (a)",
                "the keyword form of a public verb passes its keywords as slots")
    elseif has_nothing_arm(t, name) && n_values == 1
        return if has_nothing_arm(t, callee)
            ("nothing pair", "go",
             "case 1: `$callee` has a `::Nothing` method, so the pair goes (#2077)")
        else
            ("nothing pair", "review",
             "case 2 or 3: `$callee` has no `::Nothing` method. Its `::Nothing` method is right when `nothing` is a true answer for every caller of `$callee`, and otherwise the pair is (b) (#2077)")
        end
    elseif is_struct(t, callee)
        if name in t.public
            return ("preset", "keep (a)", "a public preset or keyword constructor (#2079)")
        elseif n_values >= 2 || name in t.hooks
            return ("interface", "keep (interface)",
                    "the callee is the constructor of a type (rule 3)")
        end
        # One private method holds a form that the constructor can hold (#2112).
        return ("interface", "go",
                "one private method that builds a `$callee`: a second name for the constructor. Its form goes to `$callee` as a method, and its callers call `$callee` (#2112)")
    elseif name in t.public && callee in t.hooks && has_driver(t, name)
        return ("worker", "go",
                "goes around the driver of `$name` to the hook `$callee`, and the driver gives the same answer. A measured loss keeps it as (b) (rule 4)")
    elseif name in t.public &&
           !(callee in t.public) &&
           !(callee in t.hooks) &&
           isempty(others)
        return ("worker", "go",
                "only `$name` calls the private worker `$callee`: its body moves into `$name` (rule 4)")
    end
    shape = if length(unique(c for (n, c) in keys(t.forwards) if n === name)) >= 2
        "per-form"
    else
        "interface"
    end
    if callee in t.public
        return (shape, "keep (interface)", "`$callee` is a shared public verb (rule 3)")
    elseif callee in t.hooks
        return (shape, "keep (interface)",
                "`$callee` is a seam that `# Interfaces` names (rule 3)")
    elseif n_values >= 2 || name in t.hooks
        return if isempty(others)
            (shape, "go",
             "only `$name` calls `$callee`: its body moves into `$name` as a method (rule 3)")
        else
            (shape, "keep (interface)",
             "$(length(others) + 1) functions share the callee `$callee` (rule 3)")
        end
    elseif name in t.public
        return (shape, "review",
                "one public method that forwards: a convenience for the user (a), or a second name")
    end
    return (shape, "go", "one private method that forwards: its callers call `$callee`")
end

function forward_findings!(out, t::Tree, rulings, done)
    for d in t.defs
        if (d.file, d.line, d.name) in done
            continue
        end
        for ((name, callee), sites) in t.forwards
            if !(name === d.name && (d.file, d.line) in sites)
                continue
            end
            if callee === name
                continue
            end
            if !(CH.forward_call(d.body).args[1] === callee)
                continue
            end
            shape, verdict, why = forward_verdict(t, name, callee, d)
            push!(out,
                  SurveyFinding(d.file, d.line, shape, name, callee, verdict, why,
                                ruling_of(rulings, name, callee)))
        end
    end
    return out
end

# --- default (#2074, rule 2) ---------------------------------------------------

function single_statement(body)
    stmts = if body isa Expr && body.head === :block
        filter(a -> !(a isa LineNumberNode), body.args)
    else
        Any[body]
    end
    if !(length(stmts) == 1)
        return missing
    end
    s = stmts[1]
    if s isa Expr && s.head === :return
        return isempty(s.args) ? nothing : s.args[1]
    end
    return s
end

"""
The verdicts of #2074, #2108 and #2109 that the parse cannot read. With no owner, an identity on
two or more slots is a traversal step or a pass-through that stays, or a fallback that only a
closed set reaches (#2108). In front of an owner, a step function that returns its input goes
when the owner is a traversal or a repair, and stays as (b) when it holds a policy of a container
(#2074, #2109). Each entry is a name, with its verdict and its reason. An entry whose name is no
longer a `default`
finding is stale, and `test/test_89_forward_survey_census.jl` fails on it, so a build ticket that
types or removes the fallback deletes the entry.
"""
const RULED_DEFAULTS = Dict{Symbol, Tuple{String, String}}(#
                                                           :_wprop => ("keep (default)",
                                                                       "one step of the `factory` traversal: a field that no threaded argument replaces keeps its value (#2108)"),
                                                           :bind_data_benchmark =>
                                                               ("keep (default)",
                                                                "a traversal owns the identity on a leaf and ignores its context slot, as `adapt(to, x) = x` does in Adapt.jl (#2108)"),
                                                           :drift_observations => ("go",
                                                                                   "only a `VecDate` reaches the fallback, so the slot takes that type (#2108)"),
                                                           :online_unreached_path => ("go",
                                                                                      "the typed methods share one body that answers every estimator: one untyped method replaces them and the fallback (#2108)"),
                                                           :resolve_deferred_quantities =>
                                                               ("go",
                                                                "only the tests reach the fallback, and the method on `AbstractPriorResult` holds the identity on a leaf (#2108)"),
                                                           :timestamp_positions =>
                                                               ("keep (default)",
                                                                "an index that is not a timestamp or a name passes to `view`, which is the gate of the index (#2108)"),
                                                           :regression_factor_names =>
                                                               ("keep (default)",
                                                                "a query of the names of the loadings axis owns `nothing`, the answer \"none\". Its callee reads fields of the input, so it owns no default for the input (#2109)"),
                                                           :factory_child => ("go",
                                                                              "`factory` is a traversal and owns the identity on every leaf, as `adapt_structure(to, x) = x` does in Adapt.jl: `factory(a, args...; kwargs...) = a` on `::Any` replaces this function (#2109)"),
                                                           :maybe_count_or_fraction =>
                                                               ("go",
                                                                "`count_or_fraction` is a repair at a constructor, and a repair owns the identity on a value that needs none: `count_or_fraction(x) = x` on `::Any` replaces this function (#2109)"),
                                                           :update_online_member => ("go",
                                                                                     "`update_online_estimator` owns the identity on a leaf (#2074), so it replaces this function (#2109)"),
                                                           :update_online_step => ("go",
                                                                                   "`update_online_estimator` takes the identity and a `PipelineStep` method (#2074)"),
                                                           :copy_step_states => ("go",
                                                                                 "`copy_states(x) = x` owns the identity, so only a `PipelineStep` method moves (#2074)"),
                                                           :pipeline_step_factory =>
                                                               ("keep (b)",
                                                                "a policy of the pipeline: only the optimiser and a nested pipeline take the weights (#2074)"),
                                                           :reset_time_dependent_step =>
                                                               ("keep (b)",
                                                                "its `TimeDependent` arm throws for a fit with no folds, a job that differs from the method of the owner (#2074)"))

# Whether a positional argument of the signature is a splat, `args...`.
function has_splat(sig)
    sig = CH.call_signature(sig)
    if !(sig isa Expr && sig.head === :call)
        return false
    end
    return any(a -> a isa Expr && a.head === :(...), sig.args[2:end])
end

# Whether another method of the function of `d` throws: the function is a check.
function throws_elsewhere(t::Tree, d)
    return any(m -> m !== d &&
                    !isempty(intersect(names_in(m.body), (:throw, Symbol("@argcheck")))),
               get(t.by_name, d.name, CH.MethodDef[]))
end

"""
    fallback_verdict(t::Tree, d, kind, args) -> Tuple{String, String}

The verdict and the reason for an untyped fallback with no owner that returns `nothing`, or its
input on two or more slots, by the rulings of #2108:

 1. A name of [`RULED_DEFAULTS`](@ref): its verdict.
 2. An `args...` fallback goes: it takes every arity. A contract verb takes a `::Nothing` method
    on the empty slot, and a check takes one open method with the typed arity.
 3. A check keeps its open method: it owns "no violation" for a type that it has no rule for.
 4. One slot: a query keeps `nothing`, the answer "none". A verb whose name ends in `!` does a
    job, so it owns no default and takes a `::Nothing` method.
 5. Two or more slots that return `nothing`: a typed method on the set that reaches it.
 6. An identity on two or more slots that the list does not hold: a person rules it.
"""
function fallback_verdict(t::Tree, d, kind, args)
    if haskey(RULED_DEFAULTS, d.name)
        return RULED_DEFAULTS[d.name]
    end
    if kind == "its input"
        return ("review",
                "a fallback returns its input on $(length(args)) slots. A traversal step or a pass-through to a gate keeps it, and a fallback that a closed set reaches takes a typed method. Rule it, and add it to `RULED_DEFAULTS` (#2108)")
    end
    if has_splat(d.sig)
        return ("go",
                "an `args...` fallback takes every arity: a contract verb takes a `::Nothing` method on the empty slot, and a check takes one open method with the typed arity (#2108)")
    end
    if occursin(r"assert|validate", string(d.name)) ||
       (length(args) > 1 && throws_elsewhere(t, d))
        return ("keep (default)",
                "a check owns \"no violation\" for a type that it has no rule for, and the field bound is the gate of the type (#2108)")
    end
    if length(args) == 1
        return if endswith(string(d.name), "!")
            ("go",
             "a verb that does a job owns no default: a `::Nothing` method on the empty slot (#2108)")
        else
            ("keep (default)",
             "a query owns `nothing`, the answer \"none\" for every type (#2108)")
        end
    end
    return ("go",
            "a fallback of $(length(args)) slots returns `nothing`: a typed method on the set that reaches it (#2108)")
end

# Whether another method of the function of `d` names the function: the function is a walk, and
# `d` is its leaf.
function is_walk(t::Tree, d)
    return any(m -> m !== d && d.name in names_in(m.body),
               get(t.by_name, d.name, CH.MethodDef[]))
end

"""
    step_verdict(t::Tree, d, kind, owner) -> Tuple{String, String}

The verdict and the reason for an untyped fallback that returns `kind` in a function with a
forward method to `owner`, by the rulings of #2074 and #2109:

 1. A name of [`RULED_DEFAULTS`](@ref): its verdict.
 2. `false`: a query trait owns `false`, so the owner takes the fallback.
 3. A walk, a function that another of its methods calls: it owns its leaf answer, the identity,
    `nothing` for "not found" or `nothing` for "pass". The owner is a callee.
 4. `nothing`: a query or a check owns `nothing`, so the owner takes the fallback. An owner whose
    name ends in `!` does a job and owns no default, so a person rules it.
 5. The input in front of a constructor: a constructor owns no default. The method on the wrapper
    type is an interface method (rule 3), and the bare form is a choice of the container (b).
 6. The input in front of a function: a traversal or a repair owns the identity, a contract verb
    owns none, and a policy of a container stays as (b). A person rules it.
"""
function step_verdict(t::Tree, d, kind, owner)
    if haskey(RULED_DEFAULTS, d.name)
        return RULED_DEFAULTS[d.name]
    end
    if kind == "`false`"
        return ("go",
                "a step function returns `false` in front of `$owner`: a query trait owns `false`, so `$owner` takes the fallback (rule 2)")
    end
    if is_walk(t, d)
        return ("keep (default)",
                "a walk owns its leaf answer: the identity, or `nothing` for \"not found\" or \"pass\". `$owner` is a callee of the walk, not its owner (#2109)")
    end
    if kind == "`nothing`"
        return if endswith(string(owner), "!")
            ("review",
             "a step function returns `nothing` in front of `$owner`, a verb that does a job and owns no default. Rule it, and add it to `RULED_DEFAULTS` (#2109)")
        else
            ("go",
             "a step function returns `nothing` in front of `$owner`: a query or a check owns `nothing`, so `$owner` takes the fallback (#2109)")
        end
    end
    if is_struct(t, owner)
        return ("keep (b)",
                "a step function returns its input in front of the constructor `$owner`, which owns no default: its method on `$owner` is an interface method (rule 3), and the bare form is a choice of the container (#2109)")
    end
    return ("review",
            "a step function returns its input in front of `$owner`. A traversal or a repair owns the identity, and then `$owner` takes the fallback. A contract verb owns none, and a policy of a container stays as (b). Rule it, and add it to `RULED_DEFAULTS` (#2109)")
end

function default_findings!(out, t::Tree, rulings)
    for d in t.defs
        if (d.qualified || is_struct(t, d.name))
            continue
        end
        args = positional_arguments(d.sig)
        if isempty(args)
            continue
        end
        _, typed, _ = CH.forward_signature(d.sig)
        if typed
            continue
        end
        s = single_statement(d.body)
        kind = if s === false
            "`false`"
        elseif s === nothing || s === :nothing
            "`nothing`"
        elseif s isa Symbol && any(a -> a[1] === s, args)
            "its input"
        else
            continue
        end
        owners = unique(c for (n, c) in keys(t.forwards) if n === d.name && c !== d.name)
        if !isempty(owners)
            owner = sort!(owners)[1]
            verdict, why = step_verdict(t, d, kind, owner)
            push!(out,
                  SurveyFinding(d.file, d.line, "default", d.name, owner, verdict, why,
                                ruling_of(rulings, d.name, owner)))
            continue
        end
        verdict, why = if kind == "`false`"
            ("keep (default)", "a query trait owns `false` (rule 2)")
        elseif kind == "its input" && length(args) == 1
            ("keep (default)", "a traversal owns the identity on a leaf (rule 2)")
        else
            fallback_verdict(t, d, kind, args)
        end
        push!(out,
              SurveyFinding(d.file, d.line, "default", d.name, nothing, verdict, why,
                            ruling_of(rulings, d.name, nothing)))
    end
    return out
end

# --- the survey ------------------------------------------------------------------

"""
    survey(; root, files, census) -> Vector{SurveyFinding}

Every finding in `files`, relative to `root`, sorted by file and line. `census` is the file whose
lists rule a finding, and a census that does not exist rules nothing. A method that one detector
reports, no later detector reports, in the order rebuild, self-forward, forward, default.
"""
function survey(; root = CH.REPO_ROOT, files = CH.source_files(; root),
                census = joinpath(root, CENSUS))
    t = read_tree(files; root)
    rulings = isfile(census) ? read_rulings(census) : Rulings()
    out = SurveyFinding[]
    rebuild_findings!(out, t, rulings)
    done() = Set((f.file, f.line, f.name) for f in out)
    self_findings!(out, t, rulings, done())
    forward_findings!(out, t, rulings, done())
    default_findings!(out, t, rulings)
    return sort!(out; by = f -> (f.file, f.line, string(f.name), string(f.callee)))
end

"""
    in_scope(f::SurveyFinding, paths) -> Bool

Whether the finding is in one of `paths`: the file itself, or a file under the directory. An empty
`paths` is the whole tree.
"""
function in_scope(f::SurveyFinding, paths)
    if isempty(paths)
        return true
    end
    return any(p -> f.file == p || startswith(f.file, rstrip(p, '/') * "/"), paths)
end

function render(io::IO, f::SurveyFinding)
    target = f.callee === nothing ? "" : " → " * string(f.callee)
    print(io, f.file, ":", f.line, "  ", f.verdict, "  ", f.shape, "  `", f.name, "`",
          target, "\n    ", f.why)
    if !(isempty(f.ruling))
        print(io, "\n    [", f.ruling, "]")
    end
    println(io)
    return nothing
end

"""
    main(args; io = stdout, root, files) -> Int

Print the findings in the paths `args`, relative to `root`, or in the tree when `args` is empty,
and a summary line. Return 1 when a candidate is printed, and 0 otherwise.
"""
function main(args; io::IO = stdout, root = CH.REPO_ROOT, files = CH.source_files(; root))
    paths = String[isabspath(a) ? relpath(a, root) : normpath(a) for a in args]
    found = filter(f -> in_scope(f, paths), survey(; root, files))
    foreach(f -> render(io, f), found)
    n_files = length(unique(f.file for f in found))
    candidates = count(is_candidate, found)
    verdicts = sort!(unique(f.verdict for f in found))
    tally = join(("$(count(f -> f.verdict == v, found)) $v" for v in verdicts), ", ")
    # A survey that finds nothing must still say so. Silence reads as a run that did not happen.
    println(io, "\n", length(found), " finding(s) in ", n_files, " file(s)",
            isempty(tally) ? "" : ": " * tally, ". ", candidates,
            " candidate(s) without a ruling.")
    return candidates == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
