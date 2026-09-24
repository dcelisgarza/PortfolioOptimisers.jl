#!/usr/bin/env julia
#
# The performance-trap half of the code-health gate.
#
#     julia --project=code_health code_health/perf.jl scan
#     julia --project=code_health code_health/perf.jl scan src/05_Moments/03_Covariance.jl
#     julia --project=code_health code_health/perf.jl scan --rule slice_copy
#     julia --project=code_health code_health/perf.jl check
#     julia --project=code_health code_health/perf.jl refresh
#     julia --project=code_health code_health/perf.jl refresh --accept-rise
#
# It reads the source text of every file under `src/` and `ext/` and finds the shapes that cost
# time or memory for no gain: a slice that copies where a view reads the same values, a reduction
# over a temporary array, a search that materialises every hit to use one, a linear-algebra chain
# that builds a matrix to read a scalar or a diagonal, and one expensive call made twice on the
# same unmutated inputs. Each rule flags a shape only where a replacement with the same meaning
# exists, and `scan` prints that replacement beside each Finding.
#
# `scan` is the review. It prints every Finding with its file, line, definition and hint, and it
# writes nothing. `check` and `refresh` are the gate, and they work as `jet.jl` does: a Finding that
# is not a defect is dismissed in `code_health/rulings.toml` under `[[perf_dismissal]]`, the
# baseline records the count per file and per rule after the Dismissals, and a count may fall and
# may not rise. ADR 0175.
#
# This script parses with `Meta.parseall` and loads only `TOML`, so it loads no package under
# measurement, costs a few seconds, and `test/test_73_performance_trap_census.jl` includes it.

# `code_health/triage.jl` includes the measuring scripts into a module of their own, so the include
# happens once and always into `Main`. CodeHealth must be ONE module.
if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

using Main.CodeHealth
using TOML

const NAME = "perf_baseline.toml"

"""
The version of the rule set. A change to what a rule matches changes every count it reads, so the
number is part of the provenance, and a baseline written under another version is refused rather
than compared. Raise it in the commit that changes a rule, and refresh the baseline in the same
commit.
"""
const RULES_VERSION = 1

"""
The rules, in the order a baseline row prints them. The value is the one-line statement `scan`
prints above the Findings of the rule.
"""
const RULES = ("slice_copy" => "A strided slice passed to a function that only reads it copies the slice. A view reads the same values.",
               "reduce_temporary" => "A reduction over a broadcast or an array comprehension builds an array it reads once.",
               "unfused_broadcast" => "A plain `+` or `-` with a broadcast operand breaks the fusion and builds one more array.",
               "search_temporary" => "A search or a sort builds a whole array, and the code uses one element or the length of it.",
               "linalg_temporary" => "A linear-algebra chain builds a matrix, and the code reads a scalar, a diagonal or a solve from it.",
               "repeated_call" => "One definition makes the same expensive call on the same unmutated inputs more than once.",
               "loop_allocation" => "A loop allocates a new array on every iteration and does not keep it.")

const RULE_NAMES = first.(RULES)

"""
The ratchet binds on every rule column. A row also carries `raw`, the count before Dismissals,
which is context and never binds.
"""
const BINDING = RULE_NAMES

# `Finding` is `CodeHealth`'s, because `code_health/triage.jl` reads it too.

fingerprint(f::Finding) = (f.file, f.rule, f.definition, f.code)

# --- reading expressions ---------------------------------------------------

"""
    callee(e) -> Union{Symbol, Nothing}

The name a call calls. `LinearAlgebra.inv(A)` is named `inv`. A call through anything that is not a
name, such as `f(x)(y)`, has no name and no rule reads it.
"""
callee(x::Symbol) = x
callee(x::QuoteNode) = callee(x.value)
callee(x::GlobalRef) = x.name
function callee(e::Expr)
    if e.head === :. && length(e.args) == 2 && e.args[2] isa QuoteNode
        return callee(e.args[2])
    end
    return nothing
end
callee(x) = nothing

iscall(e, names) = e isa Expr && e.head === :call && callee(e.args[1]) in names
iscall(e, name::Symbol) = iscall(e, (name,))

"""
    positional(e) -> Vector{Any}

The positional arguments of a call, without the callee, a keyword argument or a `;` block.
"""
function positional(e::Expr)
    return [a for a in e.args[2:end] if !(a isa Expr && a.head in (:parameters, :kw))]
end

haskeywords(e::Expr) = any(a -> a isa Expr && a.head in (:parameters, :kw), e.args[2:end])

"""
    isbroadcast(e) -> Bool

`f.(x)` and a dotted operator such as `x .* y`. Each builds an array.
"""
function isbroadcast(e)
    if !(e isa Expr)
        return false
    end
    if e.head === :. &&
       length(e.args) == 2 &&
       e.args[2] isa Expr &&
       e.args[2].head === :tuple
        return true
    end
    if e.head === :call && e.args[1] isa Symbol
        s = String(e.args[1])
        return length(s) >= 2 && s[1] == '.' && s != ".."
    end
    return false
end

"""
    strip_lines(e)

The expression without its `LineNumberNode`s, so two copies of one call written on two lines
compare and print the same.
"""
strip_lines(x) = x
function strip_lines(e::Expr)
    return Expr(e.head, (strip_lines(a) for a in e.args if !(a isa LineNumberNode))...)
end

# A Dismissal is written by copying the code `scan` prints, so the code is one line in the words a
# reader wrote. `string` prints `end` inside a range index as `var"end"`, and a closure over several
# lines.
code(e) = replace(string(strip_lines(e)), "var\"end\"" => "end", r"\s+" => " ")

# --- scalar indices --------------------------------------------------------

"""
The iterators whose elements are integer indices. A loop or generator variable over one of them is
a scalar, so `A[:, j]` with such a `j` is a strided slice and its view is as fast to read as its
copy.
"""
const INDEX_ITERATORS = (:(:), :eachindex, :axes, :OneTo, :reverse, :findall,
                         :LinearIndices)

const INTEGER_TYPES = (:Integer, :Int, :Int64, :Int32, :Signed, :Unsigned, :UInt)

function collect_scalars!(scalars::Set{Symbol}, e)
    CodeHealth.walk_ast(e) do x
        if x.head === :for
            spec = x.args[1]
            for s in (spec.head === :block ? spec.args : (spec,))
                bind_scalar!(scalars, s)
            end
        elseif x.head in (:generator, :filter)
            for s in x.args[2:end]
                bind_scalar!(scalars, s)
            end
        elseif x.head === :(::) &&
               length(x.args) == 2 &&
               x.args[1] isa Symbol &&
               x.args[2] in INTEGER_TYPES
            push!(scalars, x.args[1])
        end
    end
    return scalars
end

function bind_scalar!(scalars, s)
    if !(s isa Expr && s.head === :(=) && length(s.args) == 2)
        return nothing
    end
    var, iter = s.args
    if var isa Symbol && iscall(iter, INDEX_ITERATORS)
        push!(scalars, var)
    elseif var isa Expr &&
           var.head === :tuple &&
           !isempty(var.args) &&
           var.args[1] isa Symbol &&
           iscall(iter, :enumerate)
        push!(scalars, var.args[1])
    end
    return nothing
end

function isscalar_index(i, scalars)
    if i isa Integer || i === :end || i === :begin
        return true
    end
    if i isa Symbol
        return i in scalars
    end
    if iscall(i, (:+, :-, :*, :div, :÷)) || iscall(i, (:first, :last, :length, :size))
        return all(a -> isscalar_index(a, scalars) || a isa Symbol, positional(i))
    end
    return false
end

isrange_index(i) = i === :(:) || iscall(i, :(:))

"""
    isstrided_slice(e, scalars) -> Bool

`A[:, j]`, `x[a:b]` or `X[i, :]`: a `ref` with at least one colon or range index and no index that
could be a vector. A vector index gives a view that BLAS cannot read with a stride, which can be
slower than the copy, so `A[:, idx]` is not flagged.
"""
function isstrided_slice(e, scalars)
    if !(e isa Expr && e.head === :ref && length(e.args) >= 2)
        return false
    end
    # `size(A)[1:2]` slices a tuple, and a tuple has no view.
    if iscall(e.args[1], (:size, :axes, :strides, :tuple))
        return false
    end
    idx = e.args[2:end]
    return any(isrange_index, idx) &&
           all(i -> isrange_index(i) || isscalar_index(i, scalars), idx)
end

# --- the rules -------------------------------------------------------------

"""
A call that makes a copy on purpose, or whose meaning changes when its argument aliases the caller's
array. A `!` function mutates, and a view would carry the mutation back to the parent. A constructor,
spelled with a capital, may keep its argument in a field, and a view would alias it.
"""
const SLICE_KEEPERS = (:copy, :collect, :deepcopy, :view, :getindex, :setindex!, :vec,
                       :reshape, :convert, :identity, :tuple, :Ref, :push!, :append!,
                       :similar)

function slice_reader(e::Expr)
    name = callee(e.args[1])
    if name === nothing || name in SLICE_KEEPERS
        return false
    end
    s = String(name)
    return !endswith(s, "!") && !isuppercase(first(s))
end

strip_adjoint(a) = a isa Expr && a.head === Symbol("'") ? a.args[1] : a

function rule_slice_copy(e, scalars)
    args = if e.head === :call && slice_reader(e)
        positional(e)
    elseif isbroadcast(e) && e.head === :.
        e.args[2].args
    else
        return nothing
    end
    hits = [strip_adjoint(a) for a in args if isstrided_slice(strip_adjoint(a), scalars)]
    return isempty(hits) ? nothing : hits
end

const REDUCERS = (:sum, :prod, :maximum, :minimum, :mean, :any, :all, :count, :extrema,
                  :findmax, :findmin)

"""
The leaves of a broadcast that are scalars whatever the data: a number literal, or a call such as
`zero(T)` that returns one.
"""
const SCALAR_CALLS = (:zero, :one, :eltype, :typemin, :typemax, :eps, :floatmin, :floatmax)

isscalar_leaf(x) = x isa Number || iscall(x, SCALAR_CALLS)

"""
    broadcast_leaves(e) -> Vector{Any}

The operands a fused broadcast reads, through every nested broadcast. `sqrt.(max.(z, 0))` reads
`z` and `0`.
"""
function broadcast_leaves(e)
    if !isbroadcast(e)
        return Any[e]
    end
    ops = e.head === :. ? e.args[2].args : e.args[2:end]
    return reduce(vcat, (broadcast_leaves(a) for a in ops); init = Any[])
end

"""
    rule_reduce_temporary(e) -> Union{Nothing, String}

A reduction whose only positional argument is a broadcast or an array comprehension. The hint
depends on the shape, because only some shapes have a replacement with the same meaning:

  - A broadcast over one array, `sum(abs.(x))` or `any(x .< zero(T))`, becomes `sum(abs, x)` or
    `any(<(zero(T)), x)`, with or without `dims`.
  - A broadcast over two or more operands becomes a generator. A generator takes no `dims`, so a
    reduction with keywords over such a broadcast has no replacement and is not flagged.
  - A comprehension becomes a generator. A generator over an empty collection has no element type,
    so `sum` throws where it returned zero: the hint says to pass `init`.
"""
function rule_reduce_temporary(e)
    if !(e.head === :call && iscall(e, REDUCERS))
        return nothing
    end
    args = positional(e)
    if length(args) != 1
        return nothing
    end
    a = args[1]
    name = callee(e.args[1])
    if a isa Expr && a.head === :comprehension
        return "`$name(f(x) for x in xs)` builds no array; pass `init` when the generator can be empty"
    end
    if !isbroadcast(a)
        return nothing
    end
    if count(!isscalar_leaf, broadcast_leaves(a)) <= 1
        return "`$name(f, x)` applies the function inside the reduction and builds no array"
    end
    if haskeywords(e)
        return nothing
    end
    return "a generator, `$name(f(x[i], y[i]) for i in eachindex(x, y))`, builds no array; " *
           "`sum(x .* y)` is `dot(x, y)` for real vectors"
end

nokw_call(e, names) = iscall(e, names) && !haskeywords(e)

function rule_search_temporary(e)
    if e.head === :call
        args = positional(e)
        if !(length(args) == 1)
            return nothing
        end
        a = args[1]
        if iscall(e, :length) && iscall(a, :findall)
            return "`count(...)` counts without building the index vector"
        elseif iscall(e, :isempty) && iscall(a, :findall)
            return "`!any(...)` stops at the first hit and builds nothing"
        elseif iscall(e, :first) && iscall(a, :findall)
            return "`findfirst(...)` stops at the first hit"
        elseif iscall(e, :last) && iscall(a, :findall)
            return "`findlast(...)` stops at the last hit"
        elseif iscall(e, :first) && nokw_call(a, :sort)
            return "`minimum(x)` reads the same value without the sort"
        elseif iscall(e, :last) && nokw_call(a, :sort)
            return "`maximum(x)` reads the same value without the sort"
        elseif iscall(e, :reverse) && nokw_call(a, :sort)
            return "`sort(x; rev = true)` sorts once and does not copy"
        elseif iscall(e, :length) && iscall(a, :collect)
            return "`length` of the iterator, or `count(Returns(true), itr)`, builds nothing"
        end
    elseif e.head === :ref && length(e.args) == 2
        obj, i = e.args
        if iscall(obj, :findall) && i == 1
            return "`findfirst(...)` stops at the first hit"
        elseif iscall(obj, :findall) && i === :end
            return "`findlast(...)` stops at the last hit"
        elseif nokw_call(obj, :sort) && i == 1
            return "`minimum(x)` reads the same value without the sort"
        elseif nokw_call(obj, :sort) && i === :end
            return "`maximum(x)` reads the same value without the sort"
        elseif iscall(obj, (:sort, :sortperm)) && iscall(i, :(:))
            return "`partialsort`/`partialsortperm(x, a:b)` sorts only the part it returns"
        elseif !(obj isa Expr) &&
               iscall(i, :findall) &&
               length(positional(i)) == 1 &&
               !haskeywords(i)
            return "index with the mask itself, `x[mask]`, and skip the index vector"
        end
    elseif e.head === :for
        spec = e.args[1]
        for s in (spec.head === :block ? spec.args : (spec,))
            if s isa Expr && s.head === :(=) && iscall(s.args[2], :collect)
                return "iterate the range or iterator itself; `collect` builds an array the loop reads once"
            end
        end
    end
    return nothing
end

isadjoint(a) = (a isa Expr && a.head === Symbol("'")) || iscall(a, (:transpose, :adjoint))
islowercase_name(a) = a isa Symbol && islowercase(first(String(a)))
adjoint_operand(a) = a isa Expr && a.head === Symbol("'") ? a.args[1] : positional(a)[1]

"""
    ismatrix_like(a) -> Bool

An operand this library writes as a matrix: a name with a capital first letter, such as `X` or
`Σ`, or a product, a transpose, a `Symmetric` wrapper or a covariance. `inv(alpha)` and `inv(pi)`
are scalars, and `inv(x) * y` for a scalar is not a trap.
"""
function ismatrix_like(a)
    if a isa Symbol
        return isuppercase(first(String(a)))
    end
    return iscall(a, (:*, :transpose, :adjoint, :Symmetric, :Hermitian, :cov, :cor)) ||
           (a isa Expr && a.head === Symbol("'"))
end

function isinverse(a)
    return iscall(a, :inv) && length(positional(a)) == 1 && ismatrix_like(positional(a)[1])
end

function rule_linalg_temporary(e)
    if e.head === :call && iscall(e, :*)
        args = positional(e)
        if any(isinverse, args)
            return "solve with `A \\ B` or `B / A`: no inverse is formed, and the solve is more accurate"
        end
        if any(a -> iscall(a, :diagm), args)
            return "`Diagonal(v)` multiplies in O(n²) and builds no dense matrix"
        end
        # `(A * B) * x` parses as a nested product, and the parentheses force the matrix-matrix
        # product first. `A * B * x` parses as one n-ary call, and Julia picks the cheaper order.
        if length(args) == 2 &&
           iscall(args[1], :*) &&
           all(ismatrix_like, positional(args[1])) &&
           islowercase_name(args[2])
            return "`A * B * x`, without the parentheses, multiplies right to left and builds no matrix"
        end
        # `d' * d * l` is a scalar times a vector, not a quadratic form, so the middle operand must
        # differ from both ends.
        if length(args) == 3 &&
           isadjoint(args[1]) &&
           islowercase_name(adjoint_operand(args[1])) &&
           islowercase_name(args[3]) &&
           args[2] != adjoint_operand(args[1]) &&
           args[2] != args[3]
            return "`dot(x, A, y)` reads the quadratic form without the vector `A * y`, when `x` and `y` are vectors"
        end
    elseif e.head === :call && length(positional(e)) == 1
        a = positional(e)[1]
        if iscall(e, :tr) && iscall(a, :*)
            return "`tr(A * B)` is `dot(A', B)`, which builds no product"
        elseif iscall(e, :diag) && iscall(a, :*)
            return "the diagonal of a product needs n dot products, not the n² entries of the product"
        end
    elseif e.head === :ref && iscall(e.args[1], :*)
        return "compute the entries the index reads, not the whole product"
    end
    return nothing
end

"""
    rule_unfused_broadcast(e) -> Union{Nothing, String}

A plain `+` or `-` with a broadcast operand. `x .* y + z` builds `x .* y`, then a second array
for the sum, where `x .* y .+ z` fuses both into one loop and one array. `-(log.(x))` builds two
arrays where `.-log.(x)` builds one.

Only `+` and `-` are read. Between two arrays they are elementwise, so the dotted form means the
same. `*` and `/` are not read, because between two matrices they are a product and a solve.

The operand must be a broadcast, because a broadcast is an array. `f.(w - v)` is not read: the
parser cannot tell `w - v` from the scalar `1 - alpha` in `(1 - alpha) .* u`, and the scalar form
is the common one. `x - I` is not read either, because `I` does not broadcast.
"""
function rule_unfused_broadcast(e)
    if !iscall(e, (:+, :-))
        return nothing
    end
    args = positional(e)
    if any(a -> a === :I || callee(a) === :I, args)
        return nothing
    end
    if any(isbroadcast, args)
        return "dot the operator too, `.+` or `.-`, so the broadcast fuses into one array"
    end
    return nothing
end

"""
The reductions whose cost is linear in the size of an argument and whose answer, without a `dims`
keyword, is a scalar. Binding a scalar to a name once shares nothing a later mutation could reach.
"""
const SCALAR_REDUCTIONS = (:sum, :prod, :maximum, :minimum, :extrema, :mean, :median, :var,
                           :std, :norm, :opnorm, :det, :logdet, :logabsdet, :tr, :dot)

"""
The factorisations and the covariance estimates, whose cost is at least quadratic. Each returns a
new object, so two calls give two objects: bind it once only where neither use mutates it.

A call that copies on purpose, such as `copy`, `collect` or a broadcast, is absent. Two copies of
one array are often the point, since each is later mutated alone.
"""
const FACTORISATIONS = (:eigen, :eigvals, :eigvecs, :svd, :svdvals, :cholesky, :qr, :lu,
                        :factorize, :schur, :bunchkaufman, :ldlt, :cov, :cor)

function isexpensive(e)
    if !(e isa Expr && e.head === :call) || isempty(positional(e))
        return false
    end
    return iscall(e, FACTORISATIONS) || (iscall(e, SCALAR_REDUCTIONS) && !haskeywords(e))
end

"""
    mutated_names(body) -> Set{Symbol}

Every name the definition assigns, updates in place, or passes to a `!` call. The set is
deliberately wide: a repeated call on any of these names may read a different value the second
time, and `repeated_call` does not flag it.
"""
function mutated_names(body)
    out = Set{Symbol}()
    root(x) =
        if x isa Symbol
            x
        elseif x isa Expr && x.head in (:ref, :., :(::)) && !isempty(x.args)
            root(x.args[1])
        else
            nothing
        end
    CodeHealth.walk_ast(body) do x
        if x.head in
           (:(=), :.=, :+=, :-=, :*=, :/=, :.+=, :.-=, :.*=, :./=, :^=, :local, :global)
            if !(x.head === :(=) && CodeHealth.is_signature(x.args[1]))
                lhs = x.args[1]
                for t in (lhs isa Expr && lhs.head === :tuple ? lhs.args : (lhs,))
                    r = root(t)
                    r === nothing || push!(out, r)
                end
            end
        elseif x.head === :call &&
               (n = callee(x.args[1])) !== nothing &&
               endswith(String(n), "!")
            for a in positional(x)
                r = root(a)
                r === nothing || push!(out, r)
            end
        end
    end
    return out
end

function free_names(e)
    out = Set{Symbol}()
    add(x) = x isa Symbol && push!(out, x)
    add(e)
    CodeHealth.walk_ast(e) do x
        start = x.head === :call ? 2 : 1
        for a in x.args[start:end]
            add(a)
        end
    end
    return out
end

"""
    collect_calls!(seen, x, path)

Record every expensive call under `x` with its branch path: one `(objectid(if), arm)` pair per `if`,
`elseif` or ternary it sits in. A cold path is skipped, as `visit` skips it.
"""
function collect_calls!(seen, x, path)
    if !(x isa Expr) || iscold(x)
        return nothing
    end
    if x.head in (:if, :elseif)
        collect_calls!(seen, x.args[1], path)
        for k in 2:length(x.args)
            collect_calls!(seen, x.args[k], (path..., (objectid(x), k)))
        end
        return nothing
    end
    if isexpensive(x)
        push!(get!(Vector{Any}, seen, code(x)), (x, path))
    end
    for a in x.args
        collect_calls!(seen, a, path)
    end
    return nothing
end

"""
Two calls are exclusive when they sit in different arms of one `if`: at most one of them runs.
"""
function exclusive(p, q)
    return any(((i, k),) -> any(((j, l),) -> i == j && k != l, q), p)
end

"""
    runs_twice(occurrences) -> Union{Nothing, Any}

The second of two occurrences that can both run in one call, or `nothing` when every pair is
exclusive.
"""
function runs_twice(occ)
    for b in 2:length(occ), a in 1:(b - 1)
        if !exclusive(occ[a][2], occ[b][2])
            return occ[b][1]
        end
    end
    return nothing
end

function rule_repeated_call(body, lines::Dict{UInt, Int}, file, def)
    seen = Dict{String, Vector{Any}}()
    collect_calls!(seen, body, ())
    mutated = mutated_names(body)
    second = Dict{String, Any}()
    for (k, v) in seen
        if !(length(v) >= 2 && isdisjoint(free_names(v[1][1]), mutated))
            continue
        end
        s = runs_twice(v)
        s === nothing || (second[k] = s)
    end
    repeated = collect(keys(second))
    out = Finding[]
    for k in sort!(repeated)
        n = length(seen[k])
        # A call inside a larger repeated call is repeated because the larger one is. Report the
        # larger one alone.
        if any(o -> o != k && occursin(k, o) && length(seen[o]) >= n, repeated)
            continue
        end
        line = get(lines, objectid(second[k]), 0)
        push!(out,
              Finding(file, line, "repeated_call", def, k,
                      "computed $n times; bind it to a local name once"))
    end
    return out
end

"""
The calls that allocate a new array whatever their arguments are.
"""
const ALLOCATORS = (:zeros, :ones, :fill, :similar, :copy, :deepcopy, :collect, :vcat,
                    :hcat, :hvcat, :cat, :falses, :trues, :Vector, :Matrix, :Array,
                    :BitVector, :BitMatrix)

"""
The calls that keep the array they are given, so an allocation passed to one is the loop's output
and not a temporary.
"""
const KEEPERS = (:push!, :pushfirst!, :append!, :prepend!, :insert!, :setindex!,
                 :setproperty!, :setfield!, :put!, :return)

isallocation(e) = (e isa Expr && e.head === :comprehension) || iscall(e, ALLOCATORS)

"""
    growth_calls(e) -> Vector{Expr}

The concatenations of `x = vcat(x, y)`: the array is copied into a larger one on every iteration,
so a loop of `n` iterations copies O(n²) entries. The right side may choose between several, as
`x = if a vcat(x, y) else vcat(x, z) end` does, and each is returned.
"""
function growth_calls(e)
    out = Expr[]
    if !(e isa Expr && e.head === :(=) && e.args[1] isa Symbol)
        return out
    end
    lhs = e.args[1]
    CodeHealth.walk_ast(e.args[2]) do x
        if iscall(x, (:vcat, :hcat, :cat)) && lhs in positional(x)
            push!(out, x)
            return CodeHealth.PRUNE
        end
    end
    return out
end

const LOOP_HINTS = Dict(:invariant => "the array is the same every iteration; allocate it once before the loop, and reset it with `fill!` or `copyto!` if the loop mutates it",
                        :varying => "allocate one buffer before the loop and fill it in place with `.=` or a `!` call",
                        :growth => "the array is copied into a larger one every iteration; `push!` or `append!` into it, or concatenate the pieces once after the loop")

"""
    loop_names(spec) -> Set{Symbol}

The names a `for` loop binds, through a tuple destructuring and a `for a in x, b in y` block.
"""
function loop_names(spec)
    out = Set{Symbol}()
    for s in (spec isa Expr && spec.head === :block ? spec.args : (spec,))
        if !(s isa Expr && s.head === :(=))
            continue
        end
        CodeHealth.walk_ast(Expr(:tuple, s.args[1])) do x
            for a in x.args
                a isa Symbol && push!(out, a)
            end
        end
        s.args[1] isa Symbol && push!(out, s.args[1])
    end
    return out
end

"""
    collect_loop_allocations!(out, x, varying, kept)

Every allocation under `x` that runs once per iteration of an enclosing loop. `varying` is `nothing`
outside every loop, and inside one it holds the names that change from one iteration to the next:
the loop variables and every name the loop body assigns. `kept` is true for an expression the loop
keeps: an argument of a `KEEPERS` call, or the right side of `x[i] = …` or `x.f = …`.
"""
function collect_loop_allocations!(out, x, varying, kept::Bool)
    if !(x isa Expr) || iscold(x)
        return nothing
    end
    if x.head in (:for, :while)
        body = x.args[end]
        v = union(something(varying, Set{Symbol}()),
                  x.head === :for ? loop_names(x.args[1]) : Set{Symbol}(),
                  mutated_names(body))
        # The iterator and the condition are read once per loop for a `for` and once per
        # iteration for a `while`.
        collect_loop_allocations!(out, x.args[1], x.head === :while ? v : varying, false)
        collect_loop_allocations!(out, body, v, false)
        return nothing
    end
    if varying !== nothing && !kept && !isempty(growth_calls(x))
        # The concatenation and every allocation inside it are one site.
        for g in growth_calls(x)
            push!(out, (g, :growth))
        end
        return nothing
    end
    if varying !== nothing && !kept && isallocation(x)
        # An allocation inside this one is part of it, so the site is reported once and the walk
        # stops here.
        push!(out, (x, isdisjoint(free_names(x), varying) ? :invariant : :varying))
        return nothing
    end
    keeps = x.head === :return || (x.head === :call && callee(x.args[1]) in KEEPERS)
    stores = x.head === :(=) && x.args[1] isa Expr && x.args[1].head in (:ref, :.)
    for (i, a) in enumerate(x.args)
        collect_loop_allocations!(out, a, varying, keeps || (stores && i == 2))
    end
    return nothing
end

"""
    rule_loop_allocation(body, lines, file, def) -> Vector{Finding}

An array allocated once per iteration of a loop: a call of `ALLOCATORS` or an array comprehension
inside a `for` or `while` body. An allocation the loop keeps, `push!(out, zeros(n))` or
`out[i] = copy(w)`, is the loop's output and is not flagged.

The hint depends on the arguments. An allocation that reads no loop variable and no name the loop
assigns builds the same array each time, and it moves out of the loop. Any other one is filled in
place in a buffer allocated before the loop. An allocation that must stay, because the iteration
hands the array on, takes a Dismissal: one without `code` covers every allocation of the
definition.
"""
function rule_loop_allocation(body, lines::Dict{UInt, Int}, file, def)
    found = Any[]
    collect_loop_allocations!(found, body, nothing, false)
    return [Finding(file, get(lines, objectid(x), 0), "loop_allocation", def, code(x),
                    LOOP_HINTS[kind]) for (x, kind) in found]
end

# --- walking one file ------------------------------------------------------

"""
    toplevel_items(ex) -> Vector{Any}

The top-level expressions of a parsed file, with a `module` block opened, since the package's own
entry file holds its definitions inside one.
"""
function toplevel_items(ex)
    out = Any[]
    for a in ex.args
        if a isa Expr && a.head === :module
            append!(out, toplevel_items(a.args[3]))
        else
            push!(out, a)
        end
    end
    return out
end

"""
    scan_file(file; root) -> Vector{Finding}

Every Finding of one file, before Dismissals, in source order within each rule.
"""
function scan_file(file; root = CodeHealth.REPO_ROOT)
    ex = CodeHealth.parse_file(file; root)
    out = Finding[]
    line = 0
    for item in toplevel_items(ex)
        if item isa LineNumberNode
            line = item.line
            continue
        end
        def = CodeHealth.definition_name(item)
        def = isempty(def) ? "<toplevel>" : def
        scalars = collect_scalars!(Set{Symbol}(), item)
        lines = Dict{UInt, Int}()
        visit(item, line, false) do e, l, inview
            lines[objectid(e)] = l
            add(rule, x, hint) = push!(out, Finding(file, l, rule, def, code(x), hint))
            if !inview && (hits = rule_slice_copy(e, scalars)) !== nothing
                for h in hits
                    add("slice_copy", h,
                        "`@view` it, or `@views` the expression, unless the callee keeps the argument")
                end
            end
            if !((h = rule_reduce_temporary(e)) === nothing)
                add("reduce_temporary", e, h)
            end
            if !((h = rule_unfused_broadcast(e)) === nothing)
                add("unfused_broadcast", e, h)
            end
            if !((h = rule_search_temporary(e)) === nothing)
                add("search_temporary", e, h)
            end
            return (h = rule_linalg_temporary(e)) === nothing ||
                   add("linalg_temporary", e, h)
        end
        append!(out, rule_repeated_call(item, lines, file, def))
        append!(out, rule_loop_allocation(item, lines, file, def))
    end
    return out
end

"""
The macros under which `slice_copy` is silent. `@view` and `@views` already read a view. A JuMP
macro rewrites the expression it holds, and `x[1:N] >= 0` in `@variables` declares a variable: it
is not a slice at all.
"""
const VIEW_MACROS = Symbol.(("@view", "@views", "@variable", "@variables", "@constraint",
                             "@constraints", "@expression", "@expressions", "@objective"))

macro_symbol(x::Symbol) = x
macro_symbol(x::Expr) = x.head === :. ? macro_symbol(x.args[end]) : nothing
macro_symbol(x::QuoteNode) = macro_symbol(x.value)
macro_symbol(x) = nothing

const LOGGING_MACROS = Symbol.(("@warn", "@info", "@debug", "@error", "@logmsg"))

"""
    iscold(e) -> Bool

An expression that runs only when the call is about to fail or to log: string interpolation, a
`throw`, an `error`, the construction of an exception such as `DomainError(...)`, and a logging
macro. No rule reads under it. `@argcheck(cond, DomainError(maximum(D), …))` evaluates `cond` on
every call and the `DomainError` only on the failing one.
"""
function iscold(e)
    if !(e isa Expr)
        return false
    end
    if e.head === :string ||
       (e.head === :macrocall && macro_symbol(e.args[1]) in LOGGING_MACROS)
        return true
    end
    if e.head === :call
        n = callee(e.args[1])
        return n !== nothing &&
               (n in (:throw, :error, :rethrow) || endswith(String(n), "Error"))
    end
    return false
end

"""
    visit(f, e, line, inview)

Call `f(e, line, inview)` on every `Expr` under `e`, in source order, with the line of the nearest
`LineNumberNode` above it and whether a macro of `VIEW_MACROS` encloses it. A cold path is skipped
whole.
"""
function visit(f, e, line::Int, inview::Bool)
    if !(e isa Expr) || iscold(e)
        return line
    end
    if e.head === :macrocall && macro_symbol(e.args[1]) in VIEW_MACROS
        inview = true
    end
    f(e, line, inview)
    for a in e.args
        if a isa LineNumberNode
            line = a.line
        else
            line = visit(f, a, line, inview)
        end
    end
    return line
end

# --- dismissals ------------------------------------------------------------

"""
    dismissal_set(rulings) -> Set

The keys of every `[[perf_dismissal]]`. An entry with `code` covers the one Finding whose
Performance Fingerprint it names. An entry without `code` covers every Finding of its rule in its
definition, which is the key `(path, definition, metric)` of a complexity Exemption: a loop that
must allocate on every iteration takes one entry rather than one per call.
"""
function dismissal_set(rulings)
    out = Set{Tuple}()
    for e in get(rulings, "perf_dismissal", [])
        key = (e["file"], e["rule"], e["definition"])
        push!(out, haskey(e, "code") ? (key..., e["code"]) : key)
    end
    return out
end

function isdismissed(f::Finding, dismissed)
    return fingerprint(f) in dismissed || (f.file, f.rule, f.definition) in dismissed
end

# --- measurement -----------------------------------------------------------

"""
    measure(; root, files, rulings) -> NamedTuple

Scan `files`, read from `root`, and count each file's Findings per rule, before and after the
Dismissals of `rulings`. The defaults are the live checkout, every file in scope and the committed
Rulings, so the entry script calls `measure()` unchanged and a test passes a fixture tree.
"""
function measure(; root = CodeHealth.REPO_ROOT, files = CodeHealth.source_files(; root),
                 rulings = CodeHealth.read_rulings())
    dismissed = dismissal_set(rulings)
    counts = Dict{String, Dict{String, Int}}()
    findings = Dict{String, Vector{Finding}}()
    for f in files
        all_found = scan_file(f; root)
        kept = [x for x in all_found if !isdismissed(x, dismissed)]
        row = Dict{String, Int}(r => count(x -> x.rule == r, kept) for r in RULE_NAMES)
        row["raw"] = length(all_found)
        counts[f] = row
        findings[f] = kept
    end
    return (; files, counts, findings, provenance = provenance(; root))
end

function provenance(; root = CodeHealth.REPO_ROOT)
    return ["julia" => string(VERSION), "rules" => RULES_VERSION,
            "commit" => CodeHealth.git_short_commit(; root)]
end

const ROW_KEYS = (RULE_NAMES..., "raw")

row(counts, f) = [k => counts[f][k] for k in ROW_KEYS]

rows(m) = Dict(f => Dict(row(m.counts, f)) for f in m.files)

recorded_rows(recorded) = get(recorded, "file", Dict{String, Any}())

# --- render ----------------------------------------------------------------

function dismissal_repairs(rulings, files)
    live = Set(files)
    dead = [e for e in get(rulings, "perf_dismissal", []) if !(e["file"] in live)]
    if isempty(dead)
        return nothing
    end
    println("NOTE: ", length(dead), " performance Dismissals still name a dead path.")
    for e in dead
        println("  ", e["file"], "  ", e["rule"], ": ",
                get(e, "code", "every Finding of " * e["definition"]))
    end
    println("Edit code_health/rulings.toml, then re-run.")
    return nothing
end

reviewed_total(row) = sum(row[r] for r in RULE_NAMES)

function render(m, recorded, accept_rise::Bool)
    rulings = CodeHealth.read_rulings()
    dismissal_repairs(rulings, m.files)
    measured = rows(m)
    rec = recorded_rows(recorded)
    if !isempty(rec)
        missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
        equal(a, b) = all(k -> get(a, k, 0) == b[k], ROW_KEYS)
        pairs, _, added = CodeHealth.pair_renames(dead_rows, missing_rows, rec, measured,
                                                  equal)
        for (dead, new) in pairs
            println("Paired the row of ", dead, " with ", new, ", which measures the same.")
        end
        th = CodeHealth.thresholds(rulings)
        refusals = String[]
        for f in added
            n = reviewed_total(measured[f])
            if n >= th["perf_reviewed"]
                push!(refusals,
                      "ERROR: $f enters with $n performance Finding(s), and an added file must " *
                      "enter at 0.\n       Fix them, or add a [[perf_dismissal]] citing an " *
                      "approved Rationale. `perf.jl scan $f` lists them.")
            end
        end
        isempty(refusals) || throw(CodeHealth.RefreshRefused(join(refusals, "\n")))
    end
    rs = CodeHealth.rises(rec, measured, BINDING)
    if !isempty(rs) && !accept_rise
        CodeHealth.refuse_rise(NAME, rs)
    end
    io = IOBuffer()
    println(io, "# Generated by code_health/perf.jl. Do not edit by hand.")
    println(io,
            "# One row per file in scope. Each rule column counts the Findings that no Dismissal")
    println(io,
            "# covers, and binds. `raw` counts before the Dismissals and is context. ADR 0175.")
    println(io)
    CodeHealth.emit_provenance(io, m.provenance)
    println(io)
    CodeHealth.emit_section(io, "file", (f => row(m.counts, f) for f in m.files))
    return String(take!(io))
end

# --- verify ----------------------------------------------------------------

function verify(m, recorded)
    rulings = CodeHealth.read_rulings()
    failures = String[]
    bad_prov = CodeHealth.provenance_failures(get(recorded, "provenance", Dict()),
                                              Dict(m.provenance))
    if !isempty(bad_prov)
        push!(failures, CodeHealth.provenance_message(bad_prov))
        return failures, false
    end
    for line in CodeHealth.check_rationale_citations(rulings)
        push!(failures, "ERROR: " * line)
    end
    rec = recorded_rows(recorded)
    missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
    for f in missing_rows
        CodeHealth.annotate(f,
                            "no row in $NAME. The baseline must name every file in scope.")
    end
    for f in dead_rows
        println("  $NAME names $f, which no longer exists.")
    end
    if !(isempty(missing_rows) && isempty(dead_rows))
        push!(failures,
              "The baseline's file set and the tree's file set differ: " *
              "$(length(missing_rows)) file(s) with no row, $(length(dead_rows)) row(s) naming no file.")
    end
    rs = CodeHealth.rises(rec, rows(m), BINDING)
    for r in rs
        CodeHealth.annotate(r.key,
                            "$(r.metric) Findings rose from $(r.old) to $(r.new). " *
                            "`perf.jl scan $(r.key) --rule $(r.metric)` lists them.")
    end
    if !isempty(rs)
        push!(failures, "The performance ratchet tripped on $(length(rs)) number(s).")
        CodeHealth.step_summary("### Performance ratchet\n\n" * CodeHealth.rise_table(rs))
    end
    if !(isempty(failures))
        push!(failures, CodeHealth.routes(; dismissal = true))
    end
    return failures, true
end

function publish(m)
    io = IOBuffer()
    println(io, "### Performance gate\n")
    println(io, "| rule | Findings |")
    println(io, "| --- | --- |")
    for r in RULE_NAMES
        println(io, "| ", r, " | ", sum(c -> c[r], values(m.counts); init = 0), " |")
    end
    total = sum(reviewed_total, values(m.counts); init = 0)
    println("Green. ", length(m.files), " files scanned, ", total,
            " Findings under the baseline.")
    CodeHealth.step_summary(String(take!(io)))
    return nothing
end

# --- scan ------------------------------------------------------------------

"""
    scan(args) -> Int

The review. Print every Finding no Dismissal covers, grouped by rule, for the files named in `args`
or for every file in scope, and write nothing. `--rule NAME` keeps one rule.
"""
function scan(args)
    rule = nothing
    files = String[]
    i = 1
    while i <= length(args)
        if args[i] == "--rule" && i < length(args)
            rule = args[i + 1]
            if !(rule in RULE_NAMES)
                error("unknown rule $(repr(rule)); the rules are $(join(RULE_NAMES, ", "))")
            end
            i += 2
        else
            push!(files, args[i])
            i += 1
        end
    end
    m = isempty(files) ? measure() : measure(; files)
    found = reduce(vcat, (m.findings[f] for f in m.files); init = Finding[])
    for (r, statement) in RULES
        if !(rule === nothing || rule == r)
            continue
        end
        hits = sort!(filter(x -> x.rule == r, found); by = x -> (x.file, x.line))
        if isempty(hits)
            continue
        end
        println("## ", r, " (", length(hits), ")\n", statement, "\n")
        for x in hits
            println(x.file, ":", x.line, "  [", x.definition, "]  ", x.code)
            println("    ", x.hint)
        end
        println()
    end
    return 0
end

# The scheduled job of ADR 0078 reuses a measuring script's `measure`, so the command line runs only
# when this file is the program.
if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "scan"
        exit(scan(ARGS[2:end]))
    end
    exit(CodeHealth.run_script(ARGS; name = NAME, measure = measure, verify = verify,
                               render = render, publish = publish))
end
