# ---------------------------------------------------------------------------
# @forward_properties — standalone property-forwarding macro (ADR 0013)
# ---------------------------------------------------------------------------

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Guard one intermediate node of a [`@forward_properties`](@ref) nested path.

Return `v` unchanged when it is not `nothing`; otherwise throw a
[`PropertyPathError`](@ref) naming the receiver type `T`, the full declared path
`pathstr`, and the `nodestr` node that resolved to `nothing`. Called once per
intermediate hop in the descent generated for a depth-≥2 locator.

# Algorithm

 1. `v` is `nothing`: throw a [`PropertyPathError`](@ref) whose message names `pathstr`, the type `T` and the node `nodestr`.
 2. `v` is anything else: return `v`.

The guard runs on the **intermediate** hops only, so a path whose last hop gives `nothing` returns that `nothing` rather than raising. That is deliberate: an absent leaf is a value, and an absent intermediate is a path that cannot be walked.

# Related

  - [`@forward_properties`](@ref)
  - [`PropertyPathError`](@ref)
"""
function forward_nonnothing(v, ::Type{T}, pathstr, nodestr) where {T}
    if isnothing(v)
        throw(PropertyPathError("cannot descend path `$(pathstr)` on `$(T)`: intermediate `$(nodestr)` is `nothing`"))
    end
    return v
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Flatten a [`@forward_properties`](@ref) locator into its path of field symbols.

A bare identifier `a` becomes `[:a]`; a dotted expression `a.b.c` becomes
`[:a, :b, :c]`. Any other expression raises an error.

# Algorithm

 1. `expr` is a `Symbol`: return the one-element vector holding it.
 2. `expr` is an `Expr` with head `:.` and two arguments:
     1. Read `leaf`, its second argument, and unwrap a `QuoteNode` to its value.
     2. `leaf` is not a `Symbol`: raise an error naming the leaf.
     3. Call this function again on the first argument, and append `leaf` to the result.
 3. `expr` is anything else: raise an error naming the expression.

The recursion of step 2.3 is what makes the path any depth: `a.b.c` parses as `(a.b).c`, so the walk descends to the bare name and rebuilds the path from the left.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_walk_expr`](@ref)
"""
function forward_flatten_path(expr)
    if expr isa Symbol
        return Symbol[expr]
    elseif expr isa Expr && expr.head == :. && length(expr.args) == 2
        leaf = expr.args[2]
        leaf = leaf isa QuoteNode ? leaf.value : leaf
        if !(leaf isa Symbol)
            return error("@forward_properties: invalid locator leaf $(repr(expr.args[2]))")
        end
        return Symbol[forward_flatten_path(expr.args[1])..., leaf]
    else
        return error("@forward_properties: locator must be a bare name or a dotted path (`a.b.c`), got: $(repr(expr))")
    end
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the expression that descends a [`@forward_properties`](@ref) `path` (a
vector of field symbols) on the receiver `x`, returning the value at the path.

A depth-1 path is a single `getfield`. A depth-≥2 path descends hop by hop,
guarding every intermediate with [`forward_nonnothing`](@ref) (keyed on the
receiver type `struct_name`) so a `nothing` node throws a path-naming
[`PropertyPathError`](@ref). When `broadcast` is `true`, the final hop maps over
the penultimate value if it is an `AbstractVector` (the scalar-or-vector
solution case), otherwise it is a plain access.

# Algorithm

 1. The path holds one name: return `getfield(x, name)` and stop. `getfield` is used rather than `getproperty`, so the generated `Base.getproperty` never re-enters itself.
 2. Build `pathstr`, the whole path joined by dots, for the error message.
 3. Start `stmts` with `__v = getfield(x, first_name)`.
 4. For each further hop `k` of the path:
     1. Push `__v = forward_nonnothing(__v, struct_name, pathstr, nodestr)`, where `nodestr` names the part of the path walked so far.
     2. `k` is the last hop and `broadcast` is `true`: push an assignment that reads the leaf with `getproperty.` when `__v` is an `AbstractVector`, and with `getproperty` otherwise.
     3. Otherwise: push `__v = getproperty(__v, leaf)`.
 5. Push `__v` as the value of the block.
 6. Return the statements wrapped in a `let` block, so `__v` never escapes into the caller.

Step 4.1 runs before **every** hop after the first, so the guard covers each intermediate exactly once and never the leaf. Only the leaf of step 4.2 broadcasts, so an intermediate vector is still a path error rather than a silent map.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_nonnothing`](@ref)
"""
function forward_walk_expr(path, struct_name, broadcast::Bool)
    if length(path) == 1
        return :(getfield(x, $(QuoteNode(path[1]))))
    end
    pathstr = join(string.(path), ".")
    stmts = Any[:(__v = getfield(x, $(QuoteNode(path[1]))))]
    for k in 2:length(path)
        nodestr = join(string.(view(path, 1:(k - 1))), ".")
        push!(stmts, :(__v = $(forward_nonnothing)(__v, $struct_name, $pathstr, $nodestr)))
        leaf = QuoteNode(path[k])
        if k == length(path) && broadcast
            push!(stmts, :(__v = if isa(__v, AbstractVector)
                               getproperty.(__v, $leaf)
                           else
                               getproperty(__v, $leaf)
                           end))
        else
            push!(stmts, :(__v = getproperty(__v, $leaf)))
        end
    end
    push!(stmts, :__v)
    return Expr(:let, Expr(:block), Expr(:block, stmts...))
end

"""
    @forward_properties T begin
        forward(loc)
        forward(loc, names...)
        alias(exposed, loc)
        compute(exposed, loc; broadcast)
        compute(exposed, fn)
        swap(field, loc)
        swap(field, fn)
    end

Generate the `Base.getproperty` / `Base.propertynames` pair for type `T` from a
block of declarative forwarding rules, so the property-forwarding decision lives
in one declared surface instead of a hand-written `getproperty` body.
`T` may be a bare type name or a parametric/`UnionAll` signature
(`Foo{<:Any, Nothing, <:Any}`), so a `swap` can be specialised per type parameter.

All names are written as **bare identifiers**. Every rule names its source via a
**locator** — a bare name `a` (the field `a` of the receiver) or a dotted path
`a.b.c` (the receiver-rooted path `obj.a.b.c`, any depth). Nesting is simply more
dots; a depth-≥2 path guards each intermediate and throws a
[`PropertyPathError`](@ref) naming the path when a node is `nothing`.

`forward`/`alias`/`compute` only *add new virtual names* and so resolve **after**
the receiver's own fields; `swap` *replaces the value of an existing field* and so
resolves **before** the field check.

# Rules

  - `forward(loc)`: forward *all* properties of the value at `loc`
    (`sym in propertynames(value)` ? `getproperty(value, sym)`).
  - `forward(loc, names...)`: forward only the named subset from the value at `loc`.
  - `alias(exposed, loc)`: expose `exposed` as the value at `loc` (renaming).
  - `compute(exposed, loc; broadcast)`: expose `exposed` via a dotted locator
    (depth ≥ 2); `broadcast` maps the final hop over a vector penultimate value.
  - `compute(exposed, fn)`: expose `exposed` as `fn(obj)`; `fn` must be an
    anonymous function (a lambda), which would otherwise be ambiguous with a
    dotted path.
  - `swap(field, loc)` / `swap(field, fn)`: override an *existing* field's value
    with the value at `loc` (bare name, e.g. `swap(L, M)`, or dotted path) or with
    `fn(obj)`. Unlike the others it takes precedence over the own-field check, and
    is the only rule that may name a real field. Typically specialised on a
    parametric `T` (`swap(L, M)` on `Regression{<:Any, Nothing, <:Any}`).
    The locator form reads through `getfield` and is recursion-safe; in the
    **function form the body must read the swapped field via `getfield(obj, :field)`,
    never `obj.field`**, since dot-access on the swapped field re-enters
    `getproperty` and recurses (`StackOverflowError`). Other fields may use
    dot-access freely.

# Algorithm

 1. `block` is not a `begin … end` block: raise an error.
 2. Make three empty vectors: `swap_branches`, `getprop_branches` and `propname_contribs`.
 3. For each rule of the block, skipping a `LineNumberNode`:
     1. The rule is not a call: raise an error naming it.
     2. Read `marker`, the rule name, and `args`, its arguments. When the first argument is a `:parameters` node, read the `broadcast` option out of it and drop it from `args`. Any other option raises an error.
     3. `marker` is `forward`: flatten the locator with [`forward_flatten_path`](@ref) and build `walk` with [`forward_walk_expr`](@ref). With no further argument, push a branch that returns `getproperty(walk, sym)` when `sym` is in `propertynames(walk)`, and contribute every one of those names. With further arguments, check that each is a bare identifier, push a branch that matches `sym` against that name set, and contribute the named subset.
     4. `marker` is `alias`: check the exposed name, build `walk` from the locator, push a branch that matches the exposed name and returns `walk`, and contribute the name.
     5. `marker` is `compute`: check the exposed name. An anonymous-function source pushes a branch returning `fn(x)`, and `broadcast` with that form raises an error. A dotted source builds `walk` with the `broadcast` flag and pushes the matching branch. Any other source raises an error. Contribute the exposed name.
     6. `marker` is `swap`: as for `compute`, but a bare name is also a legal source, and the branch is pushed onto `swap_branches` rather than `getprop_branches`.
     7. `marker` is anything else: raise an error naming it.
 4. Build `Base.getproperty(x::T, sym::Symbol)` in this order: the `swap` branches; the own-field check, which returns `getfield(x, sym)`; the remaining branches in declaration order; and `getfield(x, sym)` as the fallthrough, which raises the standard error for an absent field.
 5. Build `Base.propertynames(x::T)` from `fieldnames(T)` followed by every contributed name, and return the unique names as a tuple.
 6. Return both definitions in one escaped block.

Step 4 is where the two orderings in the first paragraph come from: a `swap` runs **before** the own-field check, so it replaces a real field, and every other rule runs **after** it, so it can only add a name. Within each group the first branch that matches wins, and the order of the branches is the declaration order of the rules.

# Related

  - [`PropertyPathError`](@ref)
  - [`@propagatable`](@ref)
"""
macro forward_properties(T, block)
    if !(block isa Expr && block.head == :block)
        return error("@forward_properties: expected a `begin ... end` block of rules")
    end
    getprop_branches = Any[]
    swap_branches = Any[]
    propname_contribs = Any[]
    for stmt in block.args
        if stmt isa LineNumberNode
            continue
        end
        if !(stmt isa Expr && stmt.head == :call)
            return error("@forward_properties: each rule must be a `forward`/`alias`/`compute`/`swap` call, got: $(repr(stmt))")
        end
        marker = stmt.args[1]
        args = stmt.args[2:end]
        broadcast = false
        if !isempty(args) && args[1] isa Expr && args[1].head == :parameters
            for p in args[1].args
                if p === :broadcast
                    broadcast = true
                else
                    return error("@forward_properties: unknown option $(repr(p)) (only `broadcast` is supported)")
                end
            end
            args = args[2:end]
        end
        if marker == :forward
            if isempty(args)
                return error("@forward_properties: `forward` needs a locator")
            end
            path = forward_flatten_path(args[1])
            walk = forward_walk_expr(path, T, false)
            if length(args) == 1
                # forward all properties of the located value
                push!(getprop_branches, quote
                          let __c = $walk
                              if sym in propertynames(__c)
                                  return getproperty(__c, sym)
                              else
                                  false
                              end
                          end
                      end)
                push!(propname_contribs, Expr(:..., :(propertynames($walk))))
            else
                names = args[2:end]
                for n in names
                    if !(n isa Symbol)
                        return error("@forward_properties: `forward` names must be bare identifiers, got: $(repr(n))")
                    else
                        true
                    end
                end
                nameset = Expr(:tuple, (QuoteNode(n) for n in names)...)
                push!(getprop_branches,
                      :(sym in $nameset && return getproperty($walk, sym)))
                append!(propname_contribs, (QuoteNode(n) for n in names))
            end
        elseif marker == :alias
            if !(length(args) == 2)
                return error("@forward_properties: `alias` takes `(exposed, locator)`, got: $(repr(stmt))")
            end
            exposed = args[1]
            if !(exposed isa Symbol)
                return error("@forward_properties: `alias` exposed name must be a bare identifier, got: $(repr(exposed))")
            end
            path = forward_flatten_path(args[2])
            walk = forward_walk_expr(path, T, false)
            push!(getprop_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
            push!(propname_contribs, QuoteNode(exposed))
        elseif marker == :compute
            if !(length(args) == 2)
                return error("@forward_properties: `compute` takes `(exposed, locator|fn)`, got: $(repr(stmt))")
            end
            exposed = args[1]
            if !(exposed isa Symbol)
                return error("@forward_properties: `compute` exposed name must be a bare identifier, got: $(repr(exposed))")
            end
            src = args[2]
            if src isa Expr && src.head == :->
                if broadcast
                    return error("@forward_properties: `broadcast` does not apply to the function form of `compute`")
                end
                push!(getprop_branches,
                      :(sym === $(QuoteNode(exposed)) && return ($src)(x)))
            elseif src isa Expr && src.head == :.
                path = forward_flatten_path(src)
                walk = forward_walk_expr(path, T, broadcast)
                push!(getprop_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
            else
                return error("@forward_properties: `compute` source must be a dotted path (depth ≥ 2) or an anonymous function, got: $(repr(src))")
            end
            push!(propname_contribs, QuoteNode(exposed))
        elseif marker == :swap
            if !(length(args) == 2)
                return error("@forward_properties: `swap` takes `(field, locator|fn)`, got: $(repr(stmt))")
            end
            exposed = args[1]
            if !(exposed isa Symbol)
                return error("@forward_properties: `swap` field name must be a bare identifier, got: $(repr(exposed))")
            end
            src = args[2]
            if src isa Expr && src.head == :->
                if broadcast
                    return error("@forward_properties: `broadcast` does not apply to the function form of `swap`")
                end
                push!(swap_branches, :(sym === $(QuoteNode(exposed)) && return ($src)(x)))
            elseif (src isa Expr && src.head == :.) || src isa Symbol
                path = forward_flatten_path(src)
                walk = forward_walk_expr(path, T, broadcast)
                push!(swap_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
            else
                return error("@forward_properties: `swap` source must be a bare name, a dotted path, or an anonymous function, got: $(repr(src))")
            end
            push!(propname_contribs, QuoteNode(exposed))
        else
            return error("@forward_properties: unknown rule `$(marker)` (expected `forward`, `alias`, `compute`, or `swap`)")
        end
    end
    getproperty_def = quote
        function Base.getproperty(x::$T, sym::Symbol)
            $(swap_branches...)
            if sym in fieldnames($T)
                return getfield(x, sym)
            end
            $(getprop_branches...)
            return getfield(x, sym)
        end
    end
    propertynames_tuple = Expr(:tuple, Expr(:..., :(fieldnames($T))), propname_contribs...)
    propertynames_def = quote
        function Base.propertynames(x::$T)
            return Tuple(unique($propertynames_tuple))
        end
    end
    return esc(quote
                   $getproperty_def
                   $propertynames_def
               end)
end

public @forward_properties
