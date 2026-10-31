# ---------------------------------------------------------------------------
# @forward_properties — standalone property-forwarding macro (ADR 0013)
# ---------------------------------------------------------------------------

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check one intermediate node of a nested path of [`@forward_properties`](@ref).

Return `v` with no change when it is not `nothing`. Otherwise, throw a
[`PropertyPathError`](@ref) that names the receiver type `T`, the full declared path
`pathstr`, and the node `nodestr` that resolved to `nothing`. The generated walk of a
locator with two or more hops calls it once for each intermediate hop.

# Algorithm

 1. `v` is `nothing`: throw a [`PropertyPathError`](@ref) whose message names `pathstr`, the type `T` and the node `nodestr`.
 2. `v` is any other value: return `v`.

The check runs on the **intermediate** hops only. A path whose last hop gives `nothing` returns that `nothing`, and raises no error. This is by design. An absent leaf is a value, but an absent intermediate node is a path that the walk cannot follow.

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

Change a locator of [`@forward_properties`](@ref) into its path of field symbols.

A bare identifier `a` becomes `[:a]`. A dotted expression `a.b.c` becomes
`[:a, :b, :c]`. Any other expression raises an error.

# Algorithm

 1. `expr` is a `Symbol`: return the one-element vector that holds it.
 2. `expr` is an `Expr` with head `:.` and two arguments:
     1. Read `leaf`, its second argument, and unwrap a `QuoteNode` to its value.
     2. `leaf` is not a `Symbol`: raise an error that names the leaf.
     3. Call this function again on the first argument, and append `leaf` to the result.
 3. `expr` is any other expression: raise an error that names the expression.

The recursion of step 2.3 lets the path have any depth. `a.b.c` parses as `(a.b).c`, so the walk goes down to the bare name and builds the path again from the left.

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

Build the expression that walks a `path` of [`@forward_properties`](@ref) on the
receiver `x`, and returns the value at the end of the path. `path` is a vector of field
symbols.

A path with one hop is one `getfield`. A path with two or more hops goes one hop at a
time. It checks every intermediate node with [`forward_nonnothing`](@ref), which takes
the receiver type `struct_name`. A `nothing` node then throws a
[`PropertyPathError`](@ref) that names the path. When `broadcast` is `true`, the last
hop maps over the value before it if that value is an `AbstractVector`. This is the case
of a solution that is a scalar or a vector. Otherwise, the last hop is a plain access.

# Algorithm

 1. The path holds one name: return `getfield(x, name)` and stop. The walk uses `getfield` and not `getproperty`, so the generated `Base.getproperty` never calls itself.
 2. Build `pathstr`, the whole path joined by dots, for the error message.
 3. Start `stmts` with `__v = getfield(x, first_name)`.
 4. For each further hop `k` of the path:
     1. Push `__v = forward_nonnothing(__v, struct_name, pathstr, nodestr)`, where `nodestr` names the part of the path walked so far.
     2. `k` is the last hop and `broadcast` is `true`: push an assignment that reads the leaf with `getproperty.` when `__v` is an `AbstractVector`, and with `getproperty` otherwise.
     3. Otherwise: push `__v = getproperty(__v, leaf)`.
 5. Push `__v` as the value of the block.
 6. Return the statements wrapped in a `let` block, so `__v` never escapes into the caller.

Step 4.1 runs before **every** hop after the first, so the check reads each intermediate node one time and never reads the leaf. Only the leaf of step 4.2 broadcasts. So an intermediate vector raises an error at the next hop, and the walk does not map over it.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild `x` with the fields named in `patch` replaced, for a type that [`@forward_properties`](@ref) decorates.

This function is the body of the `ConstructionBase.setproperties` method that the macro makes. `Accessors.set` and `Accessors.@set` call that method to build again every struct on a lens path. The default method refuses a type whose `propertynames` differ from its `fieldnames`. It also reads every field through `getproperty`, so with a `swap(L, M)` rule it writes `M` into an unset `L`. This function reads the fields with `getfield`, and it patches only a field. A forwarded or computed property is a view of a nested value, with no storage of its own. The function refuses a patch that names one, and `source(sym)` gives the path to set.

# Algorithm

 1. For each key `sym` of `patch` that is not a field of `T`:
     1. `source(sym)` gives a path: refuse it, and name that path as the one to set.
     2. `sym` is another property of `x`: refuse it as a computed value, and name the fields.
     3. Otherwise: refuse it as a name that `T` does not have, and name the fields.
 2. Read every field of `x` with `getfield` into a `NamedTuple`, in declaration order.
 3. Merge `patch` over it, and call `ctor`, the keyword constructor of `T`, with the result. Every check of the constructor runs on the new value.

The macro passes `ctor` as the name of `T` without its type parameters. So step 3 calls the ordinary keyword constructor, and it needs no reflection to find it. A patch that changes a type parameter, such as a value for an unset `L`, gives the specialisation that the constructor selects. The keyword constructor must take every field of `T` by its field name.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_prior`](@ref)
"""
function forward_setproperties(x::T, patch::NamedTuple, source, ctor) where {T}
    fnames = fieldnames(T)
    for sym in keys(patch)
        if sym in fnames
            continue
        end
        src = source(sym)
        if !isnothing(src)
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is a property that `$(nameof(T))` forwards from `$(src)`, and it has no storage of its own. Set `$(src)` of the `$(nameof(T))` instead."))
        elseif sym in propertynames(x)
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is a property that `$(nameof(T))` computes from its fields, and it has no storage of its own. Set one of the fields $(fnames) instead."))
        else
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is neither a field nor a property of it. Its fields are $(fnames)."))
        end
    end
    vals = NamedTuple{fnames}(getfield.((x,), fnames))
    return ctor(; merge(vals, patch)...)
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

Make the `Base.getproperty` and `Base.propertynames` methods of the type `T` from a
block of declared forwarding rules. The forwarding of properties is then in one
declaration, not in a hand-written `getproperty` body. The macro also makes the
`ConstructionBase.setproperties` method that `Accessors.set` needs to build `T` again on
a lens path. The default method refuses a type whose `propertynames` differ from its
`fieldnames`. `T` can be a bare type name or a parametric or `UnionAll` signature, such
as `Foo{<:Any, Nothing, <:Any}`, so a `swap` can apply to one value of a type parameter.

Write all names as **bare identifiers**. Every rule names its source through a
**locator**. A locator is a bare name `a`, which is the field `a` of the receiver, or a
dotted path `a.b.c` of any depth, which is the path `obj.a.b.c` from the receiver. A
deeper path has more dots. A path with two or more hops checks each intermediate node,
and it throws a [`PropertyPathError`](@ref) that names the path when a node is `nothing`.

`forward`, `alias` and `compute` only *add new virtual names*, so they resolve **after**
the fields of the receiver. `swap` *replaces the value of an existing field*, so it
resolves **before** the field check.

# Rules

  - `forward(loc)`: forward *all* properties of the value at `loc`. A name `sym` with
    `sym in propertynames(value)` gives `getproperty(value, sym)`.
  - `forward(loc, names...)`: forward only the named properties of the value at `loc`.
  - `alias(exposed, loc)`: give the value at `loc` the new name `exposed`.
  - `compute(exposed, loc; broadcast)`: give `exposed` through a dotted locator with
    two or more hops. `broadcast` maps the last hop over a vector value before it.
  - `compute(exposed, fn)`: give `exposed` as `fn(obj)`. `fn` must be an anonymous
    function, because the macro cannot tell any other function from a dotted path.
  - `swap(field, loc)` and `swap(field, fn)`: replace the value of an *existing* field
    with the value at `loc`, or with `fn(obj)`. `loc` is a bare name, such as in
    `swap(L, M)`, or a dotted path. This rule runs before the own-field check, and it
    is the only rule that can name a real field. It usually applies to one value of a
    type parameter. For example, `swap(L, M)` applies to a `Regression` whose `L`
    parameter is `Nothing`. The locator form reads through `getfield`, so it does not
    recurse. In the **function form, the body must read the swapped field with
    `getfield(obj, :field)`, never with `obj.field`**. A dot access of the swapped field
    calls `getproperty` again, and the recursion raises a `StackOverflowError`. The body
    can use a dot access for every other field.

# Algorithm

 1. `block` is not a `begin … end` block: raise an error.
 2. Make four empty vectors: `swap_branches`, `getprop_branches`, `source_branches` and `propname_contribs`.
 3. For each rule of the block, except a `LineNumberNode`:
     1. The rule is not a call: raise an error that names it.
     2. Read `marker`, the rule name, and `args`, its arguments. When the first argument is a `:parameters` node, read the `broadcast` option from it and drop it from `args`. Any other option raises an error.
     3. `marker` is `forward`: flatten the locator with [`forward_flatten_path`](@ref), and build `walk` with [`forward_walk_expr`](@ref). With no more arguments, push a branch that returns `getproperty(walk, sym)` when `sym` is in `propertynames(walk)`, and add all of those names. With more arguments, check that each is a bare identifier. Then push a branch that matches `sym` against that set of names, and add those names. In both cases, push onto `source_branches` a branch that gives the path `loc.sym` for the same names.
     4. `marker` is `alias`: check the exposed name, and build `walk` from the locator. Push a branch that matches the exposed name and returns `walk`. Push a source branch that gives the locator path, and add the name.
     5. `marker` is `compute`: check the exposed name. An anonymous-function source pushes a branch that returns `fn(x)`, and `broadcast` with that form raises an error. A dotted source builds `walk` with the `broadcast` flag, pushes the matching branch, and pushes a source branch that gives the dotted path. Any other source raises an error. Add the exposed name.
     6. `marker` is `swap`: do as for `compute`, but a bare name is also a valid source, and the branch goes onto `swap_branches` and not onto `getprop_branches`.
     7. `marker` is any other name: raise an error that names it.
 4. Build `Base.getproperty(x::T, sym::Symbol)` in this order: the `swap` branches, then the own-field check, which returns `getfield(x, sym)`, then the other branches in declaration order. The last statement is `getfield(x, sym)`, which raises the standard error for an absent field.
 5. Build `Base.propertynames(x::T)` from `fieldnames(T)` and then every added name, and return the unique names as a tuple.
 6. Build `ConstructionBase.setproperties(x::T, patch::NamedTuple)`, which calls [`forward_setproperties`](@ref) with two more arguments. The first is a function of `sym` that runs the `source_branches` and gives `nothing` when no branch matches. The second is the name of `T` without its type parameters, which is its keyword constructor.
 7. Return the three definitions in one escaped block.

Step 4 gives the two orders of the first paragraph. A `swap` runs **before** the own-field check, so it replaces a real field. Every other rule runs **after** that check, so it can only add a name. In each group, the first branch that matches wins, and the branches are in the declaration order of the rules.

Step 6 gives no source for a `swap`, because a `swap` names a real field, and the patch sets it as a field. Step 6 also gives no source for the function form of `compute`, because that form has no path to set.

# Related

  - [`PropertyPathError`](@ref)
  - [`forward_setproperties`](@ref)
  - [`@propagatable`](@ref)
"""
macro forward_properties(T, block)
    if !(block isa Expr && block.head == :block)
        return error("@forward_properties: expected a `begin ... end` block of rules")
    end
    getprop_branches = Any[]
    swap_branches = Any[]
    source_branches = Any[]
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
                push!(source_branches,
                      :(sym in propertynames($walk) &&
                        return string($(join(string.(path), ".")), ".", sym)))
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
                push!(source_branches,
                      :(sym in $nameset &&
                        return string($(join(string.(path), ".")), ".", sym)))
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
            push!(source_branches,
                  :(sym === $(QuoteNode(exposed)) && return $(join(string.(path), "."))))
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
                push!(source_branches,
                      :(sym === $(QuoteNode(exposed)) && return $(join(string.(path), "."))))
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
    ctor = T isa Expr && T.head == :curly ? T.args[1] : T
    setproperties_def = quote
        function $(Accessors).setproperties(x::$T, patch::NamedTuple)
            return $(forward_setproperties)(x, patch, sym -> begin
                                                $(source_branches...)
                                                return nothing
                                            end, $ctor)
        end
    end
    return esc(quote
                   $getproperty_def
                   $propertynames_def
                   $setproperties_def
               end)
end

public @forward_properties
