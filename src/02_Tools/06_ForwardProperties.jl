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
    # `Any` is the last resort (#1979): `stmts` becomes the arguments of an `Expr(:block, …)`.
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

Read the options of one rule of [`@forward_properties`](@ref), and return `(broadcast, settable, args)`.

`args` are the arguments of the rule `marker`. When the first of them is a `:parameters` node, it holds the options of the rule, and the returned `args` drop it. Each option applies to some rules only. `broadcast` applies to `compute` and `swap`, and `settable` applies to `forward`. An option on any other rule raises an error, so a rule never ignores an option that it was given.

# Algorithm

 1. `args` is empty, or its first argument is not a `:parameters` node: return `(false, false, args)`.
 2. For each option `p` of the node:
     1. `p` is `broadcast` and `marker` is `compute` or `swap`: set `broadcast` to `true`.
     2. `p` is `settable` and `marker` is `forward`: set `settable` to `true`.
     3. `p` is `broadcast` or `settable`: raise an error that names the rules it applies to.
     4. Otherwise: raise an error that names the two options.
 3. Return `(broadcast, settable, args[2:end])`.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_setproperties`](@ref)
"""
function forward_rule_options(marker, args)
    broadcast = false
    settable = false
    if isempty(args) || !(args[1] isa Expr && args[1].head == :parameters)
        return broadcast, settable, args
    end
    for p in args[1].args
        if p === :broadcast && marker in (:compute, :swap)
            broadcast = true
        elseif p === :settable && marker === :forward
            settable = true
        elseif p === :broadcast || p === :settable
            rules = p === :broadcast ? "`compute` and `swap`" : "`forward`"
            return error("@forward_properties: the option `$(p)` does not apply to `$(marker)`, only to $(rules)")
        else
            return error("@forward_properties: unknown option $(repr(p)) (the options are `broadcast` and `settable`)")
        end
    end
    return broadcast, settable, args[2:end]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check the arguments `args` of a `derived` rule of [`@forward_properties`](@ref), and append them to `names`.

Each argument is the name of a field that the inner constructor computes. The macro cannot see the fields of the type when it expands, so the census of `test/test_05_tools.jl` checks that each name is a field.

# Algorithm

 1. `args` is empty: raise an error.
 2. For each argument `n` of `args`:
     1. `n` is not a `Symbol`: raise an error that names it.
     2. Otherwise: push `n` onto `names`.
 3. Return `names`.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_derived`](@ref)
"""
function forward_derived_names!(names::Vector{Symbol}, args)
    if isempty(args)
        return error("@forward_properties: `derived` needs one or more field names")
    end
    for n in args
        if !(n isa Symbol)
            return error("@forward_properties: `derived` names must be bare identifiers, got: $(repr(n))")
        end
        push!(names, n)
    end
    return names
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give the source of the property `sym` of `x`, for a type that [`@forward_properties`](@ref) does not decorate: `nothing`.

The macro adds a method of this function for each type that it decorates. That method gives `nothing` when no rule names `sym`. Otherwise it gives the pair `(path, settable)` of the first rule that names `sym`, in the order in which `getproperty` reads the rules. `path` is the tuple of the field symbols of the path that holds the value, from `x`. It is `nothing` for the function form of `compute`. `settable` is `true` for a name of a settable `forward` rule.

[`forward_setproperties`](@ref) reads the pair to decide whether a patch can set `sym`. [`forward_storage_path`](@ref) reads it for each hop of a path, so it follows a chain of rules to the value that stores the property. A value of a type that the macro does not decorate stores each of its properties, as far as this function can tell, so this method ends the chain.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_setproperties`](@ref)
  - [`forward_storage_path`](@ref)
"""
forward_source(x, sym::Symbol) = nothing
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give the names of the derived fields of the type `T`, for a type that [`@forward_properties`](@ref) does not decorate: `()`.

A derived field is a field that the inner constructor of its type computes from the other fields. The keyword constructor of the type does not take it. The macro adds a method of this function for each type that it decorates. That method gives the tuple of the names of the `derived` rules of the block, and `()` when the block has none. The function takes the type and not a value, because a derived field is a property of the type, and a census can read it with no instance.

[`forward_setproperties`](@ref) and [`rebuild_struct`](@ref) leave these fields out of the call of the keyword constructor, so the constructor computes them again from the new values. [`assert_not_derived`](@ref) refuses a patch that names one of them.

# Related

  - [`@forward_properties`](@ref)
  - [`assert_not_derived`](@ref)
  - [`forward_setproperties`](@ref)
  - [`rebuild_struct`](@ref)
"""
forward_derived(::Type{T}) where {T} = ()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a `patch` that names a derived field of `x`.

A derived field has no value of its own to set, because the inner constructor computes it from the other fields. A patch that carries a value for it would give a value that the constructor does not compute. [`forward_derived`](@ref) gives the names of the derived fields of `T`, the type of `x`.

# Validation

  - No key of `patch` is in `forward_derived(T)`. Raises an `ArgumentError` that names the derived field and the fields that the keyword constructor takes.

# Returns

  - `nothing`.

# Related

  - [`forward_derived`](@ref)
  - [`forward_setproperties`](@ref)
  - [`rebuild_struct`](@ref)
"""
function assert_not_derived(x::T, patch::NamedTuple)::Nothing where {T}
    der = forward_derived(T)
    for sym in keys(patch)
        if sym in der
            kept = Tuple(s for s in fieldnames(T) if !(s in der))
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is a derived field, which the constructor of `$(nameof(T))` computes from its other fields. Set one of the fields $(kept) instead."))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Follow the `path` of field symbols from `x` to the path that stores its value, and return `(hops, owner)`.

A forward rule of [`@forward_properties`](@ref) names the path that it reads, from the receiver. A hop of that path can be a forwarded or computed property of the value before it, with no storage either. This function asks [`forward_source`](@ref) of the value at each hop, and it puts the path of the rule in place of a hop that a rule reads. `hops` is the path that results, from `x`. `owner` is `nothing` when the value at `hops` is stored or settable. Otherwise, `owner` is the type that computes the last hop with the function form of `compute`, and `hops` ends at that hop.

# Algorithm

 1. Start `hops` empty, and start `v` at `x`.
 2. For each hop `s` of `path`:
     1. `s` is a field of `v`, or [`forward_source`](@ref) of `v` gives `nothing` for it, or the rule that reads it is settable: push `s` onto `hops`.
     2. The rule that reads `s` has no path: push `s` onto `hops`, and return `(hops, typeof(v))`.
     3. Otherwise: follow the path of the rule from `v` with this function, and append its hops. Its owner is not `nothing`: return `(hops, owner)`.
     4. `s` is not the last hop, and `v` is not `nothing`: set `v` to `getproperty(v, s)`.
 3. Return `(hops, nothing)`.

Step 2.4 reads no value after the last hop, and it reads nothing through a `nothing` node. So the walk never raises the [`PropertyPathError`](@ref) of a node that is `nothing`. A hop after such a node is a field that the walk cannot see, and step 2.1 keeps it as it is.

# Related

  - [`forward_source`](@ref)
  - [`forward_setproperties`](@ref)
"""
function forward_storage_path(x, path)
    hops = Symbol[]
    v = x
    for (k, s) in enumerate(path)
        src = hasfield(typeof(v), s) ? nothing : forward_source(v, s)
        if isnothing(src) || src[2]
            push!(hops, s)
        elseif isnothing(src[1])
            push!(hops, s)
            return hops, typeof(v)
        else
            sub, owner = forward_storage_path(v, src[1])
            append!(hops, sub)
            if !isnothing(owner)
                return hops, owner
            end
        end
        if k < length(path) && !isnothing(v)
            v = getproperty(v, s)
        end
    end
    return hops, nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild `x` with the names in `patch` replaced, for a type that [`@forward_properties`](@ref) decorates.

This function is the body of the `ConstructionBase.setproperties` method that the macro makes. `Accessors.set` and `Accessors.@set` call that method to build again every struct on a lens path. The default method refuses a type whose `propertynames` differ from its `fieldnames`. It also reads every field through `getproperty`, so with a `swap(L, M)` rule it writes `M` into an unset `L`. This function reads the fields with `getfield`, and it patches a field. A forwarded or computed property is a view of a nested value, with no storage of its own. The function refuses a patch that names one, and it names the path to set. [`forward_storage_path`](@ref) follows the path of the rule to the path that stores the value, so the path that the message names does not refuse the patch too.

A `forward` rule with the `settable` option is the one exception. The function passes a name of that rule to `ctor` as a flat keyword, beside the fields. So `Accessors.set(x, @optic(_.w), v)` works on a type that forwards `w` from a nested part.

[`forward_source`](@ref) gives the pair `(path, settable)` of the rule that reads `sym`, or `nothing` when no rule names `sym`.

# Algorithm

 1. Refuse a patch that names a derived field of `T`, with [`assert_not_derived`](@ref).
 2. For each key `sym` of `patch` that is not a field of `T`:
     1. [`forward_source`](@ref) of `sym` is `nothing`: refuse it as a name that `T` does not have, and name the fields.
     2. The pair is settable: keep `sym` in the patch.
     3. The pair has no path: refuse it as a computed value, and name the fields.
     4. Otherwise: follow its path with [`forward_storage_path`](@ref). The path ends at a stored value: refuse `sym`, and name that path as the one to set.
     5. The path ends at a computed property: refuse `sym`. Name the property, the type that computes it, and the fields of that type as the ones to set.
 3. Read every field of `x` with `getfield` into a `NamedTuple`, in declaration order.
 4. Merge `patch` over it, and remove the derived fields that [`forward_derived`](@ref) names.
 5. Call `ctor`, the keyword constructor of `T`, with the result. Every check of the constructor runs on the new value, and the constructor computes each derived field again.

The macro passes `ctor` as the name of `T` without its type parameters. So step 5 calls the ordinary keyword constructor, and it needs no reflection to find it. A patch that changes a type parameter, such as a value for an unset `L`, gives the specialisation that the constructor selects.

The keyword constructor of `T` must take every field of `T` by its field name, except a field that a `derived` rule names. A type that marks a name as `settable` must also take that name as a keyword of its keyword constructor. The keyword replaces that slot of the nested part that the constructor is given, and the default of the keyword must leave that slot as it is. Step 5 passes the old nested part and the new value of the slot in one call. [`rebuild_struct`](@ref) calls the constructor in the same way, so the two give the same value.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_rule_options`](@ref)
  - [`forward_source`](@ref)
  - [`forward_storage_path`](@ref)
  - [`forward_derived`](@ref)
  - [`assert_not_derived`](@ref)
  - [`forward_prior`](@ref)
  - [`rebuild_struct`](@ref)
"""
function forward_setproperties(x::T, patch::NamedTuple, ctor) where {T}
    assert_not_derived(x, patch)
    fnames = fieldnames(T)
    for sym in keys(patch)
        if sym in fnames
            continue
        end
        src = forward_source(x, sym)
        if isnothing(src)
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is neither a field nor a property of it. Its fields are $(fnames)."))
        end
        path, settable = src
        if settable
            continue
        elseif isnothing(path)
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is a property that `$(nameof(T))` computes from its fields, and it has no storage of its own. Set one of the fields $(fnames) instead."))
        end
        pathstr = join(path, ".")
        hops, owner = forward_storage_path(x, path)
        hopstr = join(hops, ".")
        if isnothing(owner)
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is a property that `$(nameof(T))` forwards from `$(pathstr)`, and it has no storage of its own. Set `$(hopstr)` of the `$(nameof(T))` instead."))
        else
            throw(ArgumentError("`$(sym)` cannot be set on a `$(nameof(T))`: it is a property that `$(nameof(T))` forwards from `$(pathstr)`, and it has no storage of its own. `$(hopstr)` is a property that `$(nameof(owner))` computes from its fields. Set one of the fields $(fieldnames(owner)) of `$(join(view(hops, 1:(length(hops) - 1)), "."))` of the `$(nameof(T))` instead."))
        end
    end
    vals = NamedTuple{fnames}(getfield.((x,), fnames))
    return ctor(; Base.structdiff(merge(vals, patch), NamedTuple{forward_derived(T)})...)
end

"""
    @forward_properties T begin
        forward(loc; settable)
        forward(loc, names...; settable)
        alias(exposed, loc)
        compute(exposed, loc; broadcast)
        compute(exposed, fn)
        swap(field, loc)
        swap(field, fn)
        derived(fields...)
    end

Make the `Base.getproperty`, `Base.propertynames` and `Base.hasproperty` methods of the
type `T` from a block of declared forwarding rules. The forwarding of properties is then
in one declaration, not in a hand-written `getproperty` body. The `hasproperty` method
answers without the list of names, so a read through a `forward` rule allocates nothing.
The macro also makes the
`ConstructionBase.setproperties` method that `Accessors.set` needs to build `T` again on
a lens path. The default method refuses a type whose `propertynames` differ from its
`fieldnames`. The method of [`forward_source`](@ref) that the macro makes tells that
method which rule reads a name. `T` can be a bare type name or a parametric or `UnionAll`
signature, such as `Foo{<:Any, Nothing, <:Any}`, so a `swap` can apply to one value of a
type parameter.

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
    `hasproperty(value, sym)` gives `getproperty(value, sym)`.
  - `forward(loc, names...)`: forward only the named properties of the value at `loc`.
  - `forward(loc; settable)` and `forward(loc, names...; settable)`: forward as above,
    and let a patch of `Accessors.set` set the forwarded names. Without the option, the
    patch refuses a forwarded name. With it, the patch passes the name to the keyword
    constructor of `T` as a keyword, so the constructor must take each such name. The
    keyword replaces that slot of the nested part, and its default leaves the slot as it
    is. [`forward_setproperties`](@ref) states this contract.
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
  - `derived(fields...)`: mark real fields as derived. The inner constructor of `T`
    computes a derived field from the other fields, and the keyword constructor does not
    take it. The patch of `Accessors.set` leaves a derived field out of the call of the
    keyword constructor, and it refuses a patch that names one. The rule adds no name, and
    it does not change a read. [`forward_derived`](@ref) gives the names.

# Algorithm

 1. `block` is not a `begin … end` block: raise an error.
 2. Make six empty vectors: `swap_branches`, `getprop_branches`, `source_branches`, `has_branches`, `propname_contribs` and `derived_names`.
 3. For each rule of the block, except a `LineNumberNode`:
     1. The rule is not a call: raise an error that names it.
     2. Read `marker`, the rule name. Read the `broadcast` and `settable` options and `args`, the other arguments, with [`forward_rule_options`](@ref).
     3. `marker` is `forward`: flatten the locator with [`forward_flatten_path`](@ref), and build `walk` with [`forward_walk_expr`](@ref). With no more arguments, push a branch that returns `getproperty(walk, sym)` when `hasproperty(walk, sym)` is `true`, and add all of the names of `propertynames(walk)`. With more arguments, check that each is a bare identifier. Then push a branch that matches `sym` against that set of names, and add those names. In both cases, push onto `has_branches` a branch that gives `true` for the same names, and push onto `source_branches` a branch that gives the pair of the path `loc.sym`, as a tuple of symbols, and the `settable` flag for the same names.
     4. `marker` is `alias`: check the exposed name, and build `walk` from the locator. Push a branch that matches the exposed name and returns `walk`. Push a source branch that gives the pair of the locator path and `false`. Push a branch onto `has_branches` that gives `true` for the exposed name, and add the name.
     5. `marker` is `compute`: check the exposed name. An anonymous-function source pushes a branch that returns `fn(x)`, and a source branch that gives `(nothing, false)`. `broadcast` with that form raises an error. A dotted source builds `walk` with the `broadcast` flag, pushes the matching branch, and pushes a source branch that gives the pair of the dotted path and `false`. Any other source raises an error. Push a branch onto `has_branches` that gives `true` for the exposed name, and add the name.
     6. `marker` is `swap`: do as for `compute`, but a bare name is also a valid source, and the branch goes onto `swap_branches` and not onto `getprop_branches`.
     7. `marker` is `derived`: check the field names, and append them to `derived_names`, with [`forward_derived_names!`](@ref).
     8. `marker` is any other name: raise an error that names it.
 4. Build `Base.getproperty(x::T, sym::Symbol)` in this order: the `swap` branches, then the own-field check against `fieldnames(S)`, which returns `getfield(x, sym)`, then the other branches in declaration order. `S` is the name of `T` without its type parameters. The last statement is `getfield(x, sym)`, which raises the standard error for an absent field.
 5. Build `Base.propertynames(x::T)` from `fieldnames(S)` and then every added name, and return the unique names as a tuple.
 6. Build `Base.hasproperty(x::T, sym::Symbol)`. It gives `true` when `sym` is in `fieldnames(S)`, then runs the `has_branches`, and gives `false` when no branch matches.
 7. Build `forward_source(x::T, sym::Symbol)`, which runs the `source_branches` and gives `nothing` when no branch matches.
 8. Build `ConstructionBase.setproperties(x::T, patch::NamedTuple)`, which calls [`forward_setproperties`](@ref) with one more argument: the name of `T` without its type parameters, which is its keyword constructor.
 9. Build `forward_derived(::Type{<:T})`, which gives the tuple of `derived_names`. With no `derived` rule, the tuple is empty, the same value as the fallback.
10. Return the definitions in one escaped block.

Step 4 gives the two orders of the first paragraph. A `swap` runs **before** the own-field check, so it replaces a real field. Every other rule runs **after** that check, so it can only add a name. In each group, the first branch that matches wins, and the branches are in the declaration order of the rules.

Step 6 gives the same set of names as step 5, without the tuple of step 5. That tuple removes the duplicate names with `unique`, which allocates on each call. A `forward` rule asks `hasproperty` of the child, so a read through a chain of `forward` rules runs one generated `hasproperty` method for each link, and none of them allocates. A child that `@forward_properties` does not decorate answers through the method of `Base`, `sym in propertynames(x)`. Steps 4 to 6 read the field names of `S`, not of `T`. A signature such as `Foo{<:Any, Nothing, <:Any}` is an expression that builds a new `UnionAll` on each call, and that construction allocates. `S` is a constant with the same field names, so the compiler folds the field check.

Step 7 gives no source for a `swap`, because a `swap` names a real field, and the patch sets it as a field. Every other rule pushes a source branch, in the same order as its `getproperty` branch. So the rule that reads a name also decides whether a patch can set it. A name that a `settable` rule forwards, but that an earlier rule reads, stays refused. Step 7 makes a method and not a closure inside step 8, so [`forward_storage_path`](@ref) can ask the source of a name of a nested value, and the refusal of a patch names the path that stores the value.

# Related

  - [`PropertyPathError`](@ref)
  - [`forward_rule_options`](@ref)
  - [`forward_setproperties`](@ref)
  - [`forward_source`](@ref)
  - [`forward_derived`](@ref)
  - [`@propagatable`](@ref)
"""
macro forward_properties(T, block)
    if !(block isa Expr && block.head == :block)
        return error("@forward_properties: expected a `begin ... end` block of rules")
    end
    # `Any` is the last resort (#1979): each list is spliced into the arguments of an `Expr`.
    getprop_branches = Any[]
    swap_branches = Any[]
    source_branches = Any[]
    has_branches = Any[]
    propname_contribs = Any[]
    derived_names = Symbol[]
    for stmt in block.args
        if stmt isa LineNumberNode
            continue
        end
        if !(stmt isa Expr && stmt.head == :call)
            return error("@forward_properties: each rule must be a `forward`/`alias`/`compute`/`swap`/`derived` call, got: $(repr(stmt))")
        end
        marker = stmt.args[1]
        broadcast, settable, args = forward_rule_options(marker, @view(stmt.args[2:end]))
        if marker == :forward
            if isempty(args)
                return error("@forward_properties: `forward` needs a locator")
            end
            path = forward_flatten_path(args[1])
            walk = forward_walk_expr(path, T, false)
            if length(args) == 1
                # Forward all properties of the located value. The test is `hasproperty`,
                # not `sym in propertynames(…)`: the `propertynames` of a decorated child
                # builds its unique names on each call, and allocates on each read.
                push!(getprop_branches, quote
                          let __c = $walk
                              if hasproperty(__c, sym)
                                  return getproperty(__c, sym)
                              else
                                  false
                              end
                          end
                      end)
                push!(has_branches, :(hasproperty($walk, sym) && return true))
                push!(propname_contribs, Expr(:..., :(propertynames($walk))))
                push!(source_branches,
                      :(hasproperty($walk, sym) &&
                        return ($(Expr(:tuple, QuoteNode.(path)..., :sym)), $settable)))
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
                push!(has_branches, :(sym in $nameset && return true))
                append!(propname_contribs, (QuoteNode(n) for n in names))
                push!(source_branches,
                      :(sym in $nameset &&
                        return ($(Expr(:tuple, QuoteNode.(path)..., :sym)), $settable)))
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
                  :(sym === $(QuoteNode(exposed)) &&
                    return ($(Expr(:tuple, QuoteNode.(path)...)), false)))
            push!(has_branches, :(sym === $(QuoteNode(exposed)) && return true))
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
                push!(source_branches,
                      :(sym === $(QuoteNode(exposed)) && return (nothing, false)))
            elseif src isa Expr && src.head == :.
                path = forward_flatten_path(src)
                walk = forward_walk_expr(path, T, broadcast)
                push!(getprop_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
                push!(source_branches,
                      :(sym === $(QuoteNode(exposed)) &&
                        return ($(Expr(:tuple, QuoteNode.(path)...)), false)))
            else
                return error("@forward_properties: `compute` source must be a dotted path (depth ≥ 2) or an anonymous function, got: $(repr(src))")
            end
            push!(has_branches, :(sym === $(QuoteNode(exposed)) && return true))
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
            push!(has_branches, :(sym === $(QuoteNode(exposed)) && return true))
            push!(propname_contribs, QuoteNode(exposed))
        elseif marker == :derived
            forward_derived_names!(derived_names, args)
        else
            return error("@forward_properties: unknown rule `$(marker)` (expected `forward`, `alias`, `compute`, `swap`, or `derived`)")
        end
    end
    # The name of `T` without its type parameters: the keyword constructor, and the type whose
    # `fieldnames` the methods read. A signature such as `Regression{<:Any, Nothing, …}` is an
    # expression that builds a new `UnionAll` on each call, and allocates. The bare name is a
    # constant, with the same field names, so the compiler and JET fold the field check.
    ctor = T isa Expr && T.head == :curly ? T.args[1] : T
    getproperty_def = quote
        function Base.getproperty(x::$T, sym::Symbol)
            $(swap_branches...)
            if sym in fieldnames($ctor)
                return getfield(x, sym)
            end
            $(getprop_branches...)
            return getfield(x, sym)
        end
    end
    propertynames_tuple = Expr(:tuple, Expr(:..., :(fieldnames($ctor))),
                               propname_contribs...)
    propertynames_def = quote
        function Base.propertynames(x::$T)
            return Tuple(unique($propertynames_tuple))
        end
    end
    hasproperty_def = quote
        function Base.hasproperty(x::$T, sym::Symbol)
            if sym in fieldnames($ctor)
                return true
            end
            $(has_branches...)
            return false
        end
    end
    source_def = quote
        function $(@__MODULE__).forward_source(x::$T, sym::Symbol)
            $(source_branches...)
            return nothing
        end
    end
    setproperties_def = quote
        function $(Accessors).setproperties(x::$T, patch::NamedTuple)
            return $(forward_setproperties)(x, patch, $ctor)
        end
    end
    derived_def = quote
        function $(@__MODULE__).forward_derived(::Type{<:$T})
            return $(Expr(:tuple, QuoteNode.(derived_names)...))
        end
    end
    return esc(quote
                   $getproperty_def
                   $propertynames_def
                   $hasproperty_def
                   $source_def
                   $setproperties_def
                   $derived_def
               end)
end

public @forward_properties
