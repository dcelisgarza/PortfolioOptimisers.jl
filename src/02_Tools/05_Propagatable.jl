# ---------------------------------------------------------------------------
# @propagatable — struct-definition macro for factory propagation
# ---------------------------------------------------------------------------

# --- private AST helpers ----------------------------------------------------

"""
    PROP_TAG_NAMES

The propagation tag set of [`@propagatable`](@ref), as data.

One entry per field tag. The recognition layer is derived from this tuple rather than
spelled out per tag: the macro names ([`PROP_TAG_MACRO_NAMES`](@ref)), the lookup
([`prop_tag`](@ref)), the gate ([`is_prop_tag_call`](@ref)), the peeler
([`peel_prop_tags`](@ref)) and the parser ([`propagatable_parse_body`](@ref)) all read it.
A new propagation channel is one row here, one branch in [`prop_tag_expr`](@ref), one entry
in [`PROP_TAG_CHANNELS`](@ref) and one stub macro; [`check_prop_tag_macros`](@ref) refuses
to load the module when a row lacks any of the three.

# Related

  - [`PROP_TAG_MACRO_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`check_prop_tag_macros`](@ref)
  - [`@propagatable`](@ref)
"""
const PROP_TAG_NAMES = (:fprop, :vprop, :pprop, :cprop, :wprop)

"""
    PROP_TAG_MACRO_NAMES

The macro name of every tag of [`PROP_TAG_NAMES`](@ref), in the same order.

Derived from the tag names, so a row of the table carries no second spelling.
[`prop_tag`](@ref) matches a `:macrocall` head against this tuple.

# Algorithm

 1. Map each tag of [`PROP_TAG_NAMES`](@ref) to `Symbol("@", tag)`, which is the name Julia gives that tag's macro.

The map preserves the order, so the ``k``-th entry here is the macro name of the ``k``-th tag there. [`prop_tag`](@ref) and [`check_prop_tag_macros`](@ref) both walk the two tuples together, and that pairing is what the shared order guarantees.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`prop_tag`](@ref)
"""
const PROP_TAG_MACRO_NAMES = Symbol.("@", PROP_TAG_NAMES)

"""
    PROP_TAG_CHANNELS

The propagation channels of [`@propagatable`](@ref), with their tag precedence as data.

One entry per generated method. Each entry has two tuples of tags:

  - `gate`: the tags that make [`@propagatable`](@ref) emit the method at all.
  - `precedence`: the order in which a field's tags are consulted. The first tag of this
    tuple that the field carries decides the field's transform; the rest are ignored for
    that channel.

The `factory` channel prefers `@fprop` over `@wprop`. The `prior` channel prefers `@pprop`,
then `@cprop`, then `@wprop`, then `@fprop`, so `@pprop` wins over `@fprop` on one field.
The precedence used to live in two hand-written `if`/`elseif` chains that no
comment linked; it is now read by [`prop_channel_pairs`](@ref) for every channel.

**A tag means what its channel says it means.** The `obs` channel reads the same `@wprop`
and `@fprop` tags as `factory` and gives them different transforms: `factory` **replaces** a
`@wprop` field with an incoming [`ObsWeights`](@ref) value, while `obs` **indexes** the value
already there. This is why [`prop_tag_expr`](@ref) takes the channel as well as the tag. It
also means a weights field opts into [`obs_weights_view`](@ref) by carrying `@wprop`, with no
second tag to write and no second tag to forget.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`prop_channel_active`](@ref)
  - [`@propagatable`](@ref)
"""
const PROP_TAG_CHANNELS = (factory = (gate = (:fprop, :wprop),
                                      precedence = (:fprop, :wprop)),
                           view = (gate = (:vprop,), precedence = (:vprop,)),
                           prior = (gate = (:pprop, :cprop),
                                    precedence = (:pprop, :cprop, :wprop, :fprop)),
                           obs = (gate = (:wprop,), precedence = (:wprop, :fprop)))

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the tag of [`PROP_TAG_NAMES`](@ref) that `x` names, or `nothing`.

`x` is the head of a `:macrocall` node. It is a bare `Symbol` in a struct body written by
hand, and a `GlobalRef` once another macro has expanded around it, so both spellings
resolve. A name outside [`PROP_TAG_MACRO_NAMES`](@ref) gives `nothing`, which is how the
callers tell a tag from any other macro; no tag falls through to another tag.

# Algorithm

 1. Read `name` from `x`. A `GlobalRef` gives its `name` field, a `Symbol` gives itself, and any other value returns `nothing` at once.
 2. Walk [`PROP_TAG_NAMES`](@ref) and [`PROP_TAG_MACRO_NAMES`](@ref) together. Return the tag whose macro name is identical to `name`.
 3. No macro name matches: return `nothing`.

The comparison is `===` on a `Symbol`, so a macro whose name merely resembles a tag never matches.

# Arguments

  - `x`: The first argument of a `:macrocall` expression.

# Returns

  - `tag::Symbol`: The tag name, without the `@`.
  - `nothing`: If `x` names no tag.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`is_prop_tag_call`](@ref)
  - [`peel_prop_tags`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_tag(x)
    name = if x isa GlobalRef
        x.name
    elseif x isa Symbol
        x
    else
        return nothing
    end
    for (tag, macro_name) in zip(PROP_TAG_NAMES, PROP_TAG_MACRO_NAMES)
        if name === macro_name
            return tag
        end
    end
    return nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a macro call to any tag of [`PROP_TAG_NAMES`](@ref).

Used by [`peel_prop_tags`](@ref) and [`propagatable_parse_body`](@ref) to detect tagged
fields in a struct body.

# Algorithm

 1. Return `true` when all three hold: `x` is an `Expr`, its head is `:macrocall`, and [`prop_tag`](@ref) of its first argument is not `nothing`.
 2. Return `false` otherwise.

# Arguments

  - `x`: Any expression appearing in a struct body.

# Related

  - [`prop_tag`](@ref)
  - [`peel_prop_tags`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_prop_tag_call(x)
    return x isa Expr && x.head == :macrocall && prop_tag(x.args[1]) !== nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Peel any stack of tag macrocalls off a field expression, recording which tags were present.

Tags may be stacked in either order (`@pprop @fprop field`), which parses as nested
`:macrocall` nodes; this unwraps them all and returns the bare field expression. Each tag
is looked up with [`prop_tag`](@ref), so an untagged macro stops the peel and no tag is
reached by falling through the others.

# Algorithm

 1. Make `tags`, an empty `Set{Symbol}`.
 2. While [`is_prop_tag_call`](@ref) of `expr` holds, push [`prop_tag`](@ref) of its first argument onto `tags`, and replace `expr` with its **last** argument, which is the expression the tag wraps.
 3. Return `tags` and the peeled `expr`.

`tags` is a set, so a tag written twice on one field is recorded once. The loop stops at the first node that is not a tag call, so a non-tag macro between two tags hides the tags below it.

# Arguments

  - `expr`: A field expression, with or without tag macrocalls around it.

# Returns

  - `tags::Set{Symbol}`: The tags of [`PROP_TAG_NAMES`](@ref) that `expr` carries.
  - `stripped`: The field expression with all tags removed.

# Related

  - [`prop_tag`](@ref)
  - [`is_prop_tag_call`](@ref)
  - [`propagatable_parse_body`](@ref)
"""
function peel_prop_tags(expr)
    tags = Set{Symbol}()
    while is_prop_tag_call(expr)
        push!(tags, prop_tag(expr.args[1]))
        expr = expr.args[end]
    end
    return tags, expr
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the expression that a tag substitutes for one field, inside a generated method.

This is the `name → field transform` half of the tag table: one branch per tag of
[`PROP_TAG_NAMES`](@ref), read by [`prop_channel_pairs`](@ref) for every channel. A tag of
the table with no branch here errors, so it cannot silently take another tag's transform.

# Algorithm

The channel is read first, and then the tag, because one tag has two transforms.

 1. `channel` is `:obs`:
     1. `tag` is `:wprop`: return `nothing_scalar_array_getindex(xf, thread...)`. The field **is** the weights, so it is indexed to the selected observations. Indexing keeps the `AbstractWeights` subtype, which a view would not.
     2. `tag` is `:fprop`: return `obs_weights_view(xf, thread...)`. The field is a composed child, so the verb recurses into it.
 2. `channel` is any other channel:
     1. `tag` is `:fprop`: return `factory_child(xf, thread..., args...; kwargs...)`.
     2. `tag` is `:vprop`: return `view_child(xf, thread..., args...)`, which calls [`port_opt_view`](@ref) on every value but a precomputed optimisation result. This channel forwards no keywords.
     3. `tag` is `:pprop`: return `sel(xf, getproperty(pr, fname))`, which is why the field name is an argument. The prior result supplies the property of the **same name**.
     4. `tag` is `:cprop`: return `sel(xf, _ctx(args...))`, which reads the context out of the threaded arguments rather than the prior.
     5. `tag` is `:wprop`: return `_wprop(xf, args...; kwargs...)`, which **replaces** the field with an incoming [`ObsWeights`](@ref).
 3. No branch matched: raise an error naming the tag and the channel, and ask for a branch here.

Steps 1.1 and 2.5 are the same tag with two transforms, so the channel decides what `@wprop` means. Every emitted name is qualified against `mod`, because the expansion is escaped into the caller's module.

# Arguments

  - `tag::Symbol`: A tag of [`PROP_TAG_NAMES`](@ref).
  - `fname::Symbol`: The field name, needed by `@pprop` to name the prior property.
  - `xf`: The expression that reads the field off the incoming struct.
  - `mod::Module`: The module that defines [`@propagatable`](@ref). Every emitted name is
    qualified against it, because the expansion is escaped into the caller.
  - `thread`: Extra positional arguments the channel threads before `args...`.

# Returns

  - `expr::Expr`: The value of the field in the generated constructor call.

# Related

  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`check_prop_tag_macros`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_tag_expr(channel::Symbol, tag::Symbol, fname::Symbol, xf, mod::Module, thread)
    if channel === :obs
        # The observation channel reads the same two tags as `factory` and gives them
        # different transforms. `@wprop` holds the weights themselves, so the field is
        # INDEXED rather than replaced; `@fprop` holds a composed child, so the verb
        # recurses into it. Indexing preserves the `AbstractWeights` subtype, which a
        # `view` would not.
        if tag === :wprop
            return :($mod.nothing_scalar_array_getindex($xf, $(thread...)))
        end
        tag === :fprop && return :($mod.obs_weights_view($xf, $(thread...)))
    else
        if tag === :fprop
            return :($mod.factory_child($xf, $(thread...), args...; kwargs...))
        end
        if tag === :vprop
            return :($mod.view_child($xf, $(thread...), args...))
        end
        if tag === :pprop
            return :($mod.sel($xf, getproperty(pr, $(QuoteNode(fname)))))
        end
        if tag === :cprop
            return :($mod.sel($xf, $mod._ctx(args...)))
        end
        tag === :wprop && return :($mod._wprop($xf, args...; kwargs...))
    end
    return error("@propagatable: tag `@$(tag)` has no field transform in channel " *
                 "`:$(channel)`. Add a branch to `prop_tag_expr`.")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if a channel of [`PROP_TAG_CHANNELS`](@ref) must emit a method.

A channel is active when at least one field carries a tag of the channel's `gate`.

# Algorithm

 1. Read the `gate` tuple of the channel from [`PROP_TAG_CHANNELS`](@ref).
 2. Return `true` when at least one tag of `gate` has a non-empty entry in `tagged`, and `false` otherwise.

The `precedence` tuple is **not** read here, so a tag that a channel consults but does not gate on never makes that channel emit a method on its own. The `obs` channel gates on `@wprop` alone and consults `@fprop`, so a type carrying `@fprop` and no `@wprop` gains no `obs` method.

# Arguments

  - `channel::Symbol`: A channel name of [`PROP_TAG_CHANNELS`](@ref).
  - `tagged::AbstractDict`: Tag name to the field names that carry it, from
    [`propagatable_parse_body`](@ref).

# Related

  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_channel_active(channel::Symbol, tagged::AbstractDict)
    return any(tag -> !isempty(tagged[tag]), getproperty(PROP_TAG_CHANNELS, channel).gate)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the keyword pairs of the constructor call that one channel generates.

Every declared field gets one pair, in declaration order. The field's tags are consulted in
the channel's `precedence` order; the first match gives the value through
[`prop_tag_expr`](@ref), and a field carrying no tag of the channel is passed through
unchanged.

# Algorithm

 1. Read the `precedence` tuple of the channel from [`PROP_TAG_CHANNELS`](@ref).
 2. For each field name `fname` of `all_fields`, in declaration order:
     1. Build `xf`, the expression `obj.fname` that reads the field off the incoming struct.
     2. Find `idx`, the position of the **first** tag of `precedence` that `fname` carries.
     3. When `idx` is `nothing`, the field carries no tag of this channel: the value is `xf` itself.
     4. Otherwise the value is [`prop_tag_expr`](@ref) of that tag, in this channel.
     5. Push `Expr(:kw, fname, value)` onto `pairs`.
 3. Return `pairs`.

Step 2.2 is where the precedence decides one field's transform. A field carrying `@pprop` and `@fprop` takes the `@pprop` transform on the `prior` channel and the `@fprop` transform on the `factory` channel, because the two channels order the tags differently.

# Arguments

  - `channel::Symbol`: A channel name of [`PROP_TAG_CHANNELS`](@ref).
  - `tagged::AbstractDict`: Tag name to the field names that carry it.
  - `all_fields::AbstractVector{Symbol}`: Every declared field, in declaration order.
  - `obj::Symbol`: The struct the generated method reads the fields off.
  - `mod::Module`: The module that defines [`@propagatable`](@ref).
  - `thread`: Extra positional arguments the channel threads before `args...`.

# Returns

  - `pairs::Vector{Any}`: One `Expr(:kw, field, value)` per declared field.

# Related

  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_channel_pairs(channel::Symbol, tagged::AbstractDict,
                            all_fields::AbstractVector{Symbol}, obj::Symbol, mod::Module,
                            thread)
    precedence = getproperty(PROP_TAG_CHANNELS, channel).precedence
    pairs      = Any[]
    for fname in all_fields
        xf  = Expr(:., obj, QuoteNode(fname))
        idx = findfirst(tag -> fname in tagged[tag], precedence)
        val = isnothing(idx) ? xf : prop_tag_expr(channel, precedence[idx], fname, xf, mod, thread)
        push!(pairs, Expr(:kw, fname, val))
    end
    return pairs
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to Julia's `@doc` macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to recognise docstring-prefixed fields in a struct body.

# Algorithm

 1. Return `true` when `x` is a `GlobalRef` whose `name` is `Symbol("@doc")`.
 2. Return `true` when `x` is equal to `Symbol("@doc")`.
 3. Return `false` otherwise.

Both spellings are needed for the same reason [`prop_tag`](@ref) needs both: a struct body written by hand carries the bare `Symbol`, and one that another macro has already expanded carries the `GlobalRef`.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
is_doc_macro(x) = (x isa GlobalRef && x.name == Symbol("@doc")) || x == Symbol("@doc")

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the field name `Symbol` from a bare field or `field::Type` expression.

Errors with a descriptive message when `expr` is neither a bare `Symbol` nor a
`field::Type` annotation, since only those forms are valid after [`@fprop`](@ref).

# Algorithm

 1. `expr` is a `Symbol`: return it.
 2. `expr` is an `Expr` whose head is `:(::)`: return its first argument, which is the field name.
 3. `expr` is anything else: raise an error naming the expression.

Step 3 is what separates this function from [`try_field_name`](@ref), which returns `nothing` in the same case. A tag states that the node **is** a field, so a node that is not one is a defect in the struct body and not a node to skip.

# Arguments

  - `expr`: A `Symbol`, an `Expr` with head `:(::)`, or any other expression (triggers an error).

# Returns

  - `name::Symbol`: The field name.

# Related

  - [`@fprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function extract_field_name(expr)
    if expr isa Symbol
        return expr
    end
    if expr isa Expr && expr.head == :(::)
        return expr.args[1]
    end
    return error("@propagatable: @fprop must precede a bare field name or field::Type, got: $(repr(expr))")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Recursively unwrap macro call chains to locate the innermost `:struct` node.

Returns `(struct_node, rebuild_fn)` where `rebuild_fn(new_struct)` reconstructs the
original macro chain with `new_struct` in place of the original struct. This allows
[`@propagatable`](@ref) to inject modified struct definitions back into arbitrary macro
wrappers such as `@concrete`.

# Algorithm

 1. `expr` is not an `Expr`: raise an error naming its type.
 2. `expr` has head `:struct`: return `expr` itself and `identity`, which rebuilds nothing.
 3. `expr` has head `:macrocall`: take `inner`, its **last** argument, and call this function again on it. That call gives `struct_node` and `rebuild`. Read `prefix`, every argument of `expr` except the last. Return `struct_node` and the function `s -> Expr(:macrocall, prefix..., rebuild(s))`.
 4. `expr` has any other head: raise an error naming the head.

Step 3 rebuilds the chain from the inside out, so a struct wrapped in several macros comes back wrapped in the same macros, in the same order, with the same arguments. The prefix carries the macro's own arguments and its `LineNumberNode`, so nothing of the call is lost.

# Arguments

  - `expr`: A `:struct` expression or a `:macrocall` expression wrapping one.

# Returns

  - `struct_node::Expr`: The innermost `:struct` expression.
  - `rebuild_fn::Function`: A function that, given a replacement `:struct`, returns the
    full macro chain with the replacement in place of the original.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`propagatable_bare_name`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_find_struct(expr)
    if !(expr isa Expr)
        error("@propagatable: expected a struct or macro-wrapped struct, got $(typeof(expr))")
    end
    if expr.head == :struct
        return expr, identity
    elseif expr.head == :macrocall
        inner = expr.args[end]
        struct_node, rebuild = propagatable_find_struct(inner)
        prefix = expr.args[1:(end - 1)]
        return struct_node, s -> Expr(:macrocall, prefix..., rebuild(s))
    else
        error("@propagatable: expected a struct definition (possibly wrapped in macros), " *
              "got Expr with head :$(expr.head)")
    end
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the plain struct name `Symbol` from a potentially parameterised or
supertype-constrained name expression.

Handles the forms `Name`, `Name{T, ...}`, and `Name{T, ...} <: SuperType` by
recursively peeling `:curly` and `:<:` wrappers until a bare `Symbol` is reached.

# Algorithm

 1. `n` is a `Symbol`: return it.
 2. `n` has head `:curly`: call this function again on its first argument, which drops the type parameters.
 3. `n` has head `:<:`: call this function again on its first argument, which drops the supertype.
 4. `n` is anything else: raise an error naming the expression.

Steps 2 and 3 compose, so `Name{T} <: Super` peels the supertype first and then the parameters.

# Arguments

  - `n`: A `Symbol`, or an `Expr` with head `:curly` or `:<:`.

# Returns

  - `name::Symbol`: The plain struct name.

# Related

  - [`propagatable_find_struct`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_bare_name(n)
    if n isa Symbol
        return n
    end
    if n isa Expr && n.head == :curly
        return propagatable_bare_name(n.args[1])
    end
    if n isa Expr && n.head == :<:
        return propagatable_bare_name(n.args[1])
    end
    return error("@propagatable: cannot extract struct name from: $(repr(n))")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the field name `Symbol` for a plain field declaration, or `nothing` for
non-field nodes.

Recognises bare `Symbol` fields and `field::Type` annotations. Returns `nothing` for
`LineNumberNode`s, inner constructors, and any other expression that does not declare
a single named field.

# Algorithm

 1. `expr` is a `Symbol`: return it.
 2. `expr` has head `:(::)` **and** its first argument is a `Symbol`: return that argument.
 3. `expr` is anything else: return `nothing`.

Step 2 tests the first argument as well as the head, which [`extract_field_name`](@ref) does not. A node such as `::Type`, which annotates no name, therefore gives `nothing` here and reaches step 3.

# Arguments

  - `expr`: Any expression appearing in a struct body.

# Returns

  - `name::Symbol`: The field name, if `expr` is a plain field declaration.
  - `nothing`: If `expr` is not a plain field declaration.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function try_field_name(expr)
    if expr isa Symbol
        return expr
    end
    if expr isa Expr && expr.head == :(::) && expr.args[1] isa Symbol
        return expr.args[1]
    end
    return nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Walk a struct body, collecting the tagged field names (and all field names) and stripping
the tags from the body.

Handles bare tagged fields (`@fprop field`, …), stacked tags
(`@pprop @fprop field`, in any order), and docstring-prefixed forms
(`"doc" \\n @fprop field`). Non-field nodes (line numbers, inner constructors) are carried
through unchanged. The tags are the rows of [`PROP_TAG_NAMES`](@ref), so a new tag needs no
change here.

# Algorithm

 1. Make `tagged`, one empty vector per tag of [`PROP_TAG_NAMES`](@ref); `all_fields`, an empty vector; and `new_args`, an empty vector for the stripped body.
 2. For each node `arg` of the struct body, in declaration order, take one of three branches:
     1. `arg` is a `@doc` macrocall, which is how a documented field parses. Peel the tags off `inner`, its last argument.
         1. The field carries at least one tag: record the field name under each of its tags and in `all_fields`, then push a rebuilt `@doc` node whose last argument is the **stripped** field.
         2. The field carries no tag: record its name in `all_fields` when [`try_field_name`](@ref) finds one, and push `arg` unchanged.
     2. `arg` is a tag macrocall with no docstring: peel the tags, record the field name under each of them and in `all_fields`, and push the stripped field expression.
     3. `arg` is anything else — a `LineNumberNode`, an untagged field, an inner constructor: record its name in `all_fields` when [`try_field_name`](@ref) finds one, and push `arg` unchanged.
 3. Return `tagged`, `all_fields`, and the new body as one `:block` expression.

`all_fields` holds **every** declared field, tagged or not, and it is what makes the generated constructor call name every keyword. The returned body carries no tag, so the wrapped macros and Julia itself never see one.

# Arguments

  - `body::Expr`: The `:block` expression forming the struct body.

# Returns

  - `tagged::Dict{Symbol, Vector{Symbol}}`: One entry per tag of [`PROP_TAG_NAMES`](@ref),
    holding the names of the fields that carry it, in declaration order.
  - `all_fields::Vector{Symbol}`: Names of every declared field (tagged or not).
  - `new_body::Expr`: The struct body with all tags stripped.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`peel_prop_tags`](@ref)
  - [`is_doc_macro`](@ref)
  - [`extract_field_name`](@ref)
  - [`try_field_name`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_parse_body(body)
    tagged     = Dict{Symbol, Vector{Symbol}}(tag => Symbol[] for tag in PROP_TAG_NAMES)
    all_fields = Symbol[]
    new_args   = Any[]
    function _record!(fname, tags)
        for tag in tags
            push!(tagged[tag], fname)
        end
        push!(all_fields, fname)
        return nothing
    end
    for arg in body.args
        if arg isa Expr && arg.head == :macrocall && is_doc_macro(arg.args[1])
            # Core.@doc "doc" (field or tagged field)
            inner = arg.args[end]
            tags, stripped = peel_prop_tags(inner)
            if !isempty(tags)
                _record!(extract_field_name(stripped), tags)
                # Rebuild @doc node with tags stripped: replace last arg with bare field
                push!(new_args, Expr(:macrocall, arg.args[1:(end - 1)]..., stripped))
            else
                # plain docstring'd field — carry through unchanged
                fname = try_field_name(inner)
                if fname !== nothing
                    push!(all_fields, fname)
                end
                push!(new_args, arg)
            end
        elseif is_prop_tag_call(arg)
            # Bare tagged field (tags may be stacked) — no docstring
            tags, stripped = peel_prop_tags(arg)
            _record!(extract_field_name(stripped), tags)
            push!(new_args, stripped)               # strip tags, keep field expr
        else
            # LineNumberNode, bare Symbol field, field::Type, inner constructor, …
            fname = try_field_name(arg)
            if fname !== nothing
                push!(all_fields, fname)
            end
            push!(new_args, arg)
        end
    end
    return tagged, all_fields, Expr(:block, new_args...)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    @fprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as participating in [`factory`](@ref) propagation —
`factory_child` will be called on it when `factory` is invoked on the
enclosing struct.

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:fprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.

 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `factory` channel and the `obs` channel, and each active channel makes [`@propagatable`](@ref) emit one method.

 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value:

      + `factory` channel: `factory_child(x.field, args...; kwargs...)`, which recurses into the child.
      + `obs` channel: `obs_weights_view(x.field, i)`, which recurses into the child on the observation axis.

 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. The same tag has two transforms, because a channel decides what a tag means. The `obs` channel does not gate on `@fprop`, so a struct whose only tag is `@fprop` gains no [`obs_weights_view`](@ref) method; the tag is consulted there only when a sibling field carries [`@wprop`](@ref).

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@vprop`](@ref)
  - [`@wprop`](@ref)
  - [`factory`](@ref)
  - [`factory_child`](@ref)
  - [`obs_weights_view`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
"""
macro fprop(expr)
    return error("@fprop may only appear inside a @propagatable struct body")
end

"""
    @vprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as participating in [`port_opt_view`](@ref) propagation —
`port_opt_view` will be called on it when a view (index selection) is propagated
through the enclosing struct.

Orthogonal to [`@fprop`](@ref); the two may be stacked on one field
(`@fprop @vprop field`) when it participates in both factory and view propagation.

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:vprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `view` channel alone, and each active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `view_child(x.field, i, args...)`, which is `port_opt_view(x.field, i, args...)` for every value but a precomputed optimisation result, kept as it is. The channel forwards the threaded tail and **no** keywords.
 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. `@vprop` appears in one channel, so it carries one transform and no channel can give it a second meaning. The index that the method threads selects **assets**; the observation axis has its own verb, [`obs_weights_view`](@ref).

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@fprop`](@ref)
  - [`port_opt_view`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
"""
macro vprop(expr)
    return error("@vprop may only appear inside a @propagatable struct body")
end

"""
    @pprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as **prior-selected**: when `factory(x, pr::AbstractPriorResult, …)` is
invoked, the field is set to `sel(getfield(x, :field), getproperty(pr, :field))` — the
risk-measure value if present, else the same-named moment from the prior result.

Orthogonal to, and stackable with, [`@wprop`](@ref) (`@pprop @wprop w` gives a weights
field both a prior factory and an `ObsWeights` factory) or [`@fprop`](@ref); `@pprop` wins
in the prior method. Mutually exclusive with [`@cprop`](@ref) on a single field.

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:pprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `prior` channel alone, and each active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `sel(x.field, getproperty(pr, :field))`. The prior result supplies the property of the **same name** as the field, so the tag names no source of its own.
 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. `@pprop` is first in the `prior` channel's precedence, so a field carrying both `@pprop` and [`@fprop`](@ref) takes the prior transform on that channel and the factory transform on the `factory` channel.

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@cprop`](@ref)
  - [`@wprop`](@ref)
  - [`factory`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
"""
macro pprop(expr)
    return error("@pprop may only appear inside a @propagatable struct body")
end

"""
    @cprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as **context-selected**: when `factory(x, pr::AbstractPriorResult, …)` is
invoked, the field is set to `sel(getfield(x, :field), _ctx(args...))` — the risk-measure
value if present, else the threaded optimiser value (a solver) located by type in the
variadic tail. Used for `slv` fields, whose source is a threaded argument rather than the
prior. Mutually exclusive with [`@pprop`](@ref) on a single field.

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:cprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `prior` channel alone, and each active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `sel(x.field, _ctx(args...))`. `_ctx` finds the value **by type** in the threaded tail, so the source is an argument and not the prior result.
 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. `@cprop` follows [`@pprop`](@ref) in the `prior` channel's precedence, and the two are mutually exclusive on one field.

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@pprop`](@ref)
  - [`factory`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
"""
macro cprop(expr)
    return error("@cprop may only appear inside a @propagatable struct body")
end

"""
    @wprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as an **observation-weights slot**: when `factory(x, w::ObsWeights, …)`
is invoked, the field is **replaced** by the incoming weights via `_wprop`; when no
[`ObsWeights`](@ref) is threaded, it is left unchanged.

Distinct from [`@fprop`](@ref), which recurses into a sub-estimator value and leaves a
`nothing` value untouched. A weights field defaults to `nothing` (meaning "uniform")
and must become the incoming weights — so it cannot share `@fprop`'s `nothing`-handling
without the two semantics colliding. Use `@wprop` for the `w`/weights field and `@fprop`
for sub-estimators.

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:wprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.

 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `factory` channel and the `obs` channel, and each active channel makes [`@propagatable`](@ref) emit one method.

 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value:

      + `factory` channel: `_wprop(x.field, args...; kwargs...)`, which **replaces** the field with an incoming [`ObsWeights`](@ref) and keeps it when none is threaded.
      + `obs` channel: `nothing_scalar_array_getindex(x.field, i)`, which **indexes** the value already there to the selected observations. Indexing rather than viewing is what keeps the `AbstractWeights` subtype.

 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. **The two channels do different things to the same field**, which is the one place a reader learns that [`factory`](@ref) and [`obs_weights_view`](@ref) are not two names for one operation. `@wprop` is also the only tag that gates the `obs` channel, so a field opts a struct into [`obs_weights_view`](@ref) by carrying this tag and no second one.

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@fprop`](@ref)
  - [`@pprop`](@ref)
  - [`factory`](@ref)
  - [`_wprop`](@ref)
  - [`obs_weights_view`](@ref)
  - [`ObsWeights`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
"""
macro wprop(expr)
    return error("@wprop may only appear inside a @propagatable struct body")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that every tag of [`PROP_TAG_NAMES`](@ref) is complete.

A tag row is complete when it has a stub macro, a channel of [`PROP_TAG_CHANNELS`](@ref)
that names it, and a field transform in [`prop_tag_expr`](@ref) **for every channel that
names it**. A row that lacks one of the three is a tag that parses but never propagates,
which is the failure the table exists to stop. The per-channel probe is what catches a tag
added to a second channel without a transform there, now that a tag means what its channel
says it means. All the violations are collected and reported together.

Runs once at the end of the module. Throws an
[`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError) listing
every violation, so the package refuses to precompile rather than shipping a dead tag.

The three tables are arguments, and each one defaults to the table the module ships. The
shipped tables are complete, so the call the module makes never reports a violation; a caller
that passes a table of its own drives each of the three clauses and reads the message it gives.

# Algorithm

 1. Make `violations`, an empty vector of strings.
 2. For each tag of [`PROP_TAG_NAMES`](@ref), with its macro name from [`PROP_TAG_MACRO_NAMES`](@ref):
     1. The macro name is not defined in this module: push a message that the tag declares no stub macro.
     2. The tag appears in the `precedence` of no channel of [`PROP_TAG_CHANNELS`](@ref): push a message that the tag appears in no channel.
     3. For each channel whose `precedence` names the tag, call [`prop_tag_expr`](@ref) with the probe name `:probe`. When that call raises, push a message naming the tag and the channel.
 3. `violations` is not empty: throw an `ArgumentError` listing every one of them.
 4. Return `nothing`.

Step 2.3 probes **each** channel that names the tag, not the tag alone. This is what catches a tag added to a second channel with no transform there, which the whole-tag probe of an earlier design let through.

# Arguments

  - `tags`: The tag names to check.
  - `macro_names`: The stub macro name of each tag, in the order of `tags`.
  - `channels`: The channel table whose `precedence` tuples are read.
  - `mod::Module`: The module the stub macros are looked up in, and the module that qualifies the names [`prop_tag_expr`](@ref) emits.

# Returns

  - `nothing`: Every row of `tags` is complete.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`check_propagatable_contracts`](@ref)
  - [`@propagatable`](@ref)
"""
function check_prop_tag_macros(tags = PROP_TAG_NAMES, macro_names = PROP_TAG_MACRO_NAMES,
                               channels = PROP_TAG_CHANNELS, mod::Module = @__MODULE__)
    violations = String[]
    for (tag, macro_name) in zip(tags, macro_names)
        if !isdefined(mod, macro_name)
            push!(violations, "`:$(tag)` declares no `$(macro_name)` stub macro.")
        end
        if !any(tag in channel.precedence for channel in channels)
            push!(violations, "`:$(tag)` appears in no channel of `PROP_TAG_CHANNELS`.")
        end
        for channel in keys(channels)
            if !(tag in getproperty(channels, channel).precedence)
                continue
            end
            try
                prop_tag_expr(channel, tag, :probe, :probe, mod, ())
            catch
                push!(violations,
                      "`:$(tag)` has no field transform in channel `:$(channel)`.")
            end
        end
    end
    if !isempty(violations)
        throw(ArgumentError("Incomplete `PROP_TAG_NAMES` rows:\n" *
                            join("  - " .* violations, "\n")))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# @propagatable — the declaration-time contract check
# ---------------------------------------------------------------------------

"""
    PROPAGATABLE_CONTRACTS

Every type declared with [`@propagatable`](@ref), paired with its `@pprop`-tagged field names.

One entry is appended by the macro itself, immediately after the struct it declares, so the
list is complete by the time the module finishes loading — including types declared in
external packages. [`check_propagatable_contracts`](@ref) is what reads it.

# Related

  - [`@propagatable`](@ref)
  - [`propagatable_register!`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
const PROPAGATABLE_CONTRACTS = Vector{Tuple{Type, Tuple{Vararg{Symbol}}}}()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Record type `T` and its `@pprop`-tagged field names in [`PROPAGATABLE_CONTRACTS`](@ref).

Called by [`@propagatable`](@ref) at the declaration itself. It only records: the outer
keyword constructor is written *below* the struct, so it does not exist yet and cannot be
checked here. [`check_propagatable_contracts`](@ref) does the checking once the module is
complete.

# Algorithm

 1. Push the pair `(T, pprops)` onto [`PROPAGATABLE_CONTRACTS`](@ref).
 2. Return `nothing`.

`T` is `@nospecialize`d, so one method serves every registered type and the registration costs no compilation.

# Related

  - [`@propagatable`](@ref)
  - [`PROPAGATABLE_CONTRACTS`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
function propagatable_register!(@nospecialize(T::Type), pprops::Tuple{Vararg{Symbol}})
    push!(PROPAGATABLE_CONTRACTS, (T, pprops))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the keyword names accepted by the outer constructors of `T`, unioned over its methods.

A `kwargs...` slurp is dropped rather than counted. A slurp accepts `field = value` and then
discards it, which is the silent failure this check exists to catch, so it must not satisfy
the contract.

# Algorithm

 1. Make `kws`, an empty vector of symbols.
 2. For each method `m` of the constructor `T`, append `Base.kwarg_decl(m)` to `kws`. The union runs over every outer constructor, so a keyword that any one of them names counts.
 3. Remove the repeats from `kws`.
 4. Remove every name whose string ends in `...`, which is how `Base.kwarg_decl` reports a slurp.
 5. Return `kws`.

Step 4 is the whole point of the function. A constructor that carries `kwargs...` reports the slurp as a keyword name, and counting it would let every field satisfy the contract.

# Related

  - [`check_propagatable_contracts`](@ref)
"""
function propagatable_keywords(@nospecialize(T::Type))
    kws = Symbol[]
    for m in methods(T)
        append!(kws, Base.kwarg_decl(m))
    end
    return filter!(k -> !endswith(string(k), "..."), unique!(kws))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the broken clauses of one type's [`@propagatable`](@ref) contract, as messages.

Two clauses are checked, and both are properties of the code the macro emits:

  - **Every field name is a keyword of the outer constructor.** Each generated method rebuilds
    the struct with `StructName(; field = …)` over *all* fields, tagged or not, so one field
    that the keyword constructor does not name is a `MethodError` at the first
    [`factory`](@ref) or [`port_opt_view`](@ref) call.
  - **Every `@pprop` field is a property of a prior result.** The generated
    `factory(x, pr::AbstractPriorResult, args...)` reads `getproperty(pr, :field)`, so a name
    absent from [`prior_result_property_pool`](@ref) throws when a prior is threaded.

The messages carry a [`suggest_declared_key`](@ref) suggestion, so a transposed or mistyped
field name names its intended neighbour.

# Algorithm

 1. Make `msgs`, an empty vector of strings.
 2. Read `kws`, the keywords of the outer constructors of `T`, with [`propagatable_keywords`](@ref).
 3. For each field name of `T` that is absent from `kws`, push a message naming the type, the field and the [`suggest_declared_key`](@ref) suggestion drawn from `kws`.
 4. For each name of `pprops` that is absent from `pool`, push a message naming the type, the field and the suggestion drawn from `pool`.
 5. Return `msgs`.

Every clause is collected, and none stops the walk, so one call reports every violation of one type at once. An empty result means that the type's contract holds.

# Related

  - [`check_propagatable_contracts`](@ref)
  - [`propagatable_keywords`](@ref)
  - [`prior_result_property_pool`](@ref)
"""
function propagatable_contract_violations(@nospecialize(T::Type), pprops, pool)
    msgs = String[]
    kws = propagatable_keywords(T)
    for f in fieldnames(T)
        if !(f in kws)
            push!(msgs,
                  "`$(nameof(T))`: field `$(f)` is not a keyword of its outer constructor" *
                  suggest_declared_key(f, kws) *
                  ".")
        end
    end
    for f in pprops
        if !(f in pool)
            push!(msgs,
                  "`$(nameof(T))`: `@pprop` field `$(f)` is not a property of a " *
                  "prior result" *
                  suggest_declared_key(f, pool) *
                  ".")
        end
    end
    return msgs
end
"""
    @propagatable expr

Define a struct and automatically generate its propagation methods from five
orthogonal, stackable field tags:

  - [`@fprop`](@ref) (factory propagation): tagged fields receive `factory_child`
    calls when [`factory`](@ref) is invoked, recursing runtime *values*
    (observation weights, prior results, solvers, …) down the composition tree.
    A `factory(x, args...)` method is always generated (it is the identity when no
    field is tagged `@fprop`/`@wprop`).
  - [`@wprop`](@ref) (weights replacement): tagged observation-weights fields are
    *replaced* by an incoming [`ObsWeights`](@ref) argument via `_wprop` (and left
    unchanged when none is threaded). Use `@wprop` for the weights slot and `@fprop`
    for sub-estimators — a weights field that defaults to `nothing` must become the
    incoming weights, which conflicts with `@fprop`'s `nothing`-passthrough.
  - [`@vprop`](@ref) (view propagation): tagged fields receive
    [`port_opt_view`](@ref) calls when a view (an index selection) is propagated,
    recursing into composed children and slicing data arrays. A `port_opt_view`
    method is generated **only when at least one field is tagged `@vprop`**.
  - [`@pprop`](@ref) (prior selection): tagged fields are selected from the
    same-named field on a prior result via `sel(getfield(x, :f), getproperty(pr, :f))`.
  - [`@cprop`](@ref) (context selection): tagged fields are selected against a
    threaded optimiser value (a solver) found by type via `sel(getfield(x, :f), _ctx(args...))`.

When at least one field is tagged `@pprop` or `@cprop`, a second method
`factory(x, pr::AbstractPriorResult, args...)` is generated. It selects `@pprop`/`@cprop`
fields as above and threads `@fprop`-only fields with `pr`
(`factory_child(getfield(x, :f), pr, args...)`); a field tagged both `@pprop` and `@fprop`
is prior-selected in this method (`@pprop` wins). It then calls
[`resolve_deferred_quantities`](@ref) on the **selected** struct — the identity unless the
type declares a method — so the Deferred-Quantity resolution runs **last** and a slot that
holds one sees the solver, the observation weights and the children already settled.
Because this method is more specific than the general `factory(x, args...)`, it is chosen
whenever a prior is passed.

Untagged fields pass through unchanged in every method, regardless of type —
tagging is explicit and opt-in. The tags are independent and the relevant field sets
genuinely diverge. `@pprop` and `@cprop` are mutually exclusive on one field (a value comes
from exactly one source); legal stacks are `@pprop @fprop` (sub-estimator) and
`@pprop @wprop` (weights slot).

Two consequences of the emitted code are **contracts on the declaration**, and both are checked
where the struct is written rather than at the first call:

  - Every generated method rebuilds the struct as `StructName(; field = …)` over *all* fields,
    tagged or not, so **every field name must also be a keyword of the outer constructor**. A
    `kwargs...` slurp does not satisfy this: it accepts the keyword and then discards it.
  - The prior method reads `getproperty(pr, :field)`, so **every `@pprop` field name must be a
    property of a prior result** (see [`prior_result_property_pool`](@ref)).

The macro registers each declaration in [`PROPAGATABLE_CONTRACTS`](@ref), and
[`check_propagatable_contracts`](@ref) checks the whole registry once the module is complete.
A mistyped field name therefore fails at precompilation with a
[`suggest_declared_key`](@ref) suggestion, rather than surfacing as a `MethodError` at the
first [`factory`](@ref) call. [`forward_prior`](@ref) leans on the same contract.

The tag set itself is data. [`PROP_TAG_NAMES`](@ref) holds the rows,
[`prop_tag_expr`](@ref) holds each tag's field transform, and [`PROP_TAG_CHANNELS`](@ref)
holds each channel's gate and tag precedence, so a new propagation channel is a table row
rather than an edit at seven sites.

Composes with `@concrete` (put `@propagatable` outermost):

```julia
@propagatable @concrete struct MyMeasure <: RiskMeasure
    @pprop @wprop w       # prior factory selects pr.w; ObsWeights factory fills w
    @pprop sigma          # prior-selected from pr.sigma
    @fprop alg            # threaded (recursed) with pr / args
    config                # passed through unchanged
    function MyMeasure(w, sigma, alg, config)
        return new{typeof(w), typeof(sigma), typeof(alg), typeof(config)}(w, sigma, alg,
                                                                          config)
    end
end
```

`@wprop` drives two channels at once, and they do different things to the same field:
`factory` **replaces** it with an incoming [`ObsWeights`](@ref) value, while
[`obs_weights_view`](@ref) **indexes** the value already there, to a set of observations.
A weights field therefore needs no second tag to join the observation-axis view.

The generated `factory`/`port_opt_view`/`obs_weights_view` methods are added to the
`PortfolioOptimisers` functions, so `@propagatable` works correctly for types
defined in external packages.

Docstrings on the enclosing definition are forwarded correctly via
`Base.@__doc__`.

# Algorithm

 1. Find the struct with [`propagatable_find_struct`](@ref), which gives `struct_node` and `rebuild`, the function that puts a replacement struct back inside the same chain of wrapping macros.
 2. Read `type_head` and `body` off `struct_node`, and read `struct_name` off `type_head` with [`propagatable_bare_name`](@ref), which drops the type parameters and the supertype.
 3. Parse the body with [`propagatable_parse_body`](@ref), which gives `tagged`, the field names per tag; `all_fields`, every declared field in declaration order; and `new_body`, the body with every tag stripped.
 4. Build `new_struct` from `new_body`, and `chain` from `rebuild(new_struct)`. `chain` is the original declaration with the tags gone, so `@concrete` and Julia both see an ordinary struct.
 5. Bind `POMOD` to the module that **defines** the macro, and qualify every emitted name against it. A bare name would resolve in the caller's module, where `function factory(…)` declares a new function of the caller's own and the method never reaches `PortfolioOptimisers.factory`. That failure is silent, because the declaration compiles and the type never joins the propagation chain.
 6. Emit the `factory` method. When [`prop_channel_active`](@ref) holds for the `factory` channel, the body is a call to the keyword constructor whose pairs come from [`prop_channel_pairs`](@ref); otherwise the body is `x` itself. **This method is always emitted**, so an untagged [`@propagatable`](@ref) struct still answers [`factory`](@ref) with the identity.
 7. When the `view` channel is active, emit `port_opt_view(x::StructName, i, args...)`, whose channel threads `i` before `args...`.
 8. When the `obs` channel is active, emit `obs_weights_view(x::StructName, i)`, whose channel threads `i` and takes no tail.
 9. When the `prior` channel is active, emit `factory(x::StructName, pr::AbstractPriorResult, args...; kwargs...)`. Its body selects every tagged field off `x` and then hands the selected struct to [`resolve_deferred_quantities`](@ref), so **the Deferred-Quantity resolution runs last**. A Deferred Quantity and a **Calibration Rule** therefore see the solver, the observation weights and the children in the state the optimisation settled them in, and a rule may call [`ERM`](@ref) or [`RRM`](@ref). This method is more specific than the one of step 6, so a call that threads a prior chooses it.
10. Build `pprop_tuple`, the `@pprop`-tagged field names as a tuple of quoted symbols.
11. Return one escaped block holding, in order: `Base.@__doc__ chain`, so a docstring on the declaration reaches the struct; the emitted methods; and the call to [`propagatable_register!`](@ref) that records the type and `pprop_tuple`.

Steps 6 to 9 differ only in the method head and in the arguments that the channel threads. Each reads its gate and its tag precedence off [`PROP_TAG_CHANNELS`](@ref), so a new channel is a row of that table, a branch in [`prop_tag_expr`](@ref) and a stub macro, rather than an edit at seven sites.

# Related

  - [`@fprop`](@ref)
  - [`@vprop`](@ref)
  - [`@pprop`](@ref)
  - [`@cprop`](@ref)
  - [`@wprop`](@ref)
  - [`PROP_TAG_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_channel_active`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`propagatable_register!`](@ref)
  - [`check_propagatable_contracts`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
"""
macro propagatable(expr)
    struct_node, rebuild = propagatable_find_struct(expr)

    type_head   = struct_node.args[2]
    body        = struct_node.args[3]
    struct_name = propagatable_bare_name(type_head)

    tagged, all_fields, new_body = propagatable_parse_body(body)

    new_struct = Expr(:struct, struct_node.args[1], type_head, new_body)
    chain      = rebuild(new_struct)

    # Every name the expansion emits is qualified against the module that *defines* the
    # macro, because the emitted block is escaped and a bare name resolves where the struct
    # is declared. `factory` is exported, which does not help: `using PortfolioOptimisers`
    # binds it implicitly, so `function factory(...)` in the caller declares a **new**
    # function of the caller's own, and `PortfolioOptimisers.factory` never gains the
    # method. That failure is silent -- the declaration compiles, the contract registers,
    # and the type simply never joins the propagation chain. `@__MODULE__` in a macro body
    # is the defining module, and interpolating the module object needs no binding in the
    # caller at all (ADR 0002, decision 4). `prop_tag_expr` qualifies the names it emits the
    # same way.
    POMOD = @__MODULE__
    _factory = :($POMOD.factory)
    _port_opt_view = :($POMOD.port_opt_view)
    _obs_weights_view = :($POMOD.obs_weights_view)
    _resolve_fn = :($POMOD.resolve_deferred_quantities)
    _prior_result = :($POMOD.AbstractPriorResult)
    _register_fn = :($POMOD.propagatable_register!)

    # Every channel below reads its tag precedence off `PROP_TAG_CHANNELS`. The emission
    # differs only in the method head and in the arguments the channel threads.

    # --- factory propagation (@fprop recurses sub-estimators, @wprop replaces weights) ---
    factory_body = if prop_channel_active(:factory, tagged)
        Expr(:call, struct_name,
             Expr(:parameters,
                  prop_channel_pairs(:factory, tagged, all_fields, :x, POMOD, ())...))
    else
        :x
    end
    factory_def = quote
        function $_factory(x::$struct_name, args...; kwargs...)
            return $factory_body
        end
    end

    defs = Any[factory_def]

    # --- view propagation (@vprop) — emit only when a field opts in ---
    if prop_channel_active(:view, tagged)
        view_body = Expr(:call, struct_name,
                         Expr(:parameters,
                              prop_channel_pairs(:view, tagged, all_fields, :x, POMOD,
                                                 (:i,))...))
        view_def = quote
            function $_port_opt_view(x::$struct_name, i, args...)
                return $view_body
            end
        end
        push!(defs, view_def)
    end

    # --- observation-weights view (@wprop) — emit only when a weights field exists ---
    # The same `@wprop` tag the factory channel reads, given the other transform: the
    # weights field is indexed to `i` rather than replaced, and `@fprop` children recurse.
    if prop_channel_active(:obs, tagged)
        obs_body = Expr(:call, struct_name,
                        Expr(:parameters,
                             prop_channel_pairs(:obs, tagged, all_fields, :x, POMOD,
                                                (:i,))...))
        obs_def = quote
            function $_obs_weights_view(x::$struct_name, i)
                return $obs_body
            end
        end
        push!(defs, obs_def)
    end

    # --- prior/context selection (@pprop / @cprop) — emit only when a field opts in ---
    if prop_channel_active(:prior, tagged)
        # Every field is read off the argument, the selections run on it, and the
        # Deferred-Quantity resolution runs on the selected struct, LAST. A Deferred
        # Quantity and a Calibration Rule therefore see the solver, the observation
        # weights and the children in the state the optimisation settled them in.
        prior_body = Expr(:call, struct_name,
                          Expr(:parameters,
                               prop_channel_pairs(:prior, tagged, all_fields, :x, POMOD,
                                                  (:pr,))...))
        prior_def = quote
            function $_factory(x::$struct_name, pr::$_prior_result, args...; kwargs...)
                return $_resolve_fn($prior_body, pr)
            end
        end
        push!(defs, prior_def)
    end

    pprop_tuple = Expr(:tuple, QuoteNode.(tagged[:pprop])...)

    return esc(quote
                   Base.@__doc__ $chain
                   $(defs...)
                   $_register_fn($struct_name, $pprop_tuple)
               end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check the [`@propagatable`](@ref) contract of every type in [`PROPAGATABLE_CONTRACTS`](@ref).

Runs once at the end of the module, so the contract behind every generated method is enforced
where the structs are *declared* rather than at the first [`factory`](@ref) call. The
violations of every type are collected and reported together, because a run that stops at the
first one hides the rest. See [`propagatable_contract_violations`](@ref) for the two clauses.

Throws an [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError)
listing every violation; the package refuses to precompile rather than shipping a type whose
generated methods throw on first use. A package that declares its own `@propagatable` types
calls this at the end of its own module to get the same guarantee.

# Algorithm

 1. Read `pool`, the property names that a prior result can carry, with [`prior_result_property_pool`](@ref).
 2. Make `msgs`, an empty vector of strings.
 3. For each pair `(T, pprops)` of [`PROPAGATABLE_CONTRACTS`](@ref), append the messages that [`propagatable_contract_violations`](@ref) reports for that type.
 4. `msgs` is not empty: throw an `ArgumentError` naming the count and listing every message.
 5. Return `nothing`.

Step 3 collects and never stops, so one run reports the violations of every registered type. The registry is filled by [`propagatable_register!`](@ref) at each declaration, so this function must run **after** the last one, which is why the module calls it at its end.

Both the registry and the pool are arguments, and each defaults to the value the module ships. A caller that passes a registry of its own reads the message a broken contract gives, without registering a broken type.

# Arguments

  - `contracts`: The pairs of a type and its `@pprop`-tagged field names to check.
  - `pool`: The property names a prior result can carry.

# Returns

  - `nothing`: Every pair of `contracts` satisfies the contract.

# Related

  - [`@propagatable`](@ref)
  - [`PROPAGATABLE_CONTRACTS`](@ref)
  - [`propagatable_contract_violations`](@ref)
  - [`@windowed_estimator`](@ref)
"""
function check_propagatable_contracts(contracts = PROPAGATABLE_CONTRACTS,
                                      pool = prior_result_property_pool())
    msgs = String[]
    for (T, pprops) in contracts
        append!(msgs, propagatable_contract_violations(T, pprops, pool))
    end
    if !isempty(msgs)
        throw(ArgumentError("@propagatable: $(length(msgs)) broken contract(s). The generated `factory`/`port_opt_view` methods rebuild the struct by keyword, so each of these throws at its first call:\n  - " *
                            join(msgs, "\n  - ")))
    end
    return nothing
end

public @propagatable, @fprop, @vprop, @pprop, @cprop, @wprop
