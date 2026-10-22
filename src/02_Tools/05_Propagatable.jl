# ---------------------------------------------------------------------------
# @propagatable — struct-definition macro for factory propagation
# ---------------------------------------------------------------------------

# --- private AST helpers ----------------------------------------------------

"""
    PROP_TAG_NAMES

The propagation tag set of [`@propagatable`](@ref), as data.

One entry per field tag. The code that recognises a tag comes from this tuple, and no
tag is written out in that code. The macro names ([`PROP_TAG_MACRO_NAMES`](@ref)), the
lookup ([`prop_tag`](@ref)), the test ([`is_prop_tag_call`](@ref)), the peeler
([`peel_prop_tags`](@ref)) and the parser ([`propagatable_parse_body`](@ref)) all read it.
A new tag is one row here, one branch in [`prop_tag_expr`](@ref), one entry in a channel of
[`PROP_TAG_CHANNELS`](@ref) and one stub macro. [`check_prop_tag_macros`](@ref) refuses to
load the module when a row has no stub macro, no channel or no branch.

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

The tuple comes from the tag names, so a row of the table has no second spelling.
[`prop_tag`](@ref) matches the head of a `:macrocall` against this tuple.

# Algorithm

 1. Map each tag of [`PROP_TAG_NAMES`](@ref) to `Symbol("@", tag)`, which is the name that Julia gives the macro of that tag.

The map keeps the order, so the ``k``-th entry here is the macro name of the ``k``-th tag there. [`prop_tag`](@ref) and [`check_prop_tag_macros`](@ref) both walk the two tuples together, and they need this shared order to pair each tag with its macro name.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`prop_tag`](@ref)
"""
const PROP_TAG_MACRO_NAMES = Symbol.("@", PROP_TAG_NAMES)

"""
    PROP_TAG_CHANNELS

The propagation channels of [`@propagatable`](@ref), with their tag precedence as data.

One entry per generated method. Each entry has two tuples of tags:

  - `gate`: the tags that make [`@propagatable`](@ref) emit the method.
  - `precedence`: the order in which the channel reads the tags of a field. The first tag
    of this tuple that the field carries gives the transform of the field. The channel
    ignores the other tags of that field.

The `factory` channel prefers `@fprop` to `@wprop`. The `prior` channel prefers `@pprop`,
then `@cprop`, then `@wprop`, then `@fprop`, so `@pprop` wins over `@fprop` on one field.
[`prop_channel_pairs`](@ref) reads the precedence for every channel.

**A tag means what its channel says it means.** The `obs` channel reads the same `@wprop`
and `@fprop` tags as `factory`, and it gives them different transforms. `factory`
**replaces** a `@wprop` field with an incoming [`ObsWeights`](@ref) value, and `obs`
**indexes** the value that the field holds. For this reason, [`prop_tag_expr`](@ref) takes
the channel as well as the tag. A weights field joins [`obs_weights_view`](@ref)
through its `@wprop` tag, and it needs no second tag.

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
hand, and a `GlobalRef` after another macro expanded around it, so the function resolves
both. A name outside [`PROP_TAG_MACRO_NAMES`](@ref) gives `nothing`. The callers use this to
tell a tag from any other macro. A name never resolves to a tag that it does not name.

# Algorithm

 1. Read `name` from `x`. A `GlobalRef` gives its `name` field, a `Symbol` gives itself, and any other value returns `nothing` at once.
 2. Walk [`PROP_TAG_NAMES`](@ref) and [`PROP_TAG_MACRO_NAMES`](@ref) together. Return the tag whose macro name is identical to `name`.
 3. No macro name matches: return `nothing`.

The comparison is `===` on a `Symbol`, so only an identical name matches. A macro with a similar name never matches.

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

[`peel_prop_tags`](@ref) and [`propagatable_parse_body`](@ref) use it to find the tagged
fields of a struct body.

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

Peel the tag macrocalls off a field expression, and record the tags.

A field can have a stack of tags in any order, such as `@pprop @fprop field`. The stack
parses as nested `:macrocall` nodes. This function removes all of them and returns the bare
field expression. It finds each tag with [`prop_tag`](@ref), so a macro that is not a tag
stops the peel.

# Algorithm

 1. Make `tags`, an empty `Set{Symbol}`.
 2. While [`is_prop_tag_call`](@ref) of `expr` holds, push [`prop_tag`](@ref) of its first argument onto `tags`, and replace `expr` with its **last** argument, which is the expression the tag wraps.
 3. Return `tags` and the peeled `expr`.

`tags` is a set, so a tag written two times on one field is recorded once. The loop stops at the first node that is not a tag call, and a macro between two tags that is not a tag hides the tags below it.

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

This function gives the field transform of each tag. It has one branch per tag of
[`PROP_TAG_NAMES`](@ref), and [`prop_channel_pairs`](@ref) calls it for every channel. A tag
of the table with no branch here raises an error, so it cannot take the transform of
another tag without a sign.

# Algorithm

The function reads the channel first, and then the tag, because one tag has two transforms.

 1. `channel` is `:obs`:
     1. `tag` is `:wprop`: return `nothing_scalar_array_getindex(xf, thread...)`. The field **is** the weights, so the method indexes it at the selected observations. An index keeps the `AbstractWeights` subtype, and a view does not.
     2. `tag` is `:fprop`: return `obs_weights_view(xf, thread...)`. The field is a composed child, so the verb recurses into it.
 2. `channel` is any other channel:
     1. `tag` is `:fprop`: return `factory_child(xf, thread..., args...; kwargs...)`.
     2. `tag` is `:vprop`: return `view_child(xf, thread..., args...)`, which calls [`port_opt_view`](@ref) on every value but a precomputed optimisation result. This channel forwards no keywords.
     3. `tag` is `:pprop`: return `sel(xf, getproperty(pr, fname))`. The prior result gives the property of the **same name**, so the field name is an argument.
     4. `tag` is `:cprop`: return `sel(xf, _ctx(args...))`, which reads the context from the threaded arguments and not from the prior.
     5. `tag` is `:wprop`: return `_wprop(xf, args...; kwargs...)`, which **replaces** the field with an incoming [`ObsWeights`](@ref).
 3. No branch matched: raise an error that names the tag and the channel, and asks for a branch here.

Steps 1.1 and 2.5 are the same tag with two transforms, so the channel decides what `@wprop` means. Every emitted name is qualified with `mod`, because the expansion is escaped into the module of the caller.

# Arguments

  - `tag::Symbol`: A tag of [`PROP_TAG_NAMES`](@ref).
  - `fname::Symbol`: The field name. `@pprop` needs it to name the prior property.
  - `xf`: The expression that reads the field from the incoming struct.
  - `mod::Module`: The module that defines [`@propagatable`](@ref). Every emitted name is
    qualified with it, because the expansion is escaped into the module of the caller.
  - `thread`: The positional arguments that the channel threads before `args...`.

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

A channel is active when at least one field carries a tag of the `gate` of the channel.

# Algorithm

 1. Read the `gate` tuple of the channel from [`PROP_TAG_CHANNELS`](@ref).
 2. Return `true` when at least one tag of `gate` has a non-empty entry in `tagged`, and `false` otherwise.

This function does **not** read the `precedence` tuple. So a tag that a channel reads, but does not gate on, never makes that channel emit a method alone. The `obs` channel gates on `@wprop` alone and reads `@fprop`, so a type with `@fprop` and no `@wprop` gets no `obs` method.

# Arguments

  - `channel::Symbol`: A channel name of [`PROP_TAG_CHANNELS`](@ref).
  - `tagged::AbstractDict`: Map from a tag name to the names of the fields that carry it,
    from [`propagatable_parse_body`](@ref).

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

Every declared field gets one pair, in declaration order. The function reads the tags of
a field in the `precedence` order of the channel. The first match gives the value through
[`prop_tag_expr`](@ref). A field with no tag of the channel keeps its value.

# Algorithm

 1. Read the `precedence` tuple of the channel from [`PROP_TAG_CHANNELS`](@ref).
 2. For each field name `fname` of `all_fields`, in declaration order:
     1. Build `xf`, the expression `obj.fname` that reads the field from the incoming struct.
     2. Find `idx`, the position of the **first** tag of `precedence` that `fname` carries.
     3. When `idx` is `nothing`, the field carries no tag of this channel, and the value is `xf`.
     4. Otherwise, the value is [`prop_tag_expr`](@ref) of that tag, in this channel.
     5. Push `Expr(:kw, fname, value)` onto `pairs`.
 3. Return `pairs`.

In step 2.2, the precedence decides the transform of one field. A field with `@pprop` and `@fprop` takes the `@pprop` transform on the `prior` channel and the `@fprop` transform on the `factory` channel, because the two channels put the tags in different orders.

# Arguments

  - `channel::Symbol`: A channel name of [`PROP_TAG_CHANNELS`](@ref).
  - `tagged::AbstractDict`: Map from a tag name to the names of the fields that carry it.
  - `all_fields::AbstractVector{Symbol}`: Every declared field, in declaration order.
  - `obj::Symbol`: The struct from which the generated method reads the fields.
  - `mod::Module`: The module that defines [`@propagatable`](@ref).
  - `thread`: The positional arguments that the channel threads before `args...`.

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

[`propagatable_parse_body`](@ref) uses it to find the fields of a struct body that have a docstring.

# Algorithm

 1. Return `true` when `x` is a `GlobalRef` whose `name` is `Symbol("@doc")`.
 2. Return `true` when `x` is equal to `Symbol("@doc")`.
 3. Return `false` otherwise.

The function needs both spellings for the same reason as [`prop_tag`](@ref). A struct body written by hand has the bare `Symbol`, and a struct body that another macro expanded has the `GlobalRef`.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
is_doc_macro(x) = (x isa GlobalRef && x.name == Symbol("@doc")) || x == Symbol("@doc")

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the field name `Symbol` from a bare field or `field::Type` expression.

The function raises an error that names `expr` when `expr` is neither a bare `Symbol` nor a
`field::Type` annotation. Only those two forms are valid after a tag such as
[`@fprop`](@ref).

# Algorithm

 1. `expr` is a `Symbol`: return it.
 2. `expr` is an `Expr` whose head is `:(::)`: return its first argument, which is the field name.
 3. `expr` is any other expression: raise an error that names the expression.

Step 3 is the difference from [`try_field_name`](@ref), which returns `nothing` in the same case. A tag states that the node **is** a field. So a node that is not a field is a defect in the struct body, and the function does not skip it.

# Arguments

  - `expr`: A `Symbol`, an `Expr` with head `:(::)`, or any other expression, which raises an error.

# Returns

  - `name::Symbol`: The field name.

# Related

  - [`@fprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function extract_field_name(expr)
    if expr isa Expr && expr.head == :(::)
        return expr.args[1]
    end
    return error("@propagatable: a propagation tag must precede a bare field name or field::Type, got: $(repr(expr))")
end
extract_field_name(expr::Symbol) = expr

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Find the innermost `:struct` node of a chain of macro calls, through a recursion.

The function returns `(struct_node, rebuild_fn)`. `rebuild_fn(new_struct)` builds the
original macro chain again, with `new_struct` in place of the original struct. So
[`@propagatable`](@ref) can put a changed struct definition back inside any wrapping
macro, such as `@concrete`.

# Algorithm

 1. `expr` is not an `Expr`: raise an error that names its type.
 2. `expr` has head `:struct`: return `expr` and `identity`, which rebuilds nothing.
 3. `expr` has head `:macrocall`: take `inner`, its **last** argument, and call this function again on it. That call gives `struct_node` and `rebuild`. Read `prefix`, every argument of `expr` except the last. Return `struct_node` and the function `s -> Expr(:macrocall, prefix..., rebuild(s))`.
 4. `expr` has any other head: raise an error that names the head.

Step 3 builds the chain again from the inside out. A struct inside several macros comes back inside the same macros, in the same order, with the same arguments. The prefix holds the arguments of the macro and its `LineNumberNode`, so the rebuilt call loses nothing.

# Arguments

  - `expr`: A `:struct` expression, or a `:macrocall` expression around one.

# Returns

  - `struct_node::Expr`: The innermost `:struct` expression.
  - `rebuild_fn::Function`: A function that takes a replacement `:struct` and returns the
    full macro chain, with the replacement in place of the original.

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

Get the plain struct name, as a `Symbol`, from a name expression that can have type
parameters and a supertype.

The function accepts the forms `Name`, `Name{T, ...}` and `Name{T, ...} <: SuperType`. It
peels the `:curly` and `:<:` nodes, through a recursion, until it gets a bare `Symbol`.

# Algorithm

 1. `n` is a `Symbol`: return it.
 2. `n` has head `:curly`: call this function again on its first argument, which drops the type parameters.
 3. `n` has head `:<:`: call this function again on its first argument, which drops the supertype.
 4. `n` is any other expression: raise an error that names the expression.

Steps 2 and 3 work together, so `Name{T} <: Super` loses the supertype first and then the parameters.

# Arguments

  - `n`: A `Symbol`, or an `Expr` with head `:curly` or `:<:`.

# Returns

  - `name::Symbol`: The plain struct name.

# Related

  - [`propagatable_find_struct`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_bare_name(n)
    if n isa Expr && n.head in (:curly, :<:)
        return propagatable_bare_name(n.args[1])
    end
    return error("@propagatable: cannot extract struct name from: $(repr(n))")
end
propagatable_bare_name(n::Symbol) = n

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the field name, as a `Symbol`, of a plain field declaration, or `nothing` for a
node that is not a field.

The function accepts a bare `Symbol` field and a `field::Type` annotation. It returns
`nothing` for a `LineNumberNode`, an inner constructor and every other expression that does
not declare one named field.

# Algorithm

 1. `expr` is a `Symbol`: return it.
 2. `expr` has head `:(::)` **and** its first argument is a `Symbol`: return that argument.
 3. `expr` is any other expression: return `nothing`.

Step 2 tests the first argument as well as the head, and [`extract_field_name`](@ref) does not. So a node such as `a.b::Int`, whose first argument is not a `Symbol`, gives `nothing` here.

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
    if expr isa Expr && expr.head == :(::) && expr.args[1] isa Symbol
        return expr.args[1]
    end
    return nothing
end
try_field_name(expr::Symbol) = expr

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Walk a struct body. Collect the names of the tagged fields and of all fields, and remove
the tags from the body.

The function accepts a bare tagged field such as `@fprop field`, a stack of tags in any
order such as `@pprop @fprop field`, and a field with a docstring such as
`"doc" \\n @fprop field`. It keeps a node that is not a field, such as a line number or an
inner constructor, with no change. The tags are the rows of [`PROP_TAG_NAMES`](@ref), so a
new tag needs no change here.

# Algorithm

 1. Make `tagged`, one empty vector per tag of [`PROP_TAG_NAMES`](@ref). Make `all_fields`, an empty vector, and `new_args`, an empty vector for the stripped body.
 2. For each node `arg` of the struct body, in declaration order, take one of three branches:
     1. `arg` is a `@doc` macrocall, which is how a field with a docstring parses. Peel the tags off `inner`, its last argument.
         1. The field carries at least one tag: record the field name under each of its tags and in `all_fields`. Then push a rebuilt `@doc` node whose last argument is the **stripped** field.
         2. The field carries no tag: record its name in `all_fields` when [`try_field_name`](@ref) finds one, and push `arg` with no change.
     2. `arg` is a tag macrocall with no docstring: peel the tags, record the field name under each of them and in `all_fields`, and push the stripped field expression.
     3. `arg` is any other node, such as a `LineNumberNode`, an untagged field or an inner constructor: record its name in `all_fields` when [`try_field_name`](@ref) finds one, and push `arg` with no change.
 3. Return `tagged`, `all_fields`, and the new body as one `:block` expression.

`all_fields` holds **every** declared field, with or without a tag. With it, the generated constructor call names every keyword. The returned body has no tag, so the wrapped macros and Julia never see one.

# Arguments

  - `body::Expr`: The `:block` expression of the struct body.

# Returns

  - `tagged::Dict{Symbol, Vector{Symbol}}`: One entry per tag of [`PROP_TAG_NAMES`](@ref),
    which holds the names of the fields that carry the tag, in declaration order.
  - `all_fields::Vector{Symbol}`: The names of every declared field, with or without a tag.
  - `new_body::Expr`: The struct body with no tags.

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

Field tag for a [`@propagatable`](@ref) struct body. It marks a composed child for
[`factory`](@ref) propagation. A call of `factory` on the struct calls `factory_child` on
the field.

# Algorithm

The tag never expands, because [`@propagatable`](@ref) runs first and removes it. The steps below tell what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:fprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.

 2. [`prop_channel_active`](@ref) reads the recorded name. This tag gates the `factory` channel alone, and an active channel makes [`@propagatable`](@ref) emit one method. The `obs` channel reads the tag, but it does not gate on it.

 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value:

      + `factory` channel: `factory_child(x.field, args...; kwargs...)`, which recurses into the child.
      + `obs` channel: `obs_weights_view(x.field, i)`, which recurses into the child on the observation axis.

 4. The generated method builds the struct again with its keyword constructor, so every check of the constructor runs again on the propagated value.

Step 3 gives the meaning of the tag. The same tag has two transforms, because a channel decides what a tag means. The `obs` channel does not gate on `@fprop`, so a struct whose only tag is `@fprop` gets no [`obs_weights_view`](@ref) method. That channel reads the tag only when another field of the struct carries [`@wprop`](@ref).

The body of this macro raises an error. Julia runs it only when the tag is outside a [`@propagatable`](@ref) struct body, where no macro removed it.

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

Field tag for a [`@propagatable`](@ref) struct body. It marks the field for
[`port_opt_view`](@ref) propagation. A view of the struct at an index selection calls
`port_opt_view` on the field.

The tag is independent of [`@fprop`](@ref). A field can carry both, as
`@fprop @vprop field`, when it takes part in factory propagation and in view propagation.

# Algorithm

The tag never expands, because [`@propagatable`](@ref) runs first and removes it. The steps below tell what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:vprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. This tag gates the `view` channel alone, and an active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `view_child(x.field, i, args...)`. This is `port_opt_view(x.field, i, args...)` for every value except a precomputed optimisation result, which it keeps with no change. The channel forwards the threaded tail and **no** keywords.
 4. The generated method builds the struct again with its keyword constructor, so every check of the constructor runs again on the propagated value.

Step 3 gives the meaning of the tag. `@vprop` is in one channel, so it has one transform and one meaning. The index that the method threads selects **assets**. The observation axis has its own verb, [`obs_weights_view`](@ref).

The body of this macro raises an error. Julia runs it only when the tag is outside a [`@propagatable`](@ref) struct body, where no macro removed it.

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

Field tag for a [`@propagatable`](@ref) struct body. It marks the field as
**prior-selected**. A call `factory(x, pr::AbstractPriorResult, …)` sets the field to
`sel(getfield(x, :field), getproperty(pr, :field))`. This is the value of the risk
measure when it is not `nothing`, else the moment of the same name from the prior result.

A field can carry this tag together with [`@wprop`](@ref) or [`@fprop`](@ref). For
example, `@pprop @wprop w` gives a weights field a prior method and an `ObsWeights`
method. `@pprop` wins in the prior method. Do not put this tag and [`@cprop`](@ref) on
one field, because a value comes from one source. No check refuses the pair, and
`@pprop` wins because it is first in the precedence of the `prior` channel.

# Algorithm

The tag never expands, because [`@propagatable`](@ref) runs first and removes it. The steps below tell what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:pprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. This tag gates the `prior` channel alone, and an active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `sel(x.field, getproperty(pr, :field))`. The prior result gives the property of the **same name** as the field, so the tag names no source of its own.
 4. The generated method builds the struct again with its keyword constructor, so every check of the constructor runs again on the propagated value.

Step 3 gives the meaning of the tag. `@pprop` is first in the precedence of the `prior` channel. So a field with `@pprop` and [`@fprop`](@ref) takes the prior transform on that channel and the factory transform on the `factory` channel.

The body of this macro raises an error. Julia runs it only when the tag is outside a [`@propagatable`](@ref) struct body, where no macro removed it.

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

Field tag for a [`@propagatable`](@ref) struct body. It marks the field as
**context-selected**. A call `factory(x, pr::AbstractPriorResult, …)` sets the field to
`sel(getfield(x, :field), _ctx(args...))`. This is the value of the risk measure when it
is not `nothing`, else the threaded optimiser value, a solver, that `_ctx` finds by type in
the tail. The `slv` fields use this tag, because their source is a threaded argument and
not the prior. Do not put this tag and [`@pprop`](@ref) on one field.

# Algorithm

The tag never expands, because [`@propagatable`](@ref) runs first and removes it. The steps below tell what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:cprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. This tag gates the `prior` channel alone, and an active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `sel(x.field, _ctx(args...))`. `_ctx` finds the value **by type** in the threaded tail, so the source is an argument and not the prior result.
 4. The generated method builds the struct again with its keyword constructor, so every check of the constructor runs again on the propagated value.

Step 3 gives the meaning of the tag. `@cprop` comes after [`@pprop`](@ref) in the precedence of the `prior` channel. If a field carries both, `@pprop` wins and the field ignores the context.

The body of this macro raises an error. Julia runs it only when the tag is outside a [`@propagatable`](@ref) struct body, where no macro removed it.

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

Field tag for a [`@propagatable`](@ref) struct body. It marks the field as an
**observation-weights slot**. A call `factory(x, w::ObsWeights, …)` **replaces** the field
with the incoming weights through `_wprop`. When no [`ObsWeights`](@ref) is threaded, the
field keeps its value.

This tag is different from [`@fprop`](@ref), which recurses into a sub-estimator and keeps
a `nothing` value. A weights field has the default `nothing`, which means uniform weights,
and it must take the incoming weights, which the `nothing` rule of `@fprop` does not do. Use
`@wprop` for the weights field `w`, and `@fprop` for sub-estimators.

# Algorithm

The tag never expands, because [`@propagatable`](@ref) runs first and removes it. The steps below tell what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:wprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.

 2. [`prop_channel_active`](@ref) reads the recorded name. This tag gates the `factory` channel and the `obs` channel, and each active channel makes [`@propagatable`](@ref) emit one method.

 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value:

      + `factory` channel: `_wprop(x.field, args...; kwargs...)`, which **replaces** the field with an incoming [`ObsWeights`](@ref), and keeps it when none is threaded.
      + `obs` channel: `nothing_scalar_array_getindex(x.field, i)`, which **indexes** the value of the field at the selected observations. An index keeps the `AbstractWeights` subtype, and a view does not.

 4. The generated method builds the struct again with its keyword constructor, so every check of the constructor runs again on the propagated value.

Step 3 gives the meaning of the tag. **The two channels do different operations on the same field**, so [`factory`](@ref) and [`obs_weights_view`](@ref) are not two names for one operation. `@wprop` is also the only tag that gates the `obs` channel. One `@wprop` field gives the struct an [`obs_weights_view`](@ref) method, with no second tag.

The body of this macro raises an error. Julia runs it only when the tag is outside a [`@propagatable`](@ref) struct body, where no macro removed it.

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
names it**. A row without one of the three is a tag that parses but never propagates, and
the table exists to stop that failure. A tag means what its channel says it means, so the
function probes each channel. That probe finds a tag that a second channel names without a
transform there. The function collects all the violations and reports them together.

The module calls this function once, at its end. It throws an
[`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError) that lists
every violation, so the package does not precompile with a tag that never propagates.

The three tables are arguments, and each one has the table of the module as its default.
The tables of the module are complete, so the call of the module never reports a violation.
A caller can pass a table of its own to make each of the three clauses fail and read the
message that it gives.

# Algorithm

 1. Make `violations`, an empty vector of strings.
 2. For each tag of [`PROP_TAG_NAMES`](@ref), with its macro name from [`PROP_TAG_MACRO_NAMES`](@ref):
     1. The macro name is not defined in `mod`: push a message that the tag declares no stub macro.
     2. The tag is in the `precedence` of no channel of `channels`: push a message that the tag is in no channel.
     3. For each channel whose `precedence` names the tag, call [`prop_tag_expr`](@ref) with the probe name `:probe`. When that call raises an error, push a message that names the tag and the channel.
 3. `violations` is not empty: throw an `ArgumentError` that lists all of them.
 4. Return `nothing`.

Step 2.3 probes **each** channel that names the tag, not the tag alone. This probe finds a tag that a second channel names with no transform there, and a probe of the tag alone does not.

# Arguments

  - `tags`: The tag names to check.
  - `macro_names`: The stub macro name of each tag, in the order of `tags`.
  - `channels`: The channel table whose `precedence` tuples the function reads.
  - `mod::Module`: The module in which the function finds the stub macros. It also qualifies the names that [`prop_tag_expr`](@ref) emits.

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

Every type declared with [`@propagatable`](@ref), with the names of its `@pprop` fields.

The macro adds one entry immediately after the struct that it declares. When this module
finishes its load, the list holds every type of this package, and
[`check_propagatable_contracts`](@ref) reads it at that point. An external package adds its
types when it loads, after that check, and it must call
[`check_propagatable_contracts`](@ref) at the end of its own module.

# Related

  - [`@propagatable`](@ref)
  - [`propagatable_register!`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
const PROPAGATABLE_CONTRACTS = Vector{Tuple{Type, Tuple{Vararg{Symbol}}}}()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Record the type `T` and the names of its `@pprop` fields in [`PROPAGATABLE_CONTRACTS`](@ref).

[`@propagatable`](@ref) calls it at the declaration. The function only records. The outer
keyword constructor is written *below* the struct, so it does not exist yet, and the
function cannot check it. [`check_propagatable_contracts`](@ref) does the check when the
module is complete.

# Algorithm

 1. Push the pair `(T, pprops)` onto [`PROPAGATABLE_CONTRACTS`](@ref).
 2. Return `nothing`.

`T` has `@nospecialize`, so one method serves every registered type, and the registration costs no compilation.

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

Return the keyword names that the outer constructors of `T` accept, as the union over its methods.

The function drops a `kwargs...` slurp and does not count it. A slurp accepts
`field = value` and then discards it without an error. The contract check exists to find
that failure, so a slurp must not satisfy the contract.

# Algorithm

 1. Make `kws`, an empty vector of symbols.
 2. For each method `m` of the constructor `T`, append `Base.kwarg_decl(m)` to `kws`. The union is over every outer constructor, so a keyword that any one of them names counts.
 3. Remove the repeats from `kws`.
 4. Remove every name whose string ends in `...`, because `Base.kwarg_decl` gives a slurp in that form.
 5. Return `kws`.

Step 4 is the reason for the function. A constructor with `kwargs...` gives the slurp as a keyword name. If the function counted it, every field would satisfy the contract.

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

Return the broken clauses of the [`@propagatable`](@ref) contract of one type, as messages.

The function checks two clauses. Both come from the code that the macro emits:

  - **Every field name is a keyword of the outer constructor.** Each generated method builds
    the struct again with `StructName(; field = …)` over *all* fields, with or without a
    tag. One field that the keyword constructor does not name causes a `MethodError` at
    the first call of [`factory`](@ref) or [`port_opt_view`](@ref).
  - **Every `@pprop` field is a property of a prior result.** The generated
    `factory(x, pr::AbstractPriorResult, args...)` reads `getproperty(pr, :field)`. So a
    name that is not in [`prior_result_property_pool`](@ref) throws when a prior is threaded.

Each message has a suggestion from [`suggest_declared_key`](@ref), so a field name with
a typing error names the nearest valid name.

# Algorithm

 1. Make `msgs`, an empty vector of strings.
 2. Read `kws`, the keywords of the outer constructors of `T`, with [`propagatable_keywords`](@ref).
 3. For each field name of `T` that is not in `kws`, push a message that names the type, the field and the [`suggest_declared_key`](@ref) suggestion from `kws`.
 4. For each name of `pprops` that is not in `pool`, push a message that names the type, the field and the suggestion from `pool`.
 5. Return `msgs`.

The function collects every clause, and no clause stops the walk, so one call reports every violation of one type. An empty result means that the contract of the type holds.

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

Define a struct, and make its propagation methods from five field tags. The tags are
independent, and a field can carry more than one:

  - [`@fprop`](@ref) (factory propagation): a call of [`factory`](@ref) calls
    `factory_child` on each tagged field. It passes runtime *values*, such as
    observation weights, prior results and solvers, down the tree of composed structs.
    The macro always makes a `factory(x, args...)` method. When no field carries
    `@fprop` or `@wprop`, that method returns `x`.
  - [`@wprop`](@ref) (weights replacement): an incoming [`ObsWeights`](@ref) argument
    *replaces* each tagged observation-weights field through `_wprop`. When no
    `ObsWeights` is threaded, the field keeps its value. Use `@wprop` for the weights
    field and `@fprop` for sub-estimators. A weights field with the default `nothing`
    must take the incoming weights, but `@fprop` keeps a `nothing` value.
  - [`@vprop`](@ref) (view propagation): a view of the struct at an index selection
    calls [`port_opt_view`](@ref) on each tagged field. The call recurses into composed
    children and views data arrays. The macro makes a `port_opt_view` method **only when
    at least one field carries `@vprop`**.
  - [`@pprop`](@ref) (prior selection): the prior method selects each tagged field
    against the field of the same name of a prior result, through
    `sel(getfield(x, :f), getproperty(pr, :f))`.
  - [`@cprop`](@ref) (context selection): the prior method selects each tagged field
    against a threaded optimiser value, a solver, that it finds by type, through
    `sel(getfield(x, :f), _ctx(args...))`.

When at least one field carries `@pprop` or `@cprop`, the macro makes a second method,
`factory(x, pr::AbstractPriorResult, args...)`. This method selects the `@pprop` and
`@cprop` fields as above. It threads `pr` into the fields that carry only `@fprop`, as
`factory_child(getfield(x, :f), pr, args...)`. A field with both `@pprop` and `@fprop`
takes the prior selection in this method, because `@pprop` wins. Then the method calls
[`resolve_deferred_quantities`](@ref) on the **selected** struct. That call returns the
struct with no change when the type has no deferred slot and no method of its own. So
the Deferred-Quantity resolution runs **last**, and a slot that holds a Deferred Quantity
sees the solver, the observation weights and the children in their last state. This
method is more specific than the general `factory(x, args...)`, so Julia selects it for
every call that passes a prior.

A field with no tag keeps its value in every method, whatever its type. A tag is
explicit, and the author adds it by choice. Do not put `@pprop` and `@cprop` on one
field, because a value comes from one source. The valid stacks are `@pprop @fprop`, for a
sub-estimator, and `@pprop @wprop`, for a weights field.

Two results of the emitted code are **contracts on the declaration**. The library checks
both where the author writes the struct, not at the first call:

  - Every generated method builds the struct again as `StructName(; field = …)` over *all*
    fields, with or without a tag. So **every field name must also be a keyword of the outer
    constructor**. A `kwargs...` slurp does not satisfy this, because it accepts the
    keyword and then discards it.
  - The prior method reads `getproperty(pr, :field)`, so **every `@pprop` field name must be
    a property of a prior result**. See [`prior_result_property_pool`](@ref).

The macro registers each declaration in [`PROPAGATABLE_CONTRACTS`](@ref).
[`check_propagatable_contracts`](@ref) checks the whole registry when the module is
complete. A field name with a typing error fails at precompilation with a suggestion
from [`suggest_declared_key`](@ref). It does not wait to raise a `MethodError` at the first
call of [`factory`](@ref). [`forward_prior`](@ref) uses the same contract.

The set of tags is data. [`PROP_TAG_NAMES`](@ref) holds the rows,
[`prop_tag_expr`](@ref) holds the field transform of each tag, and
[`PROP_TAG_CHANNELS`](@ref) holds the gate and the tag precedence of each channel. So a new
tag is a row of those tables and a stub macro. A new channel is also a row of
[`PROP_TAG_CHANNELS`](@ref), but it needs its own emission step in this macro, because
each channel has its own method head.

The macro works with `@concrete`. Put `@propagatable` on the outside:

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

`@wprop` gates two channels, and they do different operations on the same field.
`factory` **replaces** the field with an incoming [`ObsWeights`](@ref) value.
[`obs_weights_view`](@ref) **indexes** the value of the field at a set of observations.
A weights field needs no second tag to join the view on the observation axis.

The macro adds the generated `factory`, `port_opt_view` and `obs_weights_view` methods to
the functions of `PortfolioOptimisers`. So `@propagatable` also works for a type that an
external package defines.

`Base.@__doc__` forwards a docstring on the declaration to the struct.

# Algorithm

 1. Find the struct with [`propagatable_find_struct`](@ref), which gives `struct_node` and `rebuild`. `rebuild` puts a replacement struct back inside the same chain of wrapping macros.
 2. Read `type_head` and `body` from `struct_node`. Read `struct_name` from `type_head` with [`propagatable_bare_name`](@ref), which drops the type parameters and the supertype.
 3. Parse the body with [`propagatable_parse_body`](@ref). It gives `tagged`, the field names of each tag, `all_fields`, every declared field in declaration order, and `new_body`, the body with no tags.
 4. Build `new_struct` from `new_body`, and `chain` from `rebuild(new_struct)`. `chain` is the original declaration without the tags, so `@concrete` and Julia both see an ordinary struct.
 5. Bind `POMOD` to the module that **defines** the macro, and qualify every emitted name with it. A bare name resolves in the module of the caller. There, `function factory(…)` declares a new function of the caller, and the method never gets to `PortfolioOptimisers.factory`. That failure gives no error, because the declaration compiles and the type never joins the propagation chain.
 6. Emit the `factory` method. When [`prop_channel_active`](@ref) holds for the `factory` channel, the body is a call to the keyword constructor, with the pairs from [`prop_channel_pairs`](@ref). Otherwise, the body is `x`. **The macro always emits this method**, so a [`@propagatable`](@ref) struct with no tags returns itself from [`factory`](@ref).
 7. When the `view` channel is active, emit `port_opt_view(x::StructName, i, args...)`. Its channel threads `i` before `args...`.
 8. When the `obs` channel is active, emit `obs_weights_view(x::StructName, i)`. Its channel threads `i` and takes no tail.
 9. When the `prior` channel is active, emit `factory(x::StructName, pr::AbstractPriorResult, args...; kwargs...)`. Its body selects every tagged field from `x`, and then gives the selected struct to [`resolve_deferred_quantities`](@ref), which makes **the Deferred-Quantity resolution the last operation**. A Deferred Quantity and a **Calibration Rule** see the solver, the observation weights and the children in the state that the optimisation gave them, and a rule can call [`ERM`](@ref) or [`RRM`](@ref). This method is more specific than the method of step 6, so Julia selects it for a call that threads a prior.
10. Build `pprop_tuple`, the names of the `@pprop` fields as a tuple of quoted symbols.
11. Return one escaped block. It holds `Base.@__doc__ chain`, so a docstring on the declaration gets to the struct. Then it holds the emitted methods, and last the call to [`propagatable_register!`](@ref) that records the type and `pprop_tuple`.

Steps 6 to 9 differ only in the method head and in the arguments that the channel threads. Each step reads its gate and its tag precedence from [`PROP_TAG_CHANNELS`](@ref). A new tag is a row of the tables, a branch in [`prop_tag_expr`](@ref) and a stub macro, but a new channel also needs a new step here.

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

The module calls it once, at its end. So the library enforces the contract of every
generated method where the structs are *declared*, not at the first call of
[`factory`](@ref). The function collects the violations of every type and reports them
together, because a run that stops at the first violation hides the others. See
[`propagatable_contract_violations`](@ref) for the two clauses.

It throws an [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError)
that lists every violation. So the package does not precompile with a type whose generated
methods throw at their first use. A package that declares its own `@propagatable` types
calls this function at the end of its own module to get the same check.

# Algorithm

 1. Read `pool`, the property names that a prior result can carry, with [`prior_result_property_pool`](@ref).
 2. Make `msgs`, an empty vector of strings.
 3. For each pair `(T, pprops)` of [`PROPAGATABLE_CONTRACTS`](@ref), append the messages that [`propagatable_contract_violations`](@ref) reports for that type.
 4. `msgs` is not empty: throw an `ArgumentError` naming the count and listing every message.
 5. Return `nothing`.

Step 3 collects and never stops, so one run reports the violations of every registered type. [`propagatable_register!`](@ref) fills the registry at each declaration, so this function must run **after** the last declaration. For this reason, the module calls it at its end.

The registry and the pool are arguments, and each has the value of the module as its default. A caller can pass a registry of its own to read the message of a broken contract, and register no broken type.

# Arguments

  - `contracts`: The pairs to check, each a type and the names of its `@pprop` fields.
  - `pool`: The property names that a prior result can carry.

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
