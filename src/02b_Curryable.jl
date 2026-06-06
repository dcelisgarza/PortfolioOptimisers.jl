# ---------------------------------------------------------------------------
# @curryable — struct-definition macro for factory propagation
# ---------------------------------------------------------------------------

# --- private AST helpers ----------------------------------------------------

_is_c_macro(x) = x == Symbol("@c") || (x isa GlobalRef && x.name == Symbol("@c"))

_is_doc_macro(x) = (x isa GlobalRef && x.name == Symbol("@doc")) || x == Symbol("@doc")

function _extract_field_name(expr)
    if expr isa Symbol
        return expr
    end
    if expr isa Expr && expr.head == :(::)
        return expr.args[1]
    end
    return error("@curryable: @c must precede a bare field name or field::Type, got: $(repr(expr))")
end

# Recursively unwrap macrocall chains until the :struct node is found.
# Returns (struct_node, rebuild_fn) where rebuild_fn(new_struct) reassembles
# the original macro chain with new_struct in place of the original struct.
function _curryable_find_struct(expr)
    if !(expr isa Expr)
        error("@curryable: expected a struct or macro-wrapped struct, got $(typeof(expr))")
    end
    if expr.head == :struct
        return expr, identity
    elseif expr.head == :macrocall
        inner = expr.args[end]
        struct_node, rebuild = _curryable_find_struct(inner)
        prefix = expr.args[1:(end - 1)]
        return struct_node, s -> Expr(:macrocall, prefix..., rebuild(s))
    else
        error("@curryable: expected a struct definition (possibly wrapped in macros), " *
              "got Expr with head :$(expr.head)")
    end
end

function _curryable_bare_name(n)
    if n isa Symbol
        return n
    end
    if n isa Expr && n.head == :curly
        return _curryable_bare_name(n.args[1])
    end
    if n isa Expr && n.head == :<:
        return _curryable_bare_name(n.args[1])
    end
    return error("@curryable: cannot extract struct name from: $(repr(n))")
end

# Walk the struct body, collecting @c-tagged field names and stripping the tags.
# Handles both bare (@c field) and docstring-prefixed ("doc" \n @c field) forms.
function _curryable_parse_body(body)
    curryable = Symbol[]
    new_args = Any[]
    for arg in body.args
        if arg isa Expr && arg.head == :macrocall
            head = arg.args[1]
            if _is_c_macro(head)
                # Bare @c field — no docstring
                fname = _extract_field_name(arg.args[end])
                push!(curryable, fname)
                push!(new_args, arg.args[end])          # strip @c, keep field expr
            elseif _is_doc_macro(head)
                # Core.@doc "doc" (field or @c(field))
                inner = arg.args[end]
                if inner isa Expr && inner.head == :macrocall && _is_c_macro(inner.args[1])
                    # "doc" \n @c field
                    fname = _extract_field_name(inner.args[end])
                    push!(curryable, fname)
                    # Rebuild @doc node with @c stripped: replace last arg with bare field
                    push!(new_args,
                          Expr(:macrocall, arg.args[1:(end - 1)]..., inner.args[end]))
                else
                    push!(new_args, arg)                # plain docstring'd field
                end
            else
                push!(new_args, arg)
            end
        else
            push!(new_args, arg)                        # LineNumberNode, Symbol, ::, function, …
        end
    end
    return curryable, Expr(:block, new_args...)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    @c field

Field tag for use inside a [`@curryable`](@ref) struct body.
Marks the field as participating in [`factory`](@ref) propagation —
`_factory_child` will be called on it when `factory` is invoked on the
enclosing struct.

Raises an error if used outside a `@curryable` struct body.
"""
macro c(expr)
    return error("@c may only appear inside a @curryable struct body")
end

"""
    @curryable expr

Define a struct and automatically generate a [`factory`](@ref) propagation
method for it.

Fields tagged with [`@c`](@ref) receive `_factory_child` calls when
`factory` is invoked, recursing runtime data (observation weights, prior
results, solvers, …) down the composition tree. Untagged fields pass through
unchanged regardless of their type — tagging is explicit and opt-in.

Composes with `@concrete` (put `@curryable` outermost):

```julia
@curryable @concrete struct MyEstimator <: AbstractEstimator
    @c inner   # factory recurses into this field
    config     # passed through unchanged
    function MyEstimator(inner::AbstractEstimator, config)
        return new{typeof(inner), typeof(config)}(inner, config)
    end
end
```

The generated `factory` method is added to `PortfolioOptimisers.factory`,
so `@curryable` works correctly for types defined in external packages.

Docstrings on the enclosing definition are forwarded correctly via
`Base.@__doc__`.
"""
macro curryable(expr)
    struct_node, rebuild = _curryable_find_struct(expr)

    type_head   = struct_node.args[2]
    body        = struct_node.args[3]
    struct_name = _curryable_bare_name(type_head)

    curryable_fields, new_body = _curryable_parse_body(body)

    new_struct = Expr(:struct, struct_node.args[1], type_head, new_body)
    chain      = rebuild(new_struct)

    if isempty(curryable_fields)
        factory_body = :x
    else
        kw_pairs = [Expr(:kw, f,
                         :(_factory_child($(Expr(:., :x, QuoteNode(f))), args...;
                                          kwargs...))) for f in curryable_fields]
        factory_body = Expr(:call, struct_name, Expr(:parameters, kw_pairs...))
    end

    factory_def = quote
        function factory(x::$struct_name, args...; kwargs...)
            return $factory_body
        end
    end

    return esc(quote
                   Base.@__doc__ $chain
                   $factory_def
               end)
end

# ---------------------------------------------------------------------------
# Dummy type — validates @curryable + @concrete composition in the library
# ---------------------------------------------------------------------------

"""
$(DocStringExtensions.TYPEDEF)

Minimal curryable estimator demonstrating [`@curryable`](@ref) macro
composition with `@concrete`.

Has one `@c`-tagged inner estimator (receives factory propagation) and one
inert scalar field (passed through unchanged).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    _CurryableExample(;
        inner::AbstractEstimator = SimpleExpectedReturns(),
        config = 1
    ) -> _CurryableExample

# Examples

```jldoctest
julia> _CurryableExample()
_CurryableExample
  inner ┼ SimpleExpectedReturns
        │   w ┴ nothing
 config ┴ Int64: 1
```
"""
@curryable @concrete struct _CurryableExample <: AbstractEstimator
    """
    Inner estimator — participates in factory propagation.
    """
    @c inner
    """
    Inert scalar — passed through unchanged by factory.
    """
    config
    function _CurryableExample(inner::AbstractEstimator, config)
        return new{typeof(inner), typeof(config)}(inner, config)
    end
end
function _CurryableExample(; inner::AbstractEstimator = SimpleExpectedReturns(), config = 1)
    return _CurryableExample(inner, config)
end

export @curryable, @c, _CurryableExample
