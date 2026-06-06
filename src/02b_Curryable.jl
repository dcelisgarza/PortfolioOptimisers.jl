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
