"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses two partial-fit states that cannot be merged, naming the reason.

Two states merge only when they describe the same estimate over the same assets, so this check holds for every state family: the two operands are the same struct, and every array field they hold has the same shape. A family whose merge needs more than that adds a method of its own that calls this one first. An exponentially weighted pair is the case that needs it, because two states of different decay weight the same observation differently.

# Algorithm

 1. Refuse the pair when the two operands are not the same struct, whatever their type parameters.
 2. Walk the fields of the pair in declaration order.
 3. Refuse the pair when a field holds an array in both operands and the two shapes differ.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` are the same struct. An `ArgumentError` is thrown otherwise.
  - Every array field of `a` has the shape of its counterpart in `b`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function assert_mergeable_states(a::AbstractPartialFitState, b::AbstractPartialFitState)
    @argcheck(Base.typename(typeof(a)) === Base.typename(typeof(b)),
              ArgumentError("two partial-fit states of different types cannot be merged, but `a` is a `$(typeof(a))` and `b` is a `$(typeof(b))`."))
    for i in 1:nfields(a)
        fa = getfield(a, i)
        fb = getfield(b, i)
        if isa(fa, AbstractArray) && isa(fb, AbstractArray)
            @argcheck(size(fa) == size(fb),
                      DimensionMismatch("two partial-fit states over different numbers of assets cannot be merged, but field `$(fieldname(typeof(a), i))` has size $(size(fa)) in `a` and $(size(fb)) in `b`."))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observation count, the mean and the second-moment accumulator of two blocks into those of the concatenated block.

This is the merge of [chan1983](@cite), and it is what makes an incremental fit parallel and associative: a sample split into any set of disjoint blocks gives the state of the whole sample, whatever order the blocks are folded in. The second moment is the accumulator ``M``, not the variance, so a caller divides by the count or by the count less one after the last merge.

# Mathematical definition

```math
\\begin{align}
n &= n_A + n_B\\\\
\\boldsymbol{\\delta} &= \\boldsymbol{\\mu}_B - \\boldsymbol{\\mu}_A\\\\
\\boldsymbol{\\mu} &= \\boldsymbol{\\mu}_A + \\boldsymbol{\\delta} \\frac{n_B}{n}\\\\
M &= M_A + M_B + \\boldsymbol{\\delta} \\boldsymbol{\\delta}^{\\intercal} \\frac{n_A n_B}{n}\\, .
\\end{align}
```

Where:

  - ``n_A``, ``n_B``: observation counts of the two blocks.
  - ``\\boldsymbol{\\mu}_A``, ``\\boldsymbol{\\mu}_B``: means of the two blocks.
  - ``M_A``, ``M_B``: second-moment accumulators of the two blocks.
  - ``\\boldsymbol{\\delta}``: difference of the two means.

The outer product ``\\boldsymbol{\\delta} \\boldsymbol{\\delta}^{\\intercal}`` becomes the elementwise square ``\\boldsymbol{\\delta}^2`` when ``M`` is per-asset rather than a co-moment matrix.

# Algorithm

 1. Add the two counts, giving `n`.
 2. Subtract the two means, giving `delta`.
 3. Move the first mean along `delta` by the share of the total that the second block holds, giving `mu`.
 4. Add the two accumulators, and add the outer product of `delta` scaled by `n_a * n_b / n`, giving `M`.

# Arguments

  - `n_a`: Observation count of the first block.
  - `mu_a`: Mean of the first block.
  - `M_a`: Second-moment accumulator of the first block.
  - `n_b`: Observation count of the second block.
  - `mu_b`: Mean of the second block.
  - `M_b`: Second-moment accumulator of the second block.

# Validation

  - `n_a + n_b > 0`. A `DomainError` is thrown otherwise, because the merge divides by the total count.

# Returns

  - `(n, mu, M)::Tuple`: The count, the mean and the accumulator of the concatenated block.

# Examples

```jldoctest
julia> PortfolioOptimisers.chan_merge(1, [1.0, 2.0], [0.0, 0.0], 1, [3.0, 6.0], [0.0, 0.0])
(2, [2.0, 4.0], [2.0, 8.0])
```

# References

  - $(ref_dict[:chan1983])

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)
"""
function chan_merge(n_a::Number, mu_a::Union{<:Number, <:VecNum},
                    M_a::Union{<:Number, <:VecNum}, n_b::Number,
                    mu_b::Union{<:Number, <:VecNum}, M_b::Union{<:Number, <:VecNum})
    n = n_a + n_b
    @argcheck(n > zero(n), DomainError(n, "n_a + n_b must be positive"))
    delta = mu_b .- mu_a
    mu = mu_a .+ delta .* (n_b / n)
    M = M_a .+ M_b .+ delta .^ 2 .* (n_a * n_b / n)
    return n, mu, M
end
function chan_merge(n_a::Number, mu_a::VecNum, M_a::MatNum, n_b::Number, mu_b::VecNum,
                    M_b::MatNum)
    n = n_a + n_b
    @argcheck(n > zero(n), DomainError(n, "n_a + n_b must be positive"))
    delta = mu_b .- mu_a
    mu = mu_a .+ delta .* (n_b / n)
    M = M_a .+ M_b .+ (delta * transpose(delta)) .* (n_a * n_b / n)
    return n, mu, M
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Combines two partial-fit states fitted on disjoint blocks of observations into the state of the concatenated block.

Each state family implements its own method, so this generic method is reached only by a pair no family answers. It refuses the pair that cannot merge at all, and names the method the family still owes. The verb is deliberately not `Base.merge`, whose contract on a `Dict` and on a `NamedTuple` is that the right operand wins a key conflict, where this merge is a sum.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref), which names a type mismatch and an asset-count mismatch.
 2. Throw an `ArgumentError` naming the `merge_states` method the state's own family must implement.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).
  - The family of `a` implements `merge_states`. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::AbstractPartialFitState`: The state the two blocks give when they are fitted as one block.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`assert_mergeable_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::AbstractPartialFitState, b::AbstractPartialFitState)
    assert_mergeable_states(a, b)
    name = Base.typename(typeof(a)).name
    return throw(ArgumentError("$name is an AbstractPartialFitState with no `merge_states` method. Implement `merge_states(a::$name, b::$name)`, which calls `assert_mergeable_states` first, refuses any further mismatch of its own, and folds the two states with `chan_merge`. See the `AbstractPartialFitState` docstring for the interface."))
end
