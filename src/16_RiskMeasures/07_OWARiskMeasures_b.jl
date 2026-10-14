"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the sorting networks that [`SortingNetworkOrderedWeightsArray`](@ref) relaxes.

A sorting network is a fixed list of comparators. Each comparator reads two positions and writes the smaller value to the first position and the larger value to the second. [goemans2015](@citet) shows that the relaxation of any sorting network is an extended formulation of the permutahedron, so every subtype gives the exact OWA risk. The network sets the number of rows of the model and the cost of one solver iteration.

# Interfaces

In order to implement a new sorting network that works seamlessly with the library, subtype `AbstractSortingNetworkMerge` and implement the following method:

## `sorting_network`

  - `sorting_network(alg::MySortingNetworkMerge, T::Integer) -> Vector{Tuple{Int, Int}}`: Comparator list that sorts `T` values in ascending order.

### Arguments

  - `alg`: The concrete subtype instance.
  - `T`: Number of values to sort.

### Returns

  - `c::Vector{Tuple{Int, Int}}`: Comparators in the order of execution. A pair `(i, j)` puts the smaller value at position `i` and the larger value at position `j`. After the last comparator, the values are in ascending order for every input.

### Examples

```jldoctest
julia> struct MySortingNetworkMerge <: PortfolioOptimisers.AbstractSortingNetworkMerge end

julia> function PortfolioOptimisers.sorting_network(::MySortingNetworkMerge, T::Integer)
           return [(i, j) for j in 2:T for i in 1:(j - 1)]
       end

julia> PortfolioOptimisers.sorting_network(MySortingNetworkMerge(), 3)
3-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (1, 3)
 (2, 3)
```

# Related

  - [`OddEvenMerge`](@ref)
  - [`BitonicMerge`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)

# References

  - $(ref_dict[:goemans2015])
  - $(ref_dict[:batcher1968])
"""
abstract type AbstractSortingNetworkMerge <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sorts with Batcher's odd-even merge network, which has the fewest comparators of the two merge networks.

[batcher1968](@citet) states the network for a power of two. The network here omits each comparator that reads a position past `T`, so it sorts every `T`. It has ``O(T \\log^{2} T)`` comparators: 3756 at `T = 250` and 9511 at `T = 500`.

# Examples

```jldoctest
julia> OddEvenMerge()
OddEvenMerge()
```

# Related

  - [`AbstractSortingNetworkMerge`](@ref)
  - [`BitonicMerge`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
struct OddEvenMerge <: AbstractSortingNetworkMerge end
"""
$(DocStringExtensions.TYPEDEF)

Sorts with Batcher's bitonic merge network, which has a regular structure that a solver factorises with less work.

[batcher1968](@citet) states the network for a power of two. The network here splits a sequence of any length `T` into a descending half and an ascending half, and merges the two halves with the largest power of two below the length as the comparator span. It has ``O(T \\log^{2} T)`` comparators: 4405 at `T = 250` and 11026 at `T = 500`, which is more than [`OddEvenMerge`](@ref).

# Examples

```jldoctest
julia> BitonicMerge()
BitonicMerge()
```

# Related

  - [`AbstractSortingNetworkMerge`](@ref)
  - [`OddEvenMerge`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
struct BitonicMerge <: AbstractSortingNetworkMerge end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the comparators of Batcher's odd-even merge network for `T` values.

# Arguments

  - `alg`: The odd-even merge network.
  - `T`: Number of values to sort.

# Returns

  - `c::Vector{Tuple{Int, Int}}`: Comparators in the order of execution. A pair `(i, j)` puts the smaller value at position `i` and the larger value at position `j`.

# Examples

```jldoctest
julia> PortfolioOptimisers.sorting_network(OddEvenMerge(), 4)
5-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (3, 4)
 (1, 3)
 (2, 4)
 (2, 3)
```

# Related

  - [`OddEvenMerge`](@ref)
  - [`AbstractSortingNetworkMerge`](@ref)
  - [`odd_even_merge_pass!`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
function sorting_network(::OddEvenMerge, T::Integer)
    c = Tuple{Int, Int}[]
    p = 1
    while p < T
        k = p
        while k >= 1
            odd_even_merge_pass!(c, T, p, k)
            k = div(k, 2)
        end
        p *= 2
    end
    return c
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Append to `c` the comparators of one pass of Batcher's odd-even merge network.

The pass compares the positions that are `k` apart and that are in the same block of `2p` positions. A comparator that reads a position past `T` is omitted.

# Arguments

  - `c`: Comparator list, changed in place.
  - `T`: Number of values to sort.
  - `p`: Half the size of the blocks that the pass merges.
  - `k`: Distance between the two positions of each comparator.

# Returns

  - `c::Vector{Tuple{Int, Int}}`: The comparator list.

# Related

  - [`sorting_network`](@ref)
  - [`OddEvenMerge`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
function odd_even_merge_pass!(c::AbstractVector{Tuple{Int, Int}}, T::Integer, p::Integer,
                              k::Integer)
    for j in (k % p):(2 * k):(T - k - 1)
        for i in 0:(min(k, T - j - k) - 1)
            if div(i + j, 2 * p) == div(i + j + k, 2 * p)
                push!(c, (i + j + 1, i + j + k + 1))
            end
        end
    end
    return c
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the comparators of Batcher's bitonic merge network for `T` values.

# Arguments

  - `alg`: The bitonic merge network.
  - `T`: Number of values to sort.

# Returns

  - `c::Vector{Tuple{Int, Int}}`: Comparators in the order of execution. A pair `(i, j)` puts the smaller value at position `i` and the larger value at position `j`.

# Examples

```jldoctest
julia> PortfolioOptimisers.sorting_network(BitonicMerge(), 3)
3-element Vector{Tuple{Int64, Int64}}:
 (2, 3)
 (1, 3)
 (1, 2)
```

# Related

  - [`BitonicMerge`](@ref)
  - [`AbstractSortingNetworkMerge`](@ref)
  - [`bitonic_sort!`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
function sorting_network(::BitonicMerge, T::Integer)
    return bitonic_sort!(Tuple{Int, Int}[], 1, T, true)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Append to `c` the comparators that sort the `m` positions from `lo` with a bitonic merge network.

The first half is sorted in the opposite direction and the second half in the direction `up`, so the two halves make a bitonic sequence. [`bitonic_merge!`](@ref) then merges it.

# Arguments

  - `c`: Comparator list, changed in place.
  - `lo`: First position of the sequence.
  - `m`: Length of the sequence.
  - `up`: If `true`, the sequence is sorted in ascending order. If `false`, in descending order.

# Returns

  - `c::Vector{Tuple{Int, Int}}`: The comparator list.

# Related

  - [`sorting_network`](@ref)
  - [`bitonic_merge!`](@ref)
  - [`BitonicMerge`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
function bitonic_sort!(c::AbstractVector{Tuple{Int, Int}}, lo::Integer, m::Integer,
                       up::Bool)
    if m > 1
        h = div(m, 2)
        bitonic_sort!(c, lo, h, !up)
        bitonic_sort!(c, lo + h, m - h, up)
        bitonic_merge!(c, lo, m, up)
    end
    return c
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Append to `c` the comparators that merge the bitonic sequence of `m` positions from `lo`.

The comparator span is the largest power of two below `m`, so the merge works for every length.

# Arguments

  - `c`: Comparator list, changed in place.
  - `lo`: First position of the sequence.
  - `m`: Length of the sequence.
  - `up`: If `true`, the merge puts the smaller value first. If `false`, it puts the larger value first.

# Returns

  - `c::Vector{Tuple{Int, Int}}`: The comparator list.

# Related

  - [`bitonic_sort!`](@ref)
  - [`BitonicMerge`](@ref)

# References

  - $(ref_dict[:batcher1968])
"""
function bitonic_merge!(c::AbstractVector{Tuple{Int, Int}}, lo::Integer, m::Integer,
                        up::Bool)
    if m > 1
        p = prevpow(2, m - 1)
        for i in lo:(lo + m - p - 1)
            push!(c, up ? (i, i + p) : (i + p, i))
        end
        bitonic_merge!(c, lo, p, up)
        bitonic_merge!(c, lo + p, m - p, up)
    end
    return c
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the orders of the network variables of [`SortingNetworkOrderedWeightsArray`](@ref).

The order changes the columns of the model, not the risk. A solver that permutes the system before it factorises it gives the same cost for every order.

# Interfaces

In order to implement a new order that works seamlessly with the library, subtype `AbstractSortingNetworkOrdering` and implement the following method:

## `sorting_network_columns`

  - `sorting_network_columns(alg::MySortingNetworkOrdering, c::AbstractVector{Tuple{Int, Int}}) -> Vector{Int}`: Column of each network value.

### Arguments

  - `alg`: The concrete subtype instance.
  - `c`: Comparators of the network, from [`sorting_network`](@ref).

### Returns

  - `cols::Vector{Int}`: A permutation of `1:(2 * length(c))`. Comparator `k` writes the values `2k - 1`, the smaller one, and `2k`, the larger one. Value `m` is the network variable `cols[m]`.

### Examples

```jldoctest
julia> struct MySortingNetworkOrdering <: PortfolioOptimisers.AbstractSortingNetworkOrdering end

julia> function PortfolioOptimisers.sorting_network_columns(::MySortingNetworkOrdering,
                                                            c::AbstractVector{Tuple{Int, Int}})
           return collect((2 * length(c)):-1:1)
       end

julia> PortfolioOptimisers.sorting_network_columns(MySortingNetworkOrdering(), [(1, 2)])
2-element Vector{Int64}:
 2
 1
```

# Related

  - [`ComparatorOrdering`](@ref)
  - [`WireOrdering`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
"""
abstract type AbstractSortingNetworkOrdering <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Puts the network variables in the order of the comparators that write them.

The two values of each comparator are adjacent, and the comparators follow the order of execution.

# Examples

```jldoctest
julia> ComparatorOrdering()
ComparatorOrdering()
```

# Related

  - [`AbstractSortingNetworkOrdering`](@ref)
  - [`WireOrdering`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
"""
struct ComparatorOrdering <: AbstractSortingNetworkOrdering end
"""
$(DocStringExtensions.TYPEDEF)

Puts the network variables in the order of the positions they hold, one position after the other.

The values that one position holds are adjacent, in the order that the network writes them.

# Examples

```jldoctest
julia> WireOrdering()
WireOrdering()
```

# Related

  - [`AbstractSortingNetworkOrdering`](@ref)
  - [`ComparatorOrdering`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
"""
struct WireOrdering <: AbstractSortingNetworkOrdering end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the columns of the network values in the order of the comparators that write them.

# Arguments

  - `alg`: The comparator order.
  - `c`: Comparators of the network.

# Returns

  - `cols::Vector{Int}`: `1:(2 * length(c))`, as a vector.

# Examples

```jldoctest
julia> PortfolioOptimisers.sorting_network_columns(ComparatorOrdering(), [(1, 2), (2, 3)])
4-element Vector{Int64}:
 1
 2
 3
 4
```

# Related

  - [`ComparatorOrdering`](@ref)
  - [`AbstractSortingNetworkOrdering`](@ref)
"""
function sorting_network_columns(::ComparatorOrdering, c::AbstractVector{Tuple{Int, Int}})
    return collect(1:(2 * length(c)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the columns of the network values in the order of the positions they hold.

A stable sort of the values by their position keeps the values of one position in the order that the network writes them.

# Arguments

  - `alg`: The position order.
  - `c`: Comparators of the network.

# Returns

  - `cols::Vector{Int}`: A permutation of `1:(2 * length(c))`.

# Examples

```jldoctest
julia> PortfolioOptimisers.sorting_network_columns(WireOrdering(), [(1, 2), (2, 3)])
4-element Vector{Int64}:
 1
 2
 3
 4

julia> PortfolioOptimisers.sorting_network_columns(WireOrdering(), [(2, 3), (1, 2)])
4-element Vector{Int64}:
 2
 4
 1
 3
```

# Related

  - [`WireOrdering`](@ref)
  - [`AbstractSortingNetworkOrdering`](@ref)
"""
function sorting_network_columns(::WireOrdering, c::AbstractVector{Tuple{Int, Int}})
    wire = Vector{Int}(undef, 2 * length(c))
    for (k, (i, j)) in enumerate(c)
        wire[2 * k - 1] = i
        wire[2 * k] = j
    end
    return invperm(sortperm(wire; alg = Base.Sort.DEFAULT_STABLE))
end
"""
$(DocStringExtensions.TYPEDEF)

OWA formulation that computes the exact OWA risk through a relaxed sorting network of ``O(T \\log^{2} T)`` linear rows.

The formulation runs the net portfolio returns through a sorting network, and relaxes each comparator to three linear rows. [goemans2015](@citet) shows that the relaxation of any sorting network is an extended formulation of the permutahedron. Its linear-programming dual is this formulation. For a nondecreasing weight vector, the minimum of the weighted network outputs is the OWA risk. So the risk is exact, as the risk of [`ExactOrderedWeightsArray`](@ref) is, but the model has ``O(T \\log^{2} T)`` rows in place of ``T^{2}``. It is the exact formulation for a long sample.

The formulation needs a nondecreasing weight vector, which is also the condition for the risk measure to be convex. On any other vector the relaxed network has no minimum, so the constructor of the risk measure refuses such a vector.

One interior-point iteration costs one factorisation of the KKT matrix, and the fill of that factorisation sets the time of the solve, not the row count. QDLDL, the default linear solver of Clarabel, factorises the matrix one column at a time. A supernodal factorisation of the same matrix cuts the time of one iteration by 83 to 85 % at ``T = 500`` and by 90 % at ``T = 1000``. The iterations and the risk do not change.

Two setups give a supernodal factorisation. The times below are for the minimum risk of the Gini mean difference weights, `owa_gmd`, on eight assets, with the default network, one thread and tolerances of `1e-9`.

  - Clarabel with the setting `direct_solve_method = :mkl` factorises with MKL Pardiso. It needs `using Pardiso`, and it runs on x86-64 only. It solved ``T = 500`` in 4.5 to 4.7 s against 26.5 to 30.5 s with QDLDL, and ``T = 1000`` in 28.4 s against 298.5 s.
  - Tulip with `KKT_System = Tulip.KKT.K1()` and `KKT_Backend = Tulip.KKT.TlpCholmod.Backend()` factorises the normal equations with the supernodal Cholesky factorisation of CHOLMOD. It needs no binary past CHOLMOD. It solved ``T = 500`` in 6.3 s and ``T = 1000`` in 39.9 s.

# Mathematical definition

```math
\\begin{align}
\\mathrm{OWA}(\\boldsymbol{w}) &= \\begin{cases}
\\underset{\\boldsymbol{z}}{\\min} & s\\, \\boldsymbol{\\omega}^{\\intercal} \\boldsymbol{z}_{\\mathrm{out}}\\\\
\\text{s.t.} & z_{l(k)} + z_{h(k)} = z_{a(k)} + z_{b(k)} \\quad \\forall k = 1,\\, \\ldots,\\, K\\\\
 & z_{h(k)} \\geq z_{a(k)},\\; z_{h(k)} \\geq z_{b(k)} \\quad \\forall k = 1,\\, \\ldots,\\, K\\\\
 & \\boldsymbol{z}_{\\mathrm{in}} = \\hat{\\boldsymbol{r}} / s\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\boldsymbol{\\omega}``: Is the nondecreasing OWA weight vector.
  - ``\\hat{\\boldsymbol{r}}``: Is the `T × 1` vector of net portfolio returns.
  - ``s``: Is the largest absolute asset return, which scales the network values towards one.
  - ``K``: Is the number of comparators of the network.
  - ``a(k),\\, b(k)``: Are the two values that comparator ``k`` reads.
  - ``l(k),\\, h(k)``: Are the smaller and the larger value that comparator ``k`` writes.
  - ``\\boldsymbol{z}_{\\mathrm{in}},\\, \\boldsymbol{z}_{\\mathrm{out}}``: Are the values that the network reads and the last value that it writes to each position, in ascending position.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SortingNetworkOrderedWeightsArray(;
        merge::AbstractSortingNetworkMerge = BitonicMerge(),
        ordering::AbstractSortingNetworkOrdering = ComparatorOrdering()
    ) -> SortingNetworkOrderedWeightsArray

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> SortingNetworkOrderedWeightsArray()
SortingNetworkOrderedWeightsArray
     merge ┼ BitonicMerge()
  ordering ┴ ComparatorOrdering()
```

# Related

  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`AbstractSortingNetworkMerge`](@ref)
  - [`AbstractSortingNetworkOrdering`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)

# References

  - $(ref_dict[:goemans2015])
  - $(ref_dict[:batcher1968])
"""
@concrete struct SortingNetworkOrderedWeightsArray <: OrderedWeightsArrayFormulation
    """
    $(field_dict[:merge_owa])
    """
    merge
    """
    $(field_dict[:ordering_owa])
    """
    ordering
    function SortingNetworkOrderedWeightsArray(merge::AbstractSortingNetworkMerge,
                                               ordering::AbstractSortingNetworkOrdering)
        return new{typeof(merge), typeof(ordering)}(merge, ordering)
    end
end
function SortingNetworkOrderedWeightsArray(;
                                           merge::AbstractSortingNetworkMerge = BitonicMerge(),
                                           ordering::AbstractSortingNetworkOrdering = ComparatorOrdering())
    return SortingNetworkOrderedWeightsArray(merge, ordering)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse an OWA weight vector `w` that is not nondecreasing.

The relaxed sorting network lets a comparator raise its larger value and lower its smaller value by the same amount. A weight vector that decreases between two positions then lowers the risk without a limit, so the model has no minimum.

# Arguments

  - `alg`: The sorting-network formulation.
  - `w`: OWA weight vector, after the range measure reverses its second vector.

# Validation

  - `issorted(w)`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayRange`](@ref)
"""
function assert_owa_formulation_weights(::SortingNetworkOrderedWeightsArray, w::VecNum)
    @argcheck(issorted(w),
              DomainError(w,
                          "SortingNetworkOrderedWeightsArray needs a nondecreasing weight vector. The relaxed sorting network has no minimum for a weight vector that decreases."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Ordered Weights Array Range (OWA Range) risk measure.

Computes portfolio risk as the difference between two OWA linear combinations of sorted portfolio returns, providing a range-based risk measure.

The constructor reverses `w2` unless the caller declares that it is already reversed (see `rev`). That reversal is what makes the *difference* of the two weight vectors a *range*: `w1` addresses the lower tail of the sorted returns, and the reversed `w2` addresses the upper tail, so `w1 - w2` sums the two tails rather than cancelling them. It is the weight-space form of the convention [`ValueatRiskRange`](@ref) states on the returns themselves.

# Mathematical definition

```math
\\begin{align}
\\mathrm{OWARange}_{\\boldsymbol{w}_{1},\\boldsymbol{w}_{2}}(\\boldsymbol{x}) &= \\sum_{t=1}^{T} \\left(w_{1,t} - w_{2,t}\\right) x_{(t)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{OWARange}_{\\boldsymbol{w}_{1},\\boldsymbol{w}_{2}}(\\boldsymbol{x})``: Ordered weights array range.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\boldsymbol{w}_{1}``: Lower-tail OWA weight vector ``T \\times 1``.
  - ``\\boldsymbol{w}_{2}``: Upper-tail OWA weight vector ``T \\times 1``, held reversed by the constructor.
  - ``x_{(t)}``: ``t``-th smallest entry of ``\\boldsymbol{x}``, so ``x_{(1)} \\leq \\dots \\leq x_{(T)}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w1::OWA_Func_VecNum = owa_tg,
        w2::OWA_Func_VecNum = owa_tg,
        alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
        rev::Bool = false
    ) -> OrderedWeightsArrayRange

Keywords correspond to the struct's fields.

## Validation

  - If `w1` is a `VecNum`: `!isempty(w1)`.
  - If `w2` is a `VecNum`: `!isempty(w2)`.
  - If both `w1` and `w2` are `VecNum`: `length(w1) == length(w2)`.
  - If both `w1` and `w2` are `VecNum` and `alg` is a [`SortingNetworkOrderedWeightsArray`](@ref): `issorted(w1 - w2)`, with `w2` reversed. Raises a `DomainError`. A callable is checked when the model is built.

# Functor

    (r::OrderedWeightsArrayRange)(x::VecNum)

Computes the OWA range of a portfolio returns vector `x`, as the sorted returns weighted by `w1 - w2`. A `w1` or `w2` that is a callable is evaluated at `length(x)` first, so the weight vectors always match the sample.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> OrderedWeightsArrayRange()
OrderedWeightsArrayRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        w1 ┼ typeof(owa_tg): PortfolioOptimisers.owa_tg
        w2 ┼ ComposedFunction{typeof(reverse), typeof(owa_tg)}: reverse ∘ PortfolioOptimisers.owa_tg
       alg ┼ ApproxOrderedWeightsArray
           │   p ┴ Vector{Float64}: [2.0, 3.0, 4.0, 10.0, 50.0]
       rev ┴ Bool: true
```

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`RiskMeasureSettings`](@ref)

# References

  - $(ref_dict[:owaog])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.2.3.
"""
@concrete struct OrderedWeightsArrayRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w1_owa])
    """
    w1
    """
    $(field_dict[:w2_owa])
    """
    w2
    """
    $(field_dict[:alg])
    """
    alg
    """
    $(field_dict[:rev_owa])
    """
    rev
    function OrderedWeightsArrayRange(settings::RiskMeasureSettings, w1::OWA_Func_VecNum,
                                      w2::OWA_Func_VecNum,
                                      alg::OrderedWeightsArrayFormulation, rev::Bool)
        w1_flag = isa(w1, VecNum)
        w2_flag = isa(w2, VecNum)
        if w1_flag
            @argcheck(!isempty(w1), IsEmptyError("w1 cannot be empty"))
        end
        if w2_flag
            @argcheck(!isempty(w2), IsEmptyError("w2 cannot be empty"))
            # Non-mutating on purpose: `w2` belongs to the caller. An in-place reverse here
            # corrupts their vector, and building two measures from one vector reverses it
            # twice. It also makes the `w1 === w2` aliasing case need no special handling.
            if !rev
                w2 = reverse(w2)
            end
        else
            # Same rule for the builder branch: the closure is the caller's, so a cached
            # return value must not be reversed underneath them.
            if !rev
                w2 = reverse ∘ w2
            end
        end
        if w1_flag && w2_flag
            @argcheck(length(w1) == length(w2),
                      DimensionMismatch("w1 ($(length(w1))) must match w2 ($(length(w2)))"))
            assert_owa_formulation_weights(alg, w1 - w2)
        end
        return new{typeof(settings), typeof(w1), typeof(w2), typeof(alg), typeof(rev)}(settings,
                                                                                       w1,
                                                                                       w2,
                                                                                       alg,
                                                                                       true)
    end
end
function OrderedWeightsArrayRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  w1::OWA_Func_VecNum = owa_tg,
                                  w2::OWA_Func_VecNum = owa_tg,
                                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                                  rev::Bool = false)
    return OrderedWeightsArrayRange(settings, w1, w2, alg, rev)
end
# Deferrable slots — see `deferred_slots`. Both builders carry them. `rev` is stored as a
# done flag, so the positional rebuild of `rebuild_with_slots` passes `true` and reverses
# nothing a second time.
deferred_slots(r::OrderedWeightsArrayRange) = (; w1 = r.w1, w2 = r.w2)
# Calibration slots — see `calibration_slots`. The container names its two children, on the
# terms the single-builder container above states.
function calibration_slots(r::OrderedWeightsArrayRange)
    return merge(isa(r.w1, OWA_CalOccupant) ? (; w1 = r.w1) : (;),
                 isa(r.w2, OWA_CalOccupant) ? (; w2 = r.w2) : (;))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two weight builders of an [`OrderedWeightsArrayRange`](@ref) against prior result `pr`.

It carries the reading of [`factory`](@ref) on [`OrderedWeightsArray`](@ref) unchanged, over two builders instead of one.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
"""
function factory(x::OrderedWeightsArrayRange, pr::AbstractPriorResult, args...; kwargs...)
    return resolve_deferred_quantities(x, pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the weight builder inside the reversal that [`OrderedWeightsArrayRange`](@ref) wraps its `w2` in.

The Range constructor stores `w2` as `reverse ∘ w2` whenever the caller has not already reversed it, so the object the `w2` slot holds is a composition and the builder a rule sits in is its inner half. Without this method the recursion would stop at the composition and a rule in the gain-side builder would never resolve, while the same rule in `w1` did.

The composition is rebuilt around the resolved half, so the reversal survives. The bound names `reverse` and an [`AbstractOrderedWeightsArrayFunction`](@ref), so no other composition reaches it.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::ComposedFunction{typeof(reverse),
                                                         <:AbstractOrderedWeightsArrayFunction},
                                     pr::AbstractPriorResult, slv = nothing)
    inner = resolve_deferred_quantities(x.inner, pr, slv)
    return inner === x.inner ? x : reverse ∘ inner
end
# Calibration slots — see `calibration_slots`. The reversal states no quantity of its own,
# so it names the builder it wraps and that builder names its own slots.
calibration_slots(x::OWA_RevFunc) = (; inner = x.inner)
# Tail decomposition — see `range_tails`. Declared for the approximate formulation only: the
# exact one collapses both tails into a single constraint on `w1 - w2`, so it fuses rather
# than duplicating. `w2` is already reversed by the constructor (`rev` is a done flag), so
# each tail is an ordinary `OrderedWeightsArray` over its own weight vector.
function range_tails(r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                 <:ApproxOrderedWeightsArray})
    settings = RiskMeasureSettings(; rke = false)
    return (; loss = OrderedWeightsArray(; settings = settings, w = r.w1, alg = r.alg),
            gain = OrderedWeightsArray(; settings = settings, w = r.w2, alg = r.alg))
end
function (r::OrderedWeightsArrayRange)(x::VecNum)
    w1 = isa(r.w1, VecNum) ? r.w1 : r.w1(length(x))
    w2 = isa(r.w2, VecNum) ? r.w2 : r.w2(length(x))
    w = w1 - w2
    return LinearAlgebra.dot(w, sort(x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::OrderedWeightsArray) = NetReturnsInput()
risk_input_kind(::OrderedWeightsArrayRange) = NetReturnsInput()

# The `# Interfaces`-marked type of #1137 (ADR 0154). Its section names only the callable
# shape, so no separate verb is promoted alongside it.

export OrderedWeightsArrayRange, OddEvenMerge, BitonicMerge, ComparatorOrdering,
       WireOrdering, SortingNetworkOrderedWeightsArray
public AbstractSortingNetworkMerge, sorting_network, AbstractSortingNetworkOrdering,
       sorting_network_columns
