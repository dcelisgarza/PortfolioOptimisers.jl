"""
    abstract type AbstractPhylogenyConstraintEstimator <: AbstractConstraintEstimator end

Abstract supertype for all phylogeny-based constraint estimators in PortfolioOptimisers.jl.

All concrete types representing phylogeny-based constraint estimators should subtype `AbstractPhylogenyConstraintEstimator`. This enables a consistent, composable interface for generating constraints based on phylogenetic, clustering, or network structures among assets.

# Related

  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
abstract type AbstractPhylogenyConstraintEstimator <: AbstractConstraintEstimator end
"""
    abstract type AbstractPhylogenyConstraintResult <: AbstractConstraintResult end

Abstract supertype for all phylogeny-based constraint result types in PortfolioOptimisers.jl.

All concrete types representing the results of phylogeny-based constraint generation should subtype `AbstractPhylogenyConstraintResult`. This enables a consistent, composable interface for storing and propagating constraint matrices, vectors, or other outputs derived from phylogenetic, clustering, or network structures among assets.

# Related

  - [`SemiDefinitePhylogeny`](@ref)
  - [`IntegerPhylogeny`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
abstract type AbstractPhylogenyConstraintResult <: AbstractConstraintResult end
"""
    struct SemiDefinitePhylogenyEstimator{T1, T2} <: AbstractPhylogenyConstraintEstimator
        pe::T1
        p::T2
    end

Estimator for generating semi-definite phylogeny-based constraints in PortfolioOptimisers.jl.

`SemiDefinitePhylogenyEstimator` constructs constraints based on phylogenetic or clustering structures among assets, using a semi-definite matrix representation. The estimator wraps a phylogeny or clustering estimator and a non-negative penalty parameter `p`, which controls the strength of the constraint.

# Fields

  - `pe`: Phylogeny or clustering estimator.
  - `p`: Non-negative penalty parameter for the constraint.

# Constructor

    SemiDefinitePhylogenyEstimator(;
                                   pe::Union{<:AbstractPhylogenyEstimator,
                                             <:AbstractClusteringResult} = NetworkEstimator(),
                                   p::Real = 0.05)

## Validation

  - `p >= 0`.

# Examples

```jldoctest
julia> SemiDefinitePhylogenyEstimator()
SemiDefinitePhylogenyEstimator
  pe ┼ NetworkEstimator
     │    ce ┼ PortfolioOptimisersCovariance
     │       │   ce ┼ Covariance
     │       │      │    me ┼ SimpleExpectedReturns
     │       │      │       │   w ┴ nothing
     │       │      │    ce ┼ GeneralCovariance
     │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │      │       │    w ┴ nothing
     │       │      │   alg ┴ Full()
     │       │   mp ┼ DefaultMatrixProcessing
     │       │      │       pdm ┼ Posdef
     │       │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
     │       │      │   denoise ┼ nothing
     │       │      │    detone ┼ nothing
     │       │      │       alg ┴ nothing
     │    de ┼ Distance
     │       │   power ┼ nothing
     │       │     alg ┴ CanonicalDistance()
     │   alg ┼ KruskalTree
     │       │     args ┼ Tuple{}: ()
     │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │     n ┴ Int64: 1
   p ┴ Float64: 0.05
```

# Related

  - [`SemiDefinitePhylogeny`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
struct SemiDefinitePhylogenyEstimator{T1, T2} <: AbstractPhylogenyConstraintEstimator
    pe::T1
    p::T2
    function SemiDefinitePhylogenyEstimator(pe::Union{<:AbstractPhylogenyEstimator,
                                                      <:AbstractClusteringResult}, p::Real)
        @argcheck(p >= zero(p), DomainError("`p` must be non-negative:\np => $p"))
        return new{typeof(pe), typeof(p)}(pe, p)
    end
end
function SemiDefinitePhylogenyEstimator(;
                                        pe::Union{<:AbstractPhylogenyEstimator,
                                                  <:AbstractClusteringResult} = NetworkEstimator(),
                                        p::Real = 0.05)
    return SemiDefinitePhylogenyEstimator(pe, p)
end
"""
    struct SemiDefinitePhylogeny{T1, T2} <: AbstractPhylogenyConstraintResult
        A::T1
        p::T2
    end

Container for the result of semi-definite phylogeny-based constraint generation.

`SemiDefinitePhylogeny` stores the constraint matrix `A` and penalty parameter `p` resulting from a semi-definite phylogeny constraint estimator. This type is used to encapsulate the output of phylogeny-based constraint routines, enabling composable and modular constraint handling in portfolio optimisation workflows.

# Fields

  - `A`: Phylogeny matrix encoding a relationship graph.
  - `p`: Non-negative penalty parameter controlling the strength of the constraint.

# Constructor

    SemiDefinitePhylogeny(;
                          A::Union{<:PhylogenyResult{<:AbstractMatrix{<:Real}},
                                   <:AbstractMatrix{<:Real}}, p::Real = 0.05)

## Validation

  - `issymmetric(A)` and `all(iszero, diag(A))`.
  - `p >= 0`.

# Examples

```jldoctest
julia> SemiDefinitePhylogeny([0.0 1.0; 1.0 0.0], 0.05)
SemiDefinitePhylogeny
  A ┼ 2×2 Matrix{Float64}
  p ┴ Float64: 0.05
```

# Related

  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
struct SemiDefinitePhylogeny{T1, T2} <: AbstractPhylogenyConstraintResult
    A::T1
    p::T2
    function SemiDefinitePhylogeny(A::AbstractMatrix{<:Real}, p::Real)
        @argcheck(all(iszero, diag(A)))
        @argcheck(issymmetric(A))
        @argcheck(p >= zero(p))
        return new{typeof(A), typeof(p)}(A, p)
    end
end
function SemiDefinitePhylogeny(A::PhylogenyResult{<:AbstractMatrix{<:Real}}, p::Real)
    return SemiDefinitePhylogeny(A.X, p)
end
function SemiDefinitePhylogeny(;
                               A::Union{<:PhylogenyResult{<:AbstractMatrix{<:Real}},
                                        <:AbstractMatrix{<:Real}}, p::Real = 0.05)
    return SemiDefinitePhylogeny(A, p)
end
"""
    _validate_length_integer_phylogeny_constraint_B(alg::Union{Nothing, <:Integer},
                                                    B::AbstractVector)

Validate that the length of the vector `B` does not exceed the integer value `alg`.

This function is used internally to ensure that the number of groups or allocations specified by `B` does not exceed the allowed maximum defined by `alg`. If the validation fails, a `DomainError` is thrown.

# Arguments

  - `alg`:

      + `Nothing`: No validation is performed.
      + `Integer`: specifying the maximum allowed length for `B`.

  - `B`: Vector of integers representing group sizes or allocations.

# Returns

  - `nothing`: Returns nothing if validation passes.

# Validation

  - Throws `DomainError` if `length(B) > alg`.

# Details

  - Checks that `length(B) <= alg`.
  - Used in the construction and validation of integer phylogeny constraints.

# Related

  - [`validate_length_integer_phylogeny_constraint_B`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
"""
function _validate_length_integer_phylogeny_constraint_B(alg::Integer, B::AbstractVector)
    @argcheck(length(B) <= alg,
              DomainError("`length(B) <= alg`:\nlength(B) => $(length(B))\nalg => $(alg)"))
    return nothing
end
function _validate_length_integer_phylogeny_constraint_B(args...)
    return nothing
end
"""
    validate_length_integer_phylogeny_constraint_B(pe::ClusteringEstimator, B::AbstractVector)
    validate_length_integer_phylogeny_constraint_B(args...)

Validate that the length of the vector `B` does not exceed the maximum allowed by the clustering estimator `pe`.

# Arguments

  - `pe`: Clustering estimator containing algorithm and maximum group information.
  - `B`: Vector of integers representing group sizes or allocations.
  - `args...`: No validation is performed.

# Returns

  - `nothing`: Returns nothing if validation passes.

# Validation

  - Throws `DomainError` if `length(B) > pe.onc.max_k` (when `max_k` is set).
  - Calls internal [`_validate_length_integer_phylogeny_constraint_B`](@ref) for further checks.

# Details

  - Checks if `pe.onc.max_k` is set and validates `length(B)` accordingly.
  - Delegates to `_validate_length_integer_phylogeny_constraint_B` for algorithm-specific validation.
  - Used in the construction and validation of integer phylogeny constraints.

# Related

  - [`_validate_length_integer_phylogeny_constraint_B`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
"""
function validate_length_integer_phylogeny_constraint_B(pe::ClusteringEstimator,
                                                        B::AbstractVector)
    if !isnothing(pe.onc.max_k)
        @argcheck(length(B) <= pe.onc.max_k,
                  DomainError("`length(B) <= pe.onc.max_k`:\nlength(B) => $(length(B))\npe.onc.max_k => $(pe.onc.max_k)"))
    end
    _validate_length_integer_phylogeny_constraint_B(pe.onc.alg, B)
    return nothing
end
function validate_length_integer_phylogeny_constraint_B(args...)
    return nothing
end
"""
    struct IntegerPhylogenyEstimator{T1, T2, T3} <: AbstractPhylogenyConstraintEstimator
        pe::T1
        B::T2
        scale::T3
    end

Estimator for generating integer phylogeny-based constraints in PortfolioOptimisers.jl.

`IntegerPhylogenyEstimator` constructs constraints based on phylogenetic or clustering structures among assets, using integer or discrete representations. The estimator wraps a phylogeny or clustering estimator, a non-negative integer or vector of integers `B` specifying group sizes or allocations, and a big-M parameter `scale` used for formulating the MIP constraints.

# Fields

  - `pe`: Phylogeny or clustering estimator.
  - `B`: Non-negative integer or vector of integers specifying group sizes or allocations.
  - `scale`: Non-negative big-M parameter for the MIP formulation.

# Constructor

    IntegerPhylogenyEstimator(;
                              pe::Union{<:AbstractPhylogenyEstimator,
                                        <:AbstractClusteringResult} = NetworkEstimator(),
                              B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                              scale::Real = 100_000.0)

## Validation

  - `B` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

      + `AbstractVector`: it is additionally validated with [`validate_length_integer_phylogeny_constraint_B`](@ref).

# Examples

```jldoctest
julia> IntegerPhylogenyEstimator()
IntegerPhylogenyEstimator
     pe ┼ NetworkEstimator
        │    ce ┼ PortfolioOptimisersCovariance
        │       │   ce ┼ Covariance
        │       │      │    me ┼ SimpleExpectedReturns
        │       │      │       │   w ┴ nothing
        │       │      │    ce ┼ GeneralCovariance
        │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │       │      │       │    w ┴ nothing
        │       │      │   alg ┴ Full()
        │       │   mp ┼ DefaultMatrixProcessing
        │       │      │       pdm ┼ Posdef
        │       │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
        │       │      │   denoise ┼ nothing
        │       │      │    detone ┼ nothing
        │       │      │       alg ┴ nothing
        │    de ┼ Distance
        │       │   power ┼ nothing
        │       │     alg ┴ CanonicalDistance()
        │   alg ┼ KruskalTree
        │       │     args ┼ Tuple{}: ()
        │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │     n ┴ Int64: 1
      B ┼ Int64: 1
  scale ┴ Float64: 100000.0
```

# Related

  - [`IntegerPhylogeny`](@ref)
  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
struct IntegerPhylogenyEstimator{T1, T2, T3} <: AbstractPhylogenyConstraintEstimator
    pe::T1
    B::T2
    scale::T3
    function IntegerPhylogenyEstimator(pe::Union{<:AbstractPhylogenyEstimator,
                                                 <:AbstractClusteringResult},
                                       B::Union{<:Integer, <:AbstractVector{<:Integer}},
                                       scale::Real)
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, AbstractVector)
            validate_length_integer_phylogeny_constraint_B(pe, B)
        end
        return new{typeof(pe), typeof(B), typeof(scale)}(pe, B, scale)
    end
end
function IntegerPhylogenyEstimator(;
                                   pe::Union{<:AbstractPhylogenyEstimator,
                                             <:AbstractClusteringResult} = NetworkEstimator(),
                                   B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                                   scale::Real = 100_000.0)
    return IntegerPhylogenyEstimator(pe, B, scale)
end
"""
    struct IntegerPhylogeny{T1, T2, T3} <: AbstractPhylogenyConstraintResult
        A::T1
        B::T2
        scale::T3
    end

Container for the result of integer phylogeny-based constraint generation.

`IntegerPhylogeny` stores the constraint matrix `A`, group sizes or allocations `B`, and scaling parameter `scale` resulting from an integer phylogeny constraint estimator. This type encapsulates the output of integer/discrete phylogeny-based constraint routines, enabling composable and modular constraint handling in portfolio optimisation workflows.

# Fields

  - `A`: Phylogeny matrix encoding asset relationships.
  - `B`: Non-negative integer or vector of integers specifying group sizes or allocations.
  - `scale`: Non-negative scaling parameter (big-M) for the constraint.

# Constructor

    IntegerPhylogeny(;
                     A::Union{<:PhylogenyResult{<:AbstractMatrix{<:Real}},
                              <:AbstractMatrix{<:Real}},
                     B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                     scale::Real = 100_000.0)

## Validation

  - `issymmetric(A)` and `all(iszero, diag(A))`.

  - `B` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

      + `AbstractVector`: `size(unique(A + I; dims = 1), 1) == length(B)`.

# Examples

```jldoctest
julia> IntegerPhylogeny(; A = [0.0 1.0; 1.0 0.0], B = 2, scale = 100_000.0)
IntegerPhylogeny
      A ┼ 1×2 Matrix{Float64}
      B ┼ Int64: 2
  scale ┴ Float64: 100000.0
```

# Related

  - [`IntegerPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
struct IntegerPhylogeny{T1, T2, T3} <: AbstractPhylogenyConstraintResult
    A::T1
    B::T2
    scale::T3
    function IntegerPhylogeny(A::AbstractMatrix{<:Real},
                              B::Union{<:Integer, <:AbstractVector{<:Integer}}, scale::Real)
        @argcheck(all(iszero, diag(A)))
        @argcheck(issymmetric(A))
        A = unique(A + I; dims = 1)
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, AbstractVector)
            @argcheck(size(A, 1) == length(B))
        end
        return new{typeof(A), typeof(B), typeof(scale)}(A, B, scale)
    end
end
function IntegerPhylogeny(A::PhylogenyResult{<:AbstractMatrix{<:Real}},
                          B::Union{<:Integer, <:AbstractVector{<:Integer}}, scale::Real)
    return IntegerPhylogeny(A.X, B, scale)
end
function IntegerPhylogeny(;
                          A::Union{<:PhylogenyResult{<:AbstractMatrix{<:Real}},
                                   <:AbstractMatrix{<:Real}},
                          B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                          scale::Real = 100_000.0)
    return IntegerPhylogeny(A, B, scale)
end
"""
    phylogeny_constraints(est::Union{<:SemiDefinitePhylogenyEstimator,
                                     <:IntegerPhylogenyEstimator, <:SemiDefinitePhylogeny,
                                     <:IntegerPhylogeny, Nothing}, X::AbstractMatrix;
                          dims::Int = 1, kwargs...)

Generate phylogeny-based portfolio constraints from an estimator or result.

`phylogeny_constraints` constructs constraint objects based on phylogenetic, clustering, or network structures among assets. It supports both semi-definite and integer constraint forms, accepting either an estimator (which wraps a phylogeny or clustering model and penalty parameters) or a precomputed result. If `est` is `nothing`, returns `nothing`.

# Arguments

  - `est`: A phylogeny constraint estimator, result, or `nothing`.
  - `X`: Data matrix of asset features or returns (ignored when `est` is not an estimator).
  - `dims`: Dimension along which to compute the phylogeny (ignored when `est` is not an estimator).
  - `kwargs...`: Additional keyword arguments passed to the underlying phylogeny matrix routine (ignored when `est` is not an estimator).

# Returns

  - `SemiDefinitePhylogeny`: For semi-definite constraint estimators/results.
  - `IntegerPhylogeny`: For integer constraint estimators/results.
  - `nothing`: If `est` is `nothing`.

# Details

  - `est`:

      + `Union{<:SemiDefinitePhylogenyEstimator, <:IntegerPhylogenyEstimator}`: computes the phylogeny matrix using the estimator.
      + `Union{Nothing, <:SemiDefinitePhylogeny, <:IntegerPhylogeny}`: returns it unchanged.

# Related

  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
  - [`IntegerPhylogeny`](@ref)
  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_constraints(plc::SemiDefinitePhylogenyEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return SemiDefinitePhylogeny(; A = phylogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                                 p = plc.p)
end
function phylogeny_constraints(plc::IntegerPhylogenyEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return IntegerPhylogeny(; A = phylogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                            B = plc.B, scale = plc.scale)
end
function phylogeny_constraints(plc::Union{<:SemiDefinitePhylogeny, <:IntegerPhylogeny,
                                          Nothing}, args...; kwargs...)
    return plc
end
function phylogeny_constraints(plcs::AbstractVector{<:Union{<:AbstractPhylogenyConstraintEstimator,
                                                            <:AbstractPhylogenyConstraintResult}},
                               args...; kwargs...)
    return [phylogeny_constraints(plc, args...; kwargs...) for plc in plcs]
end
"""
    abstract type VectorToRealMeasure <: AbstractAlgorithm end

Abstract supertype for algorithms mapping a vector of real values to a single real value.

`VectorToRealMeasure` provides a unified interface for algorithms that reduce a vector of real numbers to a scalar, such as minimum, mean, median, or maximum. These are used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Related

  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
abstract type VectorToRealMeasure <: AbstractAlgorithm end
"""
    struct MinValue <: VectorToRealMeasure end

Algorithm for reducing a vector of real values to its minimum.

`MinValue` is a concrete subtype of [`VectorToRealMeasure`](@ref) that returns the minimum value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their minimum.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MinValue(), [1.2, 3.4, 0.7])
0.7
```

# Related

  - [`VectorToRealMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MinValue <: VectorToRealMeasure end
"""
    struct MeanValue <: VectorToRealMeasure end

Algorithm for reducing a vector of real values to its mean.

`MeanValue` is a concrete subtype of [`VectorToRealMeasure`](@ref) that returns the mean (average) value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their mean.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MeanValue(), [1.2, 3.4, 0.7])
1.7666666666666666
```

# Related

  - [`VectorToRealMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MeanValue <: VectorToRealMeasure end
"""
    struct MedianValue <: VectorToRealMeasure end

Algorithm for reducing a vector of real values to its median.

`MedianValue` is a concrete subtype of [`VectorToRealMeasure`](@ref) that returns the median value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their median.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MedianValue(), [1.2, 3.4, 0.7])
1.2
```

# Related

  - [`VectorToRealMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MedianValue <: VectorToRealMeasure end
"""
    struct MaxValue <: VectorToRealMeasure end

Algorithm for reducing a vector of real values to its maximum.

`MaxValue` is a concrete subtype of [`VectorToRealMeasure`](@ref) that returns the maximum value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their maximum.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MaxValue(), [1.2, 3.4, 0.7])
3.4
```

# Related

  - [`VectorToRealMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MaxValue <: VectorToRealMeasure end
"""
    vec_to_real_measure(measure::Union{<:VectorToRealMeasure, <:Real}, val::AbstractVector)

Reduce a vector of real values to a single real value using a specified measure.

`vec_to_real_measure` applies a reduction algorithm (such as minimum, mean, median, or maximum) to a vector of real numbers, as specified by the concrete subtype of [`VectorToRealMeasure`](@ref). This is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Arguments

  - `measure`: An instance of a concrete subtype of [`VectorToRealMeasure`](@ref), or the predefined value to return.
  - `val`: A vector of real values to be reduced (ignored if `measure` is a `Real`).

# Returns

  - `score::Real`: computed value according to `measure`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MaxValue(), [1.2, 3.4, 0.7])
3.4

julia> PortfolioOptimisers.vec_to_real_measure(0.9, [1.2, 3.4, 0.7])
0.9
```

# Related

  - [`VectorToRealMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
"""
function vec_to_real_measure(::MinValue, val::AbstractVector)
    return minimum(val)
end
function vec_to_real_measure(::MeanValue, val::AbstractVector)
    return mean(val)
end
function vec_to_real_measure(::MedianValue, val::AbstractVector)
    return median(val)
end
function vec_to_real_measure(::MaxValue, val::AbstractVector)
    return maximum(val)
end
function vec_to_real_measure(val::Real, ::AbstractVector)
    return val
end
"""
    struct CentralityConstraint{T1, T2, T3} <: AbstractPhylogenyConstraintEstimator
        A::T1
        B::T2
        comp::T3
    end

Estimator for generating centrality-based portfolio constraints.

`CentralityConstraint` constructs constraints based on asset centrality measures within a phylogeny or network structure. It wraps a centrality estimator `A`, a [`VectorToRealMeasure`](@ref) measure or threshold `B`, and a comparison operator `comp` [`ComparisonOperator`](@ref). This enables flexible constraint generation based on asset centrality, supporting both inequality and equality forms.

# Fields

  - `A`: Centrality estimator.
  - `B`: Real value or reduction measure.
  - `comp`: Comparison operator.

# Constructor

    CentralityConstraint(; A::CentralityEstimator = CentralityEstimator(),
                         B::Union{<:Real, <:VectorToRealMeasure} = MinValue(),
                         comp::ComparisonOperator = LEQ())

# Examples

```jldoctest
julia> CentralityConstraint()
CentralityConstraint
     A ┼ CentralityEstimator
       │     ne ┼ NetworkEstimator
       │        │    ce ┼ PortfolioOptimisersCovariance
       │        │       │   ce ┼ Covariance
       │        │       │      │    me ┼ SimpleExpectedReturns
       │        │       │      │       │   w ┴ nothing
       │        │       │      │    ce ┼ GeneralCovariance
       │        │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
       │        │       │      │       │    w ┴ nothing
       │        │       │      │   alg ┴ Full()
       │        │       │   mp ┼ DefaultMatrixProcessing
       │        │       │      │       pdm ┼ Posdef
       │        │       │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
       │        │       │      │   denoise ┼ nothing
       │        │       │      │    detone ┼ nothing
       │        │       │      │       alg ┴ nothing
       │        │    de ┼ Distance
       │        │       │   power ┼ nothing
       │        │       │     alg ┴ CanonicalDistance()
       │        │   alg ┼ KruskalTree
       │        │       │     args ┼ Tuple{}: ()
       │        │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
       │        │     n ┴ Int64: 1
       │   cent ┼ DegreeCentrality
       │        │     kind ┼ Int64: 0
       │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
     B ┼ MinValue()
  comp ┴ LEQ: LEQ()
```

# Related

  - [`CentralityEstimator`](@ref)
  - [`VectorToRealMeasure`](@ref)
  - [`ComparisonOperator`](@ref)
  - [`centrality_constraints`](@ref)
"""
struct CentralityConstraint{T1, T2, T3} <: AbstractPhylogenyConstraintEstimator
    A::T1
    B::T2
    comp::T3
    function CentralityConstraint(A::CentralityEstimator,
                                  B::Union{<:Real, <:VectorToRealMeasure},
                                  comp::ComparisonOperator)
        return new{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
    end
end
function CentralityConstraint(; A::CentralityEstimator = CentralityEstimator(),
                              B::Union{<:Real, <:VectorToRealMeasure} = MinValue(),
                              comp::ComparisonOperator = LEQ())
    return CentralityConstraint(A, B, comp)
end
"""
    centrality_constraints(ccs::Union{<:CentralityConstraint,
                                      <:AbstractVector{<:CentralityConstraint}},
                           X::AbstractMatrix; dims::Int = 1, kwargs...)

Generate centrality-based linear constraints from one or more `CentralityConstraint` estimators.

`centrality_constraints` constructs linear constraints for portfolio optimisation based on asset centrality measures within a phylogeny or network structure. It accepts one or more [`CentralityConstraint`](@ref) estimators, computes centrality vectors for the given data matrix `X`, applies the specified reduction measure or threshold, and assembles the resulting constraints into a [`LinearConstraint`](@ref) object.

# Arguments

  - `ccs`: A single [`CentralityConstraint`](@ref) or a vector of such estimators.
  - `X`: Data matrix of asset features or returns.
  - `dims`: Dimension along which to compute centrality.
  - `kwargs...`: Additional keyword arguments passed to the centrality estimator.

# Returns

  - `lc::Union{Nothing, <:LinearConstraint}`: An object containing the assembled inequality and equality constraints, or `nothing` if no constraints are present.

# Details

  - For each constraint, computes the centrality vector using the estimator in `cc.A`.
  - Applies the comparison operator and reduction measure or threshold in `cc.B` and `cc.comp`.
  - Aggregates constraints into equality and inequality forms.
  - Returns `nothing` if no valid constraints are generated.

# Related

  - [`CentralityConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_constraints(ccs::Union{<:CentralityConstraint,
                                           <:AbstractVector{<:CentralityConstraint}},
                                X::AbstractMatrix; dims::Int = 1, kwargs...)
    if isa(ccs, AbstractVector)
        @argcheck(!isempty(ccs))
    end
    A_ineq = Vector{eltype(X)}(undef, 0)
    B_ineq = Vector{eltype(X)}(undef, 0)
    A_eq = Vector{eltype(X)}(undef, 0)
    B_eq = Vector{eltype(X)}(undef, 0)
    for cc in ccs
        A = centrality_vector(cc.A, X; dims = dims, kwargs...).X
        lhs_flag = isempty(A) || all(iszero, A)
        if lhs_flag
            continue
        end
        d, flag_ineq = comparison_sign_ineq_flag(cc.comp)
        A .*= d
        B = d * vec_to_real_measure(cc.B, A)
        if flag_ineq
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    ineq_flag = !isempty(A_ineq)
    eq_flag = !isempty(A_eq)
    if ineq_flag
        A_ineq = transpose(reshape(A_ineq, size(X, 2), :))
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, size(X, 2), :))
    end
    return if !ineq_flag && !eq_flag
        nothing
    else
        LinearConstraint(; ineq = if ineq_flag
                             PartialLinearConstraint(; A = A_ineq, B = B_ineq)
                         else
                             nothing
                         end, eq = if eq_flag
                             PartialLinearConstraint(; A = A_eq, B = B_eq)
                         else
                             nothing
                         end)
    end
end
"""
    centrality_constraints(ccs::Union{Nothing, <:LinearConstraint}, args...; kwargs...)

No-op fallback for centrality-based constraint propagation.

This method returns the input [`LinearConstraint`](@ref) object or `nothing` unchanged. It is used to pass through an already constructed centrality-based constraint object, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `ccs`: An existing [`LinearConstraint`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `ccs`: The input constraint object or `nothing`, unchanged.

# Related

  - [`CentralityConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`centrality_constraints`](@ref)
"""
function centrality_constraints(ccs::Union{Nothing, <:LinearConstraint}, args...; kwargs...)
    return ccs
end

export SemiDefinitePhylogenyEstimator, SemiDefinitePhylogeny, IntegerPhylogenyEstimator,
       IntegerPhylogeny, MinValue, MeanValue, MedianValue, MaxValue, CentralityConstraint,
       phylogeny_constraints, centrality_constraints
