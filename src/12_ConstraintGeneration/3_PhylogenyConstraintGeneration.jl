"""
```julia
abstract type AbstractPhylogenyConstraintEstimator <: AbstractConstraintEstimator end
```

Abstract supertype for all phylogeny-based constraint estimators in PortfolioOptimisers.jl.

All concrete types implementing phylogeny-based constraint generation algorithms should subtype `AbstractPhylogenyConstraintEstimator`. This enables a consistent interface for phylogeny-derived constraint estimators throughout the package, supporting composable workflows for portfolio optimisation with phylogenetic structure.

# Related

  - [`AbstractConstraintEstimator`](@ref)
  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
abstract type AbstractPhylogenyConstraintEstimator <: AbstractConstraintEstimator end

"""
```julia
abstract type AbstractPhylogenyConstraintResult <: AbstractConstraintResult end
```

Abstract supertype for all phylogeny-based constraint result types in PortfolioOptimisers.jl.

All concrete types representing the result of phylogeny-based constraint generation algorithms should subtype `AbstractPhylogenyConstraintResult`. This enables a consistent interface for handling phylogeny-derived constraint results, supporting composable and extensible workflows in portfolio optimisation with phylogenetic structure.

# Related

  - [`AbstractConstraintResult`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
  - [`IntegerPhylogeny`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
abstract type AbstractPhylogenyConstraintResult <: AbstractConstraintResult end

"""
```julia
struct SemiDefinitePhylogenyEstimator{T1, T2} <: AbstractPhylogenyConstraintEstimator
    pe::T1
    p::T2
end
```

Estimator for generating semi-definite phylogeny-based constraints in portfolio optimisation.

`SemiDefinitePhylogenyEstimator` wraps a phylogeny estimator or clustering result (`pe`) and a non-negative penalty parameter (`p`). The penalty blurs the contribution of the asset phylogeny to the optimisation. It is used to construct semi-definite constraint matrices that encode phylogenetic structure among assets, supporting composable constraint generation workflows.

# Fields

  - `pe`: Phylogeny estimator or clustering result.
  - `p`: Non-negative penalty parameter.

# Constructor

```julia
SemiDefinitePhylogenyEstimator(;
                               pe::Union{<:AbstractPhylogenyEstimator,
                                         <:AbstractClusteringResult} = NetworkEstimator(),
                               p::Real = 0.05)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `pe` is a matrix, `issymmetric(pe)` and `all(x -> iszero(x), diag(pe))`.
  - `p >= 0`.

# Examples

```jldoctest
julia> SemiDefinitePhylogenyEstimator(; pe = NetworkEstimator(), p = 0.1)
SemiDefinitePhylogenyEstimator
  pe | NetworkEstimator
     |    ce | PortfolioOptimisersCovariance
     |       |   ce | Covariance
     |       |      |    me | SimpleExpectedReturns
     |       |      |       |   w | nothing
     |       |      |    ce | GeneralWeightedCovariance
     |       |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     |       |      |       |    w | nothing
     |       |      |   alg | Full()
     |       |   mp | DefaultMatrixProcessing
     |       |      |       pdm | Posdef
     |       |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
     |       |      |   denoise | nothing
     |       |      |    detone | nothing
     |       |      |       alg | nothing
     |    de | Distance
     |       |   alg | CanonicalDistance()
     |   alg | KruskalTree
     |       |     args | Tuple{}: ()
     |       |   kwargs | @NamedTuple{}: NamedTuple()
     |     n | Int64: 1
   p | Float64: 0.1
```

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
struct SemiDefinitePhylogenyEstimator{T1, T2} <: AbstractPhylogenyConstraintEstimator
    pe::T1
    p::T2
end
function SemiDefinitePhylogenyEstimator(;
                                        pe::Union{<:AbstractPhylogenyEstimator,
                                                  <:AbstractClusteringResult} = NetworkEstimator(),
                                        p::Real = 0.05)
    @argcheck(p >= zero(p), DomainError("`p` must be non-negative:\np => $p"))
    return SemiDefinitePhylogenyEstimator(pe, p)
end
struct SemiDefinitePhylogeny{T1, T2} <: AbstractPhylogenyConstraintResult
    A::T1
    p::T2
end
function SemiDefinitePhylogeny(; A::AbstractMatrix{<:Real}, p::Real = 0.05)
    @argcheck(!isempty(A))
    @argcheck(p >= zero(p))
    return SemiDefinitePhylogeny(A, p)
end
function phylogeny_constraints(plc::SemiDefinitePhylogenyEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return SemiDefinitePhylogeny(; A = phylogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                                 p = plc.p)
end
struct IntegerPhylogenyEstimator{T1, T2, T3} <: AbstractPhylogenyConstraintEstimator
    pe::T1
    B::T2
    scale::T3
end
function _validate_length_integer_phylogeny_constraint_B(alg::Integer, B::AbstractVector)
    @argcheck(length(B) <= alg,
              DomainError("`length(B) <= alg`:\nlength(B) => $(length(B))\nalg => $(alg)"))
    return nothing
end
function _validate_length_integer_phylogeny_constraint_B(args...)
    return nothing
end
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
function IntegerPhylogenyEstimator(;
                                   pe::Union{<:AbstractPhylogenyEstimator,
                                             <:AbstractClusteringResult} = NetworkEstimator(),
                                   B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                                   scale::Real = 100_000.0)
    if isa(B, AbstractVector)
        @argcheck(!isempty(B))
        @argcheck(all(x -> x >= zero(x), B))
        validate_length_integer_phylogeny_constraint_B(pe, B)
    else
        @argcheck(B >= zero(B))
    end
    return IntegerPhylogenyEstimator(pe, B, scale)
end
struct IntegerPhylogeny{T1, T2, T3} <: AbstractPhylogenyConstraintResult
    A::T1
    B::T2
    scale::T3
end
function IntegerPhylogeny(; A::AbstractMatrix{<:Real},
                          B::Union{<:Integer, <:AbstractVector{<:Integer}} = 1,
                          scale::Real = 100_000.0)
    @argcheck(!isempty(A))
    A = unique(A + I; dims = 1)
    if isa(B, AbstractVector)
        @argcheck(!isempty(B))
        @argcheck(size(A, 1) == length(B))
        @argcheck(all(x -> x >= zero(x), B))
    else
        @argcheck(B >= zero(B))
    end
    return IntegerPhylogeny(A, B, scale)
end
function phylogeny_constraints(plc::IntegerPhylogenyEstimator, X::AbstractMatrix;
                               dims::Int = 1, kwargs...)
    return IntegerPhylogeny(; A = phylogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                            B = plc.B, scale = plc.scale)
end
function phylogeny_constraints(plc::Union{<:SemiDefinitePhylogeny, <:IntegerPhylogeny},
                               args...; kwargs...)
    return plc
end
function phylogeny_constraints(::Nothing, args...; kwargs...)
    return nothing
end
abstract type VectorToRealMeasure <: AbstractAlgorithm end
struct MinValue <: VectorToRealMeasure end
function vec_to_real_measure(::MinValue, val::AbstractVector)
    return minimum(val)
end
struct MeanValue <: VectorToRealMeasure end
function vec_to_real_measure(::MeanValue, val::AbstractVector)
    return mean(val)
end
struct MedianValue <: VectorToRealMeasure end
function vec_to_real_measure(::MedianValue, val::AbstractVector)
    return median(val)
end
struct MaxValue <: VectorToRealMeasure end
function vec_to_real_measure(::MaxValue, val::AbstractVector)
    return maximum(val)
end
function vec_to_real_measure(val::Real, ::AbstractVector)
    return val
end
struct CentralityEstimator{T1, T2, T3} <: AbstractPhylogenyConstraintEstimator
    A::T1
    B::T2
    comp::T3
end
function CentralityEstimator(; A::Centrality = Centrality(),
                             B::Union{<:Real, <:VectorToRealMeasure} = MinValue(),
                             comp::ComparisonOperators = LEQ())
    return CentralityEstimator(A, B, comp)
end
function centrality_constraints(ccs::Union{<:CentralityEstimator,
                                           <:AbstractVector{<:CentralityEstimator}},
                                X::AbstractMatrix; dims::Int = 1, kwargs...)
    if isa(ccs, AbstractVector)
        @argcheck(!isempty(ccs))
    end
    A_ineq = Vector{eltype(X)}(undef, 0)
    B_ineq = Vector{eltype(X)}(undef, 0)
    A_eq = Vector{eltype(X)}(undef, 0)
    B_eq = Vector{eltype(X)}(undef, 0)
    for cc in ccs
        A = centrality_vector(cc.A, X; dims = dims, kwargs...)
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
function centrality_constraints(ccs::LinearConstraint, args...; kwargs...)
    return ccs
end
function centrality_constraints(::Nothing, args...; kwargs...)
    return nothing
end

export SemiDefinitePhylogenyEstimator, SemiDefinitePhylogeny, IntegerPhylogenyEstimator,
       IntegerPhylogeny, MinValue, MeanValue, MedianValue, MaxValue, CentralityEstimator,
       phylogeny_constraints, centrality_constraints
