"""
    abstract type AbstractPhylogenyConstraintEstimator <: AbstractConstraintEstimator end

Abstract supertype for all phylogeny-based constraint estimators in PortfolioOptimisers.jl.

All concrete types representing phylogeny-based constraint estimators should subtype `AbstractPhylogenyConstraintEstimator`. This enables a consistent, composable interface for generating constraints based on phylogenetic, res, or network structures among assets.

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

All concrete types representing the results of phylogeny-based constraint generation should subtype `AbstractPhylogenyConstraintResult`. This enables a consistent, composable interface for storing and propagating constraint matrices, vectors, or other outputs derived from phylogenetic, res, or network structures among assets.

# Related

  - [`SemiDefinitePhylogeny`](@ref)
  - [`IntegerPhylogeny`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
abstract type AbstractPhylogenyConstraintResult <: AbstractConstraintResult end
const PhCE_PhC = Union{<:AbstractPhylogenyConstraintEstimator,
                       <:AbstractPhylogenyConstraintResult}
const VecPhCE_PhC = AbstractVector{<:PhCE_PhC}
const PhCE_PhC_VecPhCE_PhC = Union{<:PhCE_PhC, <:VecPhCE_PhC}
const VecPhC = AbstractVector{<:AbstractPhylogenyConstraintResult}
const PhC_VecPhC = Union{<:AbstractPhylogenyConstraintResult, <:VecPhC}
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
                                   pe::NwE_HClE_HCl = NetworkEstimator(),
                                   p::Number = 0.05)

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
     │       │   mp ┼ DenoiseDetoneAlgMatrixProcessing
     │       │      │       pdm ┼ Posdef
     │       │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │       │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │       │      │   denoise ┼ nothing
     │       │      │    detone ┼ nothing
     │       │      │       alg ┼ nothing
     │       │      │     order ┴ DenoiseDetoneAlg()
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
    function SemiDefinitePhylogenyEstimator(pe::NwE_HClE_HCl, p::Number)
        @argcheck(p >= zero(p), DomainError("`p` must be non-negative:\np => $p"))
        return new{typeof(pe), typeof(p)}(pe, p)
    end
end
function SemiDefinitePhylogenyEstimator(; pe::NwE_HClE_HCl = NetworkEstimator(),
                                        p::Number = 0.05)
    return SemiDefinitePhylogenyEstimator(pe, p)
end
const MatNum_PhRMatNum = Union{<:PhylogenyResult{<:MatNum}, <:MatNum}
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
                          A::MatNum_PhRMatNum, p::Number = 0.05)

## Validation

  - `LinearAlgebra.issymmetric(A)` and `all(iszero, LinearAlgebra.diag(A))`.
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
    function SemiDefinitePhylogeny(A::MatNum, p::Number)
        @argcheck(all(iszero, LinearAlgebra.diag(A)))
        @argcheck(LinearAlgebra.issymmetric(A))
        @argcheck(p >= zero(p))
        return new{typeof(A), typeof(p)}(A, p)
    end
end
function SemiDefinitePhylogeny(A::PhylogenyResult{<:MatNum}, p::Number)
    return SemiDefinitePhylogeny(A.X, p)
end
function SemiDefinitePhylogeny(; A::MatNum_PhRMatNum, p::Number = 0.05)
    return SemiDefinitePhylogeny(A, p)
end
"""
    _validate_length_integer_phylogeny_constraint_B(alg::Option{<:Integer},
                                                    B::VecNum)

Validate that the length of the vector `B` does not exceed the integer value `alg`.

This function is used internally to ensure that the number of groups or allocations specified by `B` does not exceed the allowed maximum defined by `alg`. If the validation fails, a `DomainError` is thrown.

# Arguments

  - `alg`:

      + `Nothing`: No validation is performed.
      + `Integer`: Specifying the maximum allowed length for `B`.

  - `B`: Vector of integers representing group sizes or allocations.

# Returns

  - `nothing`.

# Validation

  - Throws `DomainError` if `length(B) > alg`.

# Details

  - Checks that `length(B) <= alg`.
  - Used in the construction and validation of integer phylogeny constraints.

# Related

  - [`validate_length_integer_phylogeny_constraint_B`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
"""
function _validate_length_integer_phylogeny_constraint_B(alg::Integer, B::VecNum)
    @argcheck(length(B) <= alg,
              DomainError("`length(B) <= alg`:\nlength(B) => $(length(B))\nalg => $(alg)"))
    return nothing
end
function _validate_length_integer_phylogeny_constraint_B(args...)
    return nothing
end
"""
    validate_length_integer_phylogeny_constraint_B(pe::ClustersEstimator, B::VecNum)
    validate_length_integer_phylogeny_constraint_B(args...)

Validate that the length of the vector `B` does not exceed the maximum allowed by the clustering estimator `pe`.

# Arguments

  - `pe`: Clustering estimator containing algorithm and maximum group information.
  - `B`: Vector of integers representing group sizes or allocations.
  - `args...`: No validation is performed.

# Returns

  - `nothing`.

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
function validate_length_integer_phylogeny_constraint_B(pe::ClustersEstimator, B::VecNum)
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
                              pe::NwE_HClE_HCl = NetworkEstimator(),
                              B::Int_VecInt = 1,
                              scale::Number = 100_000.0)

## Validation

  - `B` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

      + `AbstractVector`: It is additionally validated with [`validate_length_integer_phylogeny_constraint_B`](@ref).

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
        │       │   mp ┼ DenoiseDetoneAlgMatrixProcessing
        │       │      │       pdm ┼ Posdef
        │       │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │       │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │       │      │   denoise ┼ nothing
        │       │      │    detone ┼ nothing
        │       │      │       alg ┼ nothing
        │       │      │     order ┴ DenoiseDetoneAlg()
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
    function IntegerPhylogenyEstimator(pe::NwE_HClE_HCl, B::Int_VecInt, scale::Number)
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, VecInt)
            validate_length_integer_phylogeny_constraint_B(pe, B)
        end
        return new{typeof(pe), typeof(B), typeof(scale)}(pe, B, scale)
    end
end
function IntegerPhylogenyEstimator(; pe::NwE_HClE_HCl = NetworkEstimator(),
                                   B::Int_VecInt = 1, scale::Number = 100_000.0)
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
                     A::MatNum_PhRMatNum,
                     B::Int_VecInt = 1,
                     scale::Number = 100_000.0)

## Validation

  - `LinearAlgebra.issymmetric(A)` and `all(iszero, LinearAlgebra.diag(A))`.

  - `B` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

      + `AbstractVector`: `size(unique(A + LinearAlgebra.I; dims = 1), 1) == length(B)`.

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
    function IntegerPhylogeny(A::MatNum, B::Int_VecInt, scale::Number)
        @argcheck(all(iszero, LinearAlgebra.diag(A)))
        @argcheck(LinearAlgebra.issymmetric(A))
        A = unique(A + LinearAlgebra.I; dims = 1)
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, VecInt)
            @argcheck(size(A, 1) == length(B))
        end
        return new{typeof(A), typeof(B), typeof(scale)}(A, B, scale)
    end
end
function IntegerPhylogeny(A::PhylogenyResult{<:MatNum}, B::Int_VecInt, scale::Number)
    return IntegerPhylogeny(A.X, B, scale)
end
function IntegerPhylogeny(; A::MatNum_PhRMatNum, B::Int_VecInt = 1,
                          scale::Number = 100_000.0)
    return IntegerPhylogeny(A, B, scale)
end
"""
    phylogeny_constraints(plc::Option{<:PhCE_PhC}, X::MatNum; dims::Int = 1, kwargs...)
    phylogeny_constraints(plcs::VecPhCE_PhC, args...; kwargs...)

Generate phylogeny-based portfolio constraints from an estimator or result.

`phylogeny_constraints` constructs constraint objects based on phylogenetic, res, or network structures among assets. It supports both semi-definite and integer constraint forms, accepting either an estimator (which wraps a phylogeny or clustering model and penalty parameters) or a precomputed result. If `plc` is `nothing`, returns `nothing`.

If a vector broadcasts the function over each element, returning a vector of constraint results.

# Arguments

  - `plc`: A phylogeny constraint estimator, result, or `nothing`.
  - `X`: Data matrix of asset features or returns (ignored when `plc` is not an estimator).
  - `dims`: Dimension along which to compute the phylogeny (ignored when `plc` is not an estimator).
  - `kwargs...`: Additional keyword arguments passed to the underlying phylogeny matrix routine (ignored when `est` is not an estimator).

# Returns

  - `res`: Constraint result.

      + `SemiDefinitePhylogeny`: For semi-definite constraint estimators/results.
      + `IntegerPhylogeny`: For integer constraint estimators/results.
      + `nothing`: If `est` is `nothing`.

# Related

  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
  - [`IntegerPhylogeny`](@ref)
  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_constraints(plc::SemiDefinitePhylogenyEstimator, X::MatNum;
                               dims::Int = 1, kwargs...)
    return SemiDefinitePhylogeny(; A = phylogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                                 p = plc.p)
end
function phylogeny_constraints(plc::IntegerPhylogenyEstimator, X::MatNum; dims::Int = 1,
                               kwargs...)
    return IntegerPhylogeny(; A = phylogeny_matrix(plc.pe, X; dims = dims, kwargs...),
                            B = plc.B, scale = plc.scale)
end
function phylogeny_constraints(plc::Option{<:AbstractPhylogenyConstraintResult}, args...;
                               kwargs...)
    return plc
end
function phylogeny_constraints(plcs::VecPhCE_PhC, args...; kwargs...)
    return [phylogeny_constraints(plc, args...; kwargs...) for plc in plcs]
end
abstract type AbstractCentralityConstraint <: AbstractConstraintEstimator end
"""
    struct CentralityConstraint{T1, T2, T3} <: AbstractCentralityConstraint
        A::T1
        B::T2
        comp::T3
    end

Estimator for generating centrality-based portfolio constraints.

`CentralityConstraint` constructs constraints based on asset centrality measures within a phylogeny or network structure. It wraps a centrality estimator `A`, a [`VectorToScalarMeasure`](@ref) measure or threshold `B`, and a comparison operator `comp` [`ComparisonOperator`](@ref). This enables flexible constraint generation based on asset centrality, supporting both inequality and equality forms.

# Fields

  - `A`: Centrality estimator.
  - `B`: Number value or reduction measure.
  - `comp`: Comparison operator.

# Constructor

    CentralityConstraint(; A::CentralityEstimator = CentralityEstimator(),
                         B::Num_VecToScaM = MinValue(),
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
       │        │       │   mp ┼ DenoiseDetoneAlgMatrixProcessing
       │        │       │      │       pdm ┼ Posdef
       │        │       │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
       │        │       │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
       │        │       │      │   denoise ┼ nothing
       │        │       │      │    detone ┼ nothing
       │        │       │      │       alg ┼ nothing
       │        │       │      │     order ┴ DenoiseDetoneAlg()
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
  - [`VectorToScalarMeasure`](@ref)
  - [`ComparisonOperator`](@ref)
  - [`centrality_constraints`](@ref)
"""
struct CentralityConstraint{T1, T2, T3} <: AbstractCentralityConstraint
    A::T1
    B::T2
    comp::T3
    function CentralityConstraint(A::CentralityEstimator, B::Num_VecToScaM,
                                  comp::ComparisonOperator)
        return new{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
    end
end
function CentralityConstraint(; A::CentralityEstimator = CentralityEstimator(),
                              B::Num_VecToScaM = MinValue(),
                              comp::ComparisonOperator = LEQ())
    return CentralityConstraint(A, B, comp)
end
const VecCC = AbstractVector{<:CentralityConstraint}
const CC_VecCC = Union{<:CentralityConstraint, <:VecCC}
const Lc_CC_VecCC = Union{<:CC_VecCC, <:LinearConstraint}
"""
    centrality_constraints(ccs::CC_VecCC,
                           X::MatNum; dims::Int = 1, kwargs...)

Generate centrality-based linear constraints from one or more `CentralityConstraint` estimators.

`centrality_constraints` constructs linear constraints for portfolio optimisation based on asset centrality measures within a phylogeny or network structure. It accepts one or more [`CentralityConstraint`](@ref) estimators, computes centrality vectors for the given data matrix `X`, applies the specified reduction measure or t, and assembles the resulting constraints into a [`LinearConstraint`](@ref) object.

# Arguments

  - `ccs`: A single [`CentralityConstraint`](@ref) or a vector of such estimators.
  - `X`: Data matrix of asset features or returns.
  - `dims`: Dimension along which to compute centrality.
  - `kwargs...`: Additional keyword arguments passed to the centrality estimator.

# Returns

  - `lc::Option{<:LinearConstraint}`: An object containing the assembled inequality and equality constraints, or `nothing` if no constraints are present.

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
function centrality_constraints(ccs::CC_VecCC, X::MatNum; dims::Int = 1, kwargs...)
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
    centrality_constraints(ccs::Option{<:LinearConstraint}, args...; kwargs...)

No-op fallback for centrality-based constraint propagation.

This method returns the input [`LinearConstraint`](@ref) object or `nothing` unchanged. It is used to pass through an already constructed centrality-based constraint object, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `ccs`: An existing [`LinearConstraint`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `ccs::Option{<:LinearConstraint}`: The input constraint object or `nothing`, unchanged.

# Related

  - [`CentralityConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`centrality_constraints`](@ref)
"""
function centrality_constraints(ccs::Option{<:LinearConstraint}, args...; kwargs...)
    return ccs
end

export SemiDefinitePhylogenyEstimator, SemiDefinitePhylogeny, IntegerPhylogenyEstimator,
       IntegerPhylogeny, CentralityConstraint, phylogeny_constraints, centrality_constraints
