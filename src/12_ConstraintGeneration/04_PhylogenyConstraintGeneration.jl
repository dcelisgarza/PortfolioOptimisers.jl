"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny-based constraint estimators.

All concrete and/or abstract types representing phylogeny-based constraint estimators should be subtypes of `AbstractPhylogenyConstraintEstimator`.

# Related

  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
abstract type AbstractPhylogenyConstraintEstimator <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny-based constraint result types.

All concrete and/or abstract types representing the results of phylogeny-based constraint generation should be subtypes of `AbstractPhylogenyConstraintResult`.

# Related

  - [`SemiDefinitePhylogeny`](@ref)
  - [`IntegerPhylogeny`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
abstract type AbstractPhylogenyConstraintResult <: AbstractConstraintResult end
"""
    const PlCE_PlC = Union{<:AbstractPhylogenyConstraintEstimator,
                           <:AbstractPhylogenyConstraintResult}

Alias for a phylogeny constraint estimator or result.

Matches either a [`AbstractPhylogenyConstraintEstimator`](@ref) or a [`AbstractPhylogenyConstraintResult`](@ref). Used internally for dispatch in phylogeny-based constraint generation.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
"""
const PlCE_PlC = Union{<:AbstractPhylogenyConstraintEstimator,
                       <:AbstractPhylogenyConstraintResult}
"""
    const VecPlCE_PlC = AbstractVector{<:PlCE_PlC}

Alias for a vector of phylogeny constraint estimators or results.

Represents a collection of [`PlCE_PlC`](@ref) objects, enabling batch processing of multiple phylogeny-based constraint estimators or results.

# Related

  - [`PlCE_PlC`](@ref)
"""
const VecPlCE_PlC = AbstractVector{<:PlCE_PlC}
"""
    const PlCE_PhC_VecPlCE_PlC = Union{<:PlCE_PlC, <:VecPlCE_PlC}

Alias for a single or vector of phylogeny constraint estimators or results.

Matches either a single [`PlCE_PlC`](@ref) or a vector of them ([`VecPlCE_PlC`](@ref)). Used internally for dispatch on phylogeny-based constraint generation that accepts one or many constraints.

# Related

  - [`PlCE_PlC`](@ref)
  - [`VecPlCE_PlC`](@ref)
"""
const PlCE_PhC_VecPlCE_PlC = Union{<:PlCE_PlC, <:VecPlCE_PlC}
"""
    const VecPlC = AbstractVector{<:AbstractPhylogenyConstraintResult}

Alias for a vector of phylogeny constraint results.

Represents a collection of [`AbstractPhylogenyConstraintResult`](@ref) objects.

# Related

  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`PlC_VecPlC`](@ref)
"""
const VecPlC = AbstractVector{<:AbstractPhylogenyConstraintResult}
"""
    const PlC_VecPlC = Union{<:AbstractPhylogenyConstraintResult, <:VecPlC}

Alias for a single or vector of phylogeny constraint results.

Matches either a single [`AbstractPhylogenyConstraintResult`](@ref) or a vector of them ([`VecPlC`](@ref)).

# Related

  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`VecPlC`](@ref)
"""
const PlC_VecPlC = Union{<:AbstractPhylogenyConstraintResult, <:VecPlC}
"""
$(DocStringExtensions.TYPEDEF)

Forbids co-movement between related assets through a semidefinite relaxation, refitting the structure from returns.

The estimator holds the source that builds the relatedness matrix and the penalty factor `p`. [`phylogeny_constraints`](@ref) refits the source against a returns matrix and returns a [`SemiDefinitePhylogeny`](@ref), which carries the equations this constraint solves.

Which pairs a network source relates — and therefore how strong the constraint is — comes from its [`AbstractSeparationAlgorithm`](@ref), not from anything set here. The constraint is **weight-inert**: `A ⊙ W == 0` is the same constraint at any magnitude, so the separation changes the *cardinality* of the forbidden set and nothing else.

!!! warning

    `NetworkEstimator(; sep = PathLength())` relates **every reachable pair**. A bare [`PathLength`](@ref) leaves `dmax = nothing`, which resolves to the observed diameter, so nothing is outside the budget — measured, `190` of `190` pairs — and this estimator then forbids all pairwise co-movement. It is the opposite end of the dial from [`HopCount`](@ref)'s default `n = 1`. State a numeric `dmax` to select anything narrower.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SemiDefinitePhylogenyEstimator(;
        pl::NwE_ClE = NetworkEstimator(),
        p::Number = 0.05
    ) -> SemiDefinitePhylogenyEstimator

Keywords correspond to the struct's fields. The default `p = 0.05` is the value the source's own worked example uses.

## Validation

  - `p >= 0`.
  - `pl` is bounded by [`NwE_ClE`](@ref): a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref) is rejected by the type, not by a check, so the keyword constructor raises `TypeError` rather than deferring the problem to a solve. Build [`SemiDefinitePhylogeny`](@ref) instead, which is what `phylogeny_constraints(estimator, X)` returns.

# Examples

```jldoctest
julia> SemiDefinitePhylogenyEstimator()
SemiDefinitePhylogenyEstimator
  pl ┼ NetworkEstimator
     │    ce ┼ PortfolioOptimisersCovariance
     │       │   ce ┼ Covariance
     │       │      │    me ┼ SimpleExpectedReturns
     │       │      │       │   w ┴ nothing
     │       │      │    ce ┼ GeneralCovariance
     │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │      │       │    w ┴ nothing
     │       │      │   alg ┴ FullMoment()
     │       │   mp ┼ MatrixProcessing
     │       │      │     pdm ┼ Posdef
     │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │       │      │      dn ┼ nothing
     │       │      │      dt ┼ nothing
     │       │      │     alg ┼ nothing
     │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
     │    de ┼ Distance
     │       │   power ┼ nothing
     │       │     alg ┴ CanonicalDistance()
     │   alg ┼ KruskalTree
     │       │     args ┼ Tuple{}: ()
     │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │   sep ┼ HopCount
     │       │   n ┴ Int64: 1
   p ┴ Float64: 0.05
```

# Related

  - [`SemiDefinitePhylogeny`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:graphpo1])
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Sections 13.1.7.2 and 13.2.4.2.
"""
@concrete struct SemiDefinitePhylogenyEstimator <: AbstractPhylogenyConstraintEstimator
    """
    $(field_dict[:plsrc])
    """
    pl
    """
    $(field_dict[:p_phylo])
    """
    p
    function SemiDefinitePhylogenyEstimator(pl::NwE_ClE,
                                            p::Number)::SemiDefinitePhylogenyEstimator
        @argcheck(p >= zero(p), DomainError("`p` must be non-negative:\np => $p"))
        return new{typeof(pl), typeof(p)}(pl, p)
    end
end
function SemiDefinitePhylogenyEstimator(; pl::NwE_ClE = NetworkEstimator(),
                                        p::Number = 0.05)::SemiDefinitePhylogenyEstimator
    return SemiDefinitePhylogenyEstimator(pl, p)
end
"""
    const MatNum_PhRMatNum = Union{<:PhylogenyResult{<:MatNum}, <:MatNum}

Alias for a phylogeny result wrapping a numeric matrix or a numeric matrix directly.

Used internally to accept either a [`PhylogenyResult`](@ref) containing a numeric matrix or a plain numeric matrix as a constraint matrix input.

# Related

  - [`PhylogenyResult`](@ref)
  - [`MatNum`](@ref)
"""
const MatNum_PhRMatNum = Union{<:PhylogenyResult{<:MatNum}, <:MatNum}
"""
$(DocStringExtensions.TYPEDEF)

Drives the product of weights of every related pair of assets to zero through a semidefinite relaxation.

Relatedness is whatever the source that built `A` calls related: a neighbourhood over a network, or membership of one cluster.

# Mathematical definition

The relaxation replaces the outer product of the weights with a symmetric matrix variable and bounds it below through a Schur complement. Where the objective is already a trace against that variable — a variance — the constraint form applies:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\, \\mathbf{W}}{\\min}\\quad & \\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W})\\\\
\\textrm{s.t.}\\quad & \\begin{bmatrix} \\mathbf{W} & \\boldsymbol{w} \\\\ \\boldsymbol{w}^\\intercal & k \\end{bmatrix} \\succeq 0\\,,\\\\
& \\mathbf{A} \\odot \\mathbf{W} = 0\\,,\\\\
& \\mathbf{W} \\in \\mathbb{S}^{N}\\,,\\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

For every other risk measure nothing in the objective pulls the relaxation down, so a penalty term does it instead:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\, \\mathbf{W}}{\\min}\\quad & \\phi(\\boldsymbol{w}) + p\\, \\mathrm{tr}(\\mathbf{W})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\mathbf{W}``: Symmetric ``N \\times N`` matrix variable that relaxes ``\\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``.
  - ``\\mathbf{A}``: Relatedness matrix, the `A` field.
  - ``p``: Penalty factor, the `p` field.
  - $(math_dict[:k_budget])
  - ``\\mathbf{\\Sigma}``: Covariance matrix.
  - ``\\phi``: Risk measure of the optimiser.
  - ``\\odot``: Hadamard product.
  - ``\\mathbb{S}^{N}``: Set of real symmetric ``N \\times N`` matrices.
  - ``\\mathcal{W}``: Rest of the feasible set.

The two branches are what the code does: [`set_sdp_phylogeny_constraints!`](@ref) always writes `A ⊙ W == 0` and adds `p * tr(W)` to the objective penalty **only** when the model carries no variance. On a 250×6 sample the Hadamard constraint holds to `2.1e-18` under a variance objective and to `3.1e-21` under a conditional-value-at-risk objective.

`p` therefore sets the size of the **relaxation gap**, not the strength of the constraint. `A ⊙ W == 0` is weight-inert: it is the same constraint at any magnitude of `A`. What `p` buys is how closely `W` tracks the outer product it stands for. On the same sample the largest entry of ``\\mathbf{W} - \\boldsymbol{w}\\boldsymbol{w}^\\intercal`` is `2.7e-5` under the variance objective, where the objective itself closes the gap, against `0.0274` under conditional value at risk at the default `p = 0.05` — three orders of magnitude wider. A wider gap lets a pair of related assets both hold weight while their entry of `W` absorbs the product.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SemiDefinitePhylogeny(
        A::MatNum_PhRMatNum,
        p::Number
    ) -> SemiDefinitePhylogeny
    SemiDefinitePhylogeny(;
        A::MatNum_PhRMatNum,
        p::Number = 0.05
    ) -> SemiDefinitePhylogeny

Keywords correspond to the struct's fields. The default `p = 0.05` is the value the source's own worked example uses.

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

  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:graphpo1])
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Sections 13.1.7.2 and 13.2.4.2.
"""
@concrete struct SemiDefinitePhylogeny <: AbstractPhylogenyConstraintResult
    """
    $(field_dict[:A_phylo])
    """
    A
    """
    $(field_dict[:p_phylo])
    """
    p
    function SemiDefinitePhylogeny(A::MatNum, p::Number)::SemiDefinitePhylogeny
        @argcheck(all(iszero, LinearAlgebra.diag(A)),
                  ArgumentError("all diagonal entries of A must be zero"))
        @argcheck(LinearAlgebra.issymmetric(A),
                  ArgumentError("A must be a symmetric matrix"))
        @argcheck(p >= zero(p), DomainError(p, "p must be >= 0"))
        return new{typeof(A), typeof(p)}(A, p)
    end
end
function SemiDefinitePhylogeny(A::PhylogenyResult{<:MatNum},
                               p::Number)::SemiDefinitePhylogeny
    return SemiDefinitePhylogeny(A.X, p)
end
function SemiDefinitePhylogeny(; A::MatNum_PhRMatNum,
                               p::Number = 0.05)::SemiDefinitePhylogeny
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
function _validate_length_integer_phylogeny_constraint_B(alg::Integer, B::VecNum)::Nothing
    @argcheck(length(B) <= alg,
              DomainError("`length(B) <= alg`:\nlength(B) => $(length(B))\nalg => $(alg)"))
    return nothing
end
function _validate_length_integer_phylogeny_constraint_B(args...)::Nothing
    return nothing
end
"""
    validate_length_integer_phylogeny_constraint_B(cle::ClustersEstimator, B::VecNum)
    validate_length_integer_phylogeny_constraint_B(args...)

Validate that the length of the vector `B` does not exceed the maximum allowed by the clustering estimator `cle`.

# Arguments

  - `cle`: Clustering estimator containing algorithm and maximum group information.
  - `B`: Vector of integers representing group sizes or allocations.
  - `args...`: No validation is performed.

# Returns

  - `nothing`.

# Validation

  - Throws `DomainError` if `length(B) > cle.onc.max_k` (when `max_k` is set).
  - Calls internal [`_validate_length_integer_phylogeny_constraint_B`](@ref) for further checks.

# Details

  - Checks if `cle.onc.max_k` is set and validates `length(B)` accordingly.
  - Delegates to `_validate_length_integer_phylogeny_constraint_B` for algorithm-specific validation.
  - Used in the construction and validation of integer phylogeny constraints.

# Related

  - [`_validate_length_integer_phylogeny_constraint_B`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
"""
function validate_length_integer_phylogeny_constraint_B(cle::ClustersEstimator, B::VecNum)
    if !isnothing(cle.onc.max_k)
        @argcheck(length(B) <= cle.onc.max_k,
                  DomainError("`length(B) <= cle.onc.max_k`:\nlength(B) => $(length(B))\npe.onc.max_k => $(cle.onc.max_k)"))
    end
    _validate_length_integer_phylogeny_constraint_B(cle.onc.alg, B)
    return nothing
end
function validate_length_integer_phylogeny_constraint_B(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Caps how many related assets may be held at once, refitting the structure from returns.

The estimator holds the source that builds the relatedness matrix and the cap `B`. [`phylogeny_constraints`](@ref) refits the source against a returns matrix and returns an [`IntegerPhylogeny`](@ref), which carries the equations this constraint solves.

Which pairs a network source relates comes from its [`AbstractSeparationAlgorithm`](@ref), and `B` is an integer cardinality counted over them. The relatedness itself stays binary under either separation: [`PhylogenyResult`](@ref)'s matrix is `Int`, and a graded one would not be countable here.

!!! warning

    `NetworkEstimator(; sep = PathLength())` relates **every reachable pair**. A bare [`PathLength`](@ref) leaves `dmax = nothing`, which resolves to the observed diameter, so nothing is outside the budget — measured, `190` of `190` pairs. It is the opposite end of the dial from [`HopCount`](@ref)'s default `n = 1`. State a numeric `dmax` to select anything narrower.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IntegerPhylogenyEstimator(;
        pl::NwE_ClE = NetworkEstimator(),
        B::Int_VecInt = 1
    ) -> IntegerPhylogenyEstimator

Keywords correspond to the struct's fields.

## Validation

  - `B` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

      + `AbstractVector`: It is additionally validated with [`validate_length_integer_phylogeny_constraint_B`](@ref).

  - `pl` is bounded by [`NwE_ClE`](@ref): a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref) is rejected by the type, not by a check, so the keyword constructor raises `TypeError` rather than deferring the problem to a solve. Build [`IntegerPhylogeny`](@ref) instead, which is what `phylogeny_constraints(estimator, X)` returns.

# Examples

```jldoctest
julia> IntegerPhylogenyEstimator()
IntegerPhylogenyEstimator
  pl ┼ NetworkEstimator
     │    ce ┼ PortfolioOptimisersCovariance
     │       │   ce ┼ Covariance
     │       │      │    me ┼ SimpleExpectedReturns
     │       │      │       │   w ┴ nothing
     │       │      │    ce ┼ GeneralCovariance
     │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │      │       │    w ┴ nothing
     │       │      │   alg ┴ FullMoment()
     │       │   mp ┼ MatrixProcessing
     │       │      │     pdm ┼ Posdef
     │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │       │      │      dn ┼ nothing
     │       │      │      dt ┼ nothing
     │       │      │     alg ┼ nothing
     │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
     │    de ┼ Distance
     │       │   power ┼ nothing
     │       │     alg ┴ CanonicalDistance()
     │   alg ┼ KruskalTree
     │       │     args ┼ Tuple{}: ()
     │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │   sep ┼ HopCount
     │       │   n ┴ Int64: 1
   B ┴ Int64: 1
```

# Related

  - [`IntegerPhylogeny`](@ref)
  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:riccascozzari2024])
  - $(ref_dict[:graphpo1])
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Sections 13.1.7.1 and 13.2.4.1.
"""
@concrete struct IntegerPhylogenyEstimator <: AbstractPhylogenyConstraintEstimator
    """
    $(field_dict[:plsrc])
    """
    pl
    """
    $(field_dict[:B_phylo])
    """
    B
    function IntegerPhylogenyEstimator(pl::NwE_ClE,
                                       B::Int_VecInt)::IntegerPhylogenyEstimator
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, VecInt)
            validate_length_integer_phylogeny_constraint_B(pl, B)
        end
        return new{typeof(pl), typeof(B)}(pl, B)
    end
end
function IntegerPhylogenyEstimator(; pl::NwE_ClE = NetworkEstimator(),
                                   B::Int_VecInt = 1)::IntegerPhylogenyEstimator
    return IntegerPhylogenyEstimator(pl, B)
end
"""
$(DocStringExtensions.TYPEDEF)

Caps at `B` the number of related assets a mixed-integer model may hold at once.

Relatedness is whatever the source that built `A` calls related: a neighbourhood over a network, or membership of one cluster.

# Mathematical definition

The cap is a mutually exclusive investment constraint on the held binary:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}}\\quad & \\phi(\\boldsymbol{w})\\\\
\\textrm{s.t.}\\quad & \\mathbf{A} \\boldsymbol{z} \\leq \\boldsymbol{B}\\,,\\\\
& \\boldsymbol{\\ell} \\odot \\boldsymbol{z} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u} \\odot \\boldsymbol{z}\\,,\\\\
& \\boldsymbol{z} \\in \\{0, 1\\}^{N}\\,,\\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{z}``: Held binary, one entry per asset.
  - ``\\mathbf{A}``: Stored relatedness rows, the `A` field.
  - ``\\boldsymbol{B}``: Cap, the `B` field.
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper weight bounds.
  - ``\\phi``: Objective function of the optimiser.
  - ``\\odot``: Hadamard product.
  - ``\\mathcal{W}``: Rest of the feasible set.

The constructor stores `unique(A + I; dims = 1)`, **not** the matrix it is given. The identity adds each asset to its own row, and the deduplication drops repeated rows, so one row survives per distinct neighbourhood or cluster. This is why the stored `A` is usually shorter than it is wide, and why a vector `B` is checked against the stored row count rather than against the number of assets. A network source with `B = 1` then forbids holding two assets that are neighbours; a clustering source with `B = 1` holds at most one asset per cluster.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IntegerPhylogeny(
        A::MatNum_PhRMatNum,
        B::Int_VecInt
    ) -> IntegerPhylogeny
    IntegerPhylogeny(;
        A::MatNum_PhRMatNum,
        B::Int_VecInt = 1
    ) -> IntegerPhylogeny

Keywords correspond to the struct's fields.

## Validation

  - `LinearAlgebra.issymmetric(A)` and `all(iszero, LinearAlgebra.diag(A))`.

  - `B` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

      + `AbstractVector`: `size(unique(A + LinearAlgebra.I; dims = 1), 1) == length(B)`.

# Examples

```jldoctest
julia> IntegerPhylogeny(; A = [0.0 1.0; 1.0 0.0], B = 2)
IntegerPhylogeny
  A ┼ 1×2 Matrix{Float64}
  B ┴ Int64: 2
```

# Related

  - [`set_iplg_constraints!`](@ref)
  - [`mip_constraints`](@ref)
  - [`IntegerPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:riccascozzari2024])
  - $(ref_dict[:graphpo1])
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Sections 13.1.7.1 and 13.2.4.1.
"""
@concrete struct IntegerPhylogeny <: AbstractPhylogenyConstraintResult
    """
    $(field_dict[:A_iphylo])
    """
    A
    """
    $(field_dict[:B_phylo])
    """
    B
    function IntegerPhylogeny(A::MatNum, B::Int_VecInt)::IntegerPhylogeny
        @argcheck(all(iszero, LinearAlgebra.diag(A)),
                  ArgumentError("all diagonal entries of A must be zero"))
        @argcheck(LinearAlgebra.issymmetric(A),
                  ArgumentError("A must be a symmetric matrix"))
        A = unique(A + LinearAlgebra.I; dims = 1)
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, VecInt)
            @argcheck(size(A, 1) == length(B),
                      DimensionMismatch("size(A, 1) ($(size(A, 1))) must match length(B) ($(length(B)))"))
        end
        return new{typeof(A), typeof(B)}(A, B)
    end
end
function IntegerPhylogeny(A::PhylogenyResult{<:MatNum}, B::Int_VecInt)::IntegerPhylogeny
    return IntegerPhylogeny(A.X, B)
end
function IntegerPhylogeny(; A::MatNum_PhRMatNum, B::Int_VecInt = 1)::IntegerPhylogeny
    return IntegerPhylogeny(A, B)
end
"""
    phylogeny_constraints(plc::Option{<:PlCE_PlC}, X::MatNum; dims::Int = 1, kwargs...)
    phylogeny_constraints(plcs::VecPlCE_PlC, args...; kwargs...)

Generate phylogeny-based portfolio constraints from an estimator or result.

`phylogeny_constraints` constructs constraint objects based on phylogenetic, res, or network structures among assets. It supports both semi-definite and integer constraint forms, accepting either an estimator (which wraps a phylogeny or clustering model and penalty parameters) or a precomputed result. If `plc` is `nothing`, returns `nothing`.

If `plcs` is a vector, this method broadcasts over each element, returning a vector of constraint results.

# Arguments

  - `plc`: A phylogeny constraint estimator, result, or `nothing`.
  - `X`: Data matrix (`observations × assets`) (ignored when `plc` is not an estimator).
  - $(arg_dict[:dims])
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
    return SemiDefinitePhylogeny(; A = phylogeny_matrix(plc.pl, X; dims = dims, kwargs...),
                                 p = plc.p)
end
function phylogeny_constraints(plc::IntegerPhylogenyEstimator, X::MatNum; dims::Int = 1,
                               kwargs...)
    return IntegerPhylogeny(; A = phylogeny_matrix(plc.pl, X; dims = dims, kwargs...),
                            B = plc.B)
end
function phylogeny_constraints(plc::Option{<:AbstractPhylogenyConstraintResult}, args...;
                               kwargs...)
    return plc
end
function phylogeny_constraints(plcs::VecPlCE_PlC, args...; kwargs...)
    return [phylogeny_constraints(plc, args...; kwargs...) for plc in plcs]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all centrality-based constraint types.

All concrete types implementing centrality-based portfolio constraints should be subtypes of `AbstractCentralityConstraint`.

A subtype states a bound on a linear function of the weights whose coefficients come from a graph, so [`centrality_constraints`](@ref) reduces it to a [`LinearConstraint`](@ref) before the model sees it.

# Related

  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
  - [`centrality_constraints`](@ref)
  - [`LinearConstraint`](@ref)
"""
abstract type AbstractCentralityConstraint <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the average centrality of the portfolio against a threshold read off the centrality vector itself.

The constraint diversifies by the influence an asset has in the network, rather than by its weight. [`centrality_constraints`](@ref) turns it into one row of a [`LinearConstraint`](@ref).

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}}\\quad & \\phi(\\boldsymbol{w})\\\\
\\textrm{s.t.}\\quad & \\boldsymbol{c}^\\intercal \\boldsymbol{w} \\mathbin{\\square} \\bar{c}\\,,\\\\
& \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{c}``: Centrality vector the estimator in `A` computes.
  - ``\\bar{c}``: Threshold, the `B` field.
  - ``\\square``: Comparison operator, the `comp` field.
  - ``\\phi``: Objective function of the optimiser.
  - ``\\mathcal{W}``: Rest of the feasible set.

The source states this constraint as an equality against a desired average centrality. `comp` widens that: `==` builds an equality row and every other operator an inequality row.

When `B` is a [`VectorToScalarMeasure`](@ref), ``\\bar{c}`` is that measure of ``\\boldsymbol{c}``, so the threshold moves with the graph and the constraint always has a feasible point. The measure reads the centrality vector, never the row after `comp` has flipped its sign: `MinValue()` gives the smallest entry under `<=` and under `>=` alike. On an eight-asset degree vector both give `0.14285714285714285`, and `MaxValue()` gives `0.42857142857142855`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CentralityConstraint(;
        A::CentralityEstimator = CentralityEstimator(),
        B::Num_VecToScaM = MinValue(),
        comp::ComparisonOperator = <=
    ) -> CentralityConstraint

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CentralityConstraint()
CentralityConstraint
     A ┼ CentralityEstimator
       │   pl ┼ NetworkEstimator
       │      │    ce ┼ PortfolioOptimisersCovariance
       │      │       │   ce ┼ Covariance
       │      │       │      │    me ┼ SimpleExpectedReturns
       │      │       │      │       │   w ┴ nothing
       │      │       │      │    ce ┼ GeneralCovariance
       │      │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
       │      │       │      │       │    w ┴ nothing
       │      │       │      │   alg ┴ FullMoment()
       │      │       │   mp ┼ MatrixProcessing
       │      │       │      │     pdm ┼ Posdef
       │      │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
       │      │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
       │      │       │      │      dn ┼ nothing
       │      │       │      │      dt ┼ nothing
       │      │       │      │     alg ┼ nothing
       │      │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
       │      │    de ┼ Distance
       │      │       │   power ┼ nothing
       │      │       │     alg ┴ CanonicalDistance()
       │      │   alg ┼ KruskalTree
       │      │       │     args ┼ Tuple{}: ()
       │      │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
       │      │   sep ┼ HopCount
       │      │       │   n ┴ Int64: 1
       │   ct ┼ DegreeCentrality
       │      │     kind ┼ Int64: 0
       │      │   kwargs ┴ @NamedTuple{}: NamedTuple()
     B ┼ MinValue()
  comp ┴ typeof(<=): <=
```

# Related

  - [`CentralityEstimator`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`ComparisonOperator`](@ref)
  - [`centrality_constraints`](@ref)
  - [`LinearConstraint`](@ref)

# References

  - $(ref_dict[:graphpo1])
  - $(ref_dict[:cajas2025]) Section 13.1.6.
"""
@concrete struct CentralityConstraint <: AbstractCentralityConstraint
    """
    $(field_dict[:cc_A])
    """
    A
    """
    $(field_dict[:cc_B])
    """
    B
    """
    $(field_dict[:cc_comp])
    """
    comp
    function CentralityConstraint(A::CentralityEstimator, B::Num_VecToScaM,
                                  comp::ComparisonOperator)
        return new{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
    end
end
function CentralityConstraint(; A::CentralityEstimator = CentralityEstimator(),
                              B::Num_VecToScaM = MinValue(), comp::ComparisonOperator = <=)
    return CentralityConstraint(A, B, comp)
end
"""
    const VecCC = AbstractVector{<:CentralityConstraint}

Alias for a vector of [`CentralityConstraint`](@ref) objects.

Represents a collection of centrality-based portfolio constraints.

# Related

  - [`CentralityConstraint`](@ref)
  - [`CC_VecCC`](@ref)
"""
const VecCC = AbstractVector{<:CentralityConstraint}
"""
    const CC_VecCC = Union{<:CentralityConstraint, <:VecCC}

Alias for a single or vector of [`CentralityConstraint`](@ref) objects.

Matches either a single [`CentralityConstraint`](@ref) or a vector of them ([`VecCC`](@ref)).

# Related

  - [`CentralityConstraint`](@ref)
  - [`VecCC`](@ref)
"""
const CC_VecCC = Union{<:CentralityConstraint, <:VecCC}
"""
    const Lc_CC_VecCC = Union{<:CC_VecCC, <:LinearConstraint}

Alias for a centrality constraint or linear constraint.

Matches either a [`CentralityConstraint`](@ref), a vector of them, or a [`LinearConstraint`](@ref). Used for dispatch in centrality-based constraint generation that also accepts linear constraints.

# Related

  - [`CC_VecCC`](@ref)
  - [`LinearConstraint`](@ref)
"""
const Lc_CC_VecCC = Union{<:CC_VecCC, <:LinearConstraint}
"""
    centrality_constraints(ccs::CC_VecCC,
                           X::MatNum; dims::Int = 1, kwargs...)

Generate centrality-based linear constraints from one or more `CentralityConstraint` estimators.

`centrality_constraints` constructs linear constraints for portfolio optimisation based on asset centrality measures within a phylogeny or network structure. It accepts one or more [`CentralityConstraint`](@ref) estimators, computes centrality vectors for the given data matrix `X`, applies the specified reduction measure or threshold, and assembles the resulting constraints into a [`LinearConstraint`](@ref) object.

# Arguments

  - `ccs`: A single [`CentralityConstraint`](@ref) or a vector of such estimators.
  - `X`: Data matrix (`observations × assets`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the centrality estimator.

# Returns

  - `lc::Option{<:LinearConstraint}`: An object containing the assembled inequality and equality constraints, or `nothing` if no constraints are present.

# Details

  - For each constraint, computes the centrality vector using the estimator in `cc.A`.
  - Derives the threshold from `cc.B`. A number is the threshold itself; a [`VectorToScalarMeasure`](@ref) is applied to the centrality vector.
  - Negates the row and the threshold together for an operator that points the other way, so every inequality is stored in the `A w <= B` form.
  - Skips a constraint whose centrality vector is empty or all zero.
  - Aggregates constraints into equality and inequality forms.
  - Returns `nothing` if no valid constraints are generated.

The threshold is derived **before** the negation, and negated once with the row. Deriving it from the negated row instead negates it a second time, which cancels the flip and turns a `MinValue()` into a `MaxValue()` with the wrong sign.

# Related

  - [`CentralityConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_constraints(ccs::CC_VecCC, X::MatNum; dims::Int = 1, kwargs...)
    if isa(ccs, AbstractVector)
        @argcheck(!isempty(ccs), IsEmptyError("ccs cannot be empty"))
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
        # The measure reads the centrality vector, never the sign-flipped row. `d` negates the
        # row and the right-hand side together, once each; deriving the threshold from `d * A`
        # negates it a second time, which cancels the flip and swaps a min for a max.
        B = d * vec_to_real_measure(cc.B, A)
        A .*= d
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
function centrality_constraints(lc::Option{<:LinearConstraint}, args...; kwargs...)
    return lc
end

export SemiDefinitePhylogenyEstimator, SemiDefinitePhylogeny, IntegerPhylogenyEstimator,
       IntegerPhylogeny, CentralityConstraint, phylogeny_constraints, centrality_constraints
