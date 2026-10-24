"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny-based constraint estimators.

A subtype holds the configuration that fits the relatedness of the assets to returns. [`phylogeny_constraints`](@ref) turns it into an [`AbstractPhylogenyConstraintResult`](@ref).

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

A subtype holds a fitted relatedness matrix and the parameter of the constraint that the model writes on it.

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

Matches an [`AbstractPhylogenyConstraintEstimator`](@ref) or an [`AbstractPhylogenyConstraintResult`](@ref). The library dispatches on it where a function accepts a phylogeny constraint estimator or a precomputed result.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
"""
const PlCE_PlC = Union{<:AbstractPhylogenyConstraintEstimator,
                       <:AbstractPhylogenyConstraintResult}
"""
    const VecPlCE_PlC = AbstractVector{<:PlCE_PlC}

Alias for a vector of phylogeny constraint estimators or results.

[`phylogeny_constraints`](@ref) maps such a vector to a vector of results, in the same order.

# Related

  - [`PlCE_PlC`](@ref)
"""
const VecPlCE_PlC = AbstractVector{<:PlCE_PlC}
"""
    const PlCE_PlC_VecPlCE_PlC = Union{<:PlCE_PlC, <:VecPlCE_PlC}

Alias for one phylogeny constraint estimator or result, or a vector of them.

Matches a [`PlCE_PlC`](@ref) or a [`VecPlCE_PlC`](@ref). It bounds the `ple` keyword of [`JuMPOptimiser`](@ref) and the `frc_ple` keyword of [`FactorRiskContribution`](@ref), so several phylogeny constraints can hold at once.

# Related

  - [`PlCE_PlC`](@ref)
  - [`VecPlCE_PlC`](@ref)
"""
const PlCE_PlC_VecPlCE_PlC = Union{<:PlCE_PlC, <:VecPlCE_PlC}
"""
    const VecPlC = AbstractVector{<:AbstractPhylogenyConstraintResult}

Alias for a vector of phylogeny constraint results.

Matches the vector that [`phylogeny_constraints`](@ref) returns for a vector of estimators and results.

# Related

  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`PlC_VecPlC`](@ref)
"""
const VecPlC = AbstractVector{<:AbstractPhylogenyConstraintResult}
"""
    const PlC_VecPlC = Union{<:AbstractPhylogenyConstraintResult, <:VecPlC}

Alias for one phylogeny constraint result, or a vector of them.

Matches an [`AbstractPhylogenyConstraintResult`](@ref) or a [`VecPlC`](@ref). It bounds the fitted phylogeny constraints that the model builders read.

# Related

  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`VecPlC`](@ref)
"""
const PlC_VecPlC = Union{<:AbstractPhylogenyConstraintResult, <:VecPlC}
"""
$(DocStringExtensions.TYPEDEF)

Fits the relatedness of the assets to returns, and drives the product of the weights of each related pair to zero through a semidefinite relaxation.

The estimator holds the source that builds the relatedness matrix, and the penalty factor `p`. [`phylogeny_constraints`](@ref) fits the source to a returns matrix and returns a [`SemiDefinitePhylogeny`](@ref), which holds the matrix and `p` for the model.

The [`AbstractSeparationAlgorithm`](@ref) of a network source decides which pairs are related. No field of this estimator changes that set. The row `A ⊙ W == 0` reads only which entries of `A` are non-zero, not their size. So the separation algorithm changes the number of forbidden pairs, and nothing else.

[graphpo1](@citet) states the constraint over a network. Section 13.2.4.2 of [cajas2025](@cite) names [graphpo2](@cite) as the source of the same constraint over clusters. No copy of that work is available.

!!! warning

    `NetworkEstimator(; sep = PathLength())` relates every pair of assets that a path joins. A bare [`PathLength`](@ref) leaves `dmax = nothing`, which resolves to the observed diameter, so no pair is outside the budget. On a tree of 20 assets it relates all 190 pairs, and the constraint then drives the product of the weights of every pair to zero. [`HopCount`](@ref) with its default `n = 1` is the other extreme, and relates only the neighbours. Set a numeric `dmax` to relate fewer pairs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SemiDefinitePhylogenyEstimator(;
        pl::NwE_ClE = NetworkEstimator(),
        p::Number = 0.05,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> SemiDefinitePhylogenyEstimator

Keywords correspond to the struct's fields. The default `p = 0.05` is the penalty of the worked example in Section 6.2 of [graphpo1](@cite).

## Validation

  - `p >= 0`.
  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). Otherwise the constructor raises an `ArgumentError`.
  - `returns_source(pl) === nothing`, see [`assert_inner_reader`](@ref). The network or clustering estimator reads the matrix that this estimator selects, so it states no source of its own. Otherwise the constructor raises an `ArgumentError`.
  - The type bound [`NwE_ClE`](@ref) of `pl` refuses a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref). So the keyword constructor raises a `TypeError` at once, and not later in a solve. To use a precomputed structure, build the [`SemiDefinitePhylogeny`](@ref) that `phylogeny_constraints(estimator, X)` returns.

# Examples

```jldoctest
julia> SemiDefinitePhylogenyEstimator()
SemiDefinitePhylogenyEstimator
     pl ┼ NetworkEstimator
        │      ce ┼ PortfolioOptimisersCovariance
        │         │   ce ┼ Covariance
        │         │      │    me ┼ SimpleExpectedReturns
        │         │      │       │   w ┴ nothing
        │         │      │    ce ┼ GeneralCovariance
        │         │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │         │      │       │    w ┴ nothing
        │         │      │   alg ┼ FullMoment()
        │         │      │     w ┴ nothing
        │         │   mp ┼ MatrixProcessing
        │         │      │     pdm ┼ Posdef
        │         │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │         │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │         │      │      dn ┼ nothing
        │         │      │      dt ┼ nothing
        │         │      │     alg ┼ nothing
        │         │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │      de ┼ Distance
        │         │   power ┼ nothing
        │         │     alg ┴ CanonicalDistance()
        │     alg ┼ KruskalTree
        │         │     args ┼ Tuple{}: ()
        │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │     sep ┼ HopCount
        │         │   n ┴ Int64: 1
        │   x_src ┴ nothing
      p ┼ Float64: 0.05
  x_src ┴ nothing
```

# Related

  - [`SemiDefinitePhylogeny`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:graphpo1]) Section 5.2, Equations 10 and 11.
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Section 13.2.4.2, Equation 13.15. $(ref_dict[:no_original_source])
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
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function SemiDefinitePhylogenyEstimator(pl::NwE_ClE, p::Number,
                                            x_src::Option{<:AbstractReturnsSource})::SemiDefinitePhylogenyEstimator
        assert_nonneg(p, :p)
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :SemiDefinitePhylogenyEstimator)
        assert_inner_reader(pl, :pl, :SemiDefinitePhylogenyEstimator)
        return new{typeof(pl), typeof(p), typeof(x_src)}(pl, p, x_src)
    end
end
function SemiDefinitePhylogenyEstimator(; pl::NwE_ClE = NetworkEstimator(),
                                        p::Number = 0.05,
                                        x_src::Option{<:AbstractReturnsSource} = nothing)::SemiDefinitePhylogenyEstimator
    return SemiDefinitePhylogenyEstimator(pl, p, x_src)
end
"""
    const MatNum_PhRMatNum = Union{<:PhylogenyResult{<:MatNum}, <:MatNum}

Alias for a numeric matrix, or a phylogeny result that holds one.

The constructors of [`SemiDefinitePhylogeny`](@ref) and [`IntegerPhylogeny`](@ref) accept it for `A`. They read the matrix out of a [`PhylogenyResult`](@ref).

# Related

  - [`PhylogenyResult`](@ref)
  - [`MatNum`](@ref)
"""
const MatNum_PhRMatNum = Union{<:PhylogenyResult{<:MatNum}, <:MatNum}
"""
$(DocStringExtensions.TYPEDEF)

Drives the product of weights of every related pair of assets to zero through a semidefinite relaxation.

The source that built `A` decides which assets are related. A network source relates the assets of one neighbourhood, and a clustering source relates the assets of one cluster. [graphpo1](@citet) states the constraint over a network. Section 13.2.4.2 of [cajas2025](@cite) names [graphpo2](@cite) as the source of the same constraint over clusters. No copy of that work is available.

`p` sets the size of the relaxation gap, not the strength of the constraint. The model writes `A ⊙ W == 0` for every `p`, so a larger `p` forbids no more pairs. A larger `p` only holds `W` closer to the outer product of the weights. When the gap is wide, both assets of a related pair can hold weight while their entry of `W` is zero.

A [`Variance`](@ref) in a model with this constraint always takes the constraint form below, whatever its `alg` field states. Under [`MaximumRatio`](@ref), the `ratio_alg` of the variance selects the degree of its objective term, and so the ratio. The default gives the Sharpe ratio, and the `# Mathematical definition` of [`MaximumRatio`](@ref) states the rule. Under that objective, [`set_sdp_phylogeny_constraints!`](@ref) writes the rows on the lifted matrix of [`set_sdp_objective_lift!`](@ref).

# Mathematical definition

The relaxation replaces the outer product of the weights with a symmetric matrix variable, and bounds it below through a Schur complement. A variance objective is already a trace against that variable, so the constraint form applies:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\, \\mathbf{W}}{\\min}\\quad & \\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W})\\\\
\\textrm{s.t.}\\quad & \\begin{bmatrix} \\mathbf{W} & \\boldsymbol{w} \\\\ \\boldsymbol{w}^\\intercal & k \\end{bmatrix} \\succeq 0\\,,\\\\
& \\mathbf{A} \\odot \\mathbf{W} = 0\\,,\\\\
& \\mathbf{W} \\in \\mathbb{S}^{N}\\,,\\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

For every other risk measure, no term of the objective holds the matrix variable down to the outer product. A penalty on its trace does that instead:

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

Keywords correspond to the struct's fields. The default `p = 0.05` is the penalty of the worked example in Section 6.2 of [graphpo1](@cite).

## Validation

  - `LinearAlgebra.issymmetric(A)` and `all(iszero, LinearAlgebra.diag(A))`. A breach raises an `ArgumentError`.
  - `p >= 0`. A breach raises a `DomainError`.

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
  - [`set_sdp_phylogeny_constraints!`](@ref): writes `A ⊙ W == 0` for every result of this type, and adds `p * tr(W)` to the objective penalty unless the objective minimises a variance on the same weights.
  - [`SemiDefinitePhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyConstraintResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:graphpo1]) Section 5.2, Equations 10 and 11.
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Section 13.2.4.2, Equation 13.15. $(ref_dict[:no_original_source])
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
    _validate_length_integer_phylogeny_constraint_B(alg::Integer, B::VecNum)
    _validate_length_integer_phylogeny_constraint_B(args...)

Check the length of `B` against an integer cluster count `alg`.

The fallback method takes every other argument list and checks nothing, so a clustering algorithm that is not an integer count puts no bound on `B`. The default [`OptimalNumberClusters`](@ref) holds a [`SecondOrderDifference`](@ref), so this check does nothing for it.

# Arguments

  - `alg`: The number of clusters that the source returns.

      + `Integer`: Bounds `length(B)`.
      + Any other type: The function checks nothing.

  - `B`: The vector of bounds, one per row.

# Validation

  - `length(B) <= alg`, on the `Integer` method only. A breach raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`validate_length_integer_phylogeny_constraint_B`](@ref): the caller, which applies the `max_k` bound before it delegates here.
  - [`IntegerPhylogenyEstimator`](@ref)
"""
function _validate_length_integer_phylogeny_constraint_B(alg::Integer, B::VecNum)::Nothing
    @argcheck(length(B) <= alg,
              DomainError((length(B), alg),
                          "`B` holds $(length(B)) entries, and `alg` fixes $alg clusters. Each entry of `B` bounds one cluster, so `length(B) <= alg` must hold."))
    return nothing
end
function _validate_length_integer_phylogeny_constraint_B(args...)::Nothing
    return nothing
end
"""
    validate_length_integer_phylogeny_constraint_B(cle::ClustersEstimator, B::VecNum)
    validate_length_integer_phylogeny_constraint_B(args...)

Check the length of `B` against the number of clusters that the estimator `cle` can return.

Only the [`ClustersEstimator`](@ref) method checks `B`. Every other argument list goes to the fallback, which checks nothing. A [`NetworkEstimator`](@ref) goes there too, so this function does not check the default source of an [`IntegerPhylogenyEstimator`](@ref). The row count of a network is known only after [`phylogeny_matrix`](@ref) runs, and the constructor of [`IntegerPhylogeny`](@ref) checks the length then.

# Arguments

  - `cle`: The clustering estimator whose `onc` field states the number of clusters.
  - `B`: The vector of bounds, one per row.
  - `args...`: Any other arguments. The fallback checks nothing.

# Validation

  - `length(B) <= cle.onc.max_k`, when `max_k` is set. A breach raises a `DomainError`.
  - `length(B) <= cle.onc.alg`, through [`_validate_length_integer_phylogeny_constraint_B`](@ref), when `alg` is an `Integer`.
  - The default `OptimalNumberClusters()` sets neither bound. Its `max_k` is `nothing` and its `alg` is a [`SecondOrderDifference`](@ref), so a `B` of any length passes.

# Returns

  - `nothing`.

# Related

  - [`_validate_length_integer_phylogeny_constraint_B`](@ref): the delegate that applies the `alg` bound.
  - [`IntegerPhylogenyEstimator`](@ref)
"""
function validate_length_integer_phylogeny_constraint_B(cle::ClustersEstimator, B::VecNum)
    if !isnothing(cle.onc.max_k)
        @argcheck(length(B) <= cle.onc.max_k,
                  DomainError((length(B), cle.onc.max_k),
                              "`B` holds $(length(B)) entries, and `cle.onc.max_k` caps the clustering at $(cle.onc.max_k) clusters. Each entry of `B` bounds one cluster, so `length(B) <= cle.onc.max_k` must hold."))
    end
    _validate_length_integer_phylogeny_constraint_B(cle.onc.alg, B)
    return nothing
end
function validate_length_integer_phylogeny_constraint_B(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Fits the relatedness of the assets to returns, and caps the number of related assets that the portfolio holds at once.

The estimator holds the source that builds the relatedness matrix, and the cap `B`. [`phylogeny_constraints`](@ref) fits the source to a returns matrix and returns an [`IntegerPhylogeny`](@ref), which holds the matrix and `B` for the model.

The [`AbstractSeparationAlgorithm`](@ref) of a network source decides which pairs are related, and `B` counts the held assets over them. The relatedness is binary under every separation algorithm. The matrix of a [`PhylogenyResult`](@ref) is an `Int` matrix. A cap cannot count over a graded relation.

!!! warning

    `NetworkEstimator(; sep = PathLength())` relates every pair of assets that a path joins. A bare [`PathLength`](@ref) leaves `dmax = nothing`, which resolves to the observed diameter, so no pair is outside the budget. On a tree of 20 assets it relates all 190 pairs. [`HopCount`](@ref) with its default `n = 1` is the other extreme, and relates only the neighbours. Set a numeric `dmax` to relate fewer pairs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IntegerPhylogenyEstimator(;
        pl::NwE_ClE = NetworkEstimator(),
        B::Int_VecInt = 1,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> IntegerPhylogenyEstimator

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `B`.

      + `AbstractVector`: [`validate_length_integer_phylogeny_constraint_B`](@ref) also checks it.

  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). Otherwise the constructor raises an `ArgumentError`.

  - `returns_source(pl) === nothing`, see [`assert_inner_reader`](@ref). The network or clustering estimator reads the matrix that this estimator selects, so it states no source of its own. Otherwise the constructor raises an `ArgumentError`.

  - The type bound [`NwE_ClE`](@ref) of `pl` refuses a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref). So the keyword constructor raises a `TypeError` at once, and not later in a solve. To use a precomputed structure, build the [`IntegerPhylogeny`](@ref) that `phylogeny_constraints(estimator, X)` returns.

# Examples

```jldoctest
julia> IntegerPhylogenyEstimator()
IntegerPhylogenyEstimator
     pl ┼ NetworkEstimator
        │      ce ┼ PortfolioOptimisersCovariance
        │         │   ce ┼ Covariance
        │         │      │    me ┼ SimpleExpectedReturns
        │         │      │       │   w ┴ nothing
        │         │      │    ce ┼ GeneralCovariance
        │         │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │         │      │       │    w ┴ nothing
        │         │      │   alg ┼ FullMoment()
        │         │      │     w ┴ nothing
        │         │   mp ┼ MatrixProcessing
        │         │      │     pdm ┼ Posdef
        │         │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │         │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │         │      │      dn ┼ nothing
        │         │      │      dt ┼ nothing
        │         │      │     alg ┼ nothing
        │         │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │      de ┼ Distance
        │         │   power ┼ nothing
        │         │     alg ┴ CanonicalDistance()
        │     alg ┼ KruskalTree
        │         │     args ┼ Tuple{}: ()
        │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │     sep ┼ HopCount
        │         │   n ┴ Int64: 1
        │   x_src ┴ nothing
      B ┼ Int64: 1
  x_src ┴ nothing
```

# Related

  - [`IntegerPhylogeny`](@ref)
  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`phylogeny_constraints`](@ref)

# References

  - $(ref_dict[:riccascozzari2024])
  - $(ref_dict[:graphpo1]) Section 5.1, Equation 9.
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Section 13.2.4.1, Equation 13.14. $(ref_dict[:no_original_source])
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
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function IntegerPhylogenyEstimator(pl::NwE_ClE, B::Int_VecInt,
                                       x_src::Option{<:AbstractReturnsSource})::IntegerPhylogenyEstimator
        assert_nonempty_nonneg_finite_val(B, :B)
        if isa(B, VecInt)
            validate_length_integer_phylogeny_constraint_B(pl, B)
        end
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :IntegerPhylogenyEstimator)
        assert_inner_reader(pl, :pl, :IntegerPhylogenyEstimator)
        return new{typeof(pl), typeof(B), typeof(x_src)}(pl, B, x_src)
    end
end
function IntegerPhylogenyEstimator(; pl::NwE_ClE = NetworkEstimator(), B::Int_VecInt = 1,
                                   x_src::Option{<:AbstractReturnsSource} = nothing)::IntegerPhylogenyEstimator
    return IntegerPhylogenyEstimator(pl, B, x_src)
end
function returns_source(x::Union{<:SemiDefinitePhylogenyEstimator,
                                 <:IntegerPhylogenyEstimator})
    return x.x_src
end
function reader_frames(x::Union{<:SemiDefinitePhylogenyEstimator,
                                <:IntegerPhylogenyEstimator}, path::AbstractString, ::Bool)
    return source_frames(x.x_src, path, true)
end
"""
$(DocStringExtensions.TYPEDEF)

Caps at `B` the number of related assets that a mixed-integer model can hold at once.

The source that built `A` decides which assets are related. A network source relates the assets of one neighbourhood, and a clustering source relates the assets of one cluster. A clustering source with `B = 1` holds at most one asset per cluster.

A network source with `B = 1` holds at most one asset of each closed neighbourhood, which is an asset together with its neighbours. So it forbids two neighbours, and it also forbids two assets that share a neighbour. The held assets are thus at a distance of at least three in the network. On the graph of seven assets in Figure 13 of [graphpo1](@cite), 32 selections hold no two neighbours, but only 12 satisfy this constraint. The semidefinite form, [`SemiDefinitePhylogeny`](@ref), forbids only the pairs that `A` relates.

[riccascozzari2024](@citet) state the constraint for the neighbours of a network. [graphpo1](@citet) extends it to the assets that a walk of a given length links. Section 13.2.4.1 of [cajas2025](@cite) names [graphpo2](@cite) as the source of the constraint over clusters, and of a cap per cluster. No copy of that work is available.

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

# Algorithm

The constructor rewrites the matrix it is given, so the stored `A` is a derived quantity:

 1. Check that the diagonal of `A` is zero and that `A` is symmetric.
 2. Add the identity to `A`, which puts each asset in its own row.
 3. Drop the repeated rows of the sum with `unique(...; dims = 1)`. The result is the stored `A`. One row survives per distinct neighbourhood or cluster, so the stored matrix usually has fewer rows than columns.
 4. Check that `B` is non-empty, non-negative and finite.
 5. Check `size(A, 1) == length(B)` when `B` is a vector, against the row count of step 3 and not against the number of assets.

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

  - `LinearAlgebra.issymmetric(A)` and `all(iszero, LinearAlgebra.diag(A))`. A breach raises an `ArgumentError`.

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `B`.

      + `AbstractVector`: `size(unique(A + LinearAlgebra.I; dims = 1), 1) == length(B)`. A breach raises a `DimensionMismatch`.

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
  - $(ref_dict[:graphpo1]) Section 5.1, Equation 9.
  - $(ref_dict[:graphpo2])
  - $(ref_dict[:cajas2025]) Section 13.2.4.1, Equation 13.14. $(ref_dict[:no_original_source])
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

Build a phylogeny constraint result from an estimator, or pass a result through.

An estimator fits its network or clustering source to `X` with [`phylogeny_matrix`](@ref). A [`SemiDefinitePhylogenyEstimator`](@ref) gives a [`SemiDefinitePhylogeny`](@ref), and an [`IntegerPhylogenyEstimator`](@ref) gives an [`IntegerPhylogeny`](@ref). A precomputed result and `nothing` pass through unchanged, and the function builds nothing for them. A vector gives a vector of results in the same order.

# Arguments

  - `plc`: A phylogeny constraint estimator, a phylogeny constraint result, or `nothing`.
  - `plcs`: A vector of phylogeny constraint estimators and results.
  - `X`: The data matrix. Only an estimator reads it.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for [`phylogeny_matrix`](@ref). Only an estimator reads them.

# Returns

  - `res`: The phylogeny constraint result, or a vector of them for `plcs`.

      + `SemiDefinitePhylogeny`: For a semidefinite estimator or result.
      + `IntegerPhylogeny`: For an integer estimator or result.
      + `nothing`: When `plc` is `nothing`.

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

A subtype states a bound on a linear function of the weights whose coefficients come from a graph. [`centrality_constraints`](@ref) reduces it to a [`LinearConstraint`](@ref) before the optimiser builds the model.

# Related

  - [`CentralityConstraint`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
  - [`centrality_constraints`](@ref)
  - [`LinearConstraint`](@ref)
"""
abstract type AbstractCentralityConstraint <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the average centrality of the portfolio by a number, or by a measure of the centrality vector.

The constraint controls how central the portfolio is in the asset network, not how its weight is spread. [`centrality_constraints`](@ref) turns it into one row of a [`LinearConstraint`](@ref). Section 4 of [graphpo1](@cite) states the constraint as an equality against a target average centrality. This type also accepts either inequality.

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

When `B` is a [`VectorToScalarMeasure`](@ref), ``\\bar{c}`` is that measure of ``\\boldsymbol{c}``, so the threshold moves with the graph. Under [`MinValue`](@ref) or [`MaxValue`](@ref), ``\\bar{c}`` is an entry of ``\\boldsymbol{c}``. The portfolio fully invested in that asset then meets the constraint with equality, under every comparison operator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CentralityConstraint(;
        A::CentralityEstimator = CentralityEstimator(),
        B::Num_VecToScaM = MinValue(),
        comp::ComparisonOperator = <=,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> CentralityConstraint

Keywords correspond to the struct's fields.

## Validation

  - `A` is bounded by [`CentralityEstimator`](@ref), `B` by [`Num_VecToScaM`](@ref), and `comp` by [`ComparisonOperator`](@ref). A value outside a bound raises a `TypeError` from the keyword constructor.
  - `x_src in ALL_RETURNS_SOURCES`, see [`assert_returns_source`](@ref). Otherwise the constructor raises an `ArgumentError`.
  - `returns_source(A) === nothing`, see [`assert_inner_reader`](@ref). The centrality estimator reads the matrix that this constraint selects, so it states no source of its own. Otherwise the constructor raises an `ArgumentError`.
  - The constructor checks no value of `B`. A non-finite `B` reaches the generated row unchanged. [`centrality_constraints`](@ref) stores a `NaN` or an `Inf` right-hand side, and the model gives it to the solver.

# Examples

```jldoctest
julia> CentralityConstraint()
CentralityConstraint
      A ┼ CentralityEstimator
        │      pl ┼ NetworkEstimator
        │         │      ce ┼ PortfolioOptimisersCovariance
        │         │         │   ce ┼ Covariance
        │         │         │      │    me ┼ SimpleExpectedReturns
        │         │         │      │       │   w ┴ nothing
        │         │         │      │    ce ┼ GeneralCovariance
        │         │         │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │         │         │      │       │    w ┴ nothing
        │         │         │      │   alg ┼ FullMoment()
        │         │         │      │     w ┴ nothing
        │         │         │   mp ┼ MatrixProcessing
        │         │         │      │     pdm ┼ Posdef
        │         │         │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │         │         │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │         │         │      │      dn ┼ nothing
        │         │         │      │      dt ┼ nothing
        │         │         │      │     alg ┼ nothing
        │         │         │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │         │      de ┼ Distance
        │         │         │   power ┼ nothing
        │         │         │     alg ┴ CanonicalDistance()
        │         │     alg ┼ KruskalTree
        │         │         │     args ┼ Tuple{}: ()
        │         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │         │     sep ┼ HopCount
        │         │         │   n ┴ Int64: 1
        │         │   x_src ┴ nothing
        │      ct ┼ DegreeCentrality
        │         │     kind ┼ Int64: 0
        │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │   x_src ┴ nothing
      B ┼ MinValue()
   comp ┼ typeof(<=): <=
  x_src ┴ nothing
```

# Related

  - [`CentralityEstimator`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`ComparisonOperator`](@ref)
  - [`centrality_constraints`](@ref)
  - [`LinearConstraint`](@ref)

# References

  - $(ref_dict[:graphpo1]) Section 4, Equation 8.
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
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    function CentralityConstraint(A::CentralityEstimator, B::Num_VecToScaM,
                                  comp::ComparisonOperator,
                                  x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :CentralityConstraint)
        assert_inner_reader(A, :A, :CentralityConstraint)
        return new{typeof(A), typeof(B), typeof(comp), typeof(x_src)}(A, B, comp, x_src)
    end
end
function CentralityConstraint(; A::CentralityEstimator = CentralityEstimator(),
                              B::Num_VecToScaM = MinValue(), comp::ComparisonOperator = <=,
                              x_src::Option{<:AbstractReturnsSource} = nothing)
    return CentralityConstraint(A, B, comp, x_src)
end
function returns_source(cc::CentralityConstraint)
    return cc.x_src
end
function reader_frames(cc::CentralityConstraint, path::AbstractString, ::Bool)
    return source_frames(cc.x_src, path, true)
end
"""
    const VecCC = AbstractVector{<:CentralityConstraint}

Alias for a vector of [`CentralityConstraint`](@ref) objects.

[`centrality_constraints`](@ref) reduces such a vector to one [`LinearConstraint`](@ref), with at most one row per constraint.

# Related

  - [`CentralityConstraint`](@ref)
  - [`CC_VecCC`](@ref)
"""
const VecCC = AbstractVector{<:CentralityConstraint}
"""
    const CC_VecCC = Union{<:CentralityConstraint, <:VecCC}

Alias for one [`CentralityConstraint`](@ref), or a vector of them.

Matches a [`CentralityConstraint`](@ref) or a [`VecCC`](@ref). [`centrality_constraints`](@ref) dispatches on it.

# Related

  - [`CentralityConstraint`](@ref)
  - [`VecCC`](@ref)
"""
const CC_VecCC = Union{<:CentralityConstraint, <:VecCC}
"""
    const Lc_CC_VecCC = Union{<:CC_VecCC, <:LinearConstraint}

Alias for a centrality constraint, a vector of them, or a linear constraint.

Matches a [`CC_VecCC`](@ref) or a [`LinearConstraint`](@ref). It bounds the `cte` keyword of [`JuMPOptimiser`](@ref), which accepts the estimators or the linear constraint that they reduce to.

# Related

  - [`CC_VecCC`](@ref)
  - [`LinearConstraint`](@ref)
"""
const Lc_CC_VecCC = Union{<:CC_VecCC, <:LinearConstraint}
"""
    centrality_constraints(ccs::CC_VecCC, X::MatNum; dims::Int = 1, strict::Bool = false,
                           kwargs...)

Reduce one or more [`CentralityConstraint`](@ref) estimators to a [`LinearConstraint`](@ref).

Each estimator gives one row. The coefficients of the row are the centrality vector of the asset network. The right-hand side is a number, or a measure of that vector. A constraint whose centrality vector is empty or all zero gives no row, because its row does not depend on the weights. Two assets under [`BetweennessCentrality`](@ref) give the zero vector, and a lone such constraint makes this function return `nothing`. [`strict_diagnostic`](@ref) reports the drop, so `strict` decides whether it raises or warns.

The dropped row compares zero with `B`, so it does not depend on the weights. Every linear constraint builder drops such a row at construction time, and `strict = true` makes the drop raise an [`IsZeroError`](@ref).

# Algorithm

 1. Check that `ccs` is not empty when it is a vector.
 2. Check `dims` with [`assert_dims`](@ref), and read the asset count `N` from the axis of `X` that `dims` does not name. `N` is the width of every row the loop builds.
 3. For each `cc` in `ccs`, compute the centrality vector `A` with the estimator in `cc.A`, along `dims`.
 4. Report `cc` through [`strict_diagnostic`](@ref) with [`zero_centrality_msg`](@ref) and skip it when `A` is empty or all zero.
 5. Read the sign `d` and the inequality flag `flag_ineq` of `cc.comp` from [`comparison_sign_ineq_flag`](@ref).
 6. Derive the right-hand side `B` from `cc.B` against the unscaled `A` with [`vec_to_real_measure`](@ref), and scale it by `d`.
 7. Scale `A` by `d`, which negates a `>=` row so that every inequality is stored in the `A w <= B` sense.
 8. Append `A` and `B` to the inequality pair or to the equality pair, as `flag_ineq` selects.
 9. Reshape each flat accumulator into `N` columns, so that each surviving constraint gives one row. Build the [`LinearConstraint`](@ref) from the halves that hold a row. Return `nothing` when neither half does.

Step 6 comes before step 7 on purpose. A measure of the scaled row negates the threshold a second time. That cancels the sign flip, and turns a `MinValue()` into a `MaxValue()` with the wrong sign.

# Arguments

  - `ccs`: One [`CentralityConstraint`](@ref) or a vector of them.
  - `X`: The data matrix.
  - $(arg_dict[:dims])
  - `strict`: If `true`, a constraint whose centrality vector is empty or all zero raises an [`IsZeroError`](@ref). If `false`, it gives a warning. The function drops the constraint in both cases.
  - `kwargs...`: More keyword arguments for [`centrality_vector`](@ref).

# Validation

  - `!isempty(ccs)` when `ccs` is a vector. A breach raises an `IsEmptyError`.
  - $(val_dict[:dims]) The check is made with [`assert_dims`](@ref), which raises the `DomainError`.
  - A centrality vector that is empty or all zero raises an [`IsZeroError`](@ref) when `strict` is `true`, and gives a warning otherwise. The function drops the constraint in both cases.
  - The function checks no value of `cc.B`. A `NaN` or an `Inf` threshold reaches the returned right-hand side unchanged.

# Returns

  - `lc::Option{<:LinearConstraint}`: The inequality rows and the equality rows, or `nothing` when the function drops every constraint.

# Related

  - [`CentralityConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`centrality_vector`](@ref)
  - [`comparison_sign_ineq_flag`](@ref): the one sign and flag table, shared with [`get_linear_constraints`](@ref).
  - [`assert_dims`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`zero_centrality_msg`](@ref)
"""
function centrality_constraints(ccs::CC_VecCC, X::MatNum; dims::Int = 1,
                                strict::Bool = false, kwargs...)
    if isa(ccs, AbstractVector)
        @argcheck(!isempty(ccs), IsEmptyError("ccs cannot be empty"))
    end
    assert_dims(dims)
    N = size(X, isone(dims) ? 2 : 1)
    A_ineq = Vector{eltype(X)}(undef, 0)
    B_ineq = Vector{eltype(X)}(undef, 0)
    A_eq = Vector{eltype(X)}(undef, 0)
    B_eq = Vector{eltype(X)}(undef, 0)
    for cc in ccs
        A = centrality_vector(cc.A, X; dims = dims, kwargs...).X
        lhs_flag = isempty(A) || all(iszero, A)
        if lhs_flag
            # A zero centrality vector is a fact about the graph, not a mistyped name, so the
            # message is its own. The row it would build, `0' w <= B`, does not depend on `w`.
            strict_diagnostic(zero_centrality_msg(cc.A.ct, length(A)), strict;
                              id = @log_id(typeof(cc.A.ct)), exception = IsZeroError)
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
        A_ineq = transpose(reshape(A_ineq, N, :))
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, N, :))
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
    centrality_constraints(lc::Option{<:LinearConstraint}, args...; kwargs...)

Return a [`LinearConstraint`](@ref) or `nothing` unchanged.

A caller can give the linear constraint that the [`CentralityConstraint`](@ref) estimators reduce to, in place of the estimators. This method passes it through and builds nothing.

# Arguments

  - `lc`: A [`LinearConstraint`](@ref) or `nothing`.
  - `args...`: Any positional arguments. The method ignores them.
  - `kwargs...`: Any keyword arguments. The method ignores them.

# Returns

  - `lc::Option{<:LinearConstraint}`: The input, unchanged.

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
