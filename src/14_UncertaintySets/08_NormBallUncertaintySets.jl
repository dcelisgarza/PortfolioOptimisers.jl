"""
$(DocStringExtensions.TYPEDEF)

Holds a radius, a geometry map and a norm order, so the set is the image of a norm ball under the map, on the mean axis or on the covariance axis.

The map may have fewer columns than the quantity has entries, so the set may be flat, which is what a set confined to an Orthogonal Subspace needs, and it may have no column at all, which is the rank-zero set whose worst case is the nominal quantity. Its worst case is the radius times the dual norm of the map's transpose applied to the exposure, so no consumer factorises anything: the mean builder raises one cone on ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}``, and the covariance builder puts ``\\mathbf{L}^{\\intercal}`` where the ellipsoid's Cholesky factor goes. The cone follows the dual norm: a second-order cone at ``p = 2``, a norm-one cone at ``p = \\infty``, a norm-infinity cone at ``p = 1``, and power cones otherwise.

**`class` names the axis, and the axis fixes the row count of `L` and the index a view applies.** A [`MuUncertaintySetClass`](@ref) set carries an ``N \\times r`` map and takes the plain asset index. A [`SigmaUncertaintySetClass`](@ref) set carries an ``N^{2} \\times r`` one, because it bounds a vectorised covariance, so [`port_opt_view`](@ref) recovers ``N`` from the map and maps the asset index through [`fourth_moment_index_generator`](@ref) before it slices. The robust-return builder refuses a set that carries the covariance tag.

**Three existing shapes are diagonal or square cases of this one.** The mean [`BoxUncertaintySet`](@ref) is the ``p = \\infty`` set with ``\\mathbf{L} = \\operatorname{diag}((\\boldsymbol{u} - \\boldsymbol{\\ell})/2)`` and ``\\kappa = 1``. The [`L1UncertaintySet`](@ref) is the ``p = 1`` set with ``\\mathbf{L} = \\operatorname{diag}(\\boldsymbol{\\sigma})`` and ``\\kappa = \\epsilon``. The [`EllipsoidalUncertaintySet`](@ref) is the ``p = 2`` set with ``\\mathbf{L}`` the lower Cholesky factor of its shape matrix and ``\\kappa = k``, which is what the converter constructor builds. Each of the three reaches the same weights as the norm ball it corresponds to, and each keeps its own builder.

# Mathematical definition

```math
\\begin{align}
U^{p}_{\\boldsymbol{z}} &= \\left\\{ \\hat{\\boldsymbol{z}} + \\mathbf{L}\\boldsymbol{u} \\, \\vert \\, \\lVert \\boldsymbol{u} \\rVert_{p} \\leq \\kappa \\right\\}\\,, \\\\
\\underset{\\boldsymbol{z} \\in U^{p}_{\\boldsymbol{z}}}{\\min} \\boldsymbol{z}^{\\intercal}\\boldsymbol{e} &= \\hat{\\boldsymbol{z}}^{\\intercal}\\boldsymbol{e} - \\kappa \\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{e} \\rVert_{q}\\,, \\quad \\frac{1}{p} + \\frac{1}{q} = 1\\,.
\\end{align}
```

Where:

  - ``U^{p}_{\\boldsymbol{z}}``: Norm-ball uncertainty set of order ``p`` for the quantity ``\\boldsymbol{z}``.
  - ``\\boldsymbol{z}``: Uncertain quantity, the mean vector on the mean axis and the vectorised covariance on the covariance axis.
  - ``\\hat{\\boldsymbol{z}}``: Estimated centre of the set.
  - ``\\mathbf{L}``: Geometry map, ``N \\times r`` on the mean axis and ``N^{2} \\times r`` on the covariance axis.
  - ``\\boldsymbol{u}``: Coordinates in the ball, ``r \\times 1``.
  - ``\\kappa \\geq 0``: Radius of the ball.
  - ``p \\geq 1``: Norm order of the ball, ``\\infty`` admitted.
  - ``q``: Dual norm order, ``\\infty`` at ``p = 1`` and ``1`` at ``p = \\infty``.
  - ``\\boldsymbol{e}``: Exposure the worst case is taken against, the weights on the mean axis and the vectorised lifted weight matrix on the covariance axis.

The second line is Hölder's inequality with equality, which is why the worst case needs no factorisation and holds for a rank-deficient ``\\mathbf{L}``. A radius of zero, or a map with no column, leaves the nominal quantity. On the covariance axis the worst case is taken jointly with the positive semi-definite constraint, and the consumer states the lifted form it builds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormBallUncertaintySet(;
        kappa::Number,
        L::MatNum,
        p::Number = 2,
        class::AbstractUncertaintySetClass,
        val::Option{<:ArrNum} = nothing
    ) -> NormBallUncertaintySet
    NormBallUncertaintySet(ucs::EllipsoidalUncertaintySet) -> NormBallUncertaintySet

Keywords correspond to the struct's fields. The second constructor converts a built [`EllipsoidalUncertaintySet`](@ref) with one Cholesky factorisation at construction time: `L = LinearAlgebra.cholesky(ucs.sigma).L`, `p = 2`, and `kappa`, `class` and `val` carried through from `k`, `class` and `val`. It is the route through which every estimator that emits an ellipsoid reaches this type.

## Validation

  - `isfinite(kappa)` and `kappa >= 0`.
  - `!isnan(p)` and `p >= 1`. `Inf` is admitted.
  - `size(L, 1) > 0` and `all(isfinite, L)`. A map with no column is admitted.
  - If `class` is a [`SigmaUncertaintySetClass`](@ref): `size(L, 1)` is a perfect square, because the map lives on the vectorised covariance.
  - If `val` is provided: `length(val) == size(L, 1)`. The rule reads a length rather than a size, so it holds on both axes: `val` is a characteristic vector of length ``N`` beside a map of ``N`` rows, and an ``N \\times N`` covariance matrix beside a map of ``N^{2}`` rows. On the covariance axis `val` must also be square.

# Examples

```jldoctest
julia> NormBallUncertaintySet(; kappa = 2.0, L = [1.0 0.0; 0.0 1.0; 0.5 0.5],
                              class = MuUncertaintySetClass())
NormBallUncertaintySet
  kappa ┼ Float64: 2.0
      L ┼ 3×2 Matrix{Float64}
      p ┼ Int64: 2
  class ┼ MuUncertaintySetClass()
    val ┴ nothing
```

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetClass`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`dual_norm_order`](@ref)
  - [`set_ucs_return_constraints!`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
  - [`ucs_variance`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
@concrete struct NormBallUncertaintySet <: AbstractUncertaintySetResult
    """
    Radius ``\\kappa \\geq 0`` of the ball, the multiplier of the dual-norm penalty. `0` leaves the nominal quantity.
    """
    kappa
    """
    Geometry map ``\\mathbf{L}``, ``N \\times r`` on the mean axis and ``N^{2} \\times r`` on the covariance axis. It may be rank deficient, and `r = 0` is admitted and leaves the nominal quantity.
    """
    L
    """
    Norm order ``p \\geq 1`` of the ball, `Inf` admitted. The consumer raises the cone of the dual order.
    """
    p
    """
    $(field_dict[:class_ucs])
    """
    class
    """
    $(field_dict[:val_ucs])
    """
    val
    # `class` is a static parameter rather than an abstract bound, so the two tags never
    # form a union inside the body: with the bound, JET reads the tag's type and the
    # tag's value as two independent unions and pairs them crosswise at `new`.
    function NormBallUncertaintySet(kappa::Number, L::MatNum, p::Number, class::C,
                                    val::Option{<:ArrNum}) where {C <:
                                                                  AbstractUncertaintySetClass}
        @argcheck(isfinite(kappa) && kappa >= zero(kappa),
                  DomainError(kappa, "kappa must be finite and >= 0"))
        @argcheck(!isnan(p) && p >= one(p), DomainError(p, "p must be >= 1"))
        @argcheck(size(L, 1) > zero(Int), IsEmptyError("L must have at least one row"))
        @argcheck(all(isfinite, L), IsNonFiniteError("all entries of L must be finite"))
        assert_norm_ball_axis(class, L)
        if isa(val, ArrNum)
            @argcheck(length(val) == size(L, 1),
                      DimensionMismatch("val ($(length(val))) must match the rows of L ($(size(L, 1)))"))
            assert_norm_ball_val(class, val)
        end
        return new{typeof(kappa), typeof(L), typeof(p), C, typeof(val)}(kappa, L, p, class,
                                                                        val)
    end
end
function NormBallUncertaintySet(; kappa::Number, L::MatNum, p::Number = 2,
                                class::AbstractUncertaintySetClass,
                                val::Option{<:ArrNum} = nothing)::NormBallUncertaintySet
    return NormBallUncertaintySet(kappa, L, p, class, val)
end
function NormBallUncertaintySet(ucs::EllipsoidalUncertaintySet)::NormBallUncertaintySet
    return NormBallUncertaintySet(ucs.k, LinearAlgebra.cholesky(ucs.sigma).L, 2, ucs.class,
                                  ucs.val)
end
"""
    assert_norm_ball_axis(::MuUncertaintySetClass, L::MatNum)
    assert_norm_ball_axis(::SigmaUncertaintySetClass, L::MatNum)

Check that the rows of `L` fit the axis the tag names.

The method is a refusal rather than a procedure, so it carries no `# Algorithm` section. The mean tag accepts any row count, and the covariance tag needs a perfect square, because a set on that axis bounds a vectorised ``N \\times N`` covariance and [`port_opt_view`](@ref) recovers ``N`` from the row count.

# Arguments

  - `class`: Axis tag.
  - `L`: Geometry map.

# Validation

  - On the covariance tag, `isqrt(size(L, 1))^2 == size(L, 1)`, else a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`assert_norm_ball_val`](@ref)
"""
function assert_norm_ball_axis(::MuUncertaintySetClass, ::MatNum)::Nothing
    return nothing
end
function assert_norm_ball_axis(::SigmaUncertaintySetClass, L::MatNum)::Nothing
    n = size(L, 1)
    @argcheck(isqrt(n)^2 == n,
              DimensionMismatch("L has $n rows, which is not a perfect square, so it cannot map a vectorised covariance"))
    return nothing
end
"""
    assert_norm_ball_val(::MuUncertaintySetClass, val::ArrNum)
    assert_norm_ball_val(::SigmaUncertaintySetClass, val::ArrNum)

Check the shape of the carried centre against the axis the tag names.

The method is a refusal rather than a procedure, so it carries no `# Algorithm` section. The constructor has already matched `length(val)` to the rows of `L`, so the mean tag has nothing left to check, and the covariance tag checks that the centre is a square matrix.

# Arguments

  - `class`: Axis tag.
  - `val`: Carried centre.

# Validation

  - On the covariance tag, `val` is a matrix and `size(val, 1) == size(val, 2)`, else a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`assert_norm_ball_axis`](@ref)
"""
function assert_norm_ball_val(::MuUncertaintySetClass, ::ArrNum)::Nothing
    return nothing
end
function assert_norm_ball_val(::SigmaUncertaintySetClass, val::ArrNum)::Nothing
    @argcheck(isa(val, AbstractMatrix),
              DimensionMismatch("val must be a square matrix on the covariance axis, got a $(ndims(val))-dimensional array"))
    assert_matrix_issquare(val, :val)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the dual norm order `q` of the norm order `p`, with `1 / p + 1 / q == 1`.

# Algorithm

 1. When `p` is infinite, return `one(p)`, because the dual of the infinity norm is the one norm.
 2. Otherwise return `p / (p - 1)`. At `p = 1` the division is by zero and returns `Inf`, the dual of the one norm, and at `p = 2` it returns `2`, which is its own dual.

# Arguments

  - `p::Number`: Norm order, `p >= 1`.

# Returns

  - `q::Number`: Dual norm order.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`set_ucs_return_constraints!`](@ref)
  - [`ucs_variance`](@ref)
"""
function dual_norm_order(p::Number)
    return isinf(p) ? one(p) : p / (p - one(p))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a mean [`NormBallUncertaintySet`](@ref) restricted to assets at index `i`.

The view is the projection of the set onto the cluster's coordinates, and not a refit: a row slice of the map is the map of the projected set, because the coordinates ``\\boldsymbol{u}`` of the ball are untouched by dropping rows of ``\\mathbf{L}``. It is what the ellipsoid's view returns too, since ``\\mathbf{\\Sigma}_{ii} = \\mathbf{L}_{i}\\mathbf{L}_{i}^{\\intercal}``. A caller who wants the set the same estimator would fit on the subset alone fits the subset.

# Algorithm

 1. Take `view(risk_ucs.L, i, :)`, the rows of the map the selected assets occupy.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted characteristic vector restricted to the same assets, which passes a `nothing` through unchanged.
 3. Build a [`NormBallUncertaintySet`](@ref) from the two views, carrying `kappa`, `p` and `class` through unchanged.

# Arguments

  - `risk_ucs`: Mean norm-ball uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::NormBallUncertaintySet`: The set restricted to `i`.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`MuUncertaintySetClass`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::NormBallUncertaintySet{<:Any, <:MatNum, <:Any,
                                                        <:MuUncertaintySetClass}, i,
                       args...)::NormBallUncertaintySet
    return NormBallUncertaintySet(; kappa = risk_ucs.kappa, L = view(risk_ucs.L, i, :),
                                  p = risk_ucs.p, class = risk_ucs.class,
                                  val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a covariance [`NormBallUncertaintySet`](@ref) restricted to assets at index `i`, mapping the map's row index through the fourth-moment index generator.

The view is the projection of the set onto the cluster's coordinates, and not a refit. The set bounds a vectorised covariance, so its map lives on the ``N^{2}`` axis while its centre lives on the ``N`` axis, and the method applies two different indices, one to each field.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted ``N \\times N`` covariance restricted to the selected assets. It takes the plain asset index, and a `nothing` passes through unchanged. The step runs first, because step 2 overwrites `i`.
 2. Recover `N` as `isqrt(size(risk_ucs.L, 1))` from the map, and expand `i` with `fourth_moment_index_generator(N, i)`, giving the positions the selected assets occupy in the vectorised covariance.
 3. Take `view(risk_ucs.L, i, :)` under the expanded index, giving the rows of the map the selected pairs occupy.
 4. Build a [`NormBallUncertaintySet`](@ref) from the two views, carrying `kappa`, `p` and `class` through unchanged.

# Arguments

  - `risk_ucs`: Covariance norm-ball uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::NormBallUncertaintySet`: The set restricted to `i`.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`SigmaUncertaintySetClass`](@ref)
  - [`fourth_moment_index_generator`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::NormBallUncertaintySet{<:Any, <:MatNum, <:Any,
                                                        <:SigmaUncertaintySetClass}, i,
                       args...)::NormBallUncertaintySet
    # `val` is the N x N covariance the set is a neighbourhood of, so it takes the asset
    # index, whereas the N^2 x r map takes the fourth-moment index.
    val = nothing_scalar_array_view(risk_ucs.val, i)
    i = fourth_moment_index_generator(isqrt(size(risk_ucs.L, 1)), i)
    return NormBallUncertaintySet(; kappa = risk_ucs.kappa, L = view(risk_ucs.L, i, :),
                                  p = risk_ucs.p, class = risk_ucs.class, val = val)
end

export NormBallUncertaintySet
