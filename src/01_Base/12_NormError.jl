"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all norm-based error algorithms.

All concrete and/or abstract types representing norm-based error algorithms (such as second-order cone or norm-one error) should be subtypes of `NormError`.

# Interfaces

In order to implement a new norm-based error algorithm which will work seamlessly with the library, subtype `NormError` with all necessary parameters struct, and implement the following method:

  - `norm_factor(f::NormError, T::Number) -> Number`: Returns the divisor that scales the norm. The `T === nothing` case is already covered by a generic method that returns `1`.

The functor side is [`norm_error`](@ref), and the model side is `set_risk_constraints!` for [`TrackingRiskMeasure`](@ref) and `set_tracking_error_constraints!` for [`TrackingError`](@ref). All three must agree.

# Related

  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
"""
abstract type NormError <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone (SOC) norm-based error formulation.

`L2Norm` implements a norm-based error formulation using the Euclidean (L2) norm, scaled by the square root of the number of assets minus the degrees of freedom (`ddof`). This is commonly used for error constraints and objectives in portfolio optimisation.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:te_l2])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])

The source states the denominator as ``\\sqrt{T}``. The default `ddof = 1` gives the sample denominator ``\\sqrt{T-1}``. Set `ddof = 0` to recover the source.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L2Norm(;
        ddof::Integer = 1
    ) -> L2Norm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> L2Norm()
L2Norm
  ddof ┴ Int64: 1
```

# Related

  - [`NormError`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.16.
"""
@concrete struct L2Norm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function L2Norm(ddof::Integer)::L2Norm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function L2Norm(; ddof::Integer = 1)::L2Norm
    return L2Norm(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone (SOC) squared norm-based error formulation.

`SquaredL2Norm` implements a norm-based error formulation using the squared Euclidean (L2) norm, scaled by the number of assets minus the degrees of freedom (`ddof`). This is commonly used for norm error constraints and objectives in portfolio optimisation where squared error is preferred.

The value is the square of the [`L2Norm`](@ref) error, so a `settings.ub` on a [`TrackingRiskMeasure`](@ref) carries squared units. The JuMP model converts the bound with a square root, so the two encodings accept the same bound.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,.
\\end{align}
```

Where:

  - $(math_dict[:te_l2sq])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SquaredL2Norm(;
        ddof::Integer = 1,
    ) -> SquaredL2Norm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> SquaredL2Norm()
SquaredL2Norm
  ddof ┴ Int64: 1
```

# Related

  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.16.
"""
@concrete struct SquaredL2Norm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function SquaredL2Norm(ddof::Integer)::SquaredL2Norm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function SquaredL2Norm(; ddof::Integer = 1)::SquaredL2Norm
    return SquaredL2Norm(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

Norm-one (NOC) error formulation.

`L1Norm` implements a norm-based error formulation using the L1 (norm-one) distance between portfolio and benchmark weights. This is commonly used for error constraints and objectives in portfolio optimisation where sparsity or absolute deviations are preferred.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_1}{T}\\,.
\\end{align}
```

Where:

  - $(math_dict[:te_l1])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T]) When ``T`` is not provided the denominator is 1.

# Constructors

    L1Norm() -> L1Norm

# Examples

```jldoctest
julia> L1Norm()
L1Norm()
```

# Related

  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.17.
"""
struct L1Norm <: NormError end
"""
$(DocStringExtensions.TYPEDEF)

L-p norm error estimator.

`LpNorm` takes the Lp-norm of the difference between the portfolio and the benchmark returns, and divides it by ``(T - d)^{1/p}``. It generalises [`L1Norm`](@ref) and [`L2Norm`](@ref) to a free norm order.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p}{(T - d)^{1/p}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:te_lp])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])
  - $(math_dict[:p_norm_order])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LpNorm(; p::Number = 3, ddof::Integer = 0) -> LpNorm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`. The constructor does not bound `p`. The JuMP model does: both `set_risk_constraints!` and `set_tracking_error_constraints!` need `1 < p` for the power cone, and raise a `DomainError` otherwise. The functor accepts any `p` that `LinearAlgebra.norm` accepts.

# Examples

```jldoctest
julia> LpNorm()
LpNorm
     p ┼ Int64: 3
  ddof ┴ Int64: 0
```

# Related

  - [`NormError`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
"""
@concrete struct LpNorm <: NormError
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:ddof])
    """
    ddof
    function LpNorm(p::Number, ddof::Integer)::LpNorm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(p), typeof(ddof)}(p, ddof)
    end
end
function LpNorm(; p::Number = 3, ddof::Integer = 0)::LpNorm
    return LpNorm(p, ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

L-infinity norm (maximum absolute deviation) error estimator.

`LInfNorm` takes the largest absolute deviation between the portfolio and the benchmark returns, and divides it by ``T - d``.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - $(math_dict[:te_linf])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LInfNorm(; ddof::Integer = 0) -> LInfNorm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> LInfNorm()
LInfNorm
  ddof ┴ Int64: 0
```

# Related

  - [`NormError`](@ref)
  - [`LpNorm`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
"""
@concrete struct LInfNorm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function LInfNorm(ddof::Integer)::LInfNorm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function LInfNorm(; ddof::Integer = 0)::LInfNorm
    return LInfNorm(ddof)
end
"""
    norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(::L1Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::Option{<:NormError}, a, T::Option{<:Number} = nothing)

Compute the norm-based tracking error between portfolio and benchmark weights.

`norm_error` takes the norm that `f` selects, and divides it by the [`norm_factor`](@ref) that the same `f` declares. Each [`NormError`](@ref) subtype names one pair. The three-argument form takes the norm of `a - b`. The two-argument form takes the norm of `a` alone, for a caller that already holds the deviation vector; `f = nothing` there means an unweighted L2 norm.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,, \\\\
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,, \\\\
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_1}{T}\\,, \\\\
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p}{(T-d)^{1/p}}\\,, \\\\
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - $(math_dict[:te_l2])
  - $(math_dict[:te_l2sq])
  - $(math_dict[:te_l1])
  - $(math_dict[:te_lp])
  - $(math_dict[:te_linf])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])
  - $(math_dict[:p_norm_order])

# Algorithm

The method Julia selects on the type of `f` is the algorithm, and every method runs the same two steps.

 1. Take the norm that `f` names, of `a - b` in the three-argument form and of `a` alone in the two-argument form.
 2. Divide the norm of step 1 by [`norm_factor`](@ref) of the same `f` and `T`, which is `1` when `T` is `nothing`.

# Arguments

  - `f`: Norm-based error algorithm, a [`NormError`](@ref) subtype.
  - `a`: Portfolio weights, or the deviation vector in the two-argument form.
  - `b`: Benchmark weights.
  - `T`: Optional number of observations.

# Returns

  - `err::Number`: Norm-based tracking error.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_error(L2Norm(), [0.5, 0.5], [0.6, 0.4], 2)
0.14142135623730948

julia> PortfolioOptimisers.norm_error(L1Norm(), [0.5, 0.5], [0.6, 0.4], 2)
0.09999999999999998
```

# Related

  - [`NormError`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`Option`](@ref)
  - [`norm_factor`](@ref)
"""
function norm_error end
"""
    norm_factor(f::Union{Nothing, <:NormError}, T::Option{<:Number})

Compute the denominator that scales a norm in [`norm_error`](@ref).

The factor is the single place where the optional observation count `T` is turned into a divisor. Each [`NormError`](@ref) declares its own factor, and the `T === nothing` case is a method, not a branch inside one. A branch is what let `ifelse` evaluate `T - f.ddof` on the `nothing` path.

# Algorithm

The method Julia selects on the types of `f` and `T` is the algorithm. A `T` of `nothing` selects the method that returns `1`, so the `nothing` case is a method and never a branch inside one.

 1. `f === nothing` gives `sqrt(T)`, the unweighted L2 factor.
 2. [`L2Norm`](@ref) gives `sqrt(T - f.ddof)`.
 3. [`SquaredL2Norm`](@ref) gives `T - f.ddof`.
 4. [`L1Norm`](@ref) gives `T`, because that norm carries no degrees of freedom.
 5. [`LpNorm`](@ref) gives `(T - f.ddof)^(1/f.p)`, taken with `cbrt` when `f.p` is `3`, the default.
 6. [`LInfNorm`](@ref) gives `T - f.ddof`.

# Arguments

  - `f`: Norm-based error algorithm, a [`NormError`](@ref) subtype. `nothing` means an unweighted L2 norm.
  - `T`: Optional number of observations.

# Returns

  - `factor::Number`: Divisor for the norm. It is `1` when `T` is `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_factor(L2Norm(), 4)
1.7320508075688772

julia> PortfolioOptimisers.norm_factor(LInfNorm(), nothing)
1
```

# Related

  - [`norm_error`](@ref)
  - [`NormError`](@ref)
  - [`Option`](@ref)
"""
function norm_factor(::Union{Nothing, <:NormError}, ::Nothing)
    return 1
end
function norm_factor(::Nothing, T::Number)
    return sqrt(T)
end
function norm_factor(f::L2Norm, T::Number)
    return sqrt(T - f.ddof)
end
function norm_factor(f::SquaredL2Norm, T::Number)
    return T - f.ddof
end
function norm_factor(::L1Norm, T::Number)
    return T
end
function norm_factor(f::LpNorm, T::Number)
    factor = T - f.ddof
    return if f.p == 3
        cbrt(factor)
    else
        factor^(inv(f.p))
    end
end
function norm_factor(f::LInfNorm, T::Number)
    return T - f.ddof
end
function norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, 2) / norm_factor(f, T)
end
function norm_error(f::L2Norm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, 2) / norm_factor(f, T)
end
function norm_error(f::Nothing, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, 2) / norm_factor(f, T)
end
function norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    val = LinearAlgebra.norm(a - b, 2)
    return val^2 / norm_factor(f, T)
end
function norm_error(f::SquaredL2Norm, a, T::Option{<:Number} = nothing)
    val = LinearAlgebra.norm(a, 2)
    return val^2 / norm_factor(f, T)
end
function norm_error(f::L1Norm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, 1) / norm_factor(f, T)
end
function norm_error(f::L1Norm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, 1) / norm_factor(f, T)
end
function norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, f.p) / norm_factor(f, T)
end
function norm_error(f::LpNorm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, f.p) / norm_factor(f, T)
end
function norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, Inf) / norm_factor(f, T)
end
function norm_error(f::LInfNorm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, Inf) / norm_factor(f, T)
end

export L2Norm, SquaredL2Norm, L1Norm, LpNorm, LInfNorm
