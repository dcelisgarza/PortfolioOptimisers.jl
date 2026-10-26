"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all norm-based error algorithms.

All concrete and abstract types of norm-based error algorithms, such as the second-order cone error or the norm-one error, must subtype `NormError`.

# Interfaces

To implement a new norm-based error algorithm, subtype `NormError` with a struct that holds its parameters, and implement the method below.

  - `norm_factor(f::NormError, T::Number) -> Number`: Returns the divisor that scales the norm. A generic method already returns `1` for `T === nothing`.

`norm_error` computes the error for a value. In the JuMP model, `set_risk_constraints!` for [`TrackingRiskMeasure`](@ref) and `set_tracking_error_constraints!` for [`TrackingError`](@ref) build the error. All three read `norm_factor`, so a new norm declares its factor once.

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

`L2Norm` takes the Euclidean norm of the difference between the portfolio returns and the benchmark returns. It divides the norm by the square root of the number of observations minus the degrees of freedom `ddof`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tr_l2])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])

[rudolf1999](@cite) minimises the sum of the squared differences, ``(\\boldsymbol{a} - \\boldsymbol{b})^{\\intercal} (\\boldsymbol{a} - \\boldsymbol{b})``, and does not centre them. [beasley2003](@cite) takes the ``\\alpha``-th root of the sum of the ``\\alpha``-th powers, divides it by ``T``, and names the case ``\\alpha = 2`` the root mean squared error. The root mean square divides by ``\\sqrt{T}``, as [cajas2025](@cite) does, and this error also does. The two values differ by the factor ``\\sqrt{T}`` and have the same minimiser, but a bound on the error must use the denominator of this error. The default `ddof = 1` gives the sample denominator ``\\sqrt{T-1}``. Set `ddof = 0` to recover the root mean square.

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

  - $(ref_dict[:rudolf1999]) Section 2, Equation 2.
  - $(ref_dict[:beasley2003]) Section 3.5.1, Equation 9.
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

`SquaredL2Norm` takes the squared Euclidean norm of the difference between the portfolio returns and the benchmark returns. It divides the squared norm by the number of observations minus the degrees of freedom `ddof`.

The value is the square of the [`L2Norm`](@ref) error, so a `settings.ub` on a [`TrackingRiskMeasure`](@ref) has squared units. The JuMP model converts the bound with a square root, so the two encodings accept the same bound.

[rudolf1999](@cite) minimises the sum of the squared differences, which is this error times ``T - d``, and does not centre the differences. The division by ``T - d`` squares the root mean square of [cajas2025](@cite).

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tr_l2sq])
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

  - $(ref_dict[:rudolf1999]) Section 2, Equation 2.
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

`L1Norm` takes the L1 distance between the portfolio returns and the benchmark returns. It divides the distance by the number of observations minus the degrees of freedom `ddof`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_1}{T - d}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tr_l1])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])

[rudolf1999](@cite) states the error as the sum of the absolute differences, ``\\boldsymbol{1}^{\\intercal} \\lvert \\boldsymbol{a} - \\boldsymbol{b} \\rvert``, and names it the mean absolute deviation. [cajas2025](@cite) divides the sum by ``T``, which gives the mean that the name states and does not move the minimiser. The default `ddof = 0` gives that denominator ``T``, so the error is the mean absolute difference. [`LpNorm`](@ref) defaults to `ddof = 1`, so `LpNorm(; p = 1)` and `L1Norm()` differ by a factor of ``T / (T - 1)``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L1Norm(;
        ddof::Integer = 0
    ) -> L1Norm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> L1Norm()
L1Norm
  ddof ┴ Int64: 0
```

# Related

  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:rudolf1999]) Section 3, Equations 5 and 7a.
  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.17.
"""
@concrete struct L1Norm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function L1Norm(ddof::Integer)::L1Norm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function L1Norm(; ddof::Integer = 0)::L1Norm
    return L1Norm(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

L-p norm error estimator.

`LpNorm` takes the Lp-norm of the difference between the portfolio returns and the benchmark returns, and divides it by ``(T - d)^{1/p}``. It extends [`L1Norm`](@ref) and [`L2Norm`](@ref) to any norm order.

[beasley2003](@cite) states this family with the divisor ``T`` outside the root, ``\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p / T``. The divisor here is ``(T - d)^{1/p}``, so the error is the ``p``-th root of a mean and keeps the scale of one period as ``T`` changes. At a fixed ``T`` the two values differ by a constant factor and have the same minimiser.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p}{(T - d)^{1/p}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tr_lp])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])
  - $(math_dict[:p_norm_order])

The default `ddof = 1` matches [`L2Norm`](@ref), so `LpNorm(; p = 2)` and `L2Norm()` give the same error. [`L1Norm`](@ref) defaults to `ddof = 0`, so `LpNorm(; p = 1)` and `L1Norm()` differ by a factor of ``T / (T - 1)``. As ``p`` grows the factor goes to ``1``, which is the factor of [`LInfNorm`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LpNorm(; p::Number = 3, ddof::Integer = 1) -> LpNorm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`. The constructor does not bound `p`, and the JuMP model does. Both `set_risk_constraints!` and `set_tracking_error_constraints!` need `1 < p` for the power cone, and raise a `DomainError` for each other `p`. `norm_error` accepts each `p` that `LinearAlgebra.norm` accepts.

# Examples

```jldoctest
julia> LpNorm()
LpNorm
     p ┼ Int64: 3
  ddof ┴ Int64: 1
```

# Related

  - [`NormError`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)

# References

  - $(ref_dict[:beasley2003]) Section 3.5.1, Equation 9.
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
function LpNorm(; p::Number = 3, ddof::Integer = 1)::LpNorm
    return LpNorm(p, ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

L-infinity norm error estimator, which is the maximum absolute deviation.

`LInfNorm` takes the largest absolute deviation between the portfolio returns and the benchmark returns, and does not scale it. The factor of [`LpNorm`](@ref) is ``(T - d)^{1/p}``, which goes to ``1`` as ``p`` increases. The error is thus the largest difference in one period, and it has no degrees of freedom. It is the MinMax tracking error of [rudolf1999](@cite).

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - $(math_dict[:tr_linf])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])

# Constructors

    LInfNorm() -> LInfNorm

# Examples

```jldoctest
julia> LInfNorm()
LInfNorm()
```

# Related

  - [`NormError`](@ref)
  - [`LpNorm`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)

# References

  - $(ref_dict[:rudolf1999]) Section 3, Equations 6 and 7c.
"""
struct LInfNorm <: NormError end
"""
    norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(::L1Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::Option{<:NormError}, a, T::Option{<:Number} = nothing)

Compute the norm-based tracking error between a portfolio vector and a benchmark vector, of weights or of returns.

`norm_error` takes the norm that `f` selects, and divides it by the [`norm_factor`](@ref) that the same `f` declares. Each [`NormError`](@ref) subtype names one pair of a norm and a factor. The three-argument form takes the norm of `a - b`. The two-argument form takes the norm of `a` alone, for a caller that already holds the deviation vector. In that form, `f = nothing` means an unweighted L2 norm.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,, \\\\
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,, \\\\
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_1}{T - d}\\,, \\\\
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p}{(T-d)^{1/p}}\\,, \\\\
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - $(math_dict[:tr_l2])
  - $(math_dict[:tr_l2sq])
  - $(math_dict[:tr_l1])
  - $(math_dict[:tr_lp])
  - $(math_dict[:tr_linf])
  - $(math_dict[:a_norm_err])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])
  - $(math_dict[:p_norm_order])

# Algorithm

The method that Julia selects on the type of `f` is the algorithm, and every method does the same two steps.

 1. Take the norm that `f` names. It is the norm of `a - b` in the three-argument form, and the norm of `a` alone in the two-argument form.
 2. Divide the norm of step 1 by [`norm_factor`](@ref) of the same `f` and `T`, which is `1` when `T` is `nothing`.

# Arguments

  - `f`: The norm-based error algorithm, a [`NormError`](@ref) subtype.
  - `a`: The portfolio vector, or the deviation vector in the two-argument form.
  - `b`: The benchmark vector.
  - `T`: The optional number of observations.

# Returns

  - `err::Number`: The norm-based tracking error.

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
    norm_factor(f::Union{Nothing, NormError}, T::Option{<:Number})

Compute the denominator that scales a norm in [`norm_error`](@ref).

This function is the one place that turns the optional observation count `T` into a divisor. [`norm_error`](@ref) and both JuMP models read it. Each [`NormError`](@ref) declares its own factor. The case `T === nothing` is a method, and not a branch inside a method. An earlier branch let `ifelse` evaluate `T - f.ddof` when `T` was `nothing`.

# Algorithm

The method that Julia selects on the types of `f` and `T` is the algorithm. A `T` of `nothing` selects the method that returns `1`.

 1. `f === nothing` gives `sqrt(T)`, the unweighted L2 factor.
 2. [`L2Norm`](@ref) gives `sqrt(T - f.ddof)`.
 3. [`SquaredL2Norm`](@ref) gives `T - f.ddof`.
 4. [`L1Norm`](@ref) gives `T - f.ddof`.
 5. [`LpNorm`](@ref) gives `(T - f.ddof)^(1/f.p)`. The method uses `cbrt` when `f.p` is `3`, the default.
 6. [`LInfNorm`](@ref) gives `1`, the limit of the [`LpNorm`](@ref) factor as `f.p` increases.

# Arguments

  - `f`: The norm-based error algorithm, a [`NormError`](@ref) subtype. `nothing` means an unweighted L2 norm.
  - `T`: The optional number of observations.

# Returns

  - `factor::Number`: The divisor of the norm. It is `1` when `T` is `nothing`.

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
function norm_factor(::Union{Nothing, NormError}, ::Nothing)
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
function norm_factor(f::L1Norm, T::Number)
    return T - f.ddof
end
function norm_factor(f::LpNorm, T::Number)
    factor = T - f.ddof
    return if f.p == 3
        cbrt(factor)
    else
        factor^(inv(f.p))
    end
end
function norm_factor(::LInfNorm, T::Number)
    return one(T)
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
public NormError, norm_factor
