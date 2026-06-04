"""
    const NSkeFormulations = Union{<:NSkeQuadFormulations, <:SOCRiskExpr}

Union of valid optimisation formulations for the [`NegativeSkewness`](@ref) risk measure.

# Related

  - [`NSkeQuadFormulations`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`NegativeSkewness`](@ref)
"""
const NSkeFormulations = Union{<:NSkeQuadFormulations, <:SOCRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Represents the Negative Skewness risk measure.

`NegativeSkewness` quantifies the portfolio's exposure to negative asymmetry in returns by computing a quadratic or SOC (second-order cone) form of the coskewness matrix. It penalises portfolio constructions that exhibit heavy left-tail behaviour.

# Mathematical definition

Let ``\\boldsymbol{w}`` be the portfolio weight vector and ``\\mathbf{V}`` the negative semi-definite coskewness matrix (spectral decomposition of the negative part of the sample coskewness tensor). The Negative Skewness risk measure is:

```math
\\begin{align}
\\mathrm{NSke}(\\boldsymbol{w}) &= \\begin{cases}
  \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w}} & \\text{(SOC formulation)} \\\\
  \\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w} & \\text{(Quadratic formulation)}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{NSke}(\\boldsymbol{w})``: Negative Skewness risk measure.
  - $(math_dict[:w_port])
  - ``\\mathbf{V}``: Negative semi-definite coskewness matrix (spectral decomposition of the negative part of the sample coskewness tensor).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NegativeSkewness(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
        sk::Option{<:MatNum} = nothing,
        V::Option{<:MatNum} = nothing,
        alg::NSkeFormulations = SOCRiskExpr(),
        window::Option{<:Int_VecInt} = nothing
    ) -> NegativeSkewness

Keywords correspond to the struct's fields.

## Validation

  - If `sk` or `V` is provided, both must be provided, non-empty, with `size(sk, 1)^2 == size(sk, 2)` and `V` square.
  - `window` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Functor

    (r::NegativeSkewness)(w::VecNum)

Computes the Negative Skewness risk of a portfolio weight vector `w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.

# Examples

```jldoctest
julia> NegativeSkewness()
NegativeSkewness
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        mp ┼ DenoiseDetoneAlgMatrixProcessing
           │     pdm ┼ Posdef
           │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │      dn ┼ nothing
           │      dt ┼ nothing
           │     alg ┼ nothing
           │   order ┴ DenoiseDetoneAlg()
        sk ┼ nothing
         V ┼ nothing
       alg ┼ SOCRiskExpr()
    window ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Kurtosis`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`NSkeQuadFormulations`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
@concrete struct NegativeSkewness <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:sk])
    """
    sk
    """
    $(field_dict[:V])
    """
    V
    """
    $(field_dict[:alg])
    """
    alg
    """
    $(field_dict[:window])
    """
    window
    function NegativeSkewness(settings::RiskMeasureSettings,
                              mp::AbstractMatrixProcessingEstimator, sk::Option{<:MatNum},
                              V::Option{<:MatNum}, alg::NSkeFormulations,
                              window::Option{<:Int_VecInt})
        sk_flag = isnothing(sk)
        V_flag = isnothing(V)
        if sk_flag || V_flag
            @argcheck(sk_flag)
            @argcheck(V_flag)
        else
            @argcheck(!isempty(sk))
            @argcheck(!isempty(V))
            @argcheck(size(sk, 1)^2 == size(sk, 2))
            assert_matrix_issquare(V, :V)
        end
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(settings), typeof(mp), typeof(sk), typeof(V), typeof(alg),
                   typeof(window)}(settings, mp, sk, V, alg, window)
    end
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
                          sk::Option{<:MatNum} = nothing, V::Option{<:MatNum} = nothing,
                          alg::NSkeFormulations = SOCRiskExpr(),
                          window::Option{<:Int_VecInt} = nothing)::NegativeSkewness
    return NegativeSkewness(settings, mp, sk, V, alg, window)
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:SOCRiskExpr})(w::VecNum)
    return sqrt(LinearAlgebra.dot(w, r.V, w))
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:NSkeQuadFormulations})(w::VecNum)
    return LinearAlgebra.dot(w, r.V, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`NegativeSkewness`](@ref) by selecting the coskewness matrix and spectral decomposition matrix from the risk-measure instance or falling back to a [`HighOrderPrior`](@ref) result.

# Related

  - [`NegativeSkewness`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::NegativeSkewness, pr::HighOrderPrior, args...;
                 kwargs...)::NegativeSkewness
    sk = nothing_scalar_array_selector(r.sk, pr.sk)
    V = nothing_scalar_array_selector(r.V, pr.V)
    return NegativeSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V, alg = r.alg,
                            window = r.window)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`NegativeSkewness`](@ref) risk measure `r` unchanged.

Coskewness is not available in [`LowOrderPrior`](@ref) results; the existing risk measure is used as-is.

# Related

  - [`NegativeSkewness`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
"""
function factory(r::NegativeSkewness, ::LowOrderPrior, args...; kwargs...)::NegativeSkewness
    return r
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any}, ::Any,
                           args...)::NegativeSkewness
    return r
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:MatNum, <:MatNum}, i,
                           X::MatNum)::NegativeSkewness
    sk = r.sk
    idx = fourth_moment_index_generator(size(sk, 1), i)
    sk = nothing_scalar_array_view_odd_order(r.sk, i, idx)
    window = get_window(r.window, X)
    V = negative_spectral_coskewness(sk, view(X, window, i), r.mp)
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V,
                            window = r.window)
end

export NegativeSkewness
