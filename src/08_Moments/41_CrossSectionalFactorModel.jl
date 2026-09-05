"""
    assert_idiosyncratic_covariance(esigma::Nothing, N::Integer)
    assert_idiosyncratic_covariance(esigma::VecNum, N::Integer)
    assert_idiosyncratic_covariance(esigma::MatNum, N::Integer)

Check the idiosyncratic covariance of a [`CrossSectionalFactorModel`](@ref) against the asset count `N`.

The idiosyncratic covariance takes either of two shapes, and the shape is the dispatch rather than a branch: a vector holds the idiosyncratic variances alone, and a matrix holds the full covariance an idiosyncratic correlation threshold produces. An absent covariance is checked by the method over `Nothing`, so no caller writes an `isnothing` test.

# Arguments

  - `esigma`: Idiosyncratic covariance, a vector of variances, a square matrix, or `nothing`.
  - `N`: Number of assets the model carries.

# Validation

  - `!isempty(esigma)`.
  - `length(esigma) == N` when `esigma` is a vector.
  - `esigma` is square, and `size(esigma, 1) == N`, when `esigma` is a matrix.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
  - [`assert_matrix_issquare`](@ref)
"""
function assert_idiosyncratic_covariance(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_idiosyncratic_covariance(esigma::VecNum, N::Integer)::Nothing
    @argcheck(!isempty(esigma), IsEmptyError("esigma cannot be empty"))
    @argcheck(length(esigma) == N,
              DimensionMismatch("esigma ($(length(esigma))) must match the asset count ($N)"))
    return nothing
end
function assert_idiosyncratic_covariance(esigma::MatNum, N::Integer)::Nothing
    @argcheck(!isempty(esigma), IsEmptyError("esigma cannot be empty"))
    assert_matrix_issquare(esigma, :esigma)
    @argcheck(size(esigma, 1) == N,
              DimensionMismatch("esigma ($(size(esigma, 1))) must match the asset count ($N)"))
    return nothing
end
"""
    idiosyncratic_covariance_view(esigma::Nothing, i)
    idiosyncratic_covariance_view(esigma::VecNum, i)
    idiosyncratic_covariance_view(esigma::MatNum, i)

Return a view of an idiosyncratic covariance, selecting only the assets indexed by `i`.

The shape is the dispatch, as it is in [`assert_idiosyncratic_covariance`](@ref): a vector of variances is indexed once, and a full covariance matrix is indexed on both of its axes, which keeps the selected block square.

# Arguments

  - `esigma`: Idiosyncratic covariance, a vector of variances, a square matrix, or `nothing`.
  - `i`: Indices of the assets to select.

# Returns

  - `esigma::Option{<:VecNum_MatNum}`: A view over the selected assets, or `nothing` when the model carries no idiosyncratic covariance.

# Examples

```jldoctest
julia> PortfolioOptimisers.idiosyncratic_covariance_view([1.0, 2.0, 3.0], [1, 3])
2-element view(::Vector{Float64}, [1, 3]) with eltype Float64:
 1.0
 3.0

julia> PortfolioOptimisers.idiosyncratic_covariance_view([1.0 0.0; 0.0 2.0], [2])
1×1 view(::Matrix{Float64}, [2], [2]) with eltype Float64:
 2.0
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_idiosyncratic_covariance`](@ref)
  - [`port_opt_view`](@ref)
"""
function idiosyncratic_covariance_view(::Nothing, args...)::Nothing
    return nothing
end
function idiosyncratic_covariance_view(esigma::VecNum, i)
    return view(esigma, i)
end
function idiosyncratic_covariance_view(esigma::MatNum, i)
    return view(esigma, i, i)
end
"""
    cs_history_assets(A::Nothing, N::Integer, sym::Symbol)
    cs_history_assets(A::MatNum, N::Integer, sym::Symbol)

Check the asset axis of an optional per-asset history of a [`CrossSectionalFactorModel`](@ref), and return its observation count.

A per-asset history holds one row per observation and one column per asset, so the asset axis is the second one. The count comes back so that the caller pins the observation axis of two histories against each other with [`assert_cs_history_obs`](@ref) and needs no `isnothing` test of its own.

# Arguments

  - `A`: A per-asset history, or `nothing`.
  - `N`: Number of assets the model carries.
  - `sym`: Name of the field, which the raise reports.

# Validation

  - `!isempty(A)`.
  - `size(A, 2) == N`.

# Returns

  - `T::Option{<:Integer}`: The observation count of `A`, or `nothing` when `A` is absent.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_cs_history_obs`](@ref)
"""
function cs_history_assets(::Nothing, ::Integer, ::Symbol)::Nothing
    return nothing
end
function cs_history_assets(A::MatNum, N::Integer, sym::Symbol)
    @argcheck(!isempty(A), IsEmptyError("$sym cannot be empty"))
    @argcheck(size(A, 2) == N,
              DimensionMismatch("$sym ($(size(A, 2)) columns) must match the asset count ($N)"))
    return size(A, 1)
end
"""
    assert_cs_history_obs(a::Option{<:Integer}, b::Option{<:Integer}, asym::Symbol, bsym::Symbol)
    assert_cs_history_obs(a::Integer, b::Integer, asym::Symbol, bsym::Symbol)

Check that two per-asset histories of a [`CrossSectionalFactorModel`](@ref) agree on the observation axis.

A history that the model does not carry constrains nothing, so the pair is checked only when [`cs_history_assets`](@ref) returned a count for both.

# Arguments

  - `a`: Observation count of the first history, or `nothing`.
  - `b`: Observation count of the second history, or `nothing`.
  - `asym`: Name of the first field, which the raise reports.
  - `bsym`: Name of the second field, which the raise reports.

# Validation

  - `a == b`, when both counts are present.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`cs_history_assets`](@ref)
"""
function assert_cs_history_obs(::Option{<:Integer}, ::Option{<:Integer}, ::Symbol,
                               ::Symbol)::Nothing
    return nothing
end
function assert_cs_history_obs(a::Integer, b::Integer, asym::Symbol, bsym::Symbol)::Nothing
    @argcheck(a == b,
              DimensionMismatch("$asym ($a rows) and $bsym ($b rows) must agree on the observation axis"))
    return nothing
end
"""
    assert_exposure_history(Ms::Nothing, N::Integer, K::Integer)
    assert_exposure_history(Ms::Arr3Num, N::Integer, K::Integer)

Check the exposure history of a [`CrossSectionalFactorModel`](@ref) against the asset count `N` and the factor count `K`.

The exposure history holds one slice per observation, and a slice has the shape of the loadings matrix, so its second and third axes are the asset axis and the factor axis.

# Arguments

  - `Ms`: An exposure history `observations × assets × factors`, or `nothing`.
  - `N`: Number of assets the model carries.
  - `K`: Number of raw factors the model carries.

# Validation

  - `!isempty(Ms)`.
  - `size(Ms, 2) == N`.
  - `size(Ms, 3) == K`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
"""
function assert_exposure_history(::Nothing, ::Integer, ::Integer)::Nothing
    return nothing
end
function assert_exposure_history(Ms::Arr3Num, N::Integer, K::Integer)::Nothing
    @argcheck(!isempty(Ms), IsEmptyError("Ms cannot be empty"))
    @argcheck(size(Ms, 2) == N,
              DimensionMismatch("Ms ($(size(Ms, 2)) rows per slice) must match the asset count ($N)"))
    @argcheck(size(Ms, 3) == K,
              DimensionMismatch("Ms ($(size(Ms, 3)) columns per slice) must match the factor count ($K)"))
    return nothing
end
"""
    assert_cs_regression_assets(csr::Nothing, N::Integer)
    assert_cs_regression_assets(csr::CrossSectionalRegression, N::Integer)

Check the asset axis of the fit a [`CrossSectionalFactorModel`](@ref) nests against the asset count `N`.

The residuals of a [`CrossSectionalRegression`](@ref) hold one row per observation and one column per asset, so the asset axis is the second one, as it is for every per-asset history of the model.

# Arguments

  - `csr`: The nested cross-sectional regression result, or `nothing`.
  - `N`: Number of assets the model carries.

# Validation

  - `size(csr.eps, 2) == N`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function assert_cs_regression_assets(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_cs_regression_assets(csr::CrossSectionalRegression, N::Integer)::Nothing
    @argcheck(size(csr.eps, 2) == N,
              DimensionMismatch("csr.eps ($(size(csr.eps, 2)) columns) must match the asset count ($N)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the loadings, the factor-orthogonal expected return and the fitted history of a factor model fitted per observation across the assets.

The result is the cross-sectional member of [`AbstractLoadingsRegressionResult`](@ref), so a consumer that re-bases a constraint or decomposes risk in the factor basis reads it exactly as it reads a [`Regression`](@ref). `M` carries the **raw** loadings, whose columns are the named original factors, because a constraint must be written in names a caller can put in an equation. Every field after `b` is optional, so a model that keeps its loadings and drops its histories is a member of the family, and a caller that asks for a dropped history reads `nothing`.

**An unset `L` reads back as `M`.** A [`@forward_properties`](@ref) `swap(L, M)` rule makes `csfm.L` return `csfm.M` whenever `L` was not given, as [`Regression`](@ref) behaves, so a consumer that decomposes risk in the factor basis needs no `Nothing` branch, and `isnothing(csfm.L)` is never true. Read `getfield(csfm, :L)` when the unset case must be told apart, as [`port_opt_view`](@ref) does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x}_{t} &= b_{t} \\boldsymbol{1} + \\mathbf{M}_{t} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}\\,, \\\\
\\boldsymbol{\\mu} &= \\mathbf{M} \\boldsymbol{\\mu}_{f} + \\boldsymbol{b}\\,, \\\\
\\mathbf{M} &= \\mathbf{M}_{T}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_obs])
  - ``b_{t}``: Intercept of observation ``t``, the ``t``-th entry of the intercept vector [`CrossSectionalRegression`](@ref) carries. The term is absent when the fit carries no intercept.
  - ``\\boldsymbol{1}``: Vector of ones ``N \\times 1``.
  - ``\\mathbf{M}_{t}``: Exposure slice of observation ``t``, ``N \\times K``, the ``t``-th slice of `Ms`.
  - ``\\boldsymbol{f}_{t}``: Factor returns of observation ``t``, on the axis of ``\\mathbf{M}_{t}``.
  - ``\\boldsymbol{\\varepsilon}_{t}``: Idiosyncratic returns of observation ``t``, the part of ``\\boldsymbol{x}_{t}`` the exposures and the intercept do not explain.
  - $(math_dict[:mu_er])
  - ``\\boldsymbol{\\mu}_{f}``: Expected factor returns ``K \\times 1``. A factor prior carries it, and this result does not.
  - ``\\boldsymbol{b}``: Factor-orthogonal expected return ``N \\times 1``, `b`. It is the part of ``\\boldsymbol{\\mu}`` the factors do not span, so it is a term of the expected return and never a term of one observation.
  - ``\\mathbf{M}``: Loadings matrix ``N \\times K`` of the factor model, `M`. It is the last slice of the exposure history.
  - $(math_dict[:N])
  - $(math_dict[:K])
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalFactorModel(;
        M::MatNum,
        L::Option{<:MatNum} = nothing,
        b::VecNum,
        csr::Option{<:CrossSectionalRegression} = nothing,
        Ms::Option{<:Arr3Num} = nothing,
        vs::Option{<:MatNum} = nothing,
        esigma::Option{<:VecNum_MatNum} = nothing,
        rw::Option{<:MatNum} = nothing,
        bw::Option{<:MatNum} = nothing,
        fam::Option{<:VecStr} = nothing,
        fcb = nothing,
        lag::Option{<:Integer} = nothing
    ) -> CrossSectionalFactorModel

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(M)`, `!isempty(b)`, and `length(b) == size(M, 1)`.
  - `L` and `fcb` are present together, or absent together.
  - If provided, `!isempty(L)`, and `size(L, 1) == size(M, 1)`.
  - If provided, `!isempty(fam)`, and `length(fam) == size(M, 2)`.
  - If provided, `!isempty(Ms)`, `size(Ms, 2) == size(M, 1)`, and `size(Ms, 3) == size(M, 2)`.
  - If provided, `size(csr.eps, 2) == size(M, 1)`.
  - If provided, `!isempty(vs)`, `!isempty(rw)`, `!isempty(bw)`, and each carries `size(M, 1)` columns.
  - Every two of `vs`, `rw` and `bw` that are present agree on the observation axis, so `size(rw) == size(bw) == size(vs)` when all three are present.
  - If provided, `!isempty(esigma)`, and `esigma` carries `size(M, 1)` entries when it is a vector, or is square with `size(M, 1)` rows when it is a matrix.
  - If provided, `lag >= 0`.

## View parameters

`CrossSectionalFactorModel` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `M`, `L` and `b` are sliced on their **first** axis, which is the asset axis of a loadings result.
  - `csr` is viewed by its own [`port_opt_view`](@ref) method.
  - `Ms` is sliced on its **second** axis, which is the asset axis of a slice.
  - `vs`, `rw` and `bw` are sliced on their **second** axis, which is the asset axis of a per-asset history.
  - `esigma` is sliced by [`idiosyncratic_covariance_view`](@ref), on one axis or on both.
  - `fam`, `fcb` and `lag` pass through unchanged. Each is indexed by factor, or by nothing at all, and neither follows an asset selection.

# Examples

```jldoctest
julia> CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0; 5.0 6.0], b = [0.1, 0.2, 0.3],
                                 esigma = [0.4, 0.5, 0.6], fam = [\"style\", \"style\"], lag = 1)
CrossSectionalFactorModel
       M ┼ 3×2 Matrix{Float64}
       L ┼ 3×2 Matrix{Float64}
       b ┼ Vector{Float64}: [0.1, 0.2, 0.3]
     csr ┼ nothing
      Ms ┼ nothing
      vs ┼ nothing
  esigma ┼ Vector{Float64}: [0.4, 0.5, 0.6]
      rw ┼ nothing
      bw ┼ nothing
     fam ┼ Vector{String}: ["style", "style"]
     fcb ┼ nothing
     lag ┴ Int64: 1
```

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`port_opt_view`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
"""
@concrete struct CrossSectionalFactorModel <: AbstractLoadingsRegressionResult
    """
    $(field_dict[:M]) Its columns are the named original factors, so a constraint written in a factor's name resolves against it.
    """
    M
    """
    $(field_dict[:L]) It is set only after a family re-basis, and an unset `L` reads back as `M`.
    """
    L
    """
    Factor-orthogonal expected return, one entry per asset. It is the part of the expected return the factors do not span, and it is not the per-observation intercept of the fit, which `csr` carries.
    """
    b
    """
    The cross-sectional fit the model was built around. It carries the factor returns, the idiosyncratic returns, the eligible asset counts and the per-observation intercepts.
    """
    csr
    """
    Exposure history `observations × assets × factors`. Its last slice is the loadings matrix `M`. The constructor checks the two axes it shares with `M` and never the entries, so a caller that builds both keeps them in step itself.
    """
    Ms
    """
    Idiosyncratic variance history `observations × assets`. Row `t` holds the variances estimated from the observations up to `t`.
    """
    vs
    """
    $(field_dict[:esigma])
    """
    esigma
    """
    Regression weight history `observations × assets`. Entry `(t, i)` is the weight asset `i` carried in the fit of observation `t`, and a weight of zero excluded the pair.
    """
    rw
    """
    Benchmark weight history `observations × assets`. Entry `(t, i)` is the weight asset `i` carried in the benchmark of observation `t`.
    """
    bw
    """
    Family label of each raw factor, one entry per column of `M`.
    """
    fam
    """
    The family re-basis `L` is written in. It is present exactly when `L` is present, and this result states no other rule about it.
    """
    fcb
    """
    Number of observations by which the exposures lag the returns.
    """
    lag
    function CrossSectionalFactorModel(M::MatNum, L::Option{<:MatNum}, b::VecNum,
                                       csr::Option{<:CrossSectionalRegression},
                                       Ms::Option{<:Arr3Num}, vs::Option{<:MatNum},
                                       esigma::Option{<:VecNum_MatNum},
                                       rw::Option{<:MatNum}, bw::Option{<:MatNum},
                                       fam::Option{<:VecStr}, fcb, lag::Option{<:Integer})
        @argcheck(!isempty(M), IsEmptyError("M cannot be empty"))
        @argcheck(!isempty(b), IsEmptyError("b cannot be empty"))
        N = size(M, 1)
        K = size(M, 2)
        @argcheck(length(b) == N,
                  DimensionMismatch("b ($(length(b))) must match M ($N rows)"))
        @argcheck(isnothing(L) == isnothing(fcb),
                  ArgumentError("L and fcb must be present together or absent together"))
        if !isnothing(L)
            @argcheck(!isempty(L), IsEmptyError("L cannot be empty"))
            @argcheck(size(L, 1) == N,
                      DimensionMismatch("L ($(size(L, 1)) rows) must match M ($N rows)"))
        end
        if !isnothing(fam)
            @argcheck(!isempty(fam), IsEmptyError("fam cannot be empty"))
            @argcheck(length(fam) == K,
                      DimensionMismatch("fam ($(length(fam))) must match M ($K columns)"))
        end
        if !isnothing(lag)
            assert_nonneg(lag, :lag)
        end
        assert_exposure_history(Ms, N, K)
        assert_cs_regression_assets(csr, N)
        assert_idiosyncratic_covariance(esigma, N)
        tvs = cs_history_assets(vs, N, :vs)
        trw = cs_history_assets(rw, N, :rw)
        tbw = cs_history_assets(bw, N, :bw)
        assert_cs_history_obs(trw, tbw, :rw, :bw)
        assert_cs_history_obs(trw, tvs, :rw, :vs)
        assert_cs_history_obs(tbw, tvs, :bw, :vs)
        return new{typeof(M), typeof(L), typeof(b), typeof(csr), typeof(Ms), typeof(vs),
                   typeof(esigma), typeof(rw), typeof(bw), typeof(fam), typeof(fcb),
                   typeof(lag)}(M, L, b, csr, Ms, vs, esigma, rw, bw, fam, fcb, lag)
    end
end
function CrossSectionalFactorModel(; M::MatNum, L::Option{<:MatNum} = nothing, b::VecNum,
                                   csr::Option{<:CrossSectionalRegression} = nothing,
                                   Ms::Option{<:Arr3Num} = nothing,
                                   vs::Option{<:MatNum} = nothing,
                                   esigma::Option{<:VecNum_MatNum} = nothing,
                                   rw::Option{<:MatNum} = nothing,
                                   bw::Option{<:MatNum} = nothing,
                                   fam::Option{<:VecStr} = nothing, fcb = nothing,
                                   lag::Option{<:Integer} = nothing)::CrossSectionalFactorModel
    return CrossSectionalFactorModel(M, L, b, csr, Ms, vs, esigma, rw, bw, fam, fcb, lag)
end
"""
    idiosyncratic_variances(rr::AbstractLoadingsRegressionResult)
    idiosyncratic_variances(esigma::VecNum, rr::AbstractLoadingsRegressionResult)
    idiosyncratic_variances(esigma::MatNum, rr::AbstractLoadingsRegressionResult)
    idiosyncratic_variances(esigma::Nothing, rr::Regression)
    idiosyncratic_variances(esigma::Nothing, rr::CrossSectionalFactorModel)

Read the idiosyncratic variance vector off a loadings block, whatever shape the block stores it in.

Both members of [`AbstractLoadingsRegressionResult`](@ref) carry `esigma` under one name, and both admit the two shapes: a vector of variances, or a full covariance. A consumer that needs the variances alone — an [`AbstractUncertaintySetEstimator`](@ref) that weights the cross-section by the inverse idiosyncratic variance is the first one — asks for them here rather than testing the shape at its own site.

The shape is the dispatch, as it is in [`assert_idiosyncratic_covariance`](@ref) and [`idiosyncratic_covariance_view`](@ref). The one-argument entry reads the field and forwards it beside the block, so the two refusals name the block they came from: a [`Regression`](@ref) is filled by the prior that lifts the factor moments, so its message names `rsd`, and a [`CrossSectionalFactorModel`](@ref) is filled by its own fit, so its message names the field.

There is no fallback. A block that carries no idiosyncratic covariance cannot answer, and an answer of ones or of zeros is a different weighting rather than a missing one.

# Arguments

  - `esigma`: Idiosyncratic covariance, a vector of variances, a square matrix, or `nothing`.
  - `rr`: The loadings block the field was read from, which the raise reports.

# Validation

  - `!isnothing(esigma)`, raising an `IsNothingError`.

# Returns

  - `esigma::VecNum`: The idiosyncratic variances, one per asset. A vector comes back unchanged, and a matrix comes back as its diagonal.

# Examples

```jldoctest
julia> re = Regression(; M = [1.0 2.0; 3.0 4.0], esigma = [0.1, 0.2]);

julia> PortfolioOptimisers.idiosyncratic_variances(re)
2-element Vector{Float64}:
 0.1
 0.2
```

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_idiosyncratic_covariance`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
"""
function idiosyncratic_variances(rr::AbstractLoadingsRegressionResult)
    return idiosyncratic_variances(rr.esigma, rr)
end
function idiosyncratic_variances(esigma::VecNum, ::AbstractLoadingsRegressionResult)
    return esigma
end
function idiosyncratic_variances(esigma::MatNum, ::AbstractLoadingsRegressionResult)
    return LinearAlgebra.diag(esigma)
end
function idiosyncratic_variances(::Nothing, rr::Regression)
    return throw(IsNothingError("`esigma` is unset on this loadings block, so it carries no idiosyncratic variances to read. A time-series factor prior writes them only when it adds a residual block, and this block was built by a fit that added none.\nFit the prior with `rsd = true`, so that the lift measures the residual variances and writes them onto the block.\nGot\nrr => $(nameof(typeof(rr)))\nesigma => nothing"))
end
function idiosyncratic_variances(::Nothing, rr::CrossSectionalFactorModel)
    return throw(IsNothingError("`esigma` is unset on this loadings block, so it carries no idiosyncratic variances to read. A cross-sectional factor model fills the field from its own fit, and this block was built without it.\nBuild the block with `esigma` set, so that the idiosyncratic variances travel with the loadings.\nGot\nrr => $(nameof(typeof(rr)))\nesigma => nothing"))
end
# When `L` is unset (`Nothing` type parameter), `:L` falls back to the loadings matrix `M`;
# when `L` is a stored matrix the default field access already returns it, so only the
# `Nothing` specialisation needs a rule (see [`@forward_properties`](@ref)'s `swap`).
@forward_properties CrossSectionalFactorModel{<:Any, Nothing, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any, <:Any} begin
    swap(L, M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

State whether the model was fitted in a re-based factor family.

`L` and `fcb` are present together and absent together, and the pair is present exactly when a Factor Family was re-based before the fit. The answer is therefore the presence of `L`, read with `getfield` rather than through property access: the `swap(L, M)` rule of [`CrossSectionalFactorModel`](@ref) makes `csfm.L` return `csfm.M` when `L` is unset, so a property read would answer `true` for every model.

# Arguments

  - `csfm`: A cross-sectional factor model result.

# Returns

  - `val::Bool`: `true` when `L` is set, so the raw factor axis of `M` is a linear image of the re-based one and a factor covariance stated on it is singular; `false` when the fit ran in the raw basis.

# Examples

```jldoctest
julia> PortfolioOptimisers.has_family_rebasis(CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0],
                                                                        b = [0.1, 0.2]))
false

julia> PortfolioOptimisers.has_family_rebasis(CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0],
                                                                        L = reshape([1.0, 2.0], 2,
                                                                                    1),
                                                                        b = [0.1, 0.2],
                                                                        fcb = [1.0, -1.0]))
true
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`AbstractLoadingsRegressionResult`](@ref)
"""
function has_family_rebasis(csfm::CrossSectionalFactorModel)::Bool
    return !isnothing(getfield(csfm, :L))
end
"""
    port_opt_view(csfm::CrossSectionalFactorModel, i, args...)

Return a view of a [`CrossSectionalFactorModel`](@ref) result, selecting only the assets indexed by `i`.

# Algorithm

 1. Read `L` with `getfield`, never through property access. The `swap(L, M)` rule of [`CrossSectionalFactorModel`](@ref) makes `csfm.L` return `csfm.M` when `L` is unset, so a property read would materialise `L` as a copy of `M` and lose the unset-ness.
 2. Take a row view of `M`, of `L` when step 1 found a matrix, and an element view of `b`, giving the loadings and the factor-orthogonal expected return of the selected assets.
 3. View the nested fit with its own [`port_opt_view`](@ref) method, which cuts its residuals on the asset axis.
 4. Take a view of `Ms` on its second axis, and of `vs`, `rw` and `bw` on their second axis, giving the histories of the selected assets.
 5. View `esigma` with [`idiosyncratic_covariance_view`](@ref), which reads its shape.
 6. Build a new [`CrossSectionalFactorModel`](@ref) from the views, passing `fam`, `fcb` and `lag` through, which re-runs every guard of the constructor.

# Arguments

  - `csfm`: A cross-sectional factor model result.
  - `i`: Indices of the assets to select.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `csfm::CrossSectionalFactorModel`: A new result whose per-asset fields are restricted to the selected assets.

# Examples

```jldoctest
julia> csfm = CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0; 5.0 6.0], b = [0.1, 0.2, 0.3],
                                        esigma = [0.4, 0.5, 0.6]);

julia> PortfolioOptimisers.port_opt_view(csfm, [1, 3]).b
2-element view(::Vector{Float64}, [1, 3]) with eltype Float64:
 0.1
 0.3

julia> isnothing(getfield(PortfolioOptimisers.port_opt_view(csfm, [1, 3]), :L))
true
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(csfm::CrossSectionalFactorModel, i,
                       args...)::CrossSectionalFactorModel
    # `L` must be read with `getfield`: the `swap(L, M)` property rule above makes
    # `csfm.L` return `csfm.M` when `L` is unset, so `isnothing(csfm.L)` is never true and
    # a viewed result would materialise `L` as a copy of `M`, silently losing the
    # unset-ness the rule exists to express.
    L = getfield(csfm, :L)
    csr = csfm.csr
    Ms = csfm.Ms
    vs = csfm.vs
    rw = csfm.rw
    bw = csfm.bw
    return CrossSectionalFactorModel(; M = view(csfm.M, i, :),
                                     L = isnothing(L) ? nothing : view(L, i, :),
                                     b = view(csfm.b, i),
                                     csr = if isnothing(csr)
                                         nothing
                                     else
                                         port_opt_view(csr, i, args...)
                                     end, Ms = isnothing(Ms) ? nothing : view(Ms, :, i, :),
                                     vs = isnothing(vs) ? nothing : view(vs, :, i),
                                     esigma = idiosyncratic_covariance_view(csfm.esigma, i),
                                     rw = isnothing(rw) ? nothing : view(rw, :, i),
                                     bw = isnothing(bw) ? nothing : view(bw, :, i),
                                     fam = csfm.fam, fcb = csfm.fcb, lag = csfm.lag)
end
"""
    regression(csfm::CrossSectionalFactorModel, args...)

Return the cross-sectional factor model unchanged.

This method is a pass-through for [`CrossSectionalFactorModel`](@ref) results, as the method over [`Regression`](@ref) is for that result. A consumer that binds `RegE_Reg` takes either a result or an estimator, and calls `regression` on both.

# Arguments

  - `csfm`: A cross-sectional factor model result.
  - `args...`: Additional arguments (ignored).

# Returns

  - The input `csfm`, unchanged.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`RegE_Reg`](@ref)
"""
function regression(csfm::CrossSectionalFactorModel, args...)
    return csfm
end

export CrossSectionalFactorModel
