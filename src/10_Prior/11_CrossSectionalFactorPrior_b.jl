"""
    cross_sectional_live_dropped(dc::Nothing, lv::AbstractVector{Bool}) -> Nothing
    cross_sectional_live_dropped(dc::AbstractMatrix{Bool}, lv::AbstractVector{Bool})
        -> BitMatrix

Expand the dropped-column mask of a regression on the factors that are not empty onto every factor.

An Empty Factor takes a return of zero because no pair loads on it, not because the solve dropped its column, so its column of the mask is `false`.

# Arguments

  - `dc`: The dropped-column mask of the regression on the factors that `lv` marks, or `nothing`.
  - `lv`: `true` at each factor that is not empty.

# Returns

  - `dc::Option{<:BitMatrix}`: The mask on every factor, or `nothing`.

# Related

  - [`cross_sectional_live_regression`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function cross_sectional_live_dropped(::Nothing, ::AbstractVector{Bool})::Nothing
    return nothing
end
function cross_sectional_live_dropped(dc::AbstractMatrix{Bool},
                                      lv::AbstractVector{Bool})::BitMatrix
    d = falses(size(dc, 1), length(lv))
    d[:, lv] = dc
    return d
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Mark each factor that is not an Empty Factor over the observations of a design.

A factor is empty when its exposure is zero at every (observation, asset) pair of positive weight. [`cross_sectional_live_regression`](@ref) reads the mark over every observation of its design. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads it over its new observations, and joins it to the mark of the observations it fitted.

# Algorithm

 1. Take the pairs of positive weight with [`cross_sectional_design_mask`](@ref), giving `act`.
 2. Mark each factor whose exposure is not zero at a pair of `act`.

# Arguments

  - `Z`: Exposure tensor `assets × factors × observations`.
  - `X`: Asset returns matrix `observations × assets`.
  - `W`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `lv::BitVector`: `true` at each factor that is not empty.

# Related

  - [`cross_sectional_live_regression`](@ref)
  - [`cross_sectional_design_mask`](@ref)
"""
function cross_sectional_live_factors(Z::Arr3Num, X::MatNum, W::MatNum)::BitVector
    act = findall(cross_sectional_design_mask(Z, X, W))
    return BitVector([any(c -> !iszero(Z[c[2], k, c[1]]), act) for k in axes(Z, 2)])
end
"""
    cross_sectional_cold_factors(Ms::Arr3Num, nf::VecStr, Zo::Option{<:Arr3Num}) -> String

Returns the sentence that names the factors whose exposure history holds no finite value, or an empty string where every factor holds one, so the refusal of [`cross_sectional_warmup`](@ref) can name the factor that consumed the whole history. The factors of `Zo` follow those of `Ms`.

The sentence gives the name of the factor, as `"momentum"` or `"industry=Energy"`, not the position of its column. A one-hot family takes one column per level, so a position does not match the order of the factors that the user wrote.

# Arguments

  - `Ms`: The exposure history, `assets × factors × observations`.
  - `nf`: The name of each factor, the columns of `Ms` and then the columns of `Zo`. [`cross_sectional_warmup`](@ref) checks its length.
  - `Zo`: The exposure history of the observed factors, on the axes of `Ms`, or `nothing`.

# Returns

  - `msg::String`: The sentence, with a leading space, or `""`.

# Related

  - [`cross_sectional_warmup`](@ref)
  - [`cross_sectional_eligible`](@ref)
"""
function cross_sectional_cold_factors(Ms::Arr3Num, nf::VecStr,
                                      Zo::Option{<:Arr3Num})::String
    H = isnothing(Zo) ? Ms : cat(Ms, Zo; dims = 2)
    k = findall(j -> !any(isfinite, view(H, :, j, :)), axes(H, 2))
    return if isempty(k)
        ""
    else
        " The factor(s) $(join(map(repr, nf[k]), ", ")) have no finite Factor Exposure at any observation."
    end
end
function lookback(pe::CrossSectionalFactorPrior)::Option{<:Integer}
    return cross_sectional_lookback(pe, lookback(map(last, pe.factors)), lookback)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the look-back of a [`CrossSectionalFactorPrior`](@ref) from the look-back of its factors: that look-back plus `lag`, or the look-back of its Return Forecast Estimator when that one is larger. `nothing` from either part gives `nothing`.

[`lookback`](@ref) gives it the look-back of the factors of the batch fit and itself as `f`. [`carry_lookback`](@ref) gives it the look-back of the factors on the carry fold and itself as `f`, so a Descriptor of the scores of the forecast that carries a state counts one row.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `L`: The look-back of its factors, or `nothing`.
  - `f`: The verb that counts the look-back of the Return Forecast Estimator, [`lookback`](@ref) or [`carry_lookback`](@ref).

# Returns

  - `n::Option{<:Integer}`: The look-back of the prior, or `nothing`.

# Related

  - [`lookback`](@ref)
  - [`carry_lookback`](@ref)
"""
function cross_sectional_lookback(pe::CrossSectionalFactorPrior, L::Option{<:Integer},
                                  f::Function)::Option{<:Integer}
    lr = isnothing(pe.rfe) ? 1 : f(pe.rfe)
    return isnothing(L) || isnothing(lr) ? nothing : max(L + pe.lag, lr)
end
