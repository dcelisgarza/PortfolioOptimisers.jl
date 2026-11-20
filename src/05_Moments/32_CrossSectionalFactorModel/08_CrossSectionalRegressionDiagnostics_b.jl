"""
    cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                       W::MatNum) -> VecNum

Return the weighted coefficient of determination of every observation.

An observation whose weighted total sum of squares is zero has no defined ratio, and its entry is `NaN`. [`mean_cross_sectional_r2`](@ref) is the scalar summary that skips those entries.

# Mathematical definition

```math
\\begin{align}
R^{2}_{t} &= 1 - \\frac{\\sum_{i} w_{t,i} \\left(x_{t,i} - \\hat{x}_{t,i}\\right)^{2}}{\\sum_{i} w_{t,i} \\left(x_{t,i} - \\bar{x}_{t}\\right)^{2}} \\\\
\\bar{x}_{t} &= \\frac{\\sum_{i} w_{t,i} x_{t,i}}{\\sum_{i} w_{t,i}}\\,.
\\end{align}
```

Where:

  - ``R^{2}_{t}``: Weighted coefficient of determination of observation ``t``.
  - ``x_{t,i}``: Return of asset ``i`` at observation ``t``.
  - ``\\hat{x}_{t,i}``: Systematic return of asset ``i`` at observation ``t``.
  - ``\\bar{x}_{t}``: Weighted mean return of observation ``t``.
  - ``w_{t,i} \\geq 0``: Cross-sectional weight of asset ``i`` at observation ``t``. Both sums run over the eligible assets alone.

# Algorithm

 1. Take the eligibility mask through [`cross_sectional_design_mask`](@ref).
 2. Take the systematic returns through [`StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)`](@ref).
 3. For each observation, sum the weighted squared residuals and the weighted squared deviations from the weighted mean, over the eligible assets alone.
 4. Return `1 - rss / tss` per observation, and `NaN` where `tss` is not positive.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `assets × factors × observations`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `r2::VecNum`: Coefficient of determination of every observation, of length `observations`, in the type a division of the inputs lands in.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 3, 2, 1);

julia> csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5],
                                        ones(1, 3));

julia> cross_sectional_r2(csr, Z, [1.0 2.0 1.5], ones(1, 3))
1-element Vector{Float64}:
 1.0
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`mean_cross_sectional_r2`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                            W::MatNum)::VecNum
    act = cross_sectional_design_mask(Z, X, W)
    Xh = StatsAPI.predict(csr, Z)
    # The ratio keeps the type of the inputs, so a `Float32` fit answers a `Float32` ratio
    # rather than one widened by a `Float64` `NaN`.
    Tf = promote_type(real(eltype(Xh)), real(eltype(X)), real(eltype(W)))
    r2 = fill(convert(Tf, NaN), size(X, 1))
    for t in axes(X, 1)
        idx = findall(view(act, t, :))
        w = view(W, t, idx)
        y = view(X, t, idx)
        sw = sum(w)
        if sw <= zero(sw)
            continue
        end
        ybar = LinearAlgebra.dot(w, y) / sw
        rss = sum(w[k] * abs2(y[k] - Xh[t, idx[k]]) for k in eachindex(idx))
        tss = sum(w[k] * abs2(y[k] - ybar) for k in eachindex(idx))
        if tss > zero(tss)
            r2[t] = 1 - rss / tss
        end
    end
    return r2
end
"""
    mean_cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                            W::MatNum) -> Number

Return the mean weighted coefficient of determination across the observations.

The mean skips every observation whose ratio is undefined, and it is `NaN` when no observation defines one.

# Algorithm

 1. Take the per-observation vector through [`cross_sectional_r2`](@ref).
 2. Return the mean of its finite entries, and `NaN` when it holds none.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `assets × factors × observations`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `r2::Number`: Mean coefficient of determination across the observations.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 3, 2, 1);

julia> csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5],
                                        ones(1, 3));

julia> mean_cross_sectional_r2(csr, Z, [1.0 2.0 1.5], ones(1, 3))
1.0
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_r2`](@ref)
"""
function mean_cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                                 W::MatNum)::Number
    r2 = cross_sectional_r2(csr, Z, X, W)
    keep = filter(isfinite, r2)
    return isempty(keep) ? convert(eltype(r2), NaN) : sum(keep) / length(keep)
end

export cross_sectional_r2, mean_cross_sectional_r2
