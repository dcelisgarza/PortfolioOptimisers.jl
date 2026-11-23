"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the search algorithms of the minimum covariance determinant.

The minimum covariance determinant of [`MCDCovariance`](@ref) is the subset of `h` observations whose covariance matrix has the smallest determinant. An exact search tries every subset, so it is not practical. A subtype gives a search that starts from a set of subsets and improves each one by concentration steps, see [`mcd_csteps`](@ref). [`mcd_search`](@ref) returns the best subset that it finds.

# Interfaces

To implement a new search, subtype `AbstractMCDAlgorithm`, keep each parameter in a field, and implement one method:

  - `PortfolioOptimisers.mcd_starts(alg, Z::MatNum, h::Integer, fixed::Bool, intensity) -> Vector{Vector{Int}}`: Returns the start subsets of `h` observations of the standardised data `Z`. When `fixed` is `true`, the centre is fixed at zero. `intensity` is the shrinkage intensity or its rule, or `nothing` for a scatter with no shrinkage.

The fit does concentration steps from each start and keeps the best subset. A search with many starts can keep a part of them first, with one more method:

  - `PortfolioOptimisers.mcd_refine(alg, Z::MatNum, starts, h::Integer, fixed::Bool, reg) -> Vector{Vector{Int}}`: Returns the subsets from which the fit does concentration steps until they do not change. The default returns `starts`. `reg` is the [`SubsetShrinkage`](@ref), or `nothing`.

# Related

  - [`DetMCD`](@ref)
  - [`FastMCD`](@ref)
  - [`MCDCovariance`](@ref)
  - [`mcd_csteps`](@ref)
"""
abstract type AbstractMCDAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Searches the minimum covariance determinant from six deterministic starts.

The search of [hubert2012](@cite) computes six robust estimates of the correlation of the standardised data. Each one gives a start of `h` observations. The search then does concentration steps from each start until the subset does not change, and keeps the subset with the smallest determinant. Two calls on the same data give the same subset, so `cov(ce, X) == fit(ce, X).sigma` holds, and a cross-validation run gives the same estimate each time.

The search follows the code of the R package `robustbase`, version 0.99-7, with `nsamp = "deterministic"`, with one difference. That code scales the data by the tau scale when ``n \\geq 1000``, and [hubert2012](@cite) names the tau scale for a large sample as a faster choice. This search uses ``Q_n``, see [`qn_scale`](@ref), at every ``n``, because the selection of [`kth_pairwise_difference`](@ref) is fast at any ``n``.

# Examples

```jldoctest
julia> DetMCD()
DetMCD()
```

# Related

  - [`AbstractMCDAlgorithm`](@ref)
  - [`FastMCD`](@ref)
  - [`MCDCovariance`](@ref)
  - [`mcd_starts`](@ref)

# References

  - $(ref_dict[:hubert2012]) Sections 3.1 and 3.2.
"""
struct DetMCD <: AbstractMCDAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Searches the minimum covariance determinant from random starts.

The search of [rousseeuw1999](@cite) draws `n_starts` random subsets of ``p + 1`` observations. Each subset gives a start of `h` observations, the observations nearest to its mean in the metric of its covariance. The search does `n_csteps` concentration steps from each start. It keeps the `n_best` subsets with the smallest determinants, does concentration steps from each until the subset does not change, and returns the best.

The search draws random numbers, so the type has the fields `rng` and `seed`. [`resolve_rng`](@ref) combines them when the search runs. With `seed = nothing`, two calls can give two estimates. [`DetMCD`](@ref) is the default of [`MCDCovariance`](@ref) for that reason.

The search does not split a large sample into the nested subsamples of Section 3.3 of [rousseeuw1999](@cite). A start reads every observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FastMCD(;
        n_starts::Integer = 500,
        n_csteps::Integer = 2,
        n_best::Integer = 10,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing
    ) -> FastMCD

Keywords correspond to the struct's fields.

## Validation

  - `n_starts`, `n_csteps` and `n_best` are positive.
  - `n_best <= n_starts`.

# Examples

```jldoctest
julia> FastMCD(; seed = 42)
FastMCD
  n_starts ┼ Int64: 500
  n_csteps ┼ Int64: 2
    n_best ┼ Int64: 10
       rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
      seed ┴ Int64: 42
```

# Related

  - [`AbstractMCDAlgorithm`](@ref)
  - [`DetMCD`](@ref)
  - [`MCDCovariance`](@ref)
  - [`resolve_rng`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Sections 3.1, 3.2 and 4.
"""
@concrete struct FastMCD <: AbstractMCDAlgorithm
    """
    $(field_dict[:mcd_n_starts])
    """
    n_starts
    """
    $(field_dict[:mcd_n_csteps])
    """
    n_csteps
    """
    $(field_dict[:mcd_n_best])
    """
    n_best
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    function FastMCD(n_starts::Integer, n_csteps::Integer, n_best::Integer,
                     rng::Random.AbstractRNG, seed::Option{<:Integer})
        @argcheck(n_starts > 0 && n_csteps > 0 && n_best > 0,
                  DomainError((n_starts, n_csteps, n_best),
                              "n_starts, n_csteps and n_best must be positive"))
        @argcheck(n_best <= n_starts,
                  DomainError((n_best, n_starts),
                              "n_best must not be larger than n_starts, got n_best = $n_best and n_starts = $n_starts"))
        return new{typeof(n_starts), typeof(n_csteps), typeof(n_best), typeof(rng),
                   typeof(seed)}(n_starts, n_csteps, n_best, rng, seed)
    end
end
function FastMCD(; n_starts::Integer = 500, n_csteps::Integer = 2, n_best::Integer = 10,
                 rng::Random.AbstractRNG = Random.default_rng(),
                 seed::Option{<:Integer} = nothing)::FastMCD
    return FastMCD(n_starts, n_csteps, n_best, rng, seed)
end
"""
    MCD_MAX_CSTEPS

The largest number of concentration steps of one start of an MCD search, 200.

The concentration steps of [`mcd_csteps`](@ref) stop when the subset does not change. Theorem 1 of [rousseeuw1999](@cite) states that the determinant does not increase in a step, and the number of subsets is finite, so the steps converge. A start of real data converges in a few steps. The bound is the one of the R package `robustbase`, version 0.99-7.

# Related

  - [`mcd_csteps`](@ref)
"""
const MCD_MAX_CSTEPS = 200
"""
$(DocStringExtensions.TYPEDEF)

Holds the data that each step of the fit of an [`MCDCovariance`](@ref) reads.

[`mcd_sample`](@ref) builds it. A reweighting step reads it too, see [`mcd_reweight`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MCDSample(
        X::MatNum,
        Z::MatNum,
        tgt::Option{<:MatNum},
        intensity,
        centre::Option{<:VecNum}
    ) -> MCDSample

# Related

  - [`mcd_sample`](@ref)
  - [`mcd_fit`](@ref)
  - [`MCDCovariance`](@ref)
"""
@concrete struct MCDSample
    """
    $(field_dict[:mcd_X])
    """
    X
    """
    $(field_dict[:mcd_Z])
    """
    Z
    """
    $(field_dict[:mcd_smp_tgt])
    """
    tgt
    """
    $(field_dict[:mcd_smp_intensity])
    """
    intensity
    """
    $(field_dict[:mcd_centre])
    """
    centre
end
"""
    mcd_standardise(X::MatNum, centre::Option{<:VecNum}) -> Tuple{MatNum, VecNum}

Centres each column of the data and divides it by its robust scale.

The search of the MCD subset is affine equivariant, so the standardisation does not change the subset that a concentration step selects. It keeps the numbers near one, and the six starts of [`DetMCD`](@ref) read the standardised data.

A value that is further than `eps^(-1/4)` scales from the median of its column, 8192 scales in double precision, moves to that bound. Such a value is an outlier at either place. But the covariance matrix of a subset that holds an outlier larger than about `eps^(-1/2)` scales is singular in the arithmetic, and the search then stops although the rest of the data are regular. The search alone reads the standardised data, so the bound does not enter the estimate.

# Algorithm

 1. Refuse a constant column.
 2. Take the centre `c` of each column: the median `m` when `centre` is `nothing`, and `centre` otherwise.
 3. Take the scale `s` of each centred column with [`robust_column_scale`](@ref).
 4. Clamp each column to `m ± s * eps^(-1/4)`. Subtract `c`, and divide by `s`.

# Arguments

  - `X`: The data matrix, `observations × assets`.
  - `centre`: A fixed centre, or `nothing` for the median of each column.

# Validation

  - No column of `X` is constant. Otherwise, the function raises a `DomainError`, because the covariance matrix of each subset is singular.

# Returns

  - `(Z, s)`: The standardised data, and the scale of each column.

# Related

  - [`robust_column_scale`](@ref)
  - [`mcd_search`](@ref)
"""
function mcd_standardise(X::MatNum, centre::Option{<:VecNum})
    flat = findfirst(allequal, eachcol(X))
    @argcheck(isnothing(flat),
              DomainError(flat,
                          "column $flat of the data is constant, so its robust covariance is singular. A robust covariance needs data in general position"))
    m = Statistics.median(X; dims = 1)
    c = isnothing(centre) ? m : transpose(centre)
    s = mapslices(robust_column_scale, X .- c; dims = 1)
    b = s ./ sqrt(sqrt(eps(real(eltype(s)))))
    return (clamp.(X, m .- b, m .+ b) .- c) ./ s, vec(s)
end
"""
    mcd_subset_fit(Z::MatNum, idx::AbstractVector{<:Integer}, fixed::Bool) -> Tuple{VecNum, MatNum}

Returns the mean and the covariance matrix of the observations `idx` of `Z`.

When `fixed` is `false`, the centre is the mean of the subset and the divisor is ``h - 1``, as the code of the R package `robustbase`, version 0.99-7, has it. When `fixed` is `true`, the centre is zero and the divisor is ``h``, because the subset estimates no centre.

# Arguments

  - `Z`: The standardised data, `observations × assets`.
  - `idx`: The positions of the observations of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `(m, S)`: The centre and the covariance matrix of the subset.

# Related

  - [`mcd_csteps`](@ref)
"""
function mcd_subset_fit(Z::MatNum, idx::AbstractVector{<:Integer}, fixed::Bool)
    Zs = view(Z, idx, :)
    m = fixed ? zeros(eltype(Z), size(Z, 2)) : vec(Statistics.mean(Zs; dims = 1))
    D = Zs .- transpose(m)
    return m, (transpose(D) * D) / (length(idx) - !fixed)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the blend of the scatter of a subset with the target of an [`MCDShrinkage`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SubsetShrinkage(
        rho::Number,
        c::Number,
        tgt::MatNum
    ) -> SubsetShrinkage

# Related

  - [`shrink_scatter!`](@ref)
  - [`MCDShrinkage`](@ref)
"""
@concrete struct SubsetShrinkage
    """
    $(field_dict[:rob_rho_fit])
    """
    rho
    """
    $(field_dict[:mcd_c])
    """
    c
    """
    $(field_dict[:mcd_tgt])
    """
    tgt
end
"""
    shrink_scatter!(S::MatNum, reg::Nothing) -> MatNum
    shrink_scatter!(S::MatNum, c::Number) -> MatNum
    shrink_scatter!(S::MatNum, reg::SubsetShrinkage) -> MatNum

Scales the scatter of a subset in place, and blends it with the target.

The `nothing` method returns the scatter of a search with no shrinkage, where a common factor does not change the subset. The number method multiplies the scatter by the factor `c`. The [`SubsetShrinkage`](@ref) method gives ``\\rho \\mathbf{T} + (1 - \\rho) c \\mathbf{S}``, Equation 7 of [boudt2020](@cite).

# Arguments

  - `S`: The scatter of the subset.
  - `reg`, `c`: The shrinkage, the factor, or `nothing`.

# Returns

  - `S::MatNum`: The input matrix.

# Related

  - [`mcd_subset_fit`](@ref)
  - [`SubsetShrinkage`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 2.2, Equation 7.
"""
function shrink_scatter!(S::MatNum, ::Nothing)
    return S
end
function shrink_scatter!(S::MatNum, c::Number)
    S .*= c
    return S
end
function shrink_scatter!(S::MatNum, reg::SubsetShrinkage)
    S .*= (1 - reg.rho) * reg.c
    S .+= reg.rho .* reg.tgt
    return S
end
"""
    scatter_cholesky(S::MatNum) -> Option{<:LinearAlgebra.Cholesky}

Returns the Cholesky factorisation of a scatter matrix, or `nothing` when the matrix is singular.

A matrix of rank `p - 1` can pass the factorisation in the arithmetic, with a pivot of the size of the rounding error. A pivot over its diagonal entry is the share of the variance of a variable that the variables before it do not explain. So the function reads a share of at most `p * eps` as zero. The rule does not depend on the units of the variables.

# Arguments

  - `S`: The scatter matrix.

# Returns

  - `C`: The factorisation, or `nothing`.

# Related

  - [`squared_distances`](@ref)
  - [`mcd_csteps`](@ref)
"""
function scatter_cholesky(S::MatNum)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(S); check = false)
    tol = size(S, 1) * eps(real(eltype(S)))
    regular = LinearAlgebra.issuccess(C) &&
              all(j -> abs2(C.factors[j, j]) > tol * S[j, j], axes(S, 1))
    return regular ? C : nothing
end
"""
    squared_distances(Z::MatNum, m::VecNum, S::MatNum) -> VecNum

Returns the squared Mahalanobis distance of each observation of `Z` from the centre `m` in the metric of `S`.

# Mathematical definition

```math
\\begin{align}
d_t^2 &= (\\boldsymbol{z}_t - \\boldsymbol{m})^\\intercal \\mathbf{S}^{-1} (\\boldsymbol{z}_t - \\boldsymbol{m})\\,.
\\end{align}
```

# Arguments

  - `Z`: The data, `observations × assets`.
  - `m`: The centre.
  - `S`: The scatter matrix. It must be positive definite.

# Validation

  - `S` is positive definite, as [`scatter_cholesky`](@ref) reads it. Otherwise, the function raises a `DomainError` that states that the observations lie on a hyperplane.

# Returns

  - `d2::VecNum`: The squared distance of each observation.

# Related

  - [`mcd_csteps`](@ref)
  - [`RobustCovarianceResult`](@ref)
"""
function squared_distances(Z::MatNum, m::VecNum, S::MatNum)
    C = scatter_cholesky(S)
    @argcheck(!isnothing(C),
              DomainError(size(Z),
                          "the scatter matrix is singular, so the observations that it reads lie on a hyperplane. A robust covariance needs data in general position"))
    R = (Z .- transpose(m)) / C.U
    return vec(sum(abs2, R; dims = 2))
end
"""
    nearest_positions(d::VecNum, h::Integer) -> Vector{Int}

Returns the positions of the `h` smallest values of `d`, in increasing order of the values.

The function sorts the positions by their values, only up to the position `h`. The order of two equal values is not fixed.

# Arguments

  - `d`: The values, such as the squared distances of the observations.
  - `h`: The number of positions.

# Returns

  - `idx::Vector{Int}`: The positions.

# Related

  - [`squared_distances`](@ref)
  - [`mcd_csteps`](@ref)
"""
function nearest_positions(d::VecNum, h::Integer)
    return collect(partialsort!(collect(1:length(d)), 1:h; by = i -> d[i]))
end
"""
    mcd_csteps(Z::MatNum, idx::AbstractVector{<:Integer}, h::Integer, fixed::Bool,
               reg::Option{<:SubsetShrinkage}, maxsteps::Integer) -> Tuple{Vector{Int}, Number}

Does concentration steps from the subset `idx` and returns the last subset and the logarithm of the determinant of its covariance matrix.

A concentration step fits the mean and the covariance matrix of the subset, computes the distance of every observation, and takes the `h` nearest observations as the new subset. Theorem 1 of [rousseeuw1999](@cite) states that the determinant of the new covariance matrix is not larger than the old one. Theorem 1 of [boudt2020](@cite) states the same for a shrunk scatter. The steps stop when the subset does not change, or after `maxsteps` steps.

# Algorithm

 1. Fit the subset with [`mcd_subset_fit`](@ref), and shrink its scatter with [`shrink_scatter!`](@ref).
 2. Repeat at most `maxsteps` times. Compute the distances with [`squared_distances`](@ref). Take the `h` nearest observations with a partial sort. Stop if this subset is the old subset. Otherwise, fit the new subset.
 3. Return the subset and the logarithm of the determinant of its covariance matrix. The logarithm is `-Inf` for a singular matrix, see [`scatter_cholesky`](@ref).

# Arguments

  - `Z`: The standardised data, `observations × assets`.
  - `idx`: The positions of the start subset, `h` of them.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.
  - `reg`: The [`SubsetShrinkage`](@ref) of the subset scatter, or `nothing`.
  - `maxsteps`: The largest number of steps.

# Returns

  - `(idx, obj)`: The sorted positions of the last subset, and the logarithm of the determinant of its covariance matrix.

# Related

  - [`mcd_subset_fit`](@ref)
  - [`squared_distances`](@ref)
  - [`MCD_MAX_CSTEPS`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 2, Theorem 1.
  - $(ref_dict[:boudt2020]) Section 3, Theorem 1.
"""
function mcd_csteps(Z::MatNum, idx::AbstractVector{<:Integer}, h::Integer, fixed::Bool,
                    reg::Option{<:SubsetShrinkage}, maxsteps::Integer)
    idx = sort(idx)
    nidx = similar(idx)
    m, S = mcd_subset_fit(Z, idx, fixed)
    shrink_scatter!(S, reg)
    for _ in 1:maxsteps
        sort!(copyto!(nidx, nearest_positions(squared_distances(Z, m, S), h)))
        if nidx == idx
            break
        end
        idx, nidx = nidx, idx
        m, S = mcd_subset_fit(Z, idx, fixed)
        shrink_scatter!(S, reg)
    end
    C = scatter_cholesky(S)
    return idx, isnothing(C) ? typemin(real(eltype(S))) : LinearAlgebra.logdet(C)
end
"""
    mcd_univariate_subset(z::VecNum, h::Integer, fixed::Bool) -> Vector{Int}

Returns the exact MCD subset of one variable.

The covariance matrix of one variable is its variance. The subset of `h` values with the smallest variance is a run of `h` consecutive values of the sorted data, so a pass over the `n - h + 1` runs finds it. Each run holds the value of rank `h`, because `2h > n`. So the sums of a run are a sum to the left of that value plus a sum to its right, and an outlier outside the run does not enter them. When the centre is fixed at zero, the subset is the `h` values with the smallest absolute value.

# Arguments

  - `z`: The standardised values of the variable.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `idx::Vector{Int}`: The positions of the subset.

# Related

  - [`mcd_search`](@ref)
"""
function mcd_univariate_subset(z::VecNum, h::Integer, fixed::Bool)
    if fixed
        return nearest_positions(abs.(z), h)
    end
    o = sortperm(z)
    y = z[o]
    # Entry `k` is the sum over the run `k:(k + h - 1)`, which holds the position `h`.
    left(v) = reverse!(cumsum(reverse(view(v, 1:h))))
    right(v) = cumsum(vcat(zero(eltype(v)), view(v, (h + 1):length(v))))
    run_sums(v) = (r = right(v); view(left(v), eachindex(r)) .+ r)
    k = argmin(run_sums(abs2.(y)) .- abs2.(run_sums(y)) ./ h)
    return o[k:(k + h - 1)]
end
"""
    mcd_search(alg::AbstractMCDAlgorithm, smp::MCDSample, h::Integer, c::Number) -> Tuple{Vector{Int}, Option{<:Number}}

Returns the positions of the best subset of `h` observations that the search `alg` finds, and the shrinkage intensity.

The data of one variable have an exact subset, which [`mcd_univariate_subset`](@ref) finds for every search. A shrinkage does not change it. A subset of every observation needs no search.

# Algorithm

 1. When `h` is the number of observations, or the data have one column, take every position, or the exact subset of [`mcd_univariate_subset`](@ref). The intensity is the one of [`mcd_intensity`](@ref) for this subset.
 2. Otherwise, take the starts of the search with [`mcd_starts`](@ref), and the intensity and the starts that it permits with [`pooled_intensity`](@ref).
 3. Keep the best starts of a search with many starts with [`mcd_refine`](@ref).
 4. Do concentration steps from each start with [`mcd_csteps`](@ref) until the subset does not change, and return the subset with the smallest determinant. The first start wins a tie.

# Arguments

  - `alg`: The search algorithm.
  - `smp`: The data of the fit.
  - `h`: The size of the subset.
  - `c`: The consistency factor of the scatter of the subset.

# Returns

  - `(idx, rho)`: The sorted positions of the best subset, and the intensity, or `nothing` with no shrinkage.

# Related

  - [`AbstractMCDAlgorithm`](@ref)
  - [`MCDSample`](@ref)
  - [`mcd_csteps`](@ref)
  - [`mcd_starts`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 3, step 3 of the algorithm.
"""
function mcd_search(alg::AbstractMCDAlgorithm, smp::MCDSample, h::Integer, c::Number)
    Z = smp.Z
    n, p = size(Z)
    fixed = !isnothing(smp.centre)
    if h == n || p == 1
        idx = h == n ? collect(1:n) : sort!(mcd_univariate_subset(view(Z, :, 1), h, fixed))
        return idx, mcd_intensity(smp.intensity, Z, idx, c, fixed)
    end
    starts = mcd_starts(alg, Z, h, fixed, smp.intensity)
    rho, starts = pooled_intensity(smp.intensity, Z, starts, c, fixed)
    reg = subset_shrinkage(rho, c, p)
    starts = mcd_refine(alg, Z, starts, h, fixed, reg)
    return best_mcd_subset(Z, starts, h, fixed, reg), rho
end
"""
    best_mcd_subset(Z::MatNum, starts, h::Integer, fixed::Bool,
                    reg::Option{<:SubsetShrinkage}) -> Vector{Int}

Does concentration steps from each start until the subset does not change, and returns the subset with the smallest determinant.

# Arguments

  - `Z`: The standardised data.
  - `starts`: The start subsets.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.
  - `reg`: The [`SubsetShrinkage`](@ref) of the subset scatter, or `nothing`.

# Returns

  - `idx::Vector{Int}`: The sorted positions of the best subset. The first start wins a tie.

# Related

  - [`mcd_search`](@ref)
  - [`mcd_csteps`](@ref)
"""
function best_mcd_subset(Z::MatNum, starts, h::Integer, fixed::Bool,
                         reg::Option{<:SubsetShrinkage})
    best, bobj = Int[], typemax(real(eltype(Z)))
    for start in starts
        idx, obj = mcd_csteps(Z, start, h, fixed, reg, MCD_MAX_CSTEPS)
        if obj < bobj
            best, bobj = idx, obj
        end
    end
    return best
end
"""
    mcd_starts(alg::DetMCD, Z::MatNum, h::Integer, fixed::Bool, intensity) -> Vector{Vector{Int}}
    mcd_starts(alg::FastMCD, Z::MatNum, h::Integer, fixed::Bool, intensity) -> Vector{Vector{Int}}

Returns the start subsets of the search `alg`.

The [`DetMCD`](@ref) method returns six starts, one for each set of directions of [`detmcd_directions`](@ref), through [`detmcd_initset`](@ref). The [`FastMCD`](@ref) method resolves the generator from `alg.rng` and `alg.seed` with [`resolve_rng`](@ref), and draws `alg.n_starts` starts with [`fastmcd_random_start`](@ref).

# Arguments

  - `alg`: The search algorithm.
  - `Z`: The standardised data.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.
  - `intensity`: The shrinkage intensity or its rule, or `nothing`.

# Returns

  - `starts::Vector{Vector{Int}}`: The start subsets.

# Related

  - [`mcd_search`](@ref)
  - [`mcd_refine`](@ref)
  - [`detmcd_directions`](@ref)
  - [`detmcd_initset`](@ref)
  - [`fastmcd_random_start`](@ref)

# References

  - $(ref_dict[:hubert2012]) Section 3.
  - $(ref_dict[:rousseeuw1999]) Section 3.1.
"""
function mcd_starts(::DetMCD, Z::MatNum, h::Integer, fixed::Bool, ::Any)
    return [detmcd_initset(Z, P, h, fixed) for P in detmcd_directions(Z)]
end
function mcd_starts(alg::FastMCD, Z::MatNum, h::Integer, fixed::Bool, intensity)
    rng = resolve_rng(alg.rng, alg.seed)
    return [fastmcd_random_start(rng, Z, h, fixed, intensity) for _ in 1:(alg.n_starts)]
end
"""
    mcd_refine(alg::AbstractMCDAlgorithm, Z::MatNum, starts, h::Integer, fixed::Bool,
               reg::Option{<:SubsetShrinkage}) -> Vector{Vector{Int}}
    mcd_refine(alg::FastMCD, Z::MatNum, starts, h::Integer, fixed::Bool,
               reg::Option{<:SubsetShrinkage}) -> Vector{Vector{Int}}

Keeps the best starts of a search.

The default method keeps every start. The [`FastMCD`](@ref) method does `alg.n_csteps` concentration steps from each start with [`mcd_csteps`](@ref), and returns the `alg.n_best` subsets with the smallest determinants, in increasing order of the determinant. A start that the shrinkage excludes leaves fewer starts, and the method then keeps at most that number.

# Arguments

  - `alg`: The search algorithm.
  - `Z`: The standardised data.
  - `starts`: The start subsets.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.
  - `reg`: The [`SubsetShrinkage`](@ref) of the subset scatter, or `nothing`.

# Returns

  - `starts::Vector{Vector{Int}}`: The subsets from which the search does concentration steps until they do not change.

# Related

  - [`mcd_starts`](@ref)
  - [`mcd_csteps`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 3.2.
"""
function mcd_refine(::AbstractMCDAlgorithm, ::MatNum, starts, ::Integer, ::Bool,
                    ::Option{<:SubsetShrinkage})
    return starts
end
function mcd_refine(alg::FastMCD, Z::MatNum, starts, h::Integer, fixed::Bool,
                    reg::Option{<:SubsetShrinkage})
    runs = [mcd_csteps(Z, start, h, fixed, reg, alg.n_csteps) for start in starts]
    best = nearest_positions(map(last, runs), min(alg.n_best, length(runs)))
    return [first(runs[i]) for i in best]
end
"""
    fastmcd_random_start(rng::Random.AbstractRNG, Z::MatNum, h::Integer, fixed::Bool,
                         intensity::Nothing) -> Vector{Int}
    fastmcd_random_start(rng::Random.AbstractRNG, Z::MatNum, h::Integer, fixed::Bool,
                         intensity::Union{Number, AbstractSubsetShrinkageIntensity}) -> Vector{Int}

Draws one start subset of [`FastMCD`](@ref).

The method with no shrinkage draws ``p + 1`` observations with no repetition. While their covariance matrix is singular, it adds one more observation that it draws from the rest. Section 3.1 of [rousseeuw1999](@cite) gives this rule. The start is then the `h` observations nearest to the mean of the drawn observations in the metric of their covariance matrix.

With a shrinkage, the scatter of the drawn observations is not necessary, because the target gives a metric. The method draws two observations, the fewest that have a spread, so a start with no outlier is likely. The start is the `h` observations nearest to their mean in the metric of the target. With a fixed centre, every start is the `h` observations nearest to the centre.

# Arguments

  - `rng`: The random number generator.
  - `Z`: The standardised data.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.
  - `intensity`: The shrinkage intensity or its rule, or `nothing`.

# Validation

  - With no shrinkage, the covariance matrix of the data is not singular. Otherwise, the method raises the `DomainError` of [`squared_distances`](@ref).

# Returns

  - `idx::Vector{Int}`: The positions of the start subset.

# Related

  - [`mcd_starts`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 3.1.
"""
function fastmcd_random_start(rng::Random.AbstractRNG, Z::MatNum, h::Integer, fixed::Bool,
                              ::Nothing)
    n, p = size(Z)
    idx = StatsBase.sample(rng, 1:n, min(p + 1, n); replace = false)
    m, S = mcd_subset_fit(Z, idx, fixed)
    while length(idx) < n && isnothing(scatter_cholesky(S))
        push!(idx, rand(rng, setdiff(1:n, idx)))
        m, S = mcd_subset_fit(Z, idx, fixed)
    end
    return nearest_positions(squared_distances(Z, m, S), h)
end
function fastmcd_random_start(rng::Random.AbstractRNG, Z::MatNum, h::Integer, fixed::Bool,
                              ::Union{Number, AbstractSubsetShrinkageIntensity})
    idx = StatsBase.sample(rng, axes(Z, 1), 2; replace = false)
    m = if fixed
        zeros(eltype(Z), size(Z, 2))
    else
        vec(Statistics.mean(view(Z, idx, :); dims = 1))
    end
    return nearest_positions(vec(sum(abs2, Z .- transpose(m); dims = 2)), h)
end

export DetMCD, FastMCD
public AbstractMCDAlgorithm, mcd_starts, mcd_refine
