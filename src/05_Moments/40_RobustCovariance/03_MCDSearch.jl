"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the search algorithms of the minimum covariance determinant.

The minimum covariance determinant of [`MCDCovariance`](@ref) is the subset of `h` observations whose covariance matrix has the smallest determinant. An exact search tries every subset, so it is not practical. A subtype gives a search that starts from a set of subsets and improves each one by concentration steps, see [`mcd_csteps`](@ref). [`mcd_search`](@ref) returns the best subset that it finds.

# Interfaces

To implement a new search, subtype `AbstractMCDAlgorithm`, keep each parameter in a field, and implement one method:

  - `PortfolioOptimisers.mcd_starts(alg, Z::MatNum, h::Integer, fixed::Bool) -> Vector{Vector{Int}}`: Returns the start subsets of `h` observations of the standardised data `Z`. When `fixed` is `true`, the centre is fixed at zero. The fit does concentration steps from each start and keeps the best subset.

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
    mcd_standardise(X::MatNum, centre::Option{<:VecNum}) -> Tuple{MatNum, VecNum, VecNum}

Centres each column of the data and divides it by its robust scale.

The search of the MCD subset is affine equivariant, so the standardisation does not change the subset that a concentration step selects. It keeps the numbers near one, and the six starts of [`DetMCD`](@ref) read the standardised data.

# Algorithm

 1. Take the centre `c` of each column: the median when `centre` is `nothing`, and `centre` otherwise.
 2. Take the scale `s` of each centred column with [`robust_column_scale`](@ref).
 3. Return `(X .- c') ./ s'`, `c` and `s`.

# Arguments

  - `X`: The data matrix, `observations × assets`.
  - `centre`: A fixed centre, or `nothing` for the median of each column.

# Returns

  - `(Z, c, s)`: The standardised data, the centre and the scale.

# Related

  - [`robust_column_scale`](@ref)
  - [`mcd_search`](@ref)
"""
function mcd_standardise(X::MatNum, centre::Option{<:VecNum})
    c = isnothing(centre) ? vec(Statistics.median(X; dims = 1)) : centre
    Xc = X .- transpose(c)
    s = map(robust_column_scale, eachcol(Xc))
    return Xc ./ transpose(s), c, s
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

  - `S` is positive definite. Otherwise, the function raises a `DomainError` that states that the observations lie on a hyperplane.

# Returns

  - `d2::VecNum`: The squared distance of each observation.

# Related

  - [`mcd_csteps`](@ref)
  - [`RobustCovarianceResult`](@ref)
"""
function squared_distances(Z::MatNum, m::VecNum, S::MatNum)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(S); check = false)
    @argcheck(LinearAlgebra.issuccess(C),
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
               maxsteps::Integer) -> Tuple{Vector{Int}, Number}

Does concentration steps from the subset `idx` and returns the last subset and the logarithm of the determinant of its covariance matrix.

A concentration step fits the mean and the covariance matrix of the subset, computes the distance of every observation, and takes the `h` nearest observations as the new subset. Theorem 1 of [rousseeuw1999](@cite) states that the determinant of the new covariance matrix is not larger than the old one. The steps stop when the subset does not change, or after `maxsteps` steps.

# Algorithm

 1. Fit the subset with [`mcd_subset_fit`](@ref).
 2. Repeat at most `maxsteps` times. Compute the distances with [`squared_distances`](@ref). Take the `h` nearest observations with a partial sort. Stop if this subset is the old subset. Otherwise, fit the new subset.
 3. Return the subset and the logarithm of the determinant of its covariance matrix.

# Arguments

  - `Z`: The standardised data, `observations × assets`.
  - `idx`: The positions of the start subset, `h` of them.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.
  - `maxsteps`: The largest number of steps.

# Returns

  - `(idx, obj)`: The sorted positions of the last subset, and the logarithm of the determinant of its covariance matrix.

# Related

  - [`mcd_subset_fit`](@ref)
  - [`squared_distances`](@ref)
  - [`MCD_MAX_CSTEPS`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 2, Theorem 1.
"""
function mcd_csteps(Z::MatNum, idx::AbstractVector{<:Integer}, h::Integer, fixed::Bool,
                    maxsteps::Integer)
    idx = sort(idx)
    nidx = similar(idx)
    m, S = mcd_subset_fit(Z, idx, fixed)
    for _ in 1:maxsteps
        sort!(copyto!(nidx, nearest_positions(squared_distances(Z, m, S), h)))
        if nidx == idx
            break
        end
        idx, nidx = nidx, idx
        m, S = mcd_subset_fit(Z, idx, fixed)
    end
    return idx, LinearAlgebra.logdet(LinearAlgebra.Symmetric(S))
end
"""
    mcd_univariate_subset(z::VecNum, h::Integer, fixed::Bool) -> Vector{Int}

Returns the exact MCD subset of one variable.

The covariance matrix of one variable is its variance. The subset of `h` values with the smallest variance is a run of `h` consecutive values of the sorted data, so a pass over the `n - h + 1` runs finds it. When the centre is fixed at zero, the subset is the `h` values with the smallest absolute value.

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
    s1 = cumsum(vcat(zero(eltype(y)), y))
    s2 = cumsum(vcat(zero(eltype(y)), abs2.(y)))
    ss = [(s2[k + h] - s2[k]) - (s1[k + h] - s1[k])^2 / h for k in 1:(length(y) - h + 1)]
    k = argmin(ss)
    return o[k:(k + h - 1)]
end
"""
    mcd_search(alg::AbstractMCDAlgorithm, Z::MatNum, h::Integer, fixed::Bool) -> Vector{Int}

Returns the positions of the best subset of `h` observations of the standardised data `Z` that the search `alg` finds.

The data of one variable have an exact subset, which [`mcd_univariate_subset`](@ref) finds for every search. A subset of every observation needs no search.

# Algorithm

 1. When `h` is the number of observations, return every position.
 2. When `Z` has one column, return the exact subset of [`mcd_univariate_subset`](@ref).
 3. Otherwise, take the starts of the search with [`mcd_starts`](@ref): six deterministic starts for [`DetMCD`](@ref), and the best random starts for [`FastMCD`](@ref). Do concentration steps from each start with [`mcd_csteps`](@ref) until the subset does not change, and return the subset with the smallest determinant. The first start wins a tie.

# Arguments

  - `alg`: The search algorithm.
  - `Z`: The standardised data, `observations × assets`.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `idx::Vector{Int}`: The sorted positions of the best subset.

# Related

  - [`AbstractMCDAlgorithm`](@ref)
  - [`mcd_csteps`](@ref)
  - [`mcd_starts`](@ref)
"""
function mcd_search(alg::AbstractMCDAlgorithm, Z::MatNum, h::Integer, fixed::Bool)
    n, p = size(Z)
    if h == n
        return collect(1:n)
    elseif p == 1
        return sort!(mcd_univariate_subset(view(Z, :, 1), h, fixed))
    end
    return best_mcd_subset(Z, mcd_starts(alg, Z, h, fixed), h, fixed)
end
"""
    best_mcd_subset(Z::MatNum, starts, h::Integer, fixed::Bool) -> Vector{Int}

Does concentration steps from each start until the subset does not change, and returns the subset with the smallest determinant.

# Arguments

  - `Z`: The standardised data.
  - `starts`: The start subsets.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `idx::Vector{Int}`: The sorted positions of the best subset. The first start wins a tie.

# Related

  - [`mcd_search`](@ref)
  - [`mcd_csteps`](@ref)
"""
function best_mcd_subset(Z::MatNum, starts, h::Integer, fixed::Bool)
    best, bobj = Int[], typemax(real(eltype(Z)))
    for start in starts
        idx, obj = mcd_csteps(Z, start, h, fixed, MCD_MAX_CSTEPS)
        if obj < bobj
            best, bobj = idx, obj
        end
    end
    return best
end
"""
    mcd_starts(alg::DetMCD, Z::MatNum, h::Integer, fixed::Bool) -> Vector{Vector{Int}}
    mcd_starts(alg::FastMCD, Z::MatNum, h::Integer, fixed::Bool) -> Vector{Vector{Int}}

Returns the start subsets of the search `alg`.

# Algorithm

The [`DetMCD`](@ref) method returns six starts, one for each set of directions of [`detmcd_directions`](@ref), through [`detmcd_initset`](@ref).

The [`FastMCD`](@ref) method returns the `n_best` random starts with the smallest determinants after `n_csteps` concentration steps:

 1. Resolve the generator from `alg.rng` and `alg.seed` with [`resolve_rng`](@ref).
 2. For each of the `n_starts` starts, draw a start subset with [`fastmcd_random_start`](@ref), and do `alg.n_csteps` concentration steps from it with [`mcd_csteps`](@ref).
 3. Return the `alg.n_best` subsets with the smallest determinants, in increasing order of the determinant.

# Arguments

  - `alg`: The search algorithm.
  - `Z`: The standardised data.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `starts::Vector{Vector{Int}}`: The start subsets.

# Related

  - [`mcd_search`](@ref)
  - [`detmcd_directions`](@ref)
  - [`detmcd_initset`](@ref)
  - [`fastmcd_random_start`](@ref)
  - [`mcd_csteps`](@ref)

# References

  - $(ref_dict[:hubert2012]) Section 3.
  - $(ref_dict[:rousseeuw1999]) Sections 3.1 and 3.2.
"""
function mcd_starts(::DetMCD, Z::MatNum, h::Integer, fixed::Bool)
    return [detmcd_initset(Z, P, h, fixed) for P in detmcd_directions(Z)]
end
function mcd_starts(alg::FastMCD, Z::MatNum, h::Integer, fixed::Bool)
    rng = resolve_rng(alg.rng, alg.seed)
    runs = [mcd_csteps(Z, fastmcd_random_start(rng, Z, h, fixed), h, fixed, alg.n_csteps)
            for _ in 1:(alg.n_starts)]
    return [first(runs[i]) for i in nearest_positions(map(last, runs), alg.n_best)]
end
"""
    fastmcd_random_start(rng::Random.AbstractRNG, Z::MatNum, h::Integer, fixed::Bool) -> Vector{Int}

Draws one start subset of [`FastMCD`](@ref).

The method draws ``p + 1`` observations with no repetition. While their covariance matrix is singular, it adds one more observation that it draws from the rest. Section 3.1 of [rousseeuw1999](@cite) gives this rule. The start is then the `h` observations nearest to the mean of the drawn observations in the metric of their covariance matrix.

# Arguments

  - `rng`: The random number generator.
  - `Z`: The standardised data.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Validation

  - The covariance matrix of the data is not singular. Otherwise, the method raises the `DomainError` of [`squared_distances`](@ref).

# Returns

  - `idx::Vector{Int}`: The positions of the start subset.

# Related

  - [`mcd_starts`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 3.1.
"""
function fastmcd_random_start(rng::Random.AbstractRNG, Z::MatNum, h::Integer, fixed::Bool)
    n, p = size(Z)
    idx = StatsBase.sample(rng, 1:n, min(p + 1, n); replace = false)
    m, S = mcd_subset_fit(Z, idx, fixed)
    while length(idx) < n && !LinearAlgebra.isposdef(LinearAlgebra.Symmetric(S))
        push!(idx, rand(rng, setdiff(1:n, idx)))
        m, S = mcd_subset_fit(Z, idx, fixed)
    end
    return nearest_positions(squared_distances(Z, m, S), h)
end

export DetMCD, FastMCD
public AbstractMCDAlgorithm, mcd_starts
