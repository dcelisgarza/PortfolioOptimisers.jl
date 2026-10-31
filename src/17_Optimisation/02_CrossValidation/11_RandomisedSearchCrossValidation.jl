"""
    make_p_grid(ps::Union{AbstractDict, AbstractVector{<:Pair}}, n_iter::Integer,
                rng::Random.AbstractRNG)
    make_p_grid(pss::AbstractVector{<:Union{AbstractDict, AbstractVector{<:Pair}}},
                n_iter::Integer, rng::Random.AbstractRNG)

Draw the trials of a randomised search, at most `n_iter` points of the search space, each a joint value of every key of one parameter set.

The output is a grid of one-point parameter sets, so the grid search of the trials scores each trial once, and no two trials combine into a new candidate.

# Mathematical definition

When every value of every parameter set is a vector:

```math
\\begin{align}
\\Lambda^{(l)} &= V_1 \\times V_2 \\times \\cdots \\times V_k\\,, \\quad l = 1, \\ldots, m\\,, \\\\
\\Lambda &= \\Lambda^{(1)} \\Vert \\Lambda^{(2)} \\Vert \\cdots \\Vert \\Lambda^{(m)}\\,, \\\\
\\Theta &\\sim \\mathcal{U}\\left\\{ S \\subseteq \\Lambda : |S| = \\min\\left(n, |\\Lambda|\\right) \\right\\}\\,.
\\end{align}
```

When a value is a distribution:

```math
\\begin{align}
l_t &\\sim \\mathcal{U}\\{1, \\ldots, m\\}\\,, \\quad t = 1, \\ldots, n\\,, \\\\
\\lambda_{tj} &\\sim D_j\\,, \\quad j = 1, \\ldots, k\\,, \\\\
\\theta_t &= \\left(\\lambda_{t1}, \\ldots, \\lambda_{tk}\\right)\\,.
\\end{align}
```

In the first form ``V_j`` and ``k`` belong to the ``l``-th parameter set. In the second form ``D_j`` and ``k`` belong to the set ``l_t``.

Where:

  - $(math_dict[:Lambda_l_space])
  - $(math_dict[:V_j_grid])
  - $(math_dict[:k_grid_keys])
  - $(math_dict[:m_grid_sets])
  - $(math_dict[:Lambda_space])
  - $(math_dict[:Vert_concat])
  - $(math_dict[:Theta_trials])
  - $(math_dict[:n_iter_trials])
  - $(math_dict[:U_uniform_law])
  - $(math_dict[:l_t_set])
  - $(math_dict[:lambda_tj_value])
  - $(math_dict[:D_j_law])
  - $(math_dict[:theta_t_trial])

[bergstra2012](@cite) draws independent trials from the space that a grid spans. The first form departs from it in one point: it draws the points of a finite space without replacement, so no fit repeats. It keeps the order of ``\\Lambda``, so when ``n \\geq |\\Lambda|``, ``\\Theta`` is the grid of [`GridSearchCrossValidation`](@ref) in its own order. The second form draws the trials independently, so two trials can be equal. It picks a parameter set with equal probability, because a set that holds a distribution has no count of points to weight it by.

# Algorithm

 1. Check that no value vector is empty.
 2. If a value of any parameter set is a distribution, do steps 3 and 4. Otherwise, do steps 5 to 8.
 3. Check `n_iter` through [`assert_search_grid_cap`](@ref).
 4. For each of the `n_iter` trials, pick a parameter set uniformly, and draw one value of each of its keys through [`draw_search_point`](@ref).
 5. Count the grid points of each parameter set, giving `sizes`, and their sum `N`, both as `BigInt`.
 6. Set `M = min(n_iter, N)`, and check it through [`assert_search_grid_cap`](@ref).
 7. Draw `M` distinct positions of `1:N` in ascending order through [`sample_grid_positions`](@ref), giving `idx`. When `M == N`, `idx` is `1:N`.
 8. Decode each position into its grid point through [`search_grid_point`](@ref).
 9. Return one parameter set per trial, each key with a vector of its one value.

# Arguments

  - `ps`: One parameter set, a vector of `key => value` pairs or a dictionary from key to value. A value is a vector of candidate values or a `Distributions.Distribution`.
  - `pss`: A vector of parameter sets, each of the form of `ps`.
  - `n_iter`: The trial count ``n``.
  - `rng`: The random number generator of the draw.

# Validation

  - Every value vector holds at least one value. Otherwise an `IsEmptyError` names the key.
  - The trial count holds at most `RESOURCE_LIMITS[].max_search_grid` candidates, through [`assert_search_grid_cap`](@ref). The check reads the count of the trials, not the count of ``\\Lambda``, because the search scores the trials alone.

# Returns

  - `trials::AbstractVector{<:AbstractVector{<:Pair}}`: One parameter set per trial, in the order of ``\\Theta``. The value of each key is a vector of one value.

# Related

  - [`RandomisedSearchCrossValidation`](@ref)
  - [`draw_search_point`](@ref)
  - [`sample_grid_positions`](@ref)
  - [`search_grid_point`](@ref)
  - [`lens_val_grid`](@ref)
  - [`assert_search_grid_cap`](@ref)

# References

  - $(ref_dict[:bergstra2012]) Section 1, Equation 4.
"""
function make_p_grid(ps::Union{AbstractDict, AbstractVector{<:Pair}}, n_iter::Integer,
                     rng::Random.AbstractRNG)
    return make_p_grid([ps], n_iter, rng)
end
function make_p_grid(pss::AbstractVector{<:Union{AbstractDict, AbstractVector{<:Pair}}},
                     n_iter::Integer, rng::Random.AbstractRNG)
    has_dist = false
    for ps in pss, (key, vals) in ps
        @argcheck(!isa(vals, AbstractVector) || !isempty(vals),
                  IsEmptyError("the randomised search parameter `$key` holds no candidate value, so the search cannot draw it. Give every tuned parameter at least one value."))
        has_dist |= isa(vals, Distributions.Distribution)
    end
    trials = if has_dist
        assert_search_grid_cap(n_iter,
                               "the n_iter = $n_iter trials of the randomised search")
        Vector{Pair}[draw_search_point(rng, pss[rand(rng, eachindex(pss))])
                     for _ in 1:n_iter]
    else
        sizes = fill(big(1), length(pss))
        for (l, ps) in enumerate(pss), (_, vals) in ps
            sizes[l] *= length(vals)
        end
        N = sum(sizes)
        M = min(n_iter, N)
        assert_search_grid_cap(M,
                               "the n_iter = $n_iter trials of the randomised search, out of $N grid points")
        idx = M == N ? (1:Int(N)) : sample_grid_positions(rng, N, M)
        Vector{Pair}[search_grid_point(pss, sizes, i) for i in idx]
    end
    return concrete_typed_array([concrete_typed_array(trial) for trial in trials])
end
"""
    sample_grid_positions(rng::Random.AbstractRNG, N::Integer, M::Integer)

Draw `M` distinct positions of `1:N` uniformly, in ascending order.

Each subset of `M` positions has the same probability. When `N` fits in an `Int`, `StatsBase.sample` draws without replacement. A larger `N` overflows its sampler, so the draw repeats a uniform pick of `1:N` and skips a position that it already holds. [`make_p_grid`](@ref) caps `M` at `RESOURCE_LIMITS[].max_search_grid`, which is far below `typemax(Int)`, so a skip almost never happens on that branch.

# Algorithm

 1. If `N <= typemax(Int)`, draw `M` positions of `1:N` without replacement through `StatsBase.sample`, in ascending order, and return them.
 2. Otherwise, draw positions of `1:N` uniformly and add each new one to a set, until the set holds `M` positions.
 3. Return the positions of the set in ascending order.

# Arguments

  - `rng`: The random number generator of the draw.
  - `N`: The count of the positions.
  - `M`: The count of the positions to draw, with `M <= N`.

# Returns

  - `idx::AbstractVector{<:Integer}`: The `M` positions, in ascending order.

# Related

  - [`make_p_grid`](@ref)
  - [`search_grid_point`](@ref)
"""
function sample_grid_positions(rng::Random.AbstractRNG, N::Integer, M::Integer)
    if N <= typemax(Int)
        return StatsBase.sample(rng, 1:Int(N), Int(M); replace = false, ordered = true)
    end
    idx = Set{BigInt}()
    while length(idx) < M
        push!(idx, rand(rng, big(1):big(N)))
    end
    return sort!(collect(idx))
end
"""
    draw_search_point(rng::Random.AbstractRNG, ps)

Draw one trial of the parameter set `ps`, as a parameter set whose keys each hold a vector of one value.

Each key draws on its own with `rand`. A distribution gives one draw, and a value vector gives one of its entries with equal probability. A multivariate distribution gives one vector.

# Algorithm

 1. For each key of `ps` in its order of iteration, draw one value from its distribution or its value vector with `rand`.
 2. Return the keys, each with a vector of its value.

# Arguments

  - `rng`: The random number generator of the draw.
  - `ps`: One parameter set, a vector of `key => value` pairs or a dictionary from key to value. A value is a vector of candidate values or a `Distributions.Distribution`.

# Returns

  - `point::AbstractVector{<:Pair}`: One `key => [value]` pair per key of `ps`.

# Related

  - [`make_p_grid`](@ref)
  - [`search_grid_point`](@ref)
"""
function draw_search_point(rng::Random.AbstractRNG, ps)
    point = Pair[]
    for (key, vals) in ps
        push!(point, key => [rand(rng, vals)])
    end
    return point
end
"""
    search_grid_point(pss::AbstractVector, sizes::AbstractVector{<:Integer}, i::Integer)

Give point `i` of the concatenated grid of the parameter sets `pss`, as a parameter set whose keys each hold a vector of one value.

The order of the points is the order of [`lens_val_grid`](@ref). The sets come in the order of `pss`, and inside one set the first key varies fastest.

# Algorithm

 1. Subtract the sizes of the sets before point `i` from `i`, giving the set `l` and the position `r` of the point in that set, from `0`.
 2. For each key of set `l` in its order of iteration, take the remainder of `r` divided by the length of its value vector as the offset of the value, and set `r` to the quotient.
 3. Return the keys of set `l`, each with a vector of its value.

# Arguments

  - `pss`: The parameter sets, each a vector of `key => values` pairs or a dictionary from key to values.
  - `sizes`: The count of the grid points of each set, the product of the lengths of its value vectors.
  - `i`: The position of the point in the concatenated grid, from `1` to `sum(sizes)`.

# Returns

  - `point::AbstractVector{<:Pair}`: One `key => [value]` pair per key of set `l`.

# Related

  - [`make_p_grid`](@ref)
  - [`draw_search_point`](@ref)
  - [`sample_grid_positions`](@ref)
  - [`lens_val_grid`](@ref)
"""
function search_grid_point(pss::AbstractVector, sizes::AbstractVector{<:Integer},
                           i::Integer)
    l = 1
    r = i - 1
    while r >= sizes[l]
        r -= sizes[l]
        l += 1
    end
    point = Pair[]
    for (key, vals) in pss[l]
        r, k = divrem(r, length(vals))
        push!(point, key => [vals[firstindex(vals) + Int(k)]])
    end
    return point
end
"""
    search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                            rscv::RandomisedSearchCrossValidation, rd::ReturnsResult)

Score `rscv.n_iter` random candidates of the search space of `rscv` on every fold of `rscv.cv`, and return the winner.

The search draws the trials through [`make_p_grid`](@ref), and runs the search of [`GridSearchCrossValidation`](@ref) on them. Each trial is one candidate. So the count of fits grows with `n_iter`, not with the count of keys. The scheme, the scorer and the treatment of a failed fold are those of the grid search. Under a [`CombinatorialCrossValidation`](@ref), the grid search scores each backtest path.

# Mathematical definition

```math
\\begin{align}
S_{fi} &= s \\, \\mathcal{R}\\left(\\hat{P}_{f}(\\theta_i)\\right)\\,, \\quad f = 1, \\ldots, F\\,, \\quad \\theta_i \\in \\Theta\\,, \\\\
\\mathcal{C} &= \\left\\{ i : S_{fi} \\text{ is finite for every } f \\right\\}\\,, \\\\
i^{\\star} &= c_{\\sigma\\left(\\mathbf{S}_{:,\\,\\mathcal{C}}\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:S_fi_search])
  - $(math_dict[:P_f_search])
  - $(math_dict[:F_folds_search])
  - $(math_dict[:theta_i_cand])
  - $(math_dict[:Theta_trials])
  - $(math_dict[:s_orient_search])
  - $(math_dict[:R_search])
  - $(math_dict[:C_finite_cand])
  - $(math_dict[:S_C_search])
  - $(math_dict[:c_j_search])
  - $(math_dict[:sigma_scorer])
  - $(math_dict[:i_star_cand])

# Algorithm

 1. Resolve the random number generator from `rscv.rng` and `rscv.seed` through [`resolve_rng`](@ref), giving `rng`.
 2. Draw the trials from `rscv.p` through [`make_p_grid`](@ref), with `rscv.n_iter` and `rng`.
 3. Build a [`GridSearchCrossValidation`](@ref) with the trials as its grid and the other fields of `rscv`.
 4. Run [`search_cross_validation`](@ref) of that grid search on `opt` and `rd`, and return its result.

# Arguments

  - `opt`: The estimator to tune. The search reads it as configuration alone.
  - `rscv`: The randomised search. It gives the space `p`, the trial count `n_iter`, the generator `rng` and `seed`, and the fields of a grid search: the scheme `cv`, the risk measure `r`, the `scorer`, the executor `ex`, the `train_score` flag and the keyword arguments `kwargs` of [`expected_risk`](@ref).
  - $(arg_dict[:rd])

# Validation

  - The checks of [`make_p_grid`](@ref) on `rscv.p` and `rscv.n_iter`.
  - The checks of the grid search on `opt` and on each candidate.

# Returns

  - `res::SearchCrossValidationResult`: The result of the grid search of the trials. Its `lens_grid` and `val_grid` hold one entry per trial, in the order of ``\\Theta``, and `idx` is ``i^{\\star}``.

# Related

  - [`RandomisedSearchCrossValidation`](@ref)
  - [`GridSearchCrossValidation`](@ref)
  - [`make_p_grid`](@ref)
  - [`resolve_rng`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`ReturnsResult`](@ref)

# References

  - $(ref_dict[:bergstra2012]) Section 1, Equation 4.
"""
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 rscv::RandomisedSearchCrossValidation, rd::ReturnsResult)
    rng = resolve_rng(rscv.rng, rscv.seed)
    return search_cross_validation(opt,
                                   GridSearchCrossValidation(make_p_grid(rscv.p,
                                                                         rscv.n_iter, rng);
                                                             cv = rscv.cv, r = rscv.r,
                                                             scorer = rscv.scorer,
                                                             ex = rscv.ex,
                                                             train_score = rscv.train_score,
                                                             kwargs = rscv.kwargs), rd)
end

export RandomisedSearchCrossValidation
