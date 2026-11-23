"""
    weighted_high_median(a::AbstractVector, w::AbstractVector{<:Integer}) -> Number

Returns the smallest entry of `a` at which the cumulative weight reaches half of the total weight.

[`kth_pairwise_difference`](@ref) uses the value as its trial value. At least half of the weight is at or below the value, and at least half of the weight is at or above it. So each trial removes at least one quarter of the candidates.

# Algorithm

 1. Sort the positions of `a` by value.
 2. Add the weights in that order. Return the first value at which twice the cumulative weight is at least the total weight.

# Arguments

  - `a`: The values.
  - `w`: The integer weight of each value, of the same length as `a`.

# Returns

  - `m::Number`: The weighted high median, an entry of `a`.

# Related

  - [`kth_pairwise_difference`](@ref)
"""
function weighted_high_median(a::AbstractVector, w::AbstractVector{<:Integer})
    o = sortperm(a)
    c = cumsum(w[o])
    return a[o[findfirst(x -> 2 * x >= last(c), c)]]
end
"""
    count_pairs_below!(P::Vector{Int}, y::VecNum, t::Number, cmp) -> Int

Counts, for each row `i` of the sorted vector `y`, the differences `y[j] - y[i]` with `j > i` for which `cmp(y[j] - y[i], t)` holds.

The differences of one row grow with `j`, so the count of a row is the position of the first difference that fails the test. The difference `y[j] - y[i]` falls when `i` grows, so that position never moves back. One pass over the rows finds every count.

# Arguments

  - `P`: The output vector, of the length of `y`. Entry `i` gets the count of row `i`.
  - `y`: The data, sorted in increasing order.
  - `t`: The trial value.
  - `cmp`: `<` to count the differences below `t`, or `<=` to count the differences at or below `t`.

# Returns

  - `s::Int`: The sum of the counts.

# Related

  - [`kth_pairwise_difference`](@ref)
"""
function count_pairs_below!(P::Vector{Int}, y::VecNum, t::Number, cmp)
    n = length(y)
    jp = 1
    for i in 1:n
        jp = max(jp, i + 1)
        while jp <= n && cmp(y[jp] - y[i], t)
            jp += 1
        end
        P[i] = jp - i - 1
    end
    return sum(P)
end
"""
    kth_pairwise_difference(y::VecNum, k::Integer) -> Number

Returns the `k`-th smallest of the differences `y[j] - y[i]`, `i < j`, of a sorted vector.

The method selects the order statistic in ``O(n \\log n)`` operations and does not form the ``n(n-1)/2`` differences. The differences of each row `i` are sorted in `j`. The method keeps, for each row, the range of the positions `j` that can hold the answer. Each step takes the median candidate of each row, and gives it the weight of the size of its range. The weighted high median of these candidates is the trial value, see [`weighted_high_median`](@ref). The method then counts the differences below the trial value, and the differences at or below it, with [`count_pairs_below!`](@ref). The answer is the trial value, or it is in the ranges below or above it. Each step removes at least one quarter of the candidates. When at most `n` candidates remain, the method selects the answer among them.

[`qn_scale`](@ref) uses it for the order statistic of the scale estimator ``Q_n``.

# Algorithm

 1. Set the range of row `i` to the positions `i + 1` to `n`.
 2. While more than `n` candidates remain, do these steps:
     1. Find the trial value from the median candidates of the rows that are not empty.
     2. Count `sp`, the differences below the trial value, and `sq`, the differences at or below it.
     3. If `k <= sp`, remove from each row the candidates at or above the trial value. If `k > sq`, remove from each row the candidates at or below the trial value. Otherwise, return the trial value.
 3. Count the differences to the left of the ranges. They are below every candidate. Select the remaining rank among the candidates with `partialsort!`.

# Arguments

  - `y`: The data, sorted in increasing order.
  - `k`: The rank, `1 <= k <= n * (n - 1) ÷ 2`.

# Returns

  - `q::Number`: The `k`-th smallest pairwise difference.

# Related

  - [`qn_scale`](@ref)
  - [`weighted_high_median`](@ref)
  - [`count_pairs_below!`](@ref)
"""
function kth_pairwise_difference(y::VecNum, k::Integer)
    n = length(y)
    lo = collect(2:(n + 1))
    hi = fill(n, n)
    P = Vector{Int}(undef, n)
    while sum(i -> max(hi[i] - lo[i] + 1, 0), 1:n) > n
        trial = pairwise_trial_value(y, lo, hi)
        sp = count_pairs_below!(P, y, trial, <)
        if k <= sp
            hi .= min.(hi, (1:n) .+ P)
            continue
        end
        sq = count_pairs_below!(P, y, trial, <=)
        if k <= sq
            return trial
        end
        lo .= max.(lo, (1:n) .+ P .+ 1)
    end
    below = sum(i -> lo[i] - i - 1, 1:n)
    cand = [y[j] - y[i] for i in 1:n for j in lo[i]:hi[i]]
    return partialsort!(cand, k - below)
end
"""
    pairwise_trial_value(y::VecNum, lo::Vector{Int}, hi::Vector{Int}) -> Number

Returns the trial value of one step of [`kth_pairwise_difference`](@ref).

The trial value is the weighted high median of the median candidate of each row that is not empty. The weight of a row is the number of its candidates.

# Arguments

  - `y`: The data, sorted in increasing order.
  - `lo`: The first candidate position of each row.
  - `hi`: The last candidate position of each row.

# Returns

  - `t::Number`: The trial value, one of the pairwise differences of `y`.

# Related

  - [`kth_pairwise_difference`](@ref)
  - [`weighted_high_median`](@ref)
"""
function pairwise_trial_value(y::VecNum, lo::Vector{Int}, hi::Vector{Int})
    rows = findall(i -> lo[i] <= hi[i], eachindex(lo))
    a = [y[(lo[i] + hi[i]) >>> 1] - y[i] for i in rows]
    w = [hi[i] - lo[i] + 1 for i in rows]
    return weighted_high_median(a, w)
end
"""
    qn_finite_factor(n::Integer) -> Number

Returns the finite-sample factor of the scale estimator ``Q_n`` at `n` observations.

[`qn_scale`](@ref) multiplies the order statistic by the asymptotic constant and by this factor, so that the scale is unbiased at the normal model for a small sample. The values are those of the R package `robustbase`, version 0.99-7. For ``n \\leq 12`` they come from a table of simulated factors. For a larger ``n`` the factor is the reciprocal of a fitted polynomial in ``1/n``, with one polynomial for an odd ``n`` and one for an even ``n``.

# Arguments

  - `n`: The number of observations, `n >= 2`.

# Returns

  - `f::Number`: The factor.

# Related

  - [`qn_scale`](@ref)
"""
function qn_finite_factor(n::Integer)
    if n <= 12
        return QN_SMALL_SAMPLE_FACTORS[n - 1]
    end
    d = if isodd(n)
        1.60188 + (-2.1284 - 5.172 / n) / n
    else
        3.67561 + (1.9654 + (6.987 - 77 / n) / n) / n
    end
    return inv(d / n + 1)
end
"""
    QN_SMALL_SAMPLE_FACTORS

The finite-sample factors of the scale estimator ``Q_n`` for 2 to 12 observations, in that order.

The values are those of the R package `robustbase`, version 0.99-7. The factor for two observations is ``\\sqrt{\\pi}/(2 \\cdot 2.21914)``, because ``Q_2`` is the absolute difference of two observations. The other factors come from a simulation.

# Related

  - [`qn_finite_factor`](@ref)
"""
const QN_SMALL_SAMPLE_FACTORS = (0.399356, 0.99365, 0.51321, 0.84401, 0.61220, 0.85877,
                                 0.66993, 0.87344, 0.72014, 0.88906, 0.75743)
"""
    qn_scale(x::VecNum) -> Number

Returns the scale estimator ``Q_n`` of [rousseeuw1993](@cite) of a vector.

``Q_n`` is the ``k``-th order statistic of the absolute pairwise differences of the data, with ``k = \\binom{h}{2}`` and ``h = \\lfloor n/2 \\rfloor + 1``. Its breakdown point is one half, and its efficiency at the normal model is 82 %. It does not need a location estimate. [`DetMCD`](@ref) standardises each column of the data by its median and by this scale.

# Mathematical definition

```math
\\begin{align}
Q_n &= d_n \\, c \\, \\left\\lbrace |x_i - x_j| \\,;\\, i < j \\right\\rbrace_{(k)}\\,.
\\end{align}
```

Where:

  - ``c = 2.21914``: The constant that makes the estimator consistent at the normal model, ``1/(\\sqrt{2}\\,\\Phi^{-1}(5/8))`` to six digits.
  - ``d_n``: The finite-sample factor of [`qn_finite_factor`](@ref).
  - ``\\lbrace \\cdot \\rbrace_{(k)}``: The ``k``-th order statistic.

# Algorithm

 1. Return zero when `x` holds fewer than two values.
 2. Sort `x`, and select the `k`-th pairwise difference with [`kth_pairwise_difference`](@ref).
 3. Multiply the result by the constant and by the factor of [`qn_finite_factor`](@ref).

# Arguments

  - `x`: The data.

# Returns

  - `s::Number`: The scale, of the element type of `x`.

# Related

  - [`kth_pairwise_difference`](@ref)
  - [`qn_finite_factor`](@ref)
  - [`robust_column_scale`](@ref)

# References

  - $(ref_dict[:rousseeuw1993])
"""
function qn_scale(x::VecNum)
    n = length(x)
    if n < 2
        return zero(eltype(x))
    end
    h = div(n, 2) + 1
    q = kth_pairwise_difference(sort(x), div(h * (h - 1), 2))
    return q * convert(typeof(q), 2.21914 * qn_finite_factor(n))
end
"""
    robust_column_scale(z::VecNum) -> Number

Returns the robust scale of a column of the data, with a positive value for a column whose ``Q_n`` is zero.

``Q_n`` is zero when more than half of the values of a column are equal. A division by it then fails. The method follows the rule of the R package `robustbase`, version 0.99-7, for that case. It reads the quantiles of the absolute values of the centred column at the probabilities ``0.5, 0.55, \\ldots, 0.95`` and ``0.9875``. It takes the first quantile ``q_a`` that is not zero, and returns ``q_a / \\Phi^{-1}((a + 1)/2)``, the scale of a normal distribution with that quantile. When every quantile is zero, it returns one.

# Arguments

  - `z`: The column, centred by its median or by a fixed centre.

# Returns

  - `s::Number`: The positive scale.

# Related

  - [`qn_scale`](@ref)
  - [`mcd_standardise`](@ref)
"""
function robust_column_scale(z::VecNum)
    s = qn_scale(z)
    if !iszero(s)
        return s
    end
    a = abs.(z)
    for alpha in ((10:19) ./ 20..., 0.9875)
        qa = Statistics.quantile(a, alpha)
        if !iszero(qa)
            return qa / convert(typeof(qa),
                                Distributions.quantile(Distributions.Normal(), (alpha + 1) / 2))
        end
    end
    return one(s)
end
