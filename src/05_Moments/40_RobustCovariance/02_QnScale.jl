"""
    weighted_high_median!(a::AbstractVector, b::AbstractVector, m::Integer) -> Number

Returns the smallest value of the first `m` pairs of `a` at which the cumulative weight reaches half of the total weight.

Each entry of `a` is a pair of a value and an integer weight. The method selects the value and does not sort the pairs, so it takes ``O(m)`` operations on average. The partition has no branch on the data. It writes each pair at the two ends of the other buffer, and moves only the end that the comparison with the pivot names. A branch on the data costs more than the two writes, because the processor cannot predict it.

[`kth_pairwise_difference`](@ref) uses the value as its trial value. At least half of the weight is at or below the value, and at least half of the weight is at or above it. So each trial removes at least one quarter of the candidates.

# Algorithm

 1. Take all `m` pairs as the range. The target is half of the total weight, rounded up.
 2. While the range holds more than one pair, do these steps:
     1. Take the median of the first, the middle and the last value of the range as the pivot.
     2. Copy the pairs below the pivot to the start of the same range of the other buffer, and the pairs above it to the end. Add the weight of each group.
     3. If the weight below the pivot reaches the target, keep the pairs below it. If the weight at or below the pivot reaches the target, return the pivot. Otherwise, subtract that weight from the target, and keep the pairs above the pivot.
     4. Swap the two buffers.
 3. Return the value that is left.

# Arguments

  - `a`: The pairs of a value and a positive integer weight. The method overwrites it.
  - `b`: A buffer of the same element type and of length `m` or more. The method overwrites it.
  - `m`: The number of pairs, `1 <= m <= length(a)`.

# Returns

  - `t::Number`: The weighted high median, one of the values of `a`.

# Related

  - [`kth_pairwise_difference`](@ref)
"""
function weighted_high_median!(a::AbstractVector, b::AbstractVector, m::Integer)
    l, r = 1, m
    total = sum(last, view(a, 1:m))
    target = cld(total, 2)
    while l < r
        x, y, z = first(a[l]), first(a[(l + r) >>> 1]), first(a[r])
        t = max(min(x, y), min(max(x, y), z))
        lt, gt = l, r
        below = above = 0
        for i in l:r
            b[lt] = b[gt] = a[i]
            v, w = a[i]
            lt += v < t
            gt -= v > t
            below += (v < t) * w
            above += (v > t) * w
        end
        if target <= below
            r, total = lt - 1, below
        elseif target <= total - above
            return t
        else
            target -= total - above
            l, total = gt + 1, above
        end
        a, b = b, a
    end
    return first(a[l])
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

The method selects the order statistic in ``O(n \\log n)`` operations and does not form the ``n(n-1)/2`` differences. The differences of each row `i` are sorted in `j`. The method keeps, for each row, the number of differences that are below its range of candidates, and the number that are at or before the end of that range. These are the counts of earlier steps, so a step that narrows the ranges swaps two vectors. Each step takes the median candidate of each row, and gives it the weight of the size of its range. The weighted high median of these candidates is the trial value, see [`weighted_high_median!`](@ref). The method then counts the differences below the trial value, and the differences at or below it, with [`count_pairs_below!`](@ref). The answer is the trial value, or it is in the ranges below or above it. Each step removes at least one quarter of the candidates. When at most `n` candidates remain, the method selects the answer among them.

[`qn_scale`](@ref) uses it for the order statistic of the scale estimator ``Q_n``.

# Algorithm

 1. Set the candidates of row `i` to all its `n - i` differences.
 2. While more than `n` candidates remain, do these steps:
     1. Find the trial value from the median candidates of the rows that are not empty, with [`pairwise_trial_value!`](@ref).
     2. Count the differences below the trial value. If `k` is at most this count, the count of each row is the new end of its range. Go to the next step of the loop.
     3. Count the differences at or below the trial value. If `k` is at most this count, return the trial value. Otherwise, the count of each row is the new number below its range.
 3. Collect the candidates. Subtract the number of differences below the ranges from `k`, and select the remaining rank with `partialsort!`.

# Arguments

  - `y`: The data, sorted in increasing order.
  - `k`: The rank, `1 <= k <= n * (n - 1) ÷ 2`.

# Returns

  - `q::Number`: The `k`-th smallest pairwise difference.

# Related

  - [`qn_scale`](@ref)
  - [`pairwise_trial_value!`](@ref)
  - [`weighted_high_median!`](@ref)
  - [`count_pairs_below!`](@ref)
"""
function kth_pairwise_difference(y::VecNum, k::Integer)
    n = length(y)
    L = zeros(Int, n)
    R = collect((n - 1):-1:0)
    P = Vector{Int}(undef, n)
    a = Vector{Tuple{eltype(y), Int}}(undef, n)
    b = similar(a)
    sl, sr = 0, div(n * (n - 1), 2)
    while sr - sl > n
        trial = pairwise_trial_value!(a, b, y, L, R)
        s = count_pairs_below!(P, y, trial, <)
        if k <= s
            R, P, sr = P, R, s
            continue
        end
        s = count_pairs_below!(P, y, trial, <=)
        if k <= s
            return trial
        end
        L, P, sl = P, L, s
    end
    cand = similar(y, sr - sl)
    m = 0
    for i in 1:n, c in (L[i] + 1):R[i]
        m += 1
        cand[m] = y[i + c] - y[i]
    end
    return partialsort!(cand, k - sl)
end
"""
    pairwise_trial_value!(a::AbstractVector, b::AbstractVector, y::VecNum, L::Vector{Int}, R::Vector{Int}) -> Number

Returns the trial value of one step of [`kth_pairwise_difference`](@ref).

The trial value is the weighted high median of the median candidate of each row that is not empty. The weight of a row is the number of its candidates.

# Arguments

  - `a`: A buffer of pairs of a value and a weight, of the length of `y`. The method overwrites it.
  - `b`: A second buffer of the same element type and length. The method overwrites it.
  - `y`: The data, sorted in increasing order.
  - `L`: The number of differences of each row that are below its candidates.
  - `R`: The number of differences of each row at or before its last candidate.

# Returns

  - `t::Number`: The trial value, one of the pairwise differences of `y`.

# Related

  - [`kth_pairwise_difference`](@ref)
  - [`weighted_high_median!`](@ref)
"""
function pairwise_trial_value!(a::AbstractVector, b::AbstractVector, y::VecNum,
                               L::Vector{Int}, R::Vector{Int})
    m = 0
    for i in eachindex(y, L, R)
        if L[i] < R[i]
            m += 1
            a[m] = (y[i + (L[i] + R[i] + 1) >>> 1] - y[i], R[i] - L[i])
        end
    end
    return weighted_high_median!(a, b, m)
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

  - `s::Number`: The positive scale, of the element type of `z`.

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
            return oftype(s,
                          qa /
                          Distributions.quantile(Distributions.Normal(), (alpha + 1) / 2))
        end
    end
    return one(s)
end
