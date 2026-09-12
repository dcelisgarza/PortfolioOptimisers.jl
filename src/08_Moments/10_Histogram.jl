"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all histogram binning algorithms.

`AbstractBins` is the abstract type for all binning algorithm types used in histogram-based calculations within `PortfolioOptimisers.jl`, such as mutual information and variation of information analysis. Concrete subtypes implement specific binning strategies (e.g., Knuth, Freedman-Diaconis, Scott, Hacine-Gharbi-Ravier) and provide a consistent interface for bin selection.

A bin count is chosen per **pair** of variables, not per variable, because the measures that read it estimate a joint histogram.

# Interfaces

In order to implement a new binning algorithm which will work seamlessly with the library, subtype `AbstractBins` with all necessary parameters as part of the struct, and implement the following method:

  - `calc_num_bins(bins::AbstractBins, xj::VecNum, xi::VecNum, j::Integer, i::Integer, T::Integer) -> Integer`: The number of histogram bins for the pair `(xj, xi)`.

## Arguments

  - $(arg_dict[:bins])
  - $(arg_dict[:xj])
  - $(arg_dict[:xi])
  - $(arg_dict[:jidx])
  - $(arg_dict[:iidx])
  - $(arg_dict[:Tobs])

## Returns

  - $(ret_dict[:nbins])

# Examples

We can create a dummy binning algorithm as follows:

```jldoctest
julia> struct MyBins <: PortfolioOptimisers.AbstractBins end

julia> function PortfolioOptimisers.calc_num_bins(bins::MyBins, xj::PortfolioOptimisers.VecNum,
                                                  xi::PortfolioOptimisers.VecNum, j::Integer,
                                                  i::Integer, T::Integer)
           return 4
       end

julia> PortfolioOptimisers.calc_num_bins(MyBins(), [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 1, 2, 3)
4
```

# Related

  - [`BinWidthBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`calc_num_bins`](@ref)
"""
abstract type AbstractBins <: AbstractAlgorithm end
"""
    const Int_Bin = Union{<:AbstractBins, <:Integer}

Alias for a histogram binning algorithm or an integer number of bins.

Matches either an [`AbstractBins`](@ref) algorithm (auto-selecting bin counts) or a plain `Integer` (fixed number of bins). Used in histogram-based mutual information and variation of information calculations.

# Related

  - [`AbstractBins`](@ref)
  - [`calc_num_bins`](@ref)
  - [`mutual_info`](@ref)
  - [`variation_info`](@ref)
  - [`mutual_variation_info`](@ref)
"""
const Int_Bin = Union{<:AbstractBins, <:Integer}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all histogram binning algorithms based on a bin width selection rule.

`BinWidthBins` is the abstract type for all binning algorithm types that select the number of bins by first computing an optimal bin width from the data, such as Knuth, Freedman-Diaconis, and Scott. Concrete subtypes implement specific binning strategies and provide a consistent interface for bin selection in histogram-based calculations within `PortfolioOptimisers.jl`.

A subtype states a bin **width** for a single variable. The shared [`calc_num_bins`](@ref) method turns that width into a bin count for the pair, so a subtype implements no `calc_num_bins` method of its own.

# Interfaces

In order to implement a new bin width rule which will work seamlessly with the library, subtype `BinWidthBins` with all necessary parameters as part of the struct, and implement the following method:

  - `bin_width(bins::BinWidthBins, x::VecNum) -> Number`: The optimal histogram bin width for `x`.

## Arguments

  - $(arg_dict[:bins])
  - `x`: Data vector.

## Returns

  - $(ret_dict[:dx])

# Examples

We can create a dummy bin width rule as follows:

```jldoctest
julia> struct MyWidth <: PortfolioOptimisers.BinWidthBins end

julia> function PortfolioOptimisers.bin_width(bins::MyWidth, x::PortfolioOptimisers.VecNum)
           return (maximum(x) - minimum(x)) / 4
       end

julia> PortfolioOptimisers.calc_num_bins(MyWidth(), [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 1, 2, 3)
4
```

# Related

  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`AbstractBins`](@ref)
  - [`bin_width`](@ref)
  - [`calc_num_bins`](@ref)
"""
abstract type BinWidthBins <: AbstractBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using Knuth's rule.

`Knuth` implements Knuth's rule for selecting the optimal number of bins in a histogram [knuth2019](@cite). This method maximises the posterior probability of a piecewise-constant density model given the data, so the binning balances bias against variance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Knuth(;
      args::Tuple = (Optim.NelderMead(),),
      kwargs::NamedTuple = (;)
    ) -> Knuth

# Examples

```jldoctest
julia> Knuth()
Knuth
    args ┼ Tuple{Optim.NelderMead{Optim.AffineSimplexer, Optim.AdaptiveParameters}}: (Optim.NelderMead{Optim.AffineSimplexer, Optim.AdaptiveParameters}(Optim.AffineSimplexer(0.025, 0.5), Optim.AdaptiveParameters(1.0, 1.0, 0.75, 1.0)),)
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`BinWidthBins`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`bin_width`](@ref)

# References

  - $(ref_dict[:knuth2019])
"""
@concrete struct Knuth <: BinWidthBins
    """
    $(field_dict[:optargs])
    """
    args
    """
    $(field_dict[:optkwargs])
    """
    kwargs
    function Knuth(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function Knuth(; args::Tuple = (Optim.NelderMead(),), kwargs::NamedTuple = (;))::Knuth
    return Knuth(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using the Freedman-Diaconis rule.

`FreedmanDiaconis` implements the Freedman-Diaconis rule for selecting the number of bins in a histogram [freedman1981](@cite). This method determines bin width based on the interquartile range (IQR) and the number of data points, making it robust to outliers and suitable for skewed distributions.

# Constructors

    FreedmanDiaconis() -> FreedmanDiaconis

# Examples

```jldoctest
julia> FreedmanDiaconis()
FreedmanDiaconis()
```

# Related

  - [`BinWidthBins`](@ref)
  - [`Knuth`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`bin_width`](@ref)

# References

  - $(ref_dict[:freedman1981])
"""
struct FreedmanDiaconis <: BinWidthBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using Scott's rule.

`Scott` implements Scott's rule for selecting the number of bins in a histogram [scott1979](@cite). This method chooses bin width based on the standard deviation of the data and the number of observations, providing a good default for normally distributed data.

# Constructors

    Scott() -> Scott

# Examples

```jldoctest
julia> Scott()
Scott()
```

# Related

  - [`BinWidthBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`bin_width`](@ref)

# References

  - $(ref_dict[:scott1979])
"""
struct Scott <: BinWidthBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using the Hacine-Gharbi–Ravier rule.

`HacineGharbiRavier` selects the bin count from the sample size and the Pearson correlation of the pair, minimising the mean square error of the joint entropy estimate. It is the default for the information-theoretic measures, [`mutual_info`](@ref) and [`variation_info`](@ref).

# Mathematical definition

Two closed forms, selected by the pair's Pearson correlation ``\\rho``. For ``\\rho^2 \\neq 1`` the bi-histogram formula applies:

```math
\\begin{align}
M &= \\left[\\frac{1}{\\sqrt{2}} \\sqrt{1 + \\sqrt{1 + \\frac{24 T}{1 - \\rho^2}}}\\right]\\,.
\\end{align}
```

The form is singular at ``\\rho^2 = 1``, at both ends of the correlation range. There the pair is deterministic and carries no joint information beyond one marginal, so the univariate formula applies, which is the limit of the bi-histogram one:

```math
\\begin{align}
z &= \\sqrt[3]{8 + 324 T + 12 \\sqrt{36 T + 729 T^2}}\\,, \\\\
M &= \\left[\\frac{z}{6} + \\frac{2}{3 z} + \\frac{1}{3}\\right]\\,.
\\end{align}
```

Where:

  - ``M``: Number of bins.
  - $(math_dict[:T])
  - ``\\rho``: Pearson correlation between the two series.
  - ``[\\cdot]``: Rounding to the nearest integer.

# Constructors

    HacineGharbiRavier() -> HacineGharbiRavier

# Examples

```jldoctest
julia> HacineGharbiRavier()
HacineGharbiRavier()
```

# Related

  - [`AbstractBins`](@ref)
  - [`BinWidthBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`calc_num_bins`](@ref)

# References

  - $(ref_dict[:hacinegharbi2012])
  - $(ref_dict[:hacinegharbi2018])
  - $(ref_dict[:mlp1]) Chapter 3.
"""
struct HacineGharbiRavier <: AbstractBins end
"""
    bin_width(::Scott, x::VecNum)

Compute the optimal histogram bin width for `x` using Scott's rule [scott1979](@cite).

# Mathematical definition

```math
\\begin{align}
\\Delta_x &= \\sigma_x \\left(\\frac{24 \\sqrt{\\pi}}{n}\\right)^{1/3}\\,.
\\end{align}
```

Where:

  - ``\\Delta_x``: Bin width.
  - ``\\sigma_x``: Uncorrected standard deviation of the data.
  - ``n``: Number of observations.

# Arguments

  - `x`: Data vector.

# Returns

  - $(ret_dict[:dx])

# Related

  - [`Scott`](@ref)
  - [`BinWidthBins`](@ref)
  - [`calc_num_bins`](@ref)

# References

  - $(ref_dict[:scott1979])
"""
function bin_width(::Scott, x::VecNum)
    return Statistics.std(x; corrected = false) * cbrt(24 * sqrt(pi) / length(x))
end
"""
    bin_width(::FreedmanDiaconis, x::VecNum)

Compute the optimal histogram bin width for `x` using the Freedman-Diaconis rule [freedman1981](@cite).

# Mathematical definition

```math
\\begin{align}
\\Delta_x &= \\frac{2 \\, \\mathrm{IQR}(x)}{n^{1/3}}\\,.
\\end{align}
```

Where:

  - ``\\Delta_x``: Bin width.
  - ``\\mathrm{IQR}(x)``: Interquartile range of the data.
  - ``n``: Number of observations.

# Arguments

  - `x`: Data vector.

# Returns

  - $(ret_dict[:dx])

# Related

  - [`FreedmanDiaconis`](@ref)
  - [`BinWidthBins`](@ref)
  - [`calc_num_bins`](@ref)

# References

  - $(ref_dict[:freedman1981])
"""
function bin_width(::FreedmanDiaconis, x::VecNum)
    q25, q75 = Statistics.quantile(x, [0.25, 0.75])
    return 2 * (q75 - q25) / cbrt(length(x))
end
"""
    bin_width(bins::Knuth, x::VecNum)

Compute the optimal histogram bin width for `x` using Knuth's rule [knuth2019](@cite).

# Mathematical definition

The bin width is the range of `x` divided by the bin count ``M`` that maximises the marginal posterior probability of a piecewise-constant density model with ``M`` equal-width bins over that range:

```math
\\begin{align}
F(M) &= n \\log M + \\log\\Gamma\\!\\left(\\frac{M}{2}\\right) - M \\log\\Gamma\\!\\left(\\frac{1}{2}\\right) - \\log\\Gamma\\!\\left(n + \\frac{M}{2}\\right) + \\sum_{k=1}^{M} \\log\\Gamma\\!\\left(n_k + \\frac{1}{2}\\right)\\,.
\\end{align}
```

Where:

  - ``M``: Number of bins.
  - ``n``: Number of observations.
  - ``n_k``: Number of observations in bin ``k``.

The maximiser is an integer, so no closed form gives it. The method searches for it.

# Algorithm

 1. Read the range of `x` into `rx`, the difference of its two extrema.
 2. Build the objective `f`, which takes a one-element vector `Ms`, floors its entry into the bin count `M`, and returns `Inf` when `M` is not positive.
 3. Inside `f`, bin the data into the counts `nk` over `M` equal-width bins of the range, and return the negated posterior of the mathematical definition. The optimiser minimises, so the sign is flipped.
 4. Take the starting point `M0` from the bin count that the Freedman-Diaconis rule implies for `x`, plus one.
 5. Minimise `f` from `M0` with `Optim.optimize`, passing `bins.args` and `bins.kwargs`. The default `args` is a Nelder-Mead simplex.
 6. Floor the minimiser into a bin count, and return the range divided by it.

# Arguments

  - $(arg_dict[:bins])
  - `x`: Data vector.

# Returns

  - $(ret_dict[:dx])

# Related

  - [`Knuth`](@ref)
  - [`BinWidthBins`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`calc_num_bins`](@ref)

# References

  - $(ref_dict[:knuth2019])
"""
function bin_width(bins::Knuth, x::VecNum)
    n = length(x)
    xl, xu = extrema(x)
    rx = xu - xl
    lg_half = SpecialFunctions.loggamma(0.5)
    nk = Vector{Int}(undef, 0)
    function f(Ms)
        M = floor(Int, first(Ms))
        if M <= 0
            return Inf
        end
        resize!(nk, M)
        fill!(nk, 0)
        for xi in x
            k = min(floor(Int, (xi - xl) / rx * M) + 1, M)
            nk[k] += 1
        end
        return -(n * log(M) + SpecialFunctions.loggamma(M / 2) - M * lg_half -
                 SpecialFunctions.loggamma(n + M / 2) +
                 sum(SpecialFunctions.loggamma, nk .+ 0.5))
    end
    M0 = max(1.0, rx / bin_width(FreedmanDiaconis(), x)) + 1
    res = Optim.optimize(f, [M0], bins.args...; bins.kwargs...)
    return rx / floor(Int, first(Optim.minimizer(res)))
end
"""
    calc_num_bins(bins::BinWidthBins, xj::VecNum, xi::VecNum, j::Integer, i::Integer,
                  args...)
    calc_num_bins(bins::HacineGharbiRavier, xj::VecNum, xi::VecNum, j::Integer,
                  i::Integer, T::Integer)
    calc_num_bins(bins::Integer, args...)

Compute the number of histogram bins for a pair of variables using a specified binning algorithm.

This function determines the number of bins to use for histogram-based calculations (such as mutual information or variation of information) between two variables, based on the selected binning strategy. It dispatches on the binning algorithm type, and the three methods read different arguments: only the [`HacineGharbiRavier`](@ref) method reads `T`, and the `Integer` method reads none of them.

# Algorithm

The [`BinWidthBins`](@ref) method turns a bin width into a bin count.

 1. Read the range of `xj` into `xju - xjl`, and divide it by [`bin_width`](@ref) of `xj`, giving `k1`.
 2. When `j` and `i` differ, repeat step 1 for `xi`, giving `k2`, and select the larger of `k1` and `k2`. The joint histogram is square, so one count serves both axes, and the larger of the two keeps the finer resolution. When `j` and `i` are equal, the pair is a variable against itself, so select `k1` and read `xi` no further.
 3. Round the selected value to the nearest integer, and return it.

The [`HacineGharbiRavier`](@ref) method reads the pair instead of a width.

 1. Take the Pearson correlation of the pair into `corr`.
 2. Select the closed form of [`HacineGharbiRavier`](@ref) that `corr` falls under.
 3. Round the selected value to the nearest integer, and return it.

The `Integer` method returns `bins` unchanged.

# Arguments

  - $(arg_dict[:bins])
  - $(arg_dict[:xj])
  - $(arg_dict[:xi])
  - $(arg_dict[:jidx])
  - $(arg_dict[:iidx])
  - $(arg_dict[:Tobs])
  - `args...`: Ignored arguments, so that the three methods share one call site.

# Returns

  - $(ret_dict[:nbins])

# Related

  - [`AbstractBins`](@ref)
  - [`BinWidthBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`bin_width`](@ref)
"""
function calc_num_bins(bins::BinWidthBins, xj::VecNum, xi::VecNum, j::Integer, i::Integer,
                       args...)
    xjl, xju = extrema(xj)
    k1 = (xju - xjl) / bin_width(bins, xj)
    return round(Int, if j != i
                     xil, xiu = extrema(xi)
                     k2 = (xiu - xil) / bin_width(bins, xi)
                     max(k1, k2)
                 else
                     k1
                 end)
end
function calc_num_bins(::HacineGharbiRavier, xj::VecNum, xi::VecNum, j::Integer, i::Integer,
                       T::Integer)
    corr = Statistics.cor(xj, xi)
    #=
    The bi-histogram formula divides by `1 - corr^2`, so it is singular at BOTH ends of the
    correlation range. `isone(abs(corr))` catches them together: a perfectly anti-correlated
    pair is as deterministic as a perfectly correlated one, and the univariate formula is the
    limit of the bi-histogram one in either case. Testing `isone(corr)` alone sent `corr = -1`
    into the singular branch, where `24T/0` is `Inf` and `round(Int, Inf)` raises an
    `InexactError`.
    =#
    return round(Int, if isone(abs(corr))
                     z = cbrt(8 + 324 * T + 12 * sqrt(36 * T + 729 * T^2))
                     z / 6 + 2 / (3 * z) + 1 / 3
                 else
                     sqrt(1 + sqrt(1 + 24 * T / (1 - corr^2))) / sqrt(2)
                 end)
end
function calc_num_bins(bins::Integer, args...)
    return bins
end
"""
    calc_hist_data(xj::VecNum, xi::VecNum, bins::Integer)

Compute histogram-based marginal and joint distributions for two variables.

This function computes the normalised histograms (probability mass functions) for two variables `xj` and `xi` using the specified number of bins, as well as their joint histogram. It returns the marginal entropies and the joint histogram, which are used in mutual information and variation of information calculations.

A bin is closed on the left, so the lower edge is the minimum itself and needs no widening. The upper edge is exclusive, so it is `nextfloat` of the maximum, one unit in the last place above it at every magnitude. The largest observation therefore falls strictly inside the last bin whatever the magnitude of the data, and a constant column gives an entropy of zero rather than `NaN`.

# Algorithm

 1. Add one to `bins`, giving `bp1`, the number of bin edges.
 2. Build the edge range of `xj` from `minimum(xj)` to `nextfloat(maximum(xj))`, with `bp1` points. Repeat for `xi`.
 3. Bin `xj` over its own edges, giving the counts `hx`, and divide `hx` by its own sum to make it a probability mass function.
 4. Repeat step 3 for `xi`, giving `hy`.
 5. Take the Shannon entropy of `hx` into `ex`, and of `hy` into `ey`.
 6. Bin the pair over both edge ranges, giving the joint counts `hxy`. It is left unnormalised, because [`intrinsic_mutual_info`](@ref) normalises it itself.

# Arguments

  - $(arg_dict[:xj])
  - $(arg_dict[:xi])
  - `bins`: Number of bins to use for the histograms.

# Returns

  - `ex::Number`: Shannon entropy of the marginal probability mass function of `xj`, in nats.
  - `ey::Number`: Shannon entropy of the marginal probability mass function of `xi`, in nats.
  - `hxy::Matrix{<:Number}`: Joint histogram (counts, not normalised to probability).

# Related

  - [`variation_info`](@ref)
  - [`mutual_info`](@ref)
  - [`mutual_variation_info`](@ref)
  - [`intrinsic_mutual_info`](@ref)
  - [`calc_num_bins`](@ref)
"""
function calc_hist_data(xj::VecNum, xi::VecNum, bins::Integer)
    bp1 = bins + one(bins)

    xjl = minimum(xj)
    xjh = nextfloat(maximum(xj))

    xil = minimum(xi)
    xih = nextfloat(maximum(xi))

    hx = StatsAPI.fit(StatsBase.Histogram, xj, range(xjl, xjh; length = bp1)).weights
    hx /= sum(hx)

    hy = StatsAPI.fit(StatsBase.Histogram, xi, range(xil, xih; length = bp1)).weights
    hy /= sum(hy)

    ex = StatsBase.entropy(hx)
    ey = StatsBase.entropy(hy)

    hxy = StatsAPI.fit(StatsBase.Histogram, (xj, xi),
                       (range(xjl, xjh; length = bp1), range(xil, xih; length = bp1))).weights

    return ex, ey, hxy
end
"""
    intrinsic_mutual_info(X::MatNum)

Compute the intrinsic mutual information from a joint histogram.

This function computes the mutual information between two variables given their joint histogram matrix `X`. It is used as a core step in information-theoretic measures such as mutual information and variation of information.

**Intrinsic** names the estimate itself: the quantity the joint histogram carries, before [`mutual_info`](@ref) divides it by a marginal entropy and before either matrix-valued measure clamps it. It is therefore in nats, it is not bounded above by one, and it is the value that [`mutual_info`](@ref) returns when `normalise` is `false`. It is **not** the intrinsic conditional information of secret-key agreement, which is a different quantity under the same word.

# Mathematical definition

Given the joint histogram ``\\mathbf{X}`` (unnormalised counts), with marginals ``p_i = \\sum_j X_{ij} / n`` and ``p_j = \\sum_i X_{ij} / n``:

```math
\\begin{align}
\\hat{I}(X; Y) &= \\sum_{i,j:\\, X_{ij} > 0} \\frac{X_{ij}}{n} \\log\\!\\left(\\frac{X_{ij} / n}{p_i \\, p_j}\\right)\\,.
\\end{align}
```

Where:

  - ``\\hat{I}(X; Y)``: Estimated mutual information between ``X`` and ``Y``.
  - ``X_{ij}``: Joint histogram count at bin ``(i, j)``.
  - ``n = \\sum_{i,j} X_{ij}``: Total count.
  - ``p_i = \\sum_j X_{ij} / n``, ``p_j = \\sum_i X_{ij} / n``: Marginal probabilities.

A bin the pair never visits contributes nothing, because ``p \\log p`` tends to zero as ``p`` does. The sum therefore runs over the non-empty bins alone. When either axis has a single bin the two variables are indistinguishable under the binning, and the estimate is zero.

# Algorithm

 1. Sum `X` over its columns into `p_i`, and over its rows into `p_j`. Both are unnormalised marginal counts.
 2. When either marginal has length one, return zero.
 3. Find the indices of the non-zero entries of `X` into `mask`, and read those entries into the vector `nz`.
 4. Sum `nz` into `nz_sum`, the total count, and divide `nz` by it, giving the joint probabilities `nz_nm`.
 5. Take the outer product of the two marginal counts at the masked indices into `outer`, and turn it into the logarithm of the product of the marginal probabilities, `log_outer`, by subtracting the logarithm of each marginal total.
 6. Form the per-bin contributions `mi` from `nz_nm` and the two logarithms, and set to zero every contribution whose magnitude is below the machine epsilon.
 7. Return the sum of `mi`.

# Arguments

  - `X`: Joint histogram matrix.

# Returns

  - `mi::Number`: The intrinsic mutual information between the two variables, in nats.

# Related

  - [`calc_hist_data`](@ref)
  - [`variation_info`](@ref)
  - [`mutual_info`](@ref)
  - [`mutual_variation_info`](@ref)

# References

  - $(ref_dict[:shannon1948])
"""
function intrinsic_mutual_info(X::MatNum)
    p_i = vec(sum(X; dims = 2))
    p_j = vec(sum(X; dims = 1))

    if length(p_i) == 1 || length(p_j) == 1
        return zero(eltype(p_j))
    end

    mask = findall(.!iszero.(X))

    nz = vec(X[mask])
    nz_sum = sum(nz)
    log_nz = log.(nz)
    nz_nm = nz / nz_sum

    outer = p_i[getindex.(mask, 1)] ⊙ p_j[getindex.(mask, 2)]
    log_outer = -log.(outer) .+ (log(sum(p_i)) + log(sum(p_j)))

    mi = (nz_nm ⊙ (log_nz .- log(nz_sum)) + nz_nm ⊙ log_outer)
    mi[abs.(mi) .< eps(eltype(mi))] .= zero(eltype(X))

    return sum(mi)
end
"""
    variation_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                   normalise::Bool = true)

Compute the variation of information (VI) matrix for a set of variables.

This function computes the pairwise variation of information between all columns of the data matrix `X`, using histogram-based entropy and mutual information estimates. VI quantifies the amount of information lost and gained when moving from one variable to another, and is a true metric on the space of discrete distributions: it is non-negative, it is zero exactly when the two variables agree, it is symmetric, and it satisfies the triangle inequality. [`mutual_info`](@ref) is the complementary measure and is **not** a metric — it grows with agreement rather than with disagreement, and its diagonal is the entropy rather than zero.

# Mathematical definition

Let ``H(X)``, ``H(Y)`` denote the marginal Shannon entropies and ``I(X;Y)`` the mutual information. The variation of information is:

```math
\\begin{align}
\\mathrm{VI}(X, Y) &= H(X) + H(Y) - 2\\,I(X;Y)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VI}(X, Y)``: Variation of information between ``X`` and ``Y``.
  - ``H(X)``, ``H(Y)``: Marginal Shannon entropies.
  - ``I(X;Y)``: Mutual information.

When `normalise = true`, it is divided by the joint entropy ``H(X,Y) = H(X) + H(Y) - I(X;Y)``:

```math
\\begin{align}
\\widetilde{\\mathrm{VI}}(X, Y) &= \\frac{H(X) + H(Y) - 2\\,I(X;Y)}{H(X) + H(Y) - I(X;Y)}\\,.
\\end{align}
```

Where:

  - ``\\widetilde{\\mathrm{VI}}(X, Y)``: Normalised variation of information.
  - ``H(X,Y) = H(X) + H(Y) - I(X;Y)``: Joint entropy.

The divisor is the joint entropy, which keeps the normalised form a metric on ``[0, 1]``. Equation 6.25 of the source divides by ``\\max(H(X), H(Y))`` instead, which is bounded by the same interval but is not a metric.

# Algorithm

 1. Read the shape of `X` into `T`, the number of observations, and `N`, the number of variables.
 2. Allocate the `N × N` result `var_mtx`.
 3. For each variable `j`, write an exact zero at `var_mtx[j, j]`. ``\\mathrm{VI}(X, X)`` is zero by definition, and the histogram estimate of ``I(X; X)`` does not reproduce the estimate of ``H(X)`` bit for bit, so estimating the self pair leaves roughly `1e-16` on the diagonal. That is enough to stop the result being a distance matrix, and skipping the self pair also saves `N` histogram computations.
 4. For each pair `(j, i)` below the diagonal, take the bin count from [`calc_num_bins`](@ref), giving `nbins`.
 5. Take the two marginal entropies `ex` and `ey` and the joint histogram `hxy` from [`calc_hist_data`](@ref).
 6. Take the mutual information of `hxy` from [`intrinsic_mutual_info`](@ref), giving `mut_ixy`, and apply the definition above, giving `var_ixy`.
 7. When `normalise` is true, divide `var_ixy` by the joint entropy `vxy`.
 8. Clamp `var_ixy` below at zero, and write it into both `var_mtx[j, i]` and `var_mtx[i, j]`.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:bins])
  - $(arg_dict[:normalise])

# Returns

  - `var_mtx::Matrix{<:Number}`: Symmetric matrix of pairwise variation of information values, with an exactly zero diagonal. In nats when `normalise` is `false`, and dimensionless on `[0, 1]` when it is `true`.

# Related

  - [`mutual_info`](@ref)
  - [`mutual_variation_info`](@ref)
  - [`calc_hist_data`](@ref)
  - [`calc_num_bins`](@ref)
  - [`intrinsic_mutual_info`](@ref)
  - [`Int_Bin`](@ref)

# References

  - $(ref_dict[:shannon1948])
  - $(ref_dict[:cajas2025]) Section 6.2.2, equation 6.24.
"""
function variation_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                        normalise::Bool = true)
    T, N = size(X)
    var_mtx = Matrix{eltype(X)}(undef, N, N)
    for j in axes(X, 2)
        xj = view(X, :, j)
        #=
        VI(X, X) = H(X) + H(X) - 2*I(X; X) = 0 exactly, by definition, so the self pair is
        pinned rather than estimated. Estimating it leaves roughly 1e-16 on the diagonal --
        measured on 7 of 12 assets -- because the histogram estimate of I(X; X) does not
        reproduce the estimate of H(X) bit for bit. A distance matrix with a non-zero
        diagonal is not a distance matrix: it fails `PhylogenyResult`'s own zero-diagonal
        check, and `SimpleWeightedGraph` reads it as a self-loop. Skipping the self pair
        also saves N histogram computations.
        =#
        var_mtx[j, j] = zero(eltype(X))
        for i in 1:(j - 1)
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = intrinsic_mutual_info(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
            end
            var_ixy = clamp(var_ixy, zero(eltype(X)), typemax(eltype(X)))
            var_mtx[j, i] = var_mtx[i, j] = var_ixy
        end
    end
    return var_mtx
end
"""
    mutual_variation_info(X::MatNum, bins::Int_Bin = Knuth(), normalise::Bool = true)

Compute the pairwise mutual information and variation of information matrices from a data matrix.

Both matrices come from one pass over the pairs, so the two share a bin count and a joint histogram per pair. The result is what [`mutual_info`](@ref) and [`variation_info`](@ref) return, and the two normalisers differ: mutual information is divided by the smaller marginal entropy, variation of information by the joint entropy. The default `bins` is [`Knuth`](@ref) here and [`HacineGharbiRavier`](@ref) in the other two, so the three agree only when `bins` is given.

# Mathematical definition

```math
\\begin{align}
I(X_i; X_j) &= H(X_i) + H(X_j) - H(X_i, X_j)\\,, \\\\
\\mathrm{VI}(X_i, X_j) &= H(X_i) + H(X_j) - 2\\,I(X_i; X_j)\\,.
\\end{align}
```

When `normalise = true`, each is divided by its own normaliser:

```math
\\begin{align}
\\tilde{I}(X_i; X_j) &= \\frac{I(X_i; X_j)}{\\min\\bigl(H(X_i),\\, H(X_j)\\bigr)}\\,, \\\\
\\widetilde{\\mathrm{VI}}(X_i, X_j) &= \\frac{\\mathrm{VI}(X_i, X_j)}{H(X_i) + H(X_j) - I(X_i; X_j)}\\,.
\\end{align}
```

Where:

  - ``I(X_i; X_j)``: Mutual information between assets ``i`` and ``j``.
  - ``\\mathrm{VI}(X_i, X_j)``: Variation of information between assets ``i`` and ``j``.
  - ``H(X_i)``, ``H(X_j)``: Marginal Shannon entropies.
  - ``H(X_i, X_j) = H(X_i) + H(X_j) - I(X_i; X_j)``: Joint entropy.
  - ``\\tilde{I}(X_i; X_j)``: Normalised mutual information.
  - ``\\widetilde{\\mathrm{VI}}(X_i, X_j)``: Normalised variation of information.

# Algorithm

 1. Read the shape of `X` into `T`, the number of observations, and `N`, the number of variables.
 2. Allocate the two `N × N` results `mut_mtx` and `var_mtx`.
 3. For each pair `(j, i)` on or below the diagonal, take the bin count from [`calc_num_bins`](@ref), giving `nbins`.
 4. Take the two marginal entropies `ex` and `ey` and the joint histogram `hxy` from [`calc_hist_data`](@ref).
 5. Take the mutual information of `hxy` from [`intrinsic_mutual_info`](@ref), giving `mut_ixy`, and form `var_ixy` from it and the two entropies.
 6. When `normalise` is true, divide `var_ixy` by the joint entropy `vxy` and `mut_ixy` by the smaller of `ex` and `ey`.
 7. Clamp both below at zero, and write each into both of its symmetric positions.
 8. After the inner loop, write an exact zero at `var_mtx[j, j]`, for the reason [`variation_info`](@ref) gives. `mut_mtx[j, j]` keeps its estimate, because ``I(X; X) = H(X)`` is a real value there rather than a zero.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:bins])
  - $(arg_dict[:normalise])

# Returns

  - `mut_mtx::Matrix{<:Number}`: Symmetric matrix of pairwise mutual information values, whose diagonal is the marginal entropy when `normalise` is `false` and one when it is `true`.
  - `var_mtx::Matrix{<:Number}`: Symmetric matrix of pairwise variation of information values, with an exactly zero diagonal.

# Related

  - [`Int_Bin`](@ref)
  - [`AbstractBins`](@ref)
  - [`mutual_info`](@ref)
  - [`variation_info`](@ref)
  - [`calc_hist_data`](@ref)
  - [`calc_num_bins`](@ref)
  - [`intrinsic_mutual_info`](@ref)

# References

  - $(ref_dict[:shannon1948])
  - $(ref_dict[:cajas2025]) Sections 6.1.6 and 6.2.2, equations 6.18, 6.19 and 6.24.
"""
function mutual_variation_info(X::MatNum, bins::Int_Bin = Knuth(), normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    var_mtx = Matrix{eltype(X)}(undef, N, N)
    for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = intrinsic_mutual_info(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
                mut_ixy /= min(ex, ey)
            end

            # if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < zero(eltype(X))
            #     mut_ixy = zero(eltype(X))
            # end
            # if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < zero(eltype(X))
            #     var_ixy = zero(eltype(X))
            # end

            mut_ixy = clamp(mut_ixy, zero(eltype(X)), typemax(eltype(X)))
            var_ixy = clamp(var_ixy, zero(eltype(X)), typemax(eltype(X)))

            mut_mtx[j, i] = mut_mtx[i, j] = mut_ixy
            var_mtx[j, i] = var_mtx[i, j] = var_ixy
        end
        # As in `variation_info`: VI(X, X) is zero by definition, and the estimate only
        # approximates it. `mut_mtx`'s diagonal is left alone -- I(X; X) = H(X) is a real
        # value there, unlike VI's zero.
        var_mtx[j, j] = zero(eltype(X))
    end

    return mut_mtx, var_mtx
end
"""
    mutual_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                normalise::Bool = true)

Compute the mutual information (MI) matrix for a set of variables.

This function computes the pairwise mutual information between all columns of the data matrix `X`, using histogram-based entropy and mutual information estimates. MI quantifies the amount of shared information between pairs of variables, and is widely used in information-theoretic analysis of dependencies.

Mutual information is a measure of agreement and is **not** a metric: it grows with dependence rather than with distance, and its diagonal carries the marginal entropy rather than zero. [`variation_info`](@ref) is the metric of the pair. The diagonal is estimated here, not pinned, so it is the marginal entropy in nats when `normalise` is `false` and one when it is `true`.

# Mathematical definition

Mutual information between assets ``i`` and ``j``:

```math
\\begin{align}
I(X_i; X_j) &= H(X_i) + H(X_j) - H(X_i, X_j) = \\sum_{x,y} p(x,y) \\log\\frac{p(x,y)}{p(x)\\,p(y)}\\,.
\\end{align}
```

Where:

  - ``I(X_i; X_j)``: Mutual information between assets ``i`` and ``j``.
  - ``H(X_i)``, ``H(X_j)``: Marginal Shannon entropies.
  - ``H(X_i, X_j)``: Joint entropy.
  - ``p(x,y)``: Joint probability mass function.

When `normalise = true`, the MI is normalised by the minimum marginal entropy:

```math
\\begin{align}
\\tilde{I}(X_i; X_j) &= \\frac{I(X_i; X_j)}{\\min\\bigl(H(X_i),\\, H(X_j)\\bigr)}\\,.
\\end{align}
```

Where:

  - ``\\tilde{I}(X_i; X_j)``: Normalised mutual information.

The smaller marginal entropy is the largest value the mutual information of the pair can take, so the normalised form is bounded by ``[0, 1]`` and reaches one exactly when one variable determines the other.

# Algorithm

 1. Read the shape of `X` into `T`, the number of observations, and `N`, the number of variables.
 2. Allocate the `N × N` result `mut_mtx`.
 3. For each pair `(j, i)` on or below the diagonal, take the bin count from [`calc_num_bins`](@ref), giving `nbins`.
 4. Take the two marginal entropies `ex` and `ey` and the joint histogram `hxy` from [`calc_hist_data`](@ref).
 5. Take the mutual information of `hxy` from [`intrinsic_mutual_info`](@ref), giving `mut_ixy`.
 6. When `normalise` is true, divide `mut_ixy` by the smaller of `ex` and `ey`.
 7. Clamp `mut_ixy` below at zero, and write it into both `mut_mtx[j, i]` and `mut_mtx[i, j]`.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:bins])
  - $(arg_dict[:normalise])

# Returns

  - `mut_mtx::Matrix{<:Number}`: Symmetric matrix of pairwise mutual information values. In nats when `normalise` is `false`, and dimensionless on `[0, 1]` when it is `true`.

# Related

  - [`variation_info`](@ref)
  - [`mutual_variation_info`](@ref)
  - [`calc_hist_data`](@ref)
  - [`calc_num_bins`](@ref)
  - [`intrinsic_mutual_info`](@ref)
  - [`Int_Bin`](@ref)
  - [`MutualInfoCovariance`](@ref)

# References

  - $(ref_dict[:shannon1948])
  - $(ref_dict[:cajas2025]) Section 6.1.6, equations 6.18 and 6.19.
"""
function mutual_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                     normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)
            mut_ixy = intrinsic_mutual_info(hxy)
            if normalise
                mut_ixy /= min(ex, ey)
            end
            mut_ixy = clamp(mut_ixy, zero(eltype(X)), typemax(eltype(X)))
            mut_mtx[j, i] = mut_mtx[i, j] = mut_ixy
        end
    end
    return mut_mtx
end

export Knuth, FreedmanDiaconis, Scott, HacineGharbiRavier
