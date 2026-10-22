"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all histogram binning algorithms.

The histogram estimates of the mutual information and the variation of information get their bin count from a subtype of `AbstractBins`. The library has four of them: [`Knuth`](@ref), [`FreedmanDiaconis`](@ref), [`Scott`](@ref) and [`HacineGharbiRavier`](@ref). An `Integer` in place of an algorithm fixes the count, as [`Int_Bin`](@ref) states.

An algorithm selects one bin count for each pair of variables, not for each variable, because the measures that read the count estimate a joint histogram.

# Interfaces

To add a binning algorithm, subtype `AbstractBins`, keep its parameters in the fields of the struct, and add this method:

  - `calc_num_bins(bins::AbstractBins, xj::VecNum, xi::VecNum, j::Integer, i::Integer, T::Integer) -> Integer`: Returns the number of histogram bins for the pair `(xj, xi)`.

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

The example below defines a minimal binning algorithm that always selects four bins.

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

Matches an [`AbstractBins`](@ref) algorithm, which selects the bin count from the data, or an `Integer`, which fixes the bin count. The `bins` argument of the histogram estimates of the mutual information and the variation of information is an `Int_Bin`.

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

A subtype of `BinWidthBins` computes a bin width from the data, and the bin count follows from that width. The library has three of them: [`Knuth`](@ref), [`FreedmanDiaconis`](@ref) and [`Scott`](@ref).

A subtype computes the bin width of one variable. The shared [`calc_num_bins`](@ref) method turns that width into a bin count for the pair, so a subtype has no `calc_num_bins` method of its own.

# Interfaces

To add a bin width rule, subtype `BinWidthBins`, keep its parameters in the fields of the struct, and add this method:

  - `bin_width(bins::BinWidthBins, x::VecNum) -> Number`: Returns the histogram bin width of `x` under the rule.

## Arguments

  - $(arg_dict[:bins])
  - `x`: Data vector.

## Returns

  - $(ret_dict[:dx])

# Examples

The example below defines a minimal bin width rule that divides the range of the data into four bins.

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

`Knuth` selects the number of bins that maximises the posterior probability of a piecewise-constant density model, given the data [knuth2019](@cite). [`bin_width`](@ref) states the posterior and the search for its maximiser.

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

  - $(ref_dict[:knuth2019]) Section 2.3, Equations 31 and 32.
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

`FreedmanDiaconis` selects the bin width from the interquartile range of the data and the number of observations [freedman1981](@cite). The interquartile range reads the middle half of the sorted data, so a change to the value of the largest or the smallest observation does not change the width. The standard deviation that [`Scott`](@ref) reads does change.

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

  - $(ref_dict[:freedman1981]) Section 1, Rule 1.8.
"""
struct FreedmanDiaconis <: BinWidthBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using Scott's rule.

`Scott` selects the bin width from the standard deviation of the data and the number of observations [scott1979](@cite). The width minimises the integrated squared error of the histogram when the data are normal, as [`bin_width`](@ref) states.

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

  - $(ref_dict[:scott1979]) Section 5, Equation 6.
"""
struct Scott <: BinWidthBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using the Hacine-Gharbi-Ravier rule.

`HacineGharbiRavier` selects the bin count from the sample size and the Pearson correlation of the pair. The count minimises the mean square error of the estimate of the joint entropy. It is the default `bins` of [`mutual_info`](@ref), [`variation_info`](@ref) and [`MutualInfoCovariance`](@ref).

# Mathematical definition

The Pearson correlation ``\\rho`` of the pair selects one of two closed forms. For ``\\rho^2 \\neq 1`` the bi-histogram formula applies:

```math
\\begin{align}
M &= \\left[\\frac{1}{\\sqrt{2}} \\sqrt{1 + \\sqrt{1 + \\frac{24 T}{1 - \\rho^2}}}\\right]\\,.
\\end{align}
```

The form is singular at ``\\rho^2 = 1``, which is both ends of the correlation range, and it grows without bound as ``\\rho^2`` approaches 1. At ``\\rho^2 = 1`` one variable is a linear function of the other, so the joint histogram holds no information that one marginal does not hold. The library then applies the univariate formula, which is the formula for the marginal entropy:

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

Equation 13 of [hacinegharbi2012](@cite) states the univariate form, and its Section 3.1 says that no earlier work on histogram estimation used that approach. Equations 36 and 37 of [hacinegharbi2013](@cite) state it again, as the bin count that minimises the mean square error. Equation 19 of [hacinegharbi2018](@cite) states the bi-histogram form, and its Equation 20 restates the univariate one from [hacinegharbi2012](@cite). Snippet 3.3 in Section 3.9 of [mlp1](@cite) gives both closed forms as code, and this method follows that code with one change. At sample sizes of 50, 250 and 1000, the method returns the bin counts that the two formulas give for the correlations of the pairs. The change is the guard at ``\\rho^2 = 1``, which the library adds. No equation of the cited works covers that case, and Snippet 3.3 applies the bi-histogram form at every correlation. Equation 16 of [hacinegharbi2012](@cite) and Equation 26 of [hacinegharbi2013](@cite) give a different bi-histogram form for the mutual information. The library does not use that form.

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

  - $(ref_dict[:hacinegharbi2012]) Section 3.1, Equation 13.
  - $(ref_dict[:hacinegharbi2013]) Section 3.2, Equations 36 and 37.
  - $(ref_dict[:hacinegharbi2018]) Section 2.2, Equation 19, and Section 3.1, Equation 20.
  - $(ref_dict[:mlp1]) Section 3.9, Snippet 3.3.
"""
struct HacineGharbiRavier <: AbstractBins end
"""
    bin_width(::Scott, x::VecNum)

Compute the histogram bin width of `x` with Scott's rule [scott1979](@cite).

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

Theorem 1.6 of [freedman1981](@cite) gives the width that minimises the integrated squared error of a histogram as ``(6 / \\gamma)^{1/3} n^{-1/3}``, with ``\\gamma = \\int f'(x)^{2}\\,\\mathrm{d}x``. A normal density has ``\\gamma = 1 / (4\\sqrt{\\pi}\\,\\sigma^{3})``, which gives the formula above and the constant ``(24\\sqrt{\\pi})^{1/3} \\approx 3.4908``. Section 1 of that work names [scott1979](@cite) as a formal computation of the same result. Equation 5 of [scott1979](@cite) states the optimal width, and its Section 3 gives ``2 \\cdot 3^{1/3} \\pi^{1/6} \\sigma n^{-1/3}`` for Gaussian data, which equals ``(24\\sqrt{\\pi})^{1/3} \\sigma n^{-1/3}``. Equation 6 states the rule with the constant rounded to ``3.49`` and an estimate in place of the standard deviation. The library keeps the exact constant. The work does not say which estimate of the standard deviation to use, and the library uses the uncorrected one.

# Arguments

  - `x`: Data vector.

# Returns

  - $(ret_dict[:dx])

# Related

  - [`Scott`](@ref)
  - [`BinWidthBins`](@ref)
  - [`calc_num_bins`](@ref)

# References

  - $(ref_dict[:scott1979]) Section 3, Equation 5, and Section 5, Equation 6.
  - $(ref_dict[:freedman1981]) Section 1, Theorem 1.6.
"""
function bin_width(::Scott, x::VecNum)
    return Statistics.std(x; corrected = false) * cbrt(24 * sqrt(pi) / length(x))
end
"""
    bin_width(::FreedmanDiaconis, x::VecNum)

Compute the histogram bin width of `x` with the Freedman-Diaconis rule [freedman1981](@cite).

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

A zero interquartile range gives a zero width, and so an infinite bin count. Data with many repeated values do this, for example a return series that is zero on most observations. The width is then the range of `x` divided by five, so the range holds five bins.

# Arguments

  - `x`: Data vector.

# Returns

  - $(ret_dict[:dx])

# Related

  - [`FreedmanDiaconis`](@ref)
  - [`BinWidthBins`](@ref)
  - [`calc_num_bins`](@ref)

# References

  - $(ref_dict[:freedman1981]) Section 1, Rule 1.8.
"""
function bin_width(::FreedmanDiaconis, x::VecNum)
    q25, q75 = Statistics.quantile(x, [0.25, 0.75])
    dx = 2 * (q75 - q25) / cbrt(length(x))
    if !iszero(dx)
        return dx
    end
    # A zero interquartile range, as in a return series that is mostly zero, gives no width.
    # Five bins over the range replace it, so the count stays finite.
    xl, xu = extrema(x)
    return (xu - xl) / 5
end
"""
    bin_width(bins::Knuth, x::VecNum)

Compute the histogram bin width of `x` with Knuth's rule [knuth2019](@cite).

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

 1. Read the range of `x` into `rx`, the difference of its two extrema. A constant `x` has a zero `rx`, and the method returns it as the width.
 2. Build the objective `f`, which takes a one-element vector `Ms`, floors its entry into the bin count `M`, and returns `Inf` when `M` is outside ``[1, n]``. Data with many repeated values, such as a return series that is zero on most observations, make the posterior rise as the bins narrow, and the bound stops the count at one bin per observation.
 3. Inside `f`, bin the data into the counts `nk` over `M` equal-width bins of the range, and return the negated posterior of the mathematical definition. The optimiser minimises, so `f` changes the sign.
 4. Take the start point `M0` from the bin count that the Freedman-Diaconis rule gives for `x`, plus one, and cap it at ``n + 1/2``, so that it floors to at most ``n``.
 5. Minimise `f` from `M0` with `Optim.optimize`, which gets `bins.args` and `bins.kwargs`. The default `args` is a Nelder-Mead simplex.
 6. Floor the minimiser into a bin count, clamp it to ``[1, n]``, and return the range divided by it.

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

  - $(ref_dict[:knuth2019]) Section 2.3, Equations 31 and 32.
"""
function bin_width(bins::Knuth, x::VecNum)
    n = length(x)
    xl, xu = extrema(x)
    rx = xu - xl
    lg_half = SpecialFunctions.loggamma(0.5)
    nk = Vector{Int}(undef, 0)
    if iszero(rx)
        return rx
    end
    function f(Ms)
        M = floor(Int, first(Ms))
        if !(1 <= M <= n)
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
    M0 = min(max(1.0, rx / bin_width(FreedmanDiaconis(), x)) + 1, n + 0.5)
    res = Optim.optimize(f, [M0], bins.args...; bins.kwargs...)
    return rx / clamp(floor(Int, first(Optim.minimizer(res))), 1, n)
end
"""
    calc_num_bins(bins::BinWidthBins, xj::VecNum, xi::VecNum, j::Integer, i::Integer,
                  args...)
    calc_num_bins(bins::HacineGharbiRavier, xj::VecNum, xi::VecNum, j::Integer,
                  i::Integer, T::Integer)
    calc_num_bins(bins::Integer, args...)

Compute the number of histogram bins for a pair of variables with a binning algorithm.

The function dispatches on the type of `bins`, and the three methods read different arguments. Only the [`HacineGharbiRavier`](@ref) method reads `T`, and the `Integer` method reads none of them.

# Algorithm

The [`BinWidthBins`](@ref) method turns a bin width into a bin count.

 1. Read the range of `xj` into `rxj`, and divide it by [`bin_width`](@ref) of `xj`. The result is `k1`. A constant `xj` has a zero range and a zero width, and their quotient is `NaN`, so `k1` is one bin.
 2. When `j` and `i` differ, repeat step 1 for `xi` to get `k2`, and select the larger of `k1` and `k2`. The joint histogram is square, so one count is for both axes, and the larger of the two keeps the finer resolution. When `j` and `i` are equal, the pair is a variable against itself, so select `k1` and do not read `xi`.
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
    # A constant column has a zero range and a zero width, and 0 / 0 is NaN. One bin holds it.
    xjl, xju = extrema(xj)
    rxj = xju - xjl
    k1 = iszero(rxj) ? one(rxj) : rxj / bin_width(bins, xj)
    return round(Int, if j != i
                     xil, xiu = extrema(xi)
                     rxi = xiu - xil
                     max(k1, iszero(rxi) ? one(rxi) : rxi / bin_width(bins, xi))
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
    pair is as deterministic as a perfectly correlated one. The bi-histogram count grows without
    bound as `corr^2` tends to one, so it has no limit there, and the fallback to the univariate
    formula is the library's own guard. Testing `isone(corr)` alone sent `corr = -1`
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

The function bins `xj` and `xi` into `bins` bins each. It returns the Shannon entropy of each marginal histogram and the joint histogram of counts. [`mutual_info`](@ref), [`variation_info`](@ref) and [`mutual_variation_info`](@ref) read these three values for each pair.

A bin is closed on the left, so the lower edge is the minimum itself and needs no widening. The upper edge is exclusive, so it is `nextfloat` of the maximum, one unit in the last place above it at every magnitude. The largest observation therefore falls strictly inside the last bin whatever the magnitude of the data, and a constant column gives an entropy of zero rather than `NaN`.

# Algorithm

 1. Set `bp1` to `bins + 1`, the number of bin edges.
 2. Build the edge range of `xj` from `minimum(xj)` to `nextfloat(maximum(xj))`, with `bp1` points. Repeat for `xi`.
 3. Bin `xj` over its own edges into the counts `hx`, and divide `hx` by its own sum to make it a probability mass function.
 4. Repeat step 3 for `xi` to get `hy`.
 5. Take the Shannon entropy of `hx` into `ex`, and of `hy` into `ey`.
 6. Bin the pair over both edge ranges into the joint counts `hxy`. The function does not normalise `hxy`, because [`intrinsic_mutual_info`](@ref) normalises it.

# Arguments

  - $(arg_dict[:xj])
  - $(arg_dict[:xi])
  - `bins`: Number of bins to use for the histograms.

# Returns

  - `ex::Number`: Shannon entropy of the marginal probability mass function of `xj`, in nats.
  - `ey::Number`: Shannon entropy of the marginal probability mass function of `xi`, in nats.
  - `hxy::Matrix{<:Number}`: Joint histogram of counts, not normalised to probabilities.

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

The function computes the mutual information of two variables from their joint histogram `X`. [`mutual_info`](@ref), [`variation_info`](@ref) and [`mutual_variation_info`](@ref) call it once for each pair.

The word "intrinsic" names the raw estimate from the joint histogram. [`mutual_info`](@ref) can divide that estimate by a marginal entropy, and [`mutual_info`](@ref) and [`mutual_variation_info`](@ref) clamp it at zero, but this function does neither. The estimate is in nats, and it can be larger than one. It is the value that [`mutual_info`](@ref) returns when `normalise` is `false`, before the clamp. It is not the intrinsic conditional information of secret-key agreement, which is a different quantity with the same name.

# Mathematical definition

Let ``\\mathbf{X}`` be the joint histogram of counts, with the marginal probabilities ``p_i = \\sum_j X_{ij} / n`` and ``p_j = \\sum_i X_{ij} / n``. The estimate is:

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

The sum equals ``H(X) + H(Y) - H(X, Y)``, the third form of the rate of transmission in Section 12 of [shannon1948](@cite), with the entropies of Section 6 computed from the histogram frequencies. This is a plug-in estimate, and the library applies no bias correction.

An empty bin adds nothing to the sum, because ``p \\log p`` tends to zero as ``p`` does. The sum therefore runs over the non-empty bins alone. When either axis has a single bin, that variable has one value under the binning. It then shares no information with the other variable, and the estimate is zero.

# Algorithm

 1. Sum `X` over its columns into `p_i`, and over its rows into `p_j`. Both are unnormalised marginal counts.
 2. When either marginal has length one, return zero.
 3. Find the indices of the non-zero entries of `X` into `mask`, and read those entries into the vector `nz`.
 4. Take the logarithm of `nz` into `log_nz`. Sum `nz` into `nz_sum`, the total count, and divide `nz` by it into the joint probabilities `nz_nm`.
 5. Take the outer product of the two marginal counts at the masked indices into `outer`. Negate the logarithm of `outer` and add the logarithm of each marginal total. The result is `log_outer`, the negated logarithm of the product of the marginal probabilities.
 6. Make the contribution `mi` of each bin from `nz_nm`, `log_nz`, `nz_sum` and `log_outer`, and set to zero every contribution whose magnitude is below the machine epsilon.
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

  - $(ref_dict[:shannon1948]) Section 6 and Section 12.
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
    log_outer = .-log.(outer) .+ (log(sum(p_i)) + log(sum(p_j)))

    mi = (nz_nm ⊙ (log_nz .- log(nz_sum)) + nz_nm ⊙ log_outer)
    mi[abs.(mi) .< eps(eltype(mi))] .= zero(eltype(X))

    return sum(mi)
end
"""
    variation_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                   normalise::Bool = true)

Compute the variation of information (VI) matrix for a set of variables.

The function computes the variation of information of each pair of columns of `X` from histogram estimates of the entropies and the mutual information. The variation of information is the information that one variable has and the other does not, summed over the two directions. It is a metric on discrete distributions. It is non-negative, it is zero exactly when the two variables agree, it is symmetric, and it satisfies the triangle inequality. [`mutual_info`](@ref) is not a metric, because its diagonal is the entropy and a metric is zero on its diagonal.

# Mathematical definition

Let ``H(X)`` and ``H(Y)`` be the marginal Shannon entropies, and ``I(X;Y)`` the mutual information. The variation of information is:

```math
\\begin{align}
\\mathrm{VI}(X, Y) &= H(X) + H(Y) - 2\\,I(X;Y)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VI}(X, Y)``: Variation of information between ``X`` and ``Y``.
  - ``H(X)``, ``H(Y)``: Marginal Shannon entropies.
  - ``I(X;Y)``: Mutual information.

When `normalise = true`, the function divides it by the joint entropy ``H(X,Y) = H(X) + H(Y) - I(X;Y)``:

```math
\\begin{align}
\\widetilde{\\mathrm{VI}}(X, Y) &= \\frac{H(X) + H(Y) - 2\\,I(X;Y)}{H(X) + H(Y) - I(X;Y)}\\,.
\\end{align}
```

Where:

  - ``\\widetilde{\\mathrm{VI}}(X, Y)``: Normalised variation of information.
  - ``H(X,Y) = H(X) + H(Y) - I(X;Y)``: Joint entropy.

Equation 19 of Section 3 of [meila2007](@cite) states the unnormalised form and proves that it is a metric. The divisor of the normalised form is the joint entropy, and Equation 1 of [rajski1961](@cite) states that quotient as a distance on discrete distributions. Equation 6.25 of [cajas2025](@cite) is ``1 - I(X;Y) / \\max(H(X), H(Y))`` instead. That quotient lies in the same interval and is also a metric, the second metric of Theorem 2 of [kraskov2005](@cite). This function does not compute it.

# Algorithm

 1. Read the shape of `X` into `T`, the number of observations, and `N`, the number of variables.
 2. Allocate the `N × N` result `var_mtx`.
 3. For each variable `j`, write an exact zero at `var_mtx[j, j]`. ``\\mathrm{VI}(X, X)`` is zero by definition. The histogram estimate of ``I(X; X)`` does not reproduce the estimate of ``H(X)`` bit for bit, so an estimate of the self pair leaves about `1e-16` on the diagonal. A matrix with a non-zero diagonal is not a distance matrix. The exact zero also saves `N` histogram computations.
 4. For each pair `(j, i)` below the diagonal, take the bin count from [`calc_num_bins`](@ref) into `nbins`.
 5. Take the two marginal entropies `ex` and `ey` and the joint histogram `hxy` from [`calc_hist_data`](@ref).
 6. Take the mutual information of `hxy` from [`intrinsic_mutual_info`](@ref) into `mut_ixy`, and apply the definition above to get `var_ixy`.
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

  - $(ref_dict[:meila2007]) Section 3, Equation 19.
  - $(ref_dict[:rajski1961]) Theorem, Equation 1.
  - $(ref_dict[:kraskov2005]) Section 2.3, Theorem 2.
  - $(ref_dict[:shannon1948]) Section 6 and Section 12.
  - $(ref_dict[:cajas2025]) Section 6.2.2, Equations 6.24 and 6.25.
"""
function variation_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                        normalise::Bool = true)
    T, N = size(X)
    var_mtx = Matrix{eltype(X)}(undef, N, N)
    for j in axes(X, 2)
        xj = view(X, :, j)
        #=
        VI(X, X) = H(X) + H(X) - 2*I(X; X) = 0 exactly, by definition, so the self pair is
        set rather than estimated. An estimate leaves about 1e-16 on the diagonal of 7 of 12
        assets of a measured sample, because the histogram estimate of I(X; X) does not
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

The function computes both matrices in one pass over the pairs, so the two matrices use one bin count and one joint histogram for each pair. With the same `bins`, the two results equal what [`mutual_info`](@ref) and [`variation_info`](@ref) return. The two normalisers differ. The function divides the mutual information by the smaller marginal entropy, and the variation of information by the joint entropy. The default `bins` is [`Knuth`](@ref) here and [`HacineGharbiRavier`](@ref) in the other two functions, so a caller who wants equal results must give the same `bins` to each function.

# Mathematical definition

```math
\\begin{align}
I(X_i; X_j) &= H(X_i) + H(X_j) - H(X_i, X_j)\\,, \\\\
\\mathrm{VI}(X_i, X_j) &= H(X_i) + H(X_j) - 2\\,I(X_i; X_j)\\,.
\\end{align}
```

When `normalise = true`, the function divides each by its own normaliser:

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

The two unnormalised forms are the rate of transmission of Section 12 of [shannon1948](@cite) and Equation 19 of [meila2007](@cite). The quotient by the joint entropy is Equation 1 of [rajski1961](@cite). The quotient of the mutual information by the smaller marginal entropy is Equation 6.19 of [cajas2025](@cite), and no work that first states it was found.

# Algorithm

 1. Read the shape of `X` into `T`, the number of observations, and `N`, the number of variables.
 2. Allocate the two `N × N` results `mut_mtx` and `var_mtx`.
 3. For each pair `(j, i)` on or below the diagonal, take the bin count from [`calc_num_bins`](@ref) into `nbins`.
 4. Take the two marginal entropies `ex` and `ey` and the joint histogram `hxy` from [`calc_hist_data`](@ref).
 5. Take the mutual information of `hxy` from [`intrinsic_mutual_info`](@ref) into `mut_ixy`, and make `var_ixy` from it and the two entropies.
 6. When `normalise` is true, divide `var_ixy` by the joint entropy `vxy` and `mut_ixy` by the smaller of `ex` and `ey`.
 7. Clamp both below at zero, and write each into both of its symmetric positions.
 8. After the inner loop, write an exact zero at `var_mtx[j, j]`, for the reason that [`variation_info`](@ref) gives. `mut_mtx[j, j]` keeps its estimate, because ``I(X; X) = H(X)`` is not zero.

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

  - $(ref_dict[:shannon1948]) Section 6 and Section 12.
  - $(ref_dict[:meila2007]) Section 3, Equations 18 and 19.
  - $(ref_dict[:rajski1961]) Theorem, Equation 1.
  - $(ref_dict[:cajas2025]) Section 6.1.6, Equation 6.19. $(ref_dict[:no_original_source])
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
        # approximates it. `mut_mtx` keeps its estimated diagonal, because I(X; X) = H(X) is
        # not zero.
        var_mtx[j, j] = zero(eltype(X))
    end

    return mut_mtx, var_mtx
end
"""
    mutual_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                normalise::Bool = true)

Compute the mutual information matrix for a set of variables.

The function computes the mutual information of each pair of columns of `X` from histogram estimates. The mutual information is the information that two variables share, and it is zero when the two variables are independent.

Mutual information is not a metric, because its diagonal is the marginal entropy and a metric is zero on its diagonal. It is larger for a more dependent pair, and a distance is smaller. [`variation_info`](@ref) is the metric of the two measures. The function estimates the diagonal and does not set it, so the diagonal is the marginal entropy in nats when `normalise` is `false`, and one when it is `true`.

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

When `normalise = true`, the function divides the mutual information by the smaller marginal entropy:

```math
\\begin{align}
\\tilde{I}(X_i; X_j) &= \\frac{I(X_i; X_j)}{\\min\\bigl(H(X_i),\\, H(X_j)\\bigr)}\\,.
\\end{align}
```

Where:

  - ``\\tilde{I}(X_i; X_j)``: Normalised mutual information.

Equation 18 of [meila2007](@cite) shows that the smaller marginal entropy is the largest value that the mutual information of the pair can take. The normalised form is therefore in ``[0, 1]``, and it is one exactly when one variable determines the other. Equation 6.19 of [cajas2025](@cite) states this quotient, and no work that first states it was found.

# Algorithm

 1. Read the shape of `X` into `T`, the number of observations, and `N`, the number of variables.
 2. Allocate the `N × N` result `mut_mtx`.
 3. For each pair `(j, i)` on or below the diagonal, take the bin count from [`calc_num_bins`](@ref) into `nbins`.
 4. Take the two marginal entropies `ex` and `ey` and the joint histogram `hxy` from [`calc_hist_data`](@ref).
 5. Take the mutual information of `hxy` from [`intrinsic_mutual_info`](@ref) into `mut_ixy`.
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

  - $(ref_dict[:shannon1948]) Section 6 and Section 12.
  - $(ref_dict[:meila2007]) Section 3, Equation 18.
  - $(ref_dict[:cajas2025]) Section 6.1.6, Equation 6.19. $(ref_dict[:no_original_source])
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
public AbstractBins, BinWidthBins, calc_num_bins, bin_width
