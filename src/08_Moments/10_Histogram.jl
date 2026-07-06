"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all histogram binning algorithms.

`AbstractBins` is the abstract type for all binning algorithm types used in histogram-based calculations within `PortfolioOptimisers.jl`, such as mutual information and variation of information analysis. Concrete subtypes implement specific binning strategies (e.g., Knuth, Freedman-Diaconis, Scott, Hacine-Gharbi-Ravier) and provide a consistent interface for bin selection.

# Related

  - [`BinWidthBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
"""
abstract type AbstractBins <: AbstractAlgorithm end
"""
    const Int_Bin = Union{<:AbstractBins, <:Integer}

Alias for a histogram binning algorithm or an integer number of bins.

Matches either an [`AbstractBins`](@ref) algorithm (auto-selecting bin counts) or a plain `Integer` (fixed number of bins). Used in histogram-based mutual information and variation of information calculations.

# Related

  - [`AbstractBins`](@ref)
  - [`mutual_variation_info`](@ref)
"""
const Int_Bin = Union{<:AbstractBins, <:Integer}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all histogram binning algorithms based on a bin width selection rule.

`BinWidthBins` is the abstract type for all binning algorithm types that select the number of bins by first computing an optimal bin width from the data, such as Knuth, Freedman-Diaconis, and Scott. Concrete subtypes implement specific binning strategies and provide a consistent interface for bin selection in histogram-based calculations within `PortfolioOptimisers.jl`.

# Related

  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`AbstractBins`](@ref)
"""
abstract type BinWidthBins <: AbstractBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using Knuth's rule.

`Knuth` implements Knuth's rule for selecting the optimal number of bins in a histogram [knuth2019](@cite). This method maximises the posterior probability of a piecewise-constant density model given the data, resulting in an adaptive binning strategy that balances bias and variance.

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
"""
@concrete struct Knuth <: BinWidthBins
    args
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
"""
struct Scott <: BinWidthBins end
"""
$(DocStringExtensions.TYPEDEF)

Histogram binning algorithm using the Hacine-Gharbi–Ravier rule.

`HacineGharbiRavier` implements the Hacine-Gharbi–Ravier rule for selecting the number of bins in a histogram. This method adapts the bin count based on the correlation structure and sample size, and is particularly useful for information-theoretic measures such as mutual information and variation of information.

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

  - `dx::Number`: The optimal bin width.

# Related

  - [`Scott`](@ref)
  - [`calc_num_bins`](@ref)
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

  - `dx::Number`: The optimal bin width.

# Related

  - [`FreedmanDiaconis`](@ref)
  - [`calc_num_bins`](@ref)
"""
function bin_width(::FreedmanDiaconis, x::VecNum)
    q25, q75 = Statistics.quantile(x, [0.25, 0.75])
    return 2 * (q75 - q25) / cbrt(length(x))
end
"""
    bin_width(bins::Knuth, x::VecNum)

Compute the optimal histogram bin width for `x` using Knuth's rule [knuth2019](@cite).

Maximises the marginal posterior probability of a piecewise-constant density model with ``M`` equal-width bins over the data range,

```math
\\begin{align}
F(M) &= n \\log M + \\log\\Gamma\\!\\left(\\frac{M}{2}\\right) - M \\log\\Gamma\\!\\left(\\frac{1}{2}\\right) - \\log\\Gamma\\!\\left(n + \\frac{M}{2}\\right) + \\sum_{k=1}^{M} \\log\\Gamma\\!\\left(n_k + \\frac{1}{2}\\right)\\,.
\\end{align}
```

Where:

  - ``M``: Number of bins.
  - ``n``: Number of observations.
  - ``n_k``: Number of observations in bin ``k``.

# Arguments

  - `x`: Data vector.

# Returns

  - `dx::Number`: The optimal bin width.

# Details

  - The optimisation is performed with Nelder-Mead over a continuous relaxation of ``M`` (evaluated at ``\\lfloor M \\rfloor``), started at the bin count implied by the Freedman-Diaconis rule.

# Related

  - [`Knuth`](@ref)
  - [`calc_num_bins`](@ref)
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
    calc_num_bins(bins::Int_Bin, xj::VecNum,
                  xi::VecNum, j::Integer, i::Integer, bin_width_func, T::Integer)

Compute the number of histogram bins for a pair of variables using a specified binning algorithm.

This function determines the number of bins to use for histogram-based calculations (such as mutual information or variation of information) between two variables, based on the selected binning strategy. It dispatches on the binning algorithm type and uses the appropriate method for each:

  - For `BinWidthBins`, it computes the bin width using the provided `bin_width_func` and computes the number of bins as the range divided by the bin width, rounding to the nearest integer. For off-diagonal pairs, it uses the maximum of the two variables' bin counts.
  - For `HacineGharbiRavier`, it uses the Hacine-Gharbi–Ravier rule, which adapts the bin count based on the correlation and sample size.
  - For an integer, it returns the specified number of bins directly.

# Arguments

  - `bins`: Binning algorithm/number.
  - `xj`: Data vector for variable `j`.
  - `xi`: Data vector for variable `i`.
  - `j`: Index of variable `j`.
  - `i`: Index of variable `i`.
  - `T`: Number of observations (used by some algorithms).

# Returns

  - `nbins::Int`: The computed number of bins for the variable pair.

# Related

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
    return round(Int, if isone(corr)
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

# Arguments

  - `xj`: Data vector for variable `j`.
  - `xi`: Data vector for variable `i`.
  - `bins`: Number of bins to use for the histograms.

# Returns

  - `ex::Number`: Entropy of `xj`.
  - `ey::Number`: Entropy of `xi`.
  - `hxy::Matrix{<:Number}`: Joint histogram (counts, not normalised to probability).

# Details

  - The histograms are computed using `StatsAPI.fit(StatsBase.Histogram, ...)` over the range of each variable, with bin edges expanded slightly using `eps` to ensure all data is included.
  - The marginal histograms are normalised to sum to 1 before entropy calculation.
  - The joint histogram is not normalised, as it is used directly in mutual information calculations.

# Related

  - [`variation_info`](@ref)
  - [`mutual_info`](@ref)
"""
function calc_hist_data(xj::VecNum, xi::VecNum, bins::Integer)
    bp1 = bins + one(bins)

    xjl = minimum(xj) - eps(eltype(xj))
    xjh = maximum(xj) + eps(eltype(xj))

    xil = minimum(xi) - eps(eltype(xi))
    xih = maximum(xi) + eps(eltype(xi))

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

# Arguments

  - `X`: Joint histogram matrix.

# Returns

  - `mi::Number`: The intrinsic mutual information between the two variables.

# Details

  - The function computes marginal distributions by summing over rows and columns.
  - Only nonzero entries in the joint histogram are considered.
  - The mutual information is computed as the sum over all nonzero joint probabilities of `p(x, y) * log(p(x, y) / (p(x) * p(y)))`, with careful handling of log and normalisation.

# Related

  - [`calc_hist_data`](@ref)
  - [`variation_info`](@ref)
  - [`mutual_info`](@ref)
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
    variation_info(X::MatNum;
                   bins::Int_Bin = HacineGharbiRavier(),
                   normalise::Bool = true)

Compute the variation of information (VI) matrix for a set of variables.

This function computes the pairwise variation of information between all columns of the data matrix `X`, using histogram-based entropy and mutual information estimates. VI quantifies the amount of information lost and gained when moving from one variable to another, and is a true metric on the space of discrete distributions.

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

# Arguments

  - `X`: Data matrix (observations × variables).
  - `bins`: Binning algorithm or fixed number of bins.
  - `normalise`: Whether to normalise the VI by the joint entropy.

# Returns

  - `var_mtx::Matrix{<:Number}`: Symmetric matrix of pairwise variation of information values.

# Details

  - For each pair of variables, the function computes marginal entropies and the joint histogram using `calc_hist_data`.
  - The mutual information is computed using `intrinsic_mutual_info`.
  - VI is calculated as `H(X) + H(Y) - 2 * LinearAlgebra.I(X, Y)`. If `normalise` is `true`, it is divided by the joint entropy.
  - The result is clamped to `[0, typemax(eltype(X))]` and is symmetric.

# Related

  - [`mutual_info`](@ref)
  - [`calc_hist_data`](@ref)
  - [`intrinsic_mutual_info`](@ref)
"""
function variation_info(X::MatNum, bins::Int_Bin = HacineGharbiRavier(),
                        normalise::Bool = true)
    T, N = size(X)
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
            end
            var_ixy = clamp(var_ixy, zero(eltype(X)), typemax(eltype(X)))
            var_mtx[j, i] = var_mtx[i, j] = var_ixy
        end
    end
    return var_mtx
end
# COV_EXCL_START
"""
    mutual_variation_info(X::MatNum, bins::Int_Bin = Knuth(), normalise::Bool = true)

Compute the pairwise mutual information and variation of information matrices from a data matrix.

# Arguments

  - `X`: Data matrix of shape `(T, N)` (observations × assets).
  - `bins`: Binning algorithm or integer number of bins for histogram computation.
  - `normalise`: If `true`, normalises the mutual information and variation of information.

# Returns

  - `(mut_mtx, var_mtx)`: Tuple of symmetric matrices for mutual information and variation of information.

# Related

  - [`Int_Bin`](@ref)
  - [`AbstractBins`](@ref)
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
    end

    return mut_mtx, var_mtx
end
# COV_EXCL_STOP
"""
    mutual_info(X::MatNum;
                bins::Int_Bin = HacineGharbiRavier(),
                normalise::Bool = true)

Compute the mutual information (MI) matrix for a set of variables.

This function computes the pairwise mutual information between all columns of the data matrix `X`, using histogram-based entropy and mutual information estimates. MI quantifies the amount of shared information between pairs of variables, and is widely used in information-theoretic analysis of dependencies.

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

# Arguments

  - `X`: Data matrix (observations × variables).
  - `bins`: Binning algorithm or fixed number of bins.
  - `normalise`: Whether to normalise the MI by the minimum marginal entropy.

# Returns

  - `mut_mtx::Matrix{<:Number}`: Symmetric matrix of pairwise mutual information values.

# Details

  - For each pair of variables, the function computes marginal entropies and the joint histogram using [`calc_hist_data`](@ref).
  - The mutual information is computed using [`intrinsic_mutual_info`](@ref).
  - If `normalise` is `true`, the MI is divided by the minimum of the two marginal entropies.
  - The result is clamped to `[0, typemax(eltype(X))]` and is symmetric.

# Related

  - [`variation_info`](@ref)
  - [`calc_hist_data`](@ref)
  - [`intrinsic_mutual_info`](@ref)
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
