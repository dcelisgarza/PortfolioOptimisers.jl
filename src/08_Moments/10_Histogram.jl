"""
    abstract type AbstractBins <: AbstractAlgorithm end

Abstract supertype for all histogram binning algorithms.

`AbstractBins` is the abstract type for all binning algorithm types used in histogram-based calculations within PortfolioOptimisers.jl, such as mutual information and variation of information analysis. Concrete subtypes implement specific binning strategies (e.g., Knuth, Freedman-Diaconis, Scott, Hacine-Gharbi-Ravier) and provide a consistent interface for bin selection.

# Related

  - [`AstroPyBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
"""
abstract type AbstractBins <: AbstractAlgorithm end
const Int_Bin = Union{<:AbstractBins, <:Integer}
"""
    abstract type AstroPyBins <: AbstractBins end

Abstract supertype for all histogram binning algorithms implemented using AstroPy's bin width selection methods.

`AstroPyBins` is the abstract type for all binning algorithm types that rely on bin width selection functions from the [AstroPy](https://www.astropy.org/) Python library, such as Knuth, Freedman-Diaconis, and Scott. Concrete subtypes implement specific binning strategies and provide a consistent interface for bin selection in histogram-based calculations within PortfolioOptimisers.jl.

# Related

  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`AbstractBins`](@ref)
"""
abstract type AstroPyBins <: AbstractBins end
"""
    struct Knuth <: AstroPyBins end

Histogram binning algorithm using Knuth's rule.

`Knuth` implements Knuth's rule for selecting the optimal number of bins in a histogram, as provided by the [AstroPy](https://www.astropy.org/) library. This method aims to maximize the posterior probability of the histogram given the data, resulting in an adaptive binning strategy that balances bias and variance.

# Related

  - [`AstroPyBins`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`get_bin_width_func`](@ref)
"""
struct Knuth <: AstroPyBins end
"""
    struct FreedmanDiaconis <: AstroPyBins end

Histogram binning algorithm using the Freedman-Diaconis rule.

`FreedmanDiaconis` implements the Freedman-Diaconis rule for selecting the number of bins in a histogram, as provided by the [AstroPy](https://www.astropy.org/) library. This method determines bin width based on the interquartile range (IQR) and the number of data points, making it robust to outliers and suitable for skewed distributions.

# Related

  - [`AstroPyBins`](@ref)
  - [`Knuth`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`get_bin_width_func`](@ref)
"""
struct FreedmanDiaconis <: AstroPyBins end
"""
    struct Scott <: AstroPyBins end

Histogram binning algorithm using Scott's rule.

`Scott` implements Scott's rule for selecting the number of bins in a histogram, as provided by the [AstroPy](https://www.astropy.org/) library. This method chooses bin width based on the standard deviation of the data and the number of observations, providing a good default for normally distributed data.

# Related

  - [`AstroPyBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`HacineGharbiRavier`](@ref)
  - [`get_bin_width_func`](@ref)
"""
struct Scott <: AstroPyBins end
"""
    struct HacineGharbiRavier <: AbstractBins end

Histogram binning algorithm using the Hacine-Gharbi–Ravier rule.

`HacineGharbiRavier` implements the Hacine-Gharbi–Ravier rule for selecting the number of bins in a histogram. This method adapts the bin count based on the correlation structure and sample size, and is particularly useful for information-theoretic measures such as mutual information and variation of information.

# Related

  - [`AbstractBins`](@ref)
  - [`AstroPyBins`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`get_bin_width_func`](@ref)
"""
struct HacineGharbiRavier <: AbstractBins end
"""
    get_bin_width_func(bins::Int_Bin)

Return the bin width selection function associated with a histogram binning algorithm.

This utility dispatches on the binning algorithm type and returns the corresponding bin width function from the [AstroPy](https://www.astropy.org/) Python library for `Knuth`, `FreedmanDiaconis`, and `Scott`. For `HacineGharbiRavier` and integer bin counts, it returns `nothing`, as these strategies do not use a bin width function.

# Arguments

  - `bins::Knuth`: Use Knuth's rule (`astropy.stats.knuth_bin_width`).
  - `bins::FreedmanDiaconis`: Use the Freedman-Diaconis rule (`astropy.stats.freedman_bin_width`).
  - `bins::Scott`: Use Scott's rule (`astropy.stats.scott_bin_width`).
  - `bins::Union{<:HacineGharbiRavier, <:Integer}`: No bin width function (returns `nothing`).

# Returns

  - `bin_width_func::Function`: The corresponding bin width function (callable), or `nothing` if not applicable.

# Examples

```jldoctest; filter = r"0x[0-9a-fA-F]+" => s"..."
julia> PortfolioOptimisers.get_bin_width_func(Knuth())
Python: <function knuth_bin_width at 0x7da1178e0fe0>

julia> PortfolioOptimisers.get_bin_width_func(FreedmanDiaconis())
Python: <function freedman_bin_width at 0x7da1178e0fe0>

julia> PortfolioOptimisers.get_bin_width_func(Scott())
Python: <function scott_bin_width at 0x7da1178e0fe0>

julia> PortfolioOptimisers.get_bin_width_func(HacineGharbiRavier())

julia> PortfolioOptimisers.get_bin_width_func(10)

```

# Related

  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
"""
function get_bin_width_func(::Knuth)
    return PythonCall.pyimport("astropy.stats").knuth_bin_width
end
function get_bin_width_func(::FreedmanDiaconis)
    return PythonCall.pyimport("astropy.stats").freedman_bin_width
end
function get_bin_width_func(::Scott)
    return PythonCall.pyimport("astropy.stats").scott_bin_width
end
function get_bin_width_func(::Union{<:HacineGharbiRavier, <:Integer})
    return nothing
end
"""
    calc_num_bins(bins::Int_Bin, xj::VecNum,
                  xi::VecNum, j::Integer, i::Integer, bin_width_func, T::Integer)

Compute the number of histogram bins for a pair of variables using a specified binning algorithm.

This function determines the number of bins to use for histogram-based calculations (such as mutual information or variation of information) between two variables, based on the selected binning strategy. It dispatches on the binning algorithm type and uses the appropriate method for each:

  - For `AstroPyBins`, it computes the bin width using the provided `bin_width_func` and computes the number of bins as the range divided by the bin width, rounding to the nearest integer. For off-diagonal pairs, it uses the maximum of the two variables' bin counts.
  - For `HacineGharbiRavier`, it uses the Hacine-Gharbi–Ravier rule, which adapts the bin count based on the correlation and sample size.
  - For an integer, it returns the specified number of bins directly.

# Arguments

  - `bins`: Binning algorithm/number.
  - `xj`: Data vector for variable `j`.
  - `xi`: Data vector for variable `i`.
  - `j`: Index of variable `j`.
  - `i`: Index of variable `i`.
  - `bin_width_func`: Bin width selection function (from `get_bin_width_func`), or `nothing`.
  - `T`: Number of observations (used by some algorithms).

# Returns

  - `nbins::Int`: The computed number of bins for the variable pair.

# Related

  - [`get_bin_width_func`](@ref)
  - [`Knuth`](@ref)
  - [`FreedmanDiaconis`](@ref)
  - [`Scott`](@ref)
  - [`HacineGharbiRavier`](@ref)
"""
function calc_num_bins(::AstroPyBins, xj::VecNum, xi::VecNum, j::Integer, i::Integer,
                       bin_width_func, ::Any)
    xjl, xju = extrema(xj)
    k1 = (xju - xjl) /
         PythonCall.pyconvert(eltype(xj), bin_width_func(PythonCall.Py(xj).to_numpy()))
    return round(Int,
                 if j != i
                     xil, xiu = extrema(xi)
                     k2 = (xiu - xil) / PythonCall.pyconvert(eltype(xi),
                                                             bin_width_func(PythonCall.Py(xi).to_numpy()))
                     max(k1, k2)
                 else
                     k1
                 end)
end
function calc_num_bins(::HacineGharbiRavier, xj::VecNum, xi::VecNum, j::Integer, i::Integer,
                       ::Any, T::Integer)
    corr = cor(xj, xi)
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

  - The histograms are computed using `StatsBase.StatsAPI.fit(StatsBase.Histogram, ...)` over the range of each variable, with bin edges expanded slightly using `eps` to ensure all data is included.
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

    hx = StatsBase.StatsAPI.fit(StatsBase.Histogram, xj, range(xjl, xjh; length = bp1)).weights
    hx /= sum(hx)

    hy = StatsBase.StatsAPI.fit(StatsBase.Histogram, xi, range(xil, xih; length = bp1)).weights
    hy /= sum(hy)

    ex = StatsBase.entropy(hx)
    ey = StatsBase.entropy(hy)

    hxy = StatsBase.StatsAPI.fit(StatsBase.Histogram, (xj, xi),
                                 (range(xjl, xjh; length = bp1),
                                  range(xil, xih; length = bp1))).weights

    return ex, ey, hxy
end
"""
    intrinsic_mutual_info(X::MatNum)

Compute the intrinsic mutual information from a joint histogram.

This function computes the mutual information between two variables given their joint histogram matrix `X`. It is used as a core step in information-theoretic measures such as mutual information and variation of information.

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
    bin_width_func = get_bin_width_func(bins)
    for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
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
function mutual_variation_info(X::MatNum, bins::Int_Bin = Knuth(), normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    var_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = get_bin_width_func(bins)

    for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
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
    bin_width_func = get_bin_width_func(bins)
    for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
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
