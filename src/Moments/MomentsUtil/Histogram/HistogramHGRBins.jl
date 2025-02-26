struct B_HacineGharbiRavier <: AbstractBins end
function get_bin_width_func(::B_HacineGharbiRavier)
    return nothing
end
function calc_num_bins(::B_HacineGharbiRavier, xj::AbstractVector, xi::AbstractVector,
                       j::Integer, i::Integer, ::Any, T::Integer)
    corr = cor(xj, xi)
    return round(Int, if isone(corr)
                     z = cbrt(8 + 324 * T + 12 * sqrt(36 * T + 729 * T^2))
                     z / 6 + 2 / (3 * z) + 1 / 3
                 else
                     sqrt(1 + sqrt(1 + 24 * T / (1 - corr^2))) / sqrt(2)
                 end)
end

export B_HacineGharbiRavier
