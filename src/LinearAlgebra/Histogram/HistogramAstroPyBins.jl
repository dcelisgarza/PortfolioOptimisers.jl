abstract type AstroPyBins <: AbstractBins end
struct B_Knuth <: AstroPyBins end
struct B_FreedmanDiaconis <: AstroPyBins end
struct B_Scott <: AstroPyBins end
function get_bin_width_func(::B_Knuth)
    return pyimport("astropy.stats").knuth_bin_width
end
function get_bin_width_func(::B_FreedmanDiaconis)
    return pyimport("astropy.stats").freedman_bin_width
end
function get_bin_width_func(::B_Scott)
    return pyimport("astropy.stats").scott_bin_width
end
function calc_num_bins(::AstroPyBins, xj::AbstractVector, xi::AbstractVector, j::Integer,
                       i::Integer, bin_width_func, ::Any)
    xjl, xju = extrema(xj)
    k1 = (xju - xjl) / pyconvert(eltype(xj), bin_width_func(Py(xj).to_numpy()))
    return round(Int,
                 if j != i
                     xil, xiu = extrema(xi)
                     k2 = (xiu - xil) /
                          pyconvert(eltype(xi), bin_width_func(Py(xi).to_numpy()))
                     max(k1, k2)
                 else
                     k1
                 end)
end

export B_Knuth, B_FreedmanDiaconis, B_Scott
