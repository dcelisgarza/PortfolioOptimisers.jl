abstract type DenoiseAlgorithm end
function denoise! end
function denoise!(::Nothing, ::Union{Nothing, <:FixNonPositiveDefiniteMatrix},
                  ::AbstractMatrix, ::Real)
    return nothing
end
export denoise!
