struct NoDenoise <: DenoiseAlgorithm end
function denoise!(::NoDenoise, ::FixNonPositiveDefiniteMatrix, ::AbstractMatrix, ::Real)
    return nothing
end

export NoDenoise
