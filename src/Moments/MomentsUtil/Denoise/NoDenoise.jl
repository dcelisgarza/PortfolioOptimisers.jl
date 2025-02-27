struct NoDenoise <: DenoiseAlgorithm end
function denoise!(::NoDenoise, args...)
    return nothing
end
