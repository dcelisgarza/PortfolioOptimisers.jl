const Int_CVE = Union{<:Integer, <:CrossValidationEstimator}
function cross_val_predict(est::OptimisationEstimator, rd::ReturnsResult,
                           cv::Option{<:Int_CVE} = nothing; cols = nothing)
    if !isnothing(cols)
        rd = returns_result_view(rd, cols)
    end
    if hasfield(cv, :shuffled) && cv.shuffled
        throw(ArgumentError("Cross validation estimator must not be shuffled."))
    end
    train, test = split(cv, rd)
    for t in train
        @argcheck(all(x -> x > zero(x), diff(t)),
                  ArgumentError("Cross validation estimator must not be shuffled."))
    end
    return nothing
end

export cross_val_predict
