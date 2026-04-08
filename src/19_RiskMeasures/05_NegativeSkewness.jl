const NSkeFormulations = Union{<:NSkeQuadFormulations, <:SOCRiskExpr}
@concrete struct NegativeSkewness <: RiskMeasure
    settings
    mp
    sk
    V
    alg
    window
    function NegativeSkewness(settings::RiskMeasureSettings,
                              mp::AbstractMatrixProcessingEstimator, sk::Option{<:MatNum},
                              V::Option{<:MatNum}, alg::NSkeFormulations,
                              window::Option{<:Int_VecInt})
        sk_flag = isnothing(sk)
        V_flag = isnothing(V)
        if sk_flag || V_flag
            @argcheck(sk_flag)
            @argcheck(V_flag)
        else
            @argcheck(!isempty(sk))
            @argcheck(!isempty(V))
            @argcheck(size(sk, 1)^2 == size(sk, 2))
            assert_matrix_issquare(V, :V)
        end
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(settings), typeof(mp), typeof(sk), typeof(V), typeof(alg),
                   typeof(window)}(settings, mp, sk, V, alg, window)
    end
end
function NegativeSkewness(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
                          sk::Option{<:MatNum} = nothing, V::Option{<:MatNum} = nothing,
                          alg::NSkeFormulations = SOCRiskExpr(),
                          window::Option{<:Int_VecInt} = nothing)
    return NegativeSkewness(settings, mp, sk, V, alg, window)
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:SOCRiskExpr})(w::VecNum)
    return sqrt(LinearAlgebra.dot(w, r.V, w))
end
function (r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any, <:NSkeQuadFormulations})(w::VecNum)
    return LinearAlgebra.dot(w, r.V, w)
end
function factory(r::NegativeSkewness, pr::HighOrderPrior, args...; kwargs...)
    sk = nothing_scalar_array_selector(r.sk, pr.sk)
    V = nothing_scalar_array_selector(r.V, pr.V)
    return NegativeSkewness(; settings = r.settings, mp = r.mp, sk = sk, V = V, alg = r.alg,
                            window = r.window)
end
function factory(r::NegativeSkewness, ::LowOrderPrior, args...; kwargs...)
    return r
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:Any, <:Any}, ::Any, args...)
    return r
end
function risk_measure_view(r::NegativeSkewness{<:Any, <:Any, <:MatNum, <:MatNum}, i,
                           X::MatNum)
    sk = r.sk
    idx = fourth_moment_index_generator(size(sk, 1), i)
    sk = view(r.sk, i, idx)
    window = get_window(r.window, X)
    V = negative_spectral_coskewness(sk, view(X, window, i), r.mp)
    return NegativeSkewness(; settings = r.settings, alg = r.alg, mp = r.mp, sk = sk, V = V,
                            window = r.window)
end

export NegativeSkewness
