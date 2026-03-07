
struct RandomisedSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       AbstractSearchCrossValidationEstimator
    p::T1
    cv::T2
    r::T3
    score::T4
    ex::T5
    n_iter::T6
    rng::T7
    seed::T8
    train_score::T9
    kwargs::T10
    function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:AbstractString,
                                                                              <:Any}},
                                                      <:AbstractVector{<:AbstractVector{<:Pair{<:AbstractString,
                                                                                               <:Any}}},
                                                      <:AbstractDict{<:Pair{<:AbstractString,
                                                                            <:Any}},
                                                      <:AbstractVector{<:AbstractDict{<:Pair{<:AbstractString,
                                                                                             <:Any}}}},
                                             cv::SearchCV, r::AbstractBaseRiskMeasure,
                                             score::CrossValSearchScorer,
                                             ex::FLoops.Transducers.Executor,
                                             n_iter::Integer, rng::Random.AbstractRNG,
                                             seed::Option{<:Integer}, train_score::Bool,
                                             kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        if isa(p, VecMultiGSCVValType)
            @argcheck(all(!isempty, p), IsEmptyError)
        end
        assert_nonempty_gt0_finite_val(n_iter, "n_iter")
        return new{typeof(p), typeof(cv), typeof(r), typeof(score), typeof(ex),
                   typeof(n_iter), typeof(rng), typeof(seed), typeof(train_score),
                   typeof(kwargs)}(p, cv, r, score, ex, n_iter, rng, seed, train_score,
                                   kwargs)
    end
end
function RandomisedSearchCrossValidation(p::Union{<:AbstractVector{<:Pair{<:AbstractString,
                                                                          <:Any}},
                                                  <:AbstractVector{<:AbstractVector{<:Pair{<:AbstractString,
                                                                                           <:Any}}},
                                                  <:AbstractDict{<:Pair{<:AbstractString,
                                                                        <:Any}},
                                                  <:AbstractVector{<:AbstractDict{<:Pair{<:AbstractString,
                                                                                         <:Any}}}};
                                         cv::SearchCV = KFold(),
                                         r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                         score::CrossValSearchScorer = HighestMeanScore(),
                                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                         n_iter::Integer = 10,
                                         rng::Random.AbstractRNG = Random.default_rng(),
                                         seed::Option{<:Integer} = nothing,
                                         train_score::Bool = false,
                                         kwargs::NamedTuple = (;))
    return RandomisedSearchCrossValidation(p, cv, r, score, ex, n_iter, rng, seed,
                                           train_score, kwargs)
end
function make_p_grid(p::PairGSCV, n_iter::Integer, rng::Random.AbstractRNG,
                     replace::Bool = false)
    vals = StatsBase.sample(rng, p[2], min(length(p[2]), n_iter); replace = replace)
    return Pair(p[1], vals)
end
function make_p_grid(p::Pair{<:AbstractString, <:Any}, n_iter::Integer,
                     rng::Random.AbstractRNG, args...)
    vals = rand(rng, p[2], n_iter)
    return Pair(p[1], vals)
end
function make_p_grid(ps::AbstractVector{<:Pair{<:AbstractString, <:Any}}, n_iter::Integer,
                     rng::Random.AbstractRNG)
    replace = any(x->isa(x[2], Distributions.Distribution), ps)
    return concrete_typed_array([make_p_grid(p, n_iter, rng, replace) for p in ps])
end
function make_p_grid(ps::Dict{<:AbstractString, <:Any}, n_iter::Integer,
                     rng::Random.AbstractRNG)
    replace = any(x->isa(x[2], Distributions.Distribution), ps)
    return Dict(make_p_grid(key => val, n_iter, rng, replace) for (key, val) in ps)
end
function make_p_grid(pss::AbstractVector{<:Union{<:Dict{<:AbstractString, <:Any},
                                                 <:AbstractVector{<:Pair{<:AbstractString,
                                                                         <:Any}}}},
                     n_iter::Integer, rng::Random.AbstractRNG)
    return concrete_typed_array([make_p_grid(ps, n_iter, rng) for ps in pss])
end
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 rscv::RandomisedSearchCrossValidation, rd::ReturnsResult)
    if !isnothing(rscv.seed)
        Random.seed!(rscv.rng, rscv.seed)
    end
    return search_cross_validation(opt,
                                   GridSearchCrossValidation(make_p_grid(rscv.p,
                                                                         rscv.n_iter,
                                                                         rscv.rng);
                                                             cv = rscv.cv, r = rscv.r,
                                                             score = rscv.score,
                                                             ex = rscv.ex,
                                                             train_score = rscv.train_score,
                                                             kwargs = rscv.kwargs), rd)
end

export RandomisedSearchCrossValidation
