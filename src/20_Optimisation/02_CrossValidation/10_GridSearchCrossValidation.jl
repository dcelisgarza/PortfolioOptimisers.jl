struct GridSearchCrossValidation{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractSearchCrossValidationEstimator
    p::T1
    cv::T2
    r::T3
    score::T4
    ex::T5
    train_score::T6
    kwargs::T7
    function GridSearchCrossValidation(p::MultiGSCVValType_VecMultiGSCVValType,
                                       cv::SearchCV, r::AbstractBaseRiskMeasure,
                                       score::CrossValSearchScorer,
                                       ex::FLoops.Transducers.Executor, train_score::Bool,
                                       kwargs::NamedTuple)
        @argcheck(!isempty(p), IsEmptyError)
        if isa(p, VecMultiGSCVValType)
            @argcheck(all(!isempty, p), IsEmptyError)
        end
        return new{typeof(p), typeof(cv), typeof(r), typeof(score), typeof(ex),
                   typeof(train_score), typeof(kwargs)}(p, cv, r, score, ex, train_score,
                                                        kwargs)
    end
end
function GridSearchCrossValidation(p::MultiGSCVValType_VecMultiGSCVValType;
                                   cv::SearchCV = KFold(),
                                   r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   score::CrossValSearchScorer = HighestMeanScore(),
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   train_score::Bool = false, kwargs::NamedTuple = (;))
    return GridSearchCrossValidation(p, cv, r, score, ex, train_score, kwargs)
end
function lens_val_grid(estval::AbstractVector{<:Pair{<:String, <:AbstractVector}})
    vals = vec(collect(Iterators.product(map(x -> x[2], estval)...)))
    lenses = fill(map(x -> parse_lens(x[1]), estval), length(vals))
    return lenses, vals
end
function lens_val_grid(estval::AbstractDict{<:String, <:AbstractVector})
    vals = vec(collect(Iterators.product(values(estval)...)))
    lenses = fill(map(x -> parse_lens(x), keys(estval)), length(vals))
    return lenses, vals
end
function lens_val_grid(estvals::AbstractVector{<:Union{<:AbstractVector{<:Pair{<:String,
                                                                               <:AbstractVector}},
                                                       <:AbstractDict{<:String,
                                                                      <:AbstractVector}}})
    lenses_vals = [lens_val_grid(estval) for estval in estvals]
    lenses = mapreduce(x -> x[1], vcat, lenses_vals)
    vals = mapreduce(x -> x[2], vcat, lenses_vals)
    return lenses, vals
end
function search_cross_validation(opt::NonFiniteAllocationOptimisationEstimator,
                                 gscv::GridSearchCrossValidation, rd::ReturnsResult)
    p = gscv.p
    lens_grid, val_grid = lens_val_grid(p)
    cv = split(gscv.cv, rd)
    @argcheck(isa(cv.test_idx[1], VecInt))
    N = length(val_grid)
    M = length(cv.train_idx)
    test_scores = Matrix{eltype(rd.X)}(undef, M, N)
    train_scores = if gscv.train_score
        Matrix{eltype(rd.X)}(undef, M, N)
    else
        nothing
    end
    let opt = opt, test_scores = test_scores, train_scores = train_scores
        FLoops.@floop gscv.ex for (i, (lenses, vals)) in enumerate(zip(lens_grid, val_grid))
            local opti = opt
            for (lens, val) in zip(lenses, vals)
                opti = Accessors.set(opti, lens, val)
            end
            for (j, (train_idx, test_idx)) in enumerate(zip(cv.train_idx, cv.test_idx))
                test_score, train_score = fit_and_score(opti, gscv, rd, train_idx, test_idx)
                test_scores[j, i] = test_score
                if gscv.train_score
                    train_scores[j, i] = train_score
                end
            end
        end
    end
    opt_idx = gscv.score(test_scores)
    opt_lens = lens_grid[opt_idx]
    opt_vals = val_grid[opt_idx]
    for (lens, val) in zip(opt_lens, opt_vals)
        opt = Accessors.set(opt, lens, val)
    end
    return SearchCrossValidationResult(opt, test_scores, train_scores, lens_grid, val_grid,
                                       opt_idx)
end
export search_cross_validation, GridSearchCrossValidation
