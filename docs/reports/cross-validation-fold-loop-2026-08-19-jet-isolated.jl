using PortfolioOptimisers, StableRNGs, LinearAlgebra, JET, Printf
BLAS.set_num_threads(1)
const PO = PortfolioOptimisers
const LBL = get(ENV, "PO_LBL", "?")
const SCHEME = get(ENV, "PO_CV", "KFold")
rd = ReturnsResult(; nx = ["A$i" for i in 1:20],
                   X = randn(StableRNG(987654321), 1008, 20) ./ 100)
cv = if SCHEME == "KFold"
    KFold(; n = 10)
elseif SCHEME == "WalkForward"
    IndexWalkForward(127, 171)
elseif SCHEME == "Combinatorial"
    CombinatorialCrossValidation(; n_folds = 6, n_test_folds = 2, purged_size = 5,
                                 embargo_size = 3)
else
    MultipleRandomised(IndexWalkForward(127, 171); subset_size = 5, n_subsets = 3,
                       rng = StableRNG(666), seed = 69)
end
o = EqualWeighted()
rep = JET.report_opt(cross_val_predict, Tuple{typeof(o), typeof(rd), typeof(cv)};
                     target_modules = (PO,))
d = r_ = ot = ld = 0
for r in JET.get_reports(rep)
    msg = sprint(io -> JET.print_report_message(io, r))
    frame = isempty(r.vst) ? "" : string(last(r.vst).linfo)
    inloop = occursin("_folds", frame) || occursin("fold_loop", frame)
    if occursin("runtime dispatch", msg)
        global d += 1
        inloop && (global ld += 1)
    elseif occursin("recursion", msg)
        global r_ += 1
    else
        global ot += 1
    end
end
rc = length(JET.get_reports(JET.report_call(cross_val_predict,
                                            Tuple{typeof(o), typeof(rd), typeof(cv)};
                                            target_modules = (PO,))))
@printf("ISOLATED %s %s dispatch=%d recursion=%d other=%d loopdisp=%d callerr=%d\n", LBL,
        SCHEME, d, r_, ot, ld, rc)
