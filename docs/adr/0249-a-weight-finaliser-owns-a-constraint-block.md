---
status: accepted
---

# A weight finaliser owns a Constraint Block, and the optimiser is the one source of its bounds and names

## Context

The clustering and the naive optimisers have no solver. They took the weight bounds `wb` alone, and
a weight finaliser moved their weights into them. A caller could not apply a cardinality, a
threshold, a group constraint, a turnover ceiling or a concentration limit (ADR 0246) to such a
book. Issue #2257 asked for a finaliser with every slot of a Constraint Block (ADR 0237). The
maintainer settled its form in a grilling session on 2026-11-23 (#2258).

## Decision

1. **`JuMPWeightFinaliser` holds a block.** It gains the field `cbe` and the flat slot keywords, as
   `JuMPOptimiser` and `ProgrammeAllocationSet` have. It is not a new type, and it does not hold a
   `ProgrammeAllocationSet`. That set owns a budget of one, its `factory` replaces the reference of
   a turnover ceiling with the book of the step, and a failed solve is a Held Step. Each of these
   rules is wrong for the book of a rule.
2. **The optimiser is the one source of `wb` and of `sets`.** The constructor of the finaliser
   refuses both slots. The split of a hierarchical head, the finaliser and the result read one set
   of bounds. The bounds do not move into the finaliser, because the split of `HierarchicalRiskParity`
   reads them as an input of the rule.
3. **The owner of a block checks its names.** A slot keyed by name needs `sets`, and a block in a
   finaliser has none. So the check moves from the constructor of `ConstraintBlock` to the
   constructor of its owner: `JuMPOptimiser`, `ProgrammeAllocationSet`, or the head that holds the
   finaliser.
4. **The head gives its context.** `finalise_weight_bounds` and `opt_weight_bounds` take `pr`, `rd`,
   `sets`, `fees` and `strict` as positional arguments after the weights. A finaliser that reads no
   context takes `ctx...`. Before the reduction to the Investable Mask, the head resolves stage 1 of
   the block over its own `sets`, and the values follow the mask, as its fees do. The distance
   readers of the block join the mask of the head, and a refusal fits again with no solve lost.
   A head that fits no prior refuses a slot that reads one, in its constructor.
5. **The net budget is the sum of the input weights.** When the bounds admit a short weight and the
   block states no `sbgt`, the programme writes the row of the net budget and gives the builder
   none. The builder of a net budget with no short budget pins the long part at that budget, which
   forbids a short. A finaliser with no slots adds the bounds and the budget alone, so its book is
   the book that it gave before the block.
6. **The objective is a deviation or a Projection Geometry.** `alg` takes both families, and they
   stay two families, because `DiscreteAllocation` reads a deviation alone. The penalties of the
   block add to both. A `SquaredL2Norm` takes the square of the norm when a penalty is present,
   because a penalty added to a norm does not move its minimiser.
7. **A failed solve with a slot is an `OptimisationFailure`.** The weights are the repair of
   `IterativeWeightFinaliser`, which reads the bounds alone and can break the slot. With no slot the
   repair meets every constraint that the finaliser holds, so the test of the bounds gives the code.
   A solve that succeeds with a slot gives the success, because the solver met the rows to its own
   tolerance, as for a JuMP head.

## Consequences

- A user subtype of `WeightFinaliser` takes the context arguments, and the positional constructor
  of `JuMPWeightFinaliser` takes `cbe` after `slv`. `docs/src/next_release/weight-finaliser-block.md` names both.
- The finaliser loads after the block and the geometries, in `17_Optimisation/12_JuMPWeightFinaliser.jl`.
- The risk ceilings `r` and the return floor `ret` of a `ProgrammeAllocationSet` are not slots of
  the block, so the finaliser has neither.
- The finaliser tracks a risk as the owner of a risk constraint, so `RiskConstraintOwner` holds
  `WeightFinaliser` beside the programme Allocation Sets, through `ProgrammeBlockOwner`.
- The block builder adds the fees of the owner before the tracking errors. Before, the tracking
  error of a JuMP head registered gross net returns, and every risk measure of the head read them.
