# Direct Bubble Hierarchy Tree

```@docs
UniqueRoot
EqualRoot
MaximumDistanceSimilarity
ExponentialSimilarity
GeneralExponentialSimilarity
DBHT
LoGo
DBHTClustering
clusterise(cle::ClusteringEstimator{<:Any, <:Any, <:DBHT, <:Any},
                    X::NumMat; branchorder::Symbol = :optimal,
                    dims::Int = 1, kwargs...)
PortfolioOptimisers.DBHTRootMethod
PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
PortfolioOptimisers.InverseMatrixSparsificationAlgorithm
PortfolioOptimisers.PMFG_T2s
PortfolioOptimisers.dbht_similarity
PortfolioOptimisers.distance_wei
PortfolioOptimisers.clique3
PortfolioOptimisers.breadth
PortfolioOptimisers.FindDisjoint
PortfolioOptimisers.BuildHierarchy
PortfolioOptimisers.AdjCliq
PortfolioOptimisers.BubbleHierarchy
PortfolioOptimisers.CliqueRoot
PortfolioOptimisers.CliqHierarchyTree2s
PortfolioOptimisers.DirectHb
PortfolioOptimisers.BubbleCluster8s
PortfolioOptimisers.BubbleMember
PortfolioOptimisers.DendroConstruct
PortfolioOptimisers.LinkageFunction
PortfolioOptimisers.build_link_and_dendro
PortfolioOptimisers.HierarchyConstruct4s
PortfolioOptimisers.turn_into_Hclust_merges
PortfolioOptimisers.DBHTs
PortfolioOptimisers.jlogo!
PortfolioOptimisers.J_LoGo
PortfolioOptimisers.LoGo_dist_assert
PortfolioOptimisers.logo!
PortfolioOptimisers.matrix_processing_algorithm!(je::LoGo, pdm::Option{<:NumMat},
                                      sigma::NumMat, X::NumMat;
                                      dims::Int = 1, kwargs...)
```
