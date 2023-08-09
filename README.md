# multireader_metrics
For computing the multi-reader generalizations of the Jaccard and Sorensen Indices for object detection and instance segmentation


*overlays_to_npy.py*
converts ImageJ overlays (manual segmentations) to .npy label matrix

*calculate_reader_agreement.py* 
computes pairwise comparisons between readers and the multi-reader generalizations described in 'Generalizations of the Jaccard Index and Sørensen Index for assessing agreement across multiple readers in object detection and instance segmentation in biomedical imaging' (Durkee et. al. 2023)

*agreement_plots_and_stats...*
these scripts compute the statistical comparisons and generate plots for multi-reader comparisons

*CP_vs_consensus.py*
computes the performance of Cellpose2.0 relative to the manual annotations provided by multiple human readers. Consensus is defined by N readers agreeing on a single object
