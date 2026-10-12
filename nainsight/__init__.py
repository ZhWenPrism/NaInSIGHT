"""NaInSIGHT: knowledge-guided dynamic prediction of postoperative hyponatraemia.

Package layout
--------------
nainsight.data   site adapters, cohort construction, event store, landmarks, tokenisation
nainsight.graph  reference medication-effect graph, transfer to the development cohort, GNN prior
nainsight.model  NaFormer dual-scale temporal backbone, graph-prior adapter, multi-task heads
nainsight.train  losses, metrics, cluster bootstrap and the training loop
"""
__version__ = "1.0.0"
