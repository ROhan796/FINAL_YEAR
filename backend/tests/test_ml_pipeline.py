from __future__ import annotations

import numpy as np


def test_matryoshka_two_stage_rank() -> None:
    from ml.encoders.matryoshka import truncate_and_normalize, two_stage_matryoshka_rank

    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((50, 768)).astype(np.float32)
    ids = [f"p-{i}" for i in range(50)]
    q = corpus[7].copy()

    trunc = truncate_and_normalize(q, 128)
    assert trunc.shape == (128,)
    assert np.isclose(np.linalg.norm(trunc), 1.0, atol=1e-4)

    ranked = two_stage_matryoshka_rank(q, ids, corpus, coarse_dim=128, shortlist_k=20, top_k=5)
    assert ranked[0][0] == "p-7"


def test_pytorch_gnn_forward_and_train() -> None:
    from ml.gnn.dataset import GraphDataset
    from ml.gnn.model import CKGAI_GNN
    from ml.gnn.train import GNNTrainer

    graph = GraphDataset(None).load_graph(feature_dim=64)
    model = CKGAI_GNN(in_dim=64, hidden_dim=32, out_dim=32, num_layers=2, gat_heads=4, use_virtual_node=True)
    trainer = GNNTrainer(model, total_epochs=5)

    loss_1 = trainer.train_epoch(
        graph.node_features,
        graph.adj_by_relation,
        graph.edge_src,
        graph.edge_dst,
        graph.edge_types,
        len(graph.node_ids),
    )
    assert loss_1 >= 0.0

    embs = model.predict_embeddings(graph.node_features, graph.adj_by_relation)
    assert embs.shape == (len(graph.node_ids), 32)


def test_citation_polarity_and_platt_calibration() -> None:
    from ml.discovery.calibration import PlattCalibrator
    from ml.relation.citation_polarity import CitationPolarityClassifier

    clf = CitationPolarityClassifier()
    sup = clf.classify("Our experiments confirm and support the Transformer scaling findings.")
    dis = clf.classify("Contrary to prior work, this method fails to replicate and disputes the claim.")
    assert sup.label == "supports"
    assert dis.label == "disputes"

    cal = PlattCalibrator()
    cal.fit([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], [True, False, True, False])
    assert cal.predict_prob(0.9, cross_domain=True) > cal.predict_prob(0.1, cross_domain=False)
