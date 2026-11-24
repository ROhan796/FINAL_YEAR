from __future__ import annotations

from fastapi.testclient import TestClient


def test_insights_endpoints(client: TestClient) -> None:
    gaps = client.get("/api/v1/insights/research-gaps")
    assert gaps.status_code == 200
    assert isinstance(gaps.json(), list)

    gateways = client.get("/api/v1/insights/gateway-papers")
    assert gateways.status_code == 200
    assert isinstance(gateways.json(), list)

    debt = client.get("/api/v1/insights/research-debt")
    assert debt.status_code == 200
    assert "debtCount" in debt.json()

    affinity = client.get("/api/v1/insights/affinity")
    assert affinity.status_code == 200
    assert isinstance(affinity.json(), dict)

    vel = client.get("/api/v1/users/me/velocity")
    assert vel.status_code == 200
    assert len(vel.json()["series"]) >= 5


def test_polarity_citation_gaps_and_graph(client: TestClient) -> None:
    pol = client.post(
        "/api/v1/papers/paper-101/citation-polarity",
        params={"citationSentence": "We confirm and extend the Transformer architecture."},
    )
    assert pol.status_code == 200
    assert pol.json()["label"] == "supports"

    ms = client.post(
        "/api/v1/manuscripts",
        json={
            "title": "Transformer and CRISPR-Cas9 Analysis",
            "abstract": "We apply Transformer and Attention Mechanism without citing prior work.",
            "venue": "NeurIPS",
            "tags": ["transformers"],
        },
    )
    assert ms.status_code == 201
    ms_id = ms.json()["id"]

    cg = client.post(f"/api/v1/manuscripts/{ms_id}/citation-gaps")
    assert cg.status_code == 200
    assert "gaps" in cg.json()

    exp = client.get("/api/v1/graph/exploration")
    assert exp.status_code == 200
    assert "nodes" in exp.json()

    disc = client.get("/api/v1/discoveries")
    assert disc.status_code == 200
    assert isinstance(disc.json(), list)
