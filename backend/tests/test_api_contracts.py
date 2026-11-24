from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_meta(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    meta = client.get("/api/v1/meta")
    assert meta.status_code == 200
    assert "hybrid" in meta.json()["searchModes"]


def test_auth_dev_login_and_me(client: TestClient) -> None:
    res = client.post("/api/v1/auth/dev-login")
    assert res.status_code == 200
    data = res.json()
    assert "accessToken" in data
    assert data["user"]["email"]

    me = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {data['accessToken']}"})
    assert me.status_code == 200
    assert me.json()["id"] == data["user"]["id"]


def test_papers_search_and_explainability(client: TestClient) -> None:
    papers = client.get("/api/v1/papers")
    assert papers.status_code == 200
    items = papers.json()["items"]
    assert len(items) >= 5

    sr = client.post("/api/v1/search", json={"query": "transformer attention", "mode": "hybrid", "limit": 5})
    assert sr.status_code == 200
    body = sr.json()
    assert len(body["results"]) >= 1
    assert "relevanceSignal" in body["results"][0]
    assert "reasoningEvidence" in body["results"][0]

    pid = body["results"][0]["id"]
    exp = client.get(f"/api/v1/search/explain/{pid}", params={"q": "transformer"})
    assert exp.status_code == 200
    assert exp.json()["paperId"] == pid


def test_legacy_ai_contracts(client: TestClient) -> None:
    s = client.post("/api/summarize", json={"title": "Attention", "abstract": "Self-attention mechanisms."})
    assert s.status_code == 200
    assert "summary" in s.json() and "executiveSummary" in s.json()

    ins = client.post("/api/generate-insights", json={"title": "Attention", "abstract": "Self-attention."})
    assert ins.status_code == 200
    assert "keyContributions" in ins.json()

    draft = client.post("/api/analyze-draft", json={"title": "Novel GNN", "abstract": "Graph learning."})
    assert draft.status_code == 200
    assert "aiReview" in draft.json()

    chat = client.post("/api/chat-assistant", json={"message": "Summarize baselines"})
    assert chat.status_code == 200
    assert "reply" in chat.json()


def test_workspace_rag_stream_and_reading_room(client: TestClient) -> None:
    proj_res = client.post(
        "/api/v1/projects",
        json={"name": "Test GNN Project", "description": "Cross-domain GNN", "field": "CS"},
    )
    assert proj_res.status_code == 201
    pid = proj_res.json()["id"]

    ann = client.post(
        f"/api/v1/projects/{pid}/reading-room",
        json={
            "paperId": "paper-101",
            "sentenceIndex": 0,
            "highlightText": "The dominant sequence transduction models",
            "comment": "Key motivation sentence.",
        },
    )
    assert ann.status_code == 201
    ann_id = ann.json()["id"]

    listed = client.get(f"/api/v1/projects/{pid}/reading-room", params={"paperId": "paper-101"})
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    stream_res = client.post(f"/api/v1/projects/{pid}/ai-thread/stream", json={"message": "What is this about?"})
    assert stream_res.status_code == 200
    assert "data:" in stream_res.text

    del_res = client.delete(f"/api/v1/projects/{pid}/reading-room/{ann_id}")
    assert del_res.status_code == 200
