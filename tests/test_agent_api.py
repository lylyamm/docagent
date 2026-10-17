"""The agent over HTTP (POST /v1/agent, GET /v1/agent/{id}, POST .../resume).

Scripted models, fake translator, no network: the same set-up as test_supervisor,
behind the API. A run goes to a worker thread, so the tests poll its status.
"""

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from test_agent_tools import make_pdf, settings
from test_answer import chat, searcher
from test_supervisor import ANSWER, CORPUS, FAITHFUL, POINT

from docagent.agent import Toolbox
from docagent.agent.runs import AgentRuns
from docagent.api.app import create_app
from docagent.rag import Answerer
from docagent.translate import FakeTranslator

TWO_STEPS = {
    "steps": [
        {"route": "summarize", "doc_ids": ["d2"]},
        {"route": "question", "question": "Quelle part des émissions pour les transports ?"},
    ]
}


def api(tmp_path: Path, *supervisors, answer=(), tool=(), factory=None):
    """The API with an agent whose supervisor replies are scripted, one list per run."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
    agent_settings = settings(
        corpus_path=corpus,
        agent_confirm_pages=2,
        rag_raw_dir=tmp_path,
        agent_trajectories_dir=tmp_path / "trajectories",
    )
    answer_chat, _ = chat(*answer)
    tool_chat, _ = chat(*tool)
    tools = Toolbox(
        agent_settings,
        answerer=Answerer(searcher(), answer_chat, top_k=1),
        make_translator=FakeTranslator,
        chat=tool_chat,
        output_dir=tmp_path / "out",
    )
    scripts = [chat(*replies)[0] for replies in supervisors]  # one client per run
    made = []

    def make_chat():
        made.append(scripts[len(made)])
        return made[-1]

    runs = AgentRuns(tools, agent_settings, make_chat)
    app_settings = settings(jobs_dir=tmp_path / "jobs", translator="fake")
    app = create_app(app_settings, agent_factory=factory or (lambda _get: runs))
    return TestClient(app), made


def wait(client: TestClient, run_id: str, until=("done", "failed", "needs_confirmation")):
    for _ in range(200):
        run = client.get(f"/v1/agent/{run_id}").json()
        if run["status"] in until:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run still {run['status']}")


def test_a_request_runs_in_the_background_and_ends_with_its_message(tmp_path):
    client, made = api(
        tmp_path,
        [TWO_STEPS, FAITHFUL, FAITHFUL],
        answer=[ANSWER],
        tool=[{"points": [POINT]}],
    )
    response = client.post("/v1/agent", json={"request": "Résume le rapport 2 puis la part"})
    assert response.status_code == 202
    started = response.json()
    assert started["status"] in ("queued", "running") and started["message"] is None

    run = wait(client, started["run_id"])
    assert run["status"] == "done"
    assert [p["route"] for p in run["plans"]] == ["summarize", "question"]
    assert len(run["sections"]) == 2 and run["sections"][0].startswith("Résumé de Report 2")
    assert run["message"].startswith("Étape 1/2 : résumé")
    assert [n["node"] for n in run["nodes"]][:3] == ["understand", "summarize", "judge"]
    assert run["supervisor_calls"] == 3 and len(made) == 1
    assert Path(run["trajectory_file"]).exists()


def test_a_pause_waits_for_the_answer_then_continues(tmp_path):
    path = str(make_pdf(tmp_path / "big.pdf", pages=3))  # more than AGENT_CONFIRM_PAGES
    client, _ = api(tmp_path, [{"route": "translate", "path": path}])
    run_id = client.post("/v1/agent", json={"request": f"Traduis {path}"}).json()["run_id"]

    run = wait(client, run_id)
    assert run["status"] == "needs_confirmation"
    assert run["question"].startswith("Ce PDF fait 3 pages")

    resumed = client.post(f"/v1/agent/{run_id}/resume", json={"answer": "oui"})
    assert resumed.status_code == 202 and resumed.json()["question"] is None
    run = wait(client, run_id, until=("done", "failed"))
    assert run["status"] == "done" and run["message"].startswith("Traduction terminée")
    assert (tmp_path / "out" / "big_fr.pdf").exists()


def test_each_request_has_its_own_supervisor_client(tmp_path):
    out = {"route": "out_of_scope"}
    client, made = api(tmp_path, [out], [out])
    first = client.post("/v1/agent", json={"request": "Quel temps demain ?"}).json()
    second = client.post("/v1/agent", json={"request": "Et après-demain ?"}).json()
    for run in (first, second):
        assert wait(client, run["run_id"])["supervisor_calls"] == 1
    assert len(made) == 2 and made[0] is not made[1]  # the cost guard counts per request


def test_wrong_calls_get_clear_errors(tmp_path):
    client, _ = api(tmp_path, [{"route": "out_of_scope"}])
    assert client.get("/v1/agent/nope").status_code == 404
    assert client.post("/v1/agent/nope/resume", json={"answer": "oui"}).status_code == 404

    run_id = client.post("/v1/agent", json={"request": "Quel temps demain ?"}).json()["run_id"]
    wait(client, run_id)
    late = client.post(f"/v1/agent/{run_id}/resume", json={"answer": "oui"})
    assert late.status_code == 409 and "asks nothing" in late.json()["detail"]
    assert client.post("/v1/agent", json={"request": ""}).status_code == 422


def test_an_agent_that_cannot_start_answers_503(tmp_path):
    def broken(_get):
        raise RuntimeError("no Google sign-in found")

    client, _ = api(tmp_path, factory=broken)
    response = client.post("/v1/agent", json={"request": "Résume le rapport 2"})
    assert response.status_code == 503 and "no Google sign-in" in response.json()["detail"]
