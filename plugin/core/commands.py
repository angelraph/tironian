"""``/tironian`` slash command (CLI, TUI, Desktop, Telegram, Discord, …) and shared text views."""

from __future__ import annotations

import json
from typing import Any, Callable

from . import jobs, pipeline
from .app import App

HELP = """Tironian: your own small model, learned from your own work.

/tironian                 status: what's learned, what's served locally, savings
/tironian report          the report on your history (recurring tasks, spend)
/tironian families        task families and what each needs
/tironian good [note]     this answer was right (teaches Tironian what to learn)
/tironian bad [note]      this answer was wrong (never trained on; demotes a student)
/tironian off | on        stop / resume learning from this conversation
/tironian train <family>  build the dataset and show the cost before training
/tironian shadow|promote|demote <family>
/tironian why             why the last request did or didn't go to your student"""


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _usd(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 or x == 0 else f"${x:.4f}"


def status_text(app: App) -> str:
    s = app.engine.summary()
    st = app.settings()
    job = jobs.active_job(app.store)
    lines = [
        f"Learning from {s['turns']} turns ({s['good']} good, {s['bad']} bad) · {s['trainable']} trainable steps"
        f" · {s['redactions'] or 0} values redacted",
        f"Students: {s['families_live']} live, {s['families_shadow']} in shadow · routing "
        f"{'on' if st.routing_enabled else 'off'}",
        f"Served locally: {s['served']} requests ({_pct(s['local_share'])} of all) · {s['fallbacks']} silent "
        f"fallbacks · saved {_usd(s['saved_usd'])}",
    ]
    if job:
        row = app.poll_job(job["id"]) or dict(job)
        lines.append(f"Training {row['family_id']}: {row['status']} {_pct(row['progress'] or 0)} · {row['message']}")
    if not s["turns"]:
        lines.append("Nothing learned yet. Use Hermes as usual; `/tironian report` reads your past sessions.")
    return "\n".join(lines)


def report_text(app: App) -> str:
    r = pipeline.report(app.store, app.settings())
    if not r.turns:
        return ("No sessions to learn from yet. Tironian reads your Hermes history read-only the first time "
                "it runs; use Hermes for a while and check back.")
    out = [
        f"Your Tironian report · {r.sessions} sessions · {r.turns} requests",
        f"{_pct(r.routine_share)} of your requests are recurring tasks ({r.routine_turns} requests).",
    ]
    if r.spend_30d_usd:
        out.append(f"Last 30 days: {_usd(r.spend_30d_usd)} model spend, {_usd(r.routine_spend_30d_usd)} of it on "
                   f"recurring tasks — that's what a personal model can take over.")
    out.append("")
    out.append("Top task families:")
    for f in r.families[:8]:
        mark = "ready to train" if f["ready_to_train"] else f"needs {f['needs']} more good examples"
        if f["status"] in ("shadow", "live"):
            mark = f"student {f['status']}"
        out.append(f"  {f['id']}  {f['name'][:48]:<48}  {f['turns']:>4} req · {f['good']:>4} good · {mark}")
    if r.trainable_families:
        out.append("")
        out.append(f"{r.trainable_families} famil{'y is' if r.trainable_families == 1 else 'ies are'} ready. "
                   f"Train one with `/tironian train <family id>`.")
    return "\n".join(out)


def families_text(app: App) -> str:
    rows = app.store.q("SELECT f.*, (SELECT gate_passed || ':' || printf('%.2f', wilson_lo) FROM evals e "
                       "WHERE e.family_id=f.id ORDER BY created_at DESC LIMIT 1) AS last_eval "
                       "FROM families f ORDER BY n_turns DESC LIMIT 20")
    if not rows:
        return "No task families yet."
    out = ["id         status      req  good  last eval    name"]
    for r in rows:
        ev = r["last_eval"] or "-"
        if ev != "-":
            ok, lo = ev.split(":")
            ev = f"{'pass' if ok == '1' else 'fail'} {float(lo):.0%}"
        out.append(f"{r['id']:<10} {r['status']:<10} {r['n_turns']:>4} {r['n_good']:>5}  {ev:<11}  {r['name'][:50]}")
    return "\n".join(out)


def handle(app: App, raw: str, *, session_id: str = "", platform: str = "", ask: Callable[..., Any] | None = None,
           **_: Any) -> str:
    parts = (raw or "").strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    try:
        if cmd in ("", "status"):
            return status_text(app)
        if cmd == "help":
            return HELP
        if cmd == "report":
            if not app.store.meta_get("first_import_done"):
                app.first_run()
                return "Reading your Hermes history (read-only)… run `/tironian report` again in a few seconds."
            pipeline.rebuild_families(app.store, app.settings())
            return report_text(app)
        if cmd == "families":
            return families_text(app)
        if cmd in ("good", "bad"):
            if not session_id:
                return "Run this inside the conversation you want to mark."
            tid = app.engine.mark(session_id, 1 if cmd == "good" else -1)
            if not tid:
                return "Nothing to mark yet in this conversation."
            return ("Noted: that answer will be learned from." if cmd == "good"
                    else "Noted: Tironian won't learn from that answer.")
        if cmd == "off":
            if session_id:
                app.engine.exclude_session(session_id, "user")
            return "Tironian won't learn from this conversation (already captured turns are excluded too)."
        if cmd == "on":
            if session_id:
                app.engine.include_session(session_id)
            return "Tironian is learning from this conversation again."
        if cmd == "why":
            return why_text(app, session_id)
        if cmd in ("shadow", "promote", "demote"):
            return set_status(app, cmd, arg)
        if cmd == "train":
            return train_flow(app, arg, ask=ask, platform=platform)
        return HELP
    except Exception as exc:  # a slash command must answer, not crash
        return f"Tironian: {exc}"


def set_status(app: App, cmd: str, family_id: str) -> str:
    f = app.store.one("SELECT * FROM families WHERE id=?", (family_id,))
    if f is None:
        return f"No family '{family_id}'. See `/tironian families`."
    if cmd == "demote":
        app.engine.set_family_status(family_id, "evaluated", reason="user")
        return f"{family_id}: student stopped; your model handles this task again."
    if not f["student_id"]:
        return f"{family_id} has no evaluated student yet."
    last = app.store.one("SELECT gate_passed FROM evals WHERE family_id=? AND student_id=? ORDER BY created_at DESC LIMIT 1",
                         (family_id, f["student_id"]))
    if not last or not last["gate_passed"]:
        return f"{family_id}: the student hasn't passed its held-out test, so it can't serve requests."
    if cmd == "shadow":
        app.engine.set_family_status(family_id, "shadow", reason="user")
        return f"{family_id}: shadow mode. The student answers in the background; your model still replies."
    ok, why = pipeline.shadow_ready(app.store, family_id, threshold=app.settings().promotion_threshold)
    if not ok and f["status"] != "live":
        return f"{family_id}: not yet — {why}. It stays in shadow until then."
    app.engine.set_family_status(family_id, "live", reason="user")
    hint = "" if app.settings().routing_enabled else " Turn on routing in Settings → Plugins → Tironian."
    return f"{family_id}: live. Matching requests are answered by your student, checked, with silent fallback.{hint}"


def why_text(app: App, session_id: str) -> str:
    r = app.store.one("SELECT r.* FROM routes r JOIN turns t ON t.id=r.turn_id WHERE t.session_id=? "
                      "ORDER BY r.id DESC LIMIT 1", (session_id,)) if session_id else None
    if r is None:
        st = app.settings()
        if not st.routing_enabled:
            return "Routing is off, so every request went to your model."
        return "The last request didn't match any promoted task family closely enough, so your model answered."
    labels = {"served": "answered by your student", "fb_invalid": "student's answer failed validation → your model answered",
              "fb_error": "student unreachable → your model answered", "fb_wrong_model": "wrong model responded → your model answered",
              "fb_unsupported": "provider format not supported for live routing → your model answered",
              "shadow_agree": "shadow: student agreed with your model", "shadow_disagree": "shadow: student disagreed",
              "shadow_error": "shadow: student unreachable"}
    return f"{r['family_id']}: {labels.get(r['outcome'], r['outcome'])} ({r['latency_ms']} ms)."


def train_flow(app: App, family_id: str, *, ask: Callable[..., Any] | None = None, platform: str = "") -> str:
    words = family_id.split()
    confirmed = "confirm" in words
    family_id = words[0] if words else ""
    if not family_id:
        return "Which family? See `/tironian families`."
    if jobs.active_job(app.store):
        return "A training job is already running: `/tironian` shows its progress."
    prep = app.prepare_training(family_id)
    if not prep["enough"]:
        return (f"{family_id}: only {prep['n_train']} training / {prep['n_heldout']} held-out examples. "
                f"Use Hermes for this task a little more (skipped: {json.dumps(prep['skipped'])}).")
    cost = "free (your GPU)" if prep["backend"] == "local" else f"about ${prep['estimate_usd']:.2f} on a Modal {prep['gpu']}"
    summary = (f"{family_id}: {prep['n_train']} examples to learn from, {prep['n_heldout']} held out for the test. "
               f"Training takes ~{prep['estimate_s'] // 60} min, {cost}.")
    if not confirmed and ask is not None:
        choice = ask(summary + " Start training?", [{"value": "yes", "label": "Start training"},
                                                    {"value": "no", "label": "Not now"}])
        confirmed = choice == "yes"
        if choice is None:
            return summary + f" Run `/tironian train {family_id} confirm` to start."
    if not confirmed:
        return summary + f" Run `/tironian train {family_id} confirm` to start." if ask is None else "Okay, not now."
    jid = app.start_training(family_id, prep["dataset_id"])
    return f"Training started ({jid}). `/tironian` shows progress; you'll get the scorecard when it's done."
