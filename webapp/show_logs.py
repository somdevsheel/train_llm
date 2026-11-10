"""Print logged conversations from webapp/logs/ in a readable form.

    python webapp/show_logs.py               # today's log, grouped by conversation
    python webapp/show_logs.py --last 3      # only the last 3 conversations
    python webapp/show_logs.py --date 2026-11-10
"""
import argparse
import json
import os
import textwrap
import time
from collections import OrderedDict

LOG_DIR = os.environ.get("LLM_LOG_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    p.add_argument("--last", type=int, default=0, help="show only the last N conversations")
    a = p.parse_args()

    path = os.path.join(LOG_DIR, f"chats-{a.date}.jsonl")
    if not os.path.exists(path):
        print(f"no log for {a.date} ({path})")
        return
    convs = OrderedDict()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        convs.setdefault(r.get("conversation_id") or f"{r['mode']}-{r['time']}", []).append(r)
    items = list(convs.items())[-a.last:] if a.last else list(convs.items())

    wrap = lambda s, ind: textwrap.indent(s.strip() or "(empty)", ind)  # noqa: E731
    for cid, rows in items:
        first = rows[0]
        print("=" * 78)
        print(f"{first['time']}  {first['mode']}  model={first['model']}  id={cid[:8]}  turns={len(rows)}")
        if first.get("system"):
            print(f"system: {first['system']}")
        for r in rows:
            s = r["settings"]
            how = "greedy" if s["greedy"] else f"temp {s['temperature']}, top-k {s['top_k']}"
            print("-" * 78)
            if r["mode"] == "chat":
                user = next((m["content"] for m in reversed(r["messages_sent"]) if m["role"] == "user"), "")
                print(wrap(user, "USER  ") if not r.get("continued_from") else "USER  (continue previous reply)")
            else:
                print(wrap(r["prompt"], "PROMPT "))
            print(wrap(r["reply"], "MODEL "))
            notes = [f"{r['tokens']} tok", r["stop"], how, f"rep {s['repetition_penalty']}",
                     f"saw {len(r['messages_sent'] or [])} msgs / {r['prompt_tokens']} tok"]
            if r["dropped_messages"]:
                notes.append(f"{r['dropped_messages']} msgs dropped")
            print("      [" + ", ".join(notes) + "]")
    print("=" * 78)
    print(f"{len(items)} conversation(s) from {path}")


if __name__ == "__main__":
    main()
