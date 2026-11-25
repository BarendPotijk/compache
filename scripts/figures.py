#!/usr/bin/env python3
"""Draw the figures under docs/assets from the data files in docs/assets/data.

    python scripts/figures.py              # every figure
    python scripts/figures.py breaks thread   # some of: headline, requests, breaks, agents, sessions, thread, openclaw, preview, live, compare, workflow

Every figure paints its own surface and carries a light and a dark palette, so it
reads on either GitHub theme. The palette, typeface and marks are the calculator's.

Where the data comes from:
    policy-grid.json   python benchmarks/docchat/run.py grid --data <fast> <mixed> <slow> --output ...
    thread.json        python benchmarks/docchat/run.py thread --data <build> --thread <id> --recipe ... --output ...
    openclaw.json      node scripts/openclaw_comparison.mjs --json docs/assets/data/openclaw.json
    live.json          python benchmarks/live/run.py report --work <folder> --figure docs/assets/data/live.json
    answers.json       the graded answers on benchmarks/live/README.md
    requests.json      python scripts/request_sizes.py
    agents.json        python benchmarks/agents/public.py price <runs> --output ...
    agents-live.json   python benchmarks/agents/run.py report --work <folders> --figure ..., with its headings
    compare.json       python benchmarks/compare/run.py price --data <fast> <mixed> <slow> --sizes ... --output ...
"""

from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DATA = ASSETS / "data"

STYLE = """
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px;fill:#0b0b0b}
.surface{fill:#fcfcfb;stroke:rgba(11,11,11,.1)}
.t2{fill:#52514e}.tm{fill:#898781}.b{font-weight:600}.h{font-size:13px;font-weight:600}.big{font-size:15px;font-weight:600}
.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}.box{fill:none;stroke:#c3c2b7}
.c{fill:#2a78d6}.cs{stroke:#2a78d6}.n{fill:#898781}.ns{stroke:#898781}.o{fill:#eb6834}.os{stroke:#eb6834}
.cell{fill:#e1e0d9}.ring{stroke:#fcfcfb;stroke-width:2}.line{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.arc{fill:none;stroke-width:1.75;stroke-linecap:round}
@media (prefers-color-scheme:dark){
text{fill:#f0efec}.surface{fill:#1a1a19;stroke:rgba(255,255,255,.1)}.t2{fill:#c3c2b7}
.grid{stroke:#2c2c2a}.axis{stroke:#383835}.box{stroke:#52514e}
.c{fill:#3987e5}.cs{stroke:#3987e5}.o{fill:#d95926}.os{stroke:#d95926}
.cell{fill:#383835}.ring{stroke:#1a1a19}
}
"""


def svg(width, height, title, description, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">\n'
        f'<title id="t">{escape(title)}</title>\n<desc id="d">{escape(description)}</desc>\n'
        f"<style>{STYLE}</style>\n"
        f'<rect class="surface" x=".5" y=".5" width="{width - 1}" height="{height - 1}" rx="10"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def text(x, y, content, cls="", anchor="start"):
    cls = f' class="{cls}"' if cls else ""
    anchor = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{cls}{anchor}>{escape(str(content))}</text>'


def dollars(value):
    return f"${value:,.2f}" if value < 10 else f"${value:,.0f}"


def away(seconds):
    return f"{seconds} seconds" if seconds < 120 else f"{seconds // 60} minutes"


# -- the headline -----------------------------------------------------------


def headline(data):
    """Two panels, a chatbot and a coding agent: the bill without and with compache, and that the work held up."""
    answers = json.loads((DATA / "answers.json").read_text("utf-8"))
    ways = [w for w in data["ways"] if answers["model"] in w["detail"]]

    def mean(costs):
        # In decimals: a float mean lands on either side of half a cent, by Python version.
        costs = [Decimal(str(cost)) for cost in costs]
        return sum(costs) / len(costs)

    def billed(arm):
        return mean(
            a["cost"] for w in ways for a in w["arms"] if a["name"].startswith(arm)
        )

    task = json.loads((DATA / "agents-live.json").read_text("utf-8"))["panels"][0]

    def spent(arm):
        return mean(r["curve"][-1] for r in task["runs"] if r["arm"] == arm)

    def passed(arm):
        runs = [r for r in task["runs"] if r["arm"] == arm]
        return f"{min(r['right'] for r in runs)} of {runs[0]['of']}"

    right = (answers["no_policy"], answers["past_traces"])
    panels = [
        (
            "Chatbot",
            "1,000 messages to a document chat",
            billed("No policy"),
            billed("Fitted on past traces"),
            f"{right[0]} answers right without, {right[1]} with, of {answers['turns']}",
        ),
        (
            "Coding agent",
            f"a {len(task['runs'][0]['curve'])}-prompt task in Claude Code",
            spent("none"),
            spent("compache"),
            f"{passed('none')} tests passed without, {passed('compache')} with",
        ),
    ]
    width, height, gap = 880, 226, 40
    panel = (width - 48 - gap) / 2
    body = ["<style>.huge{font-size:34px;font-weight:650;letter-spacing:-1px}</style>"]
    said = []
    for i, (name, what, before, after, held) in enumerate(panels):
        x0 = 24 + i * (panel + gap)
        was, now = (
            f"${v.quantize(Decimal('0.01'), ROUND_HALF_UP)}" for v in (before, after)
        )
        body += [
            text(x0, 36, name, "big"),
            text(x0 + 9 * len(name) + 8, 36, what, "t2"),
            text(x0, 86, f"{1 - after / before:.0%} less", "huge"),
        ]
        if i:
            body.append(
                f'<line class="grid" x1="{x0 - gap / 2}" x2="{x0 - gap / 2}" y1="24" y2="{height - 24}"/>'
            )
        bars = [("n", before, was, "Without"), ("c", after, now, "With compache")]
        for k, (cls, value, label, side) in enumerate(bars):
            y, length = 110 + 36 * k, float(value / before) * (panel - 160)
            body += [
                text(x0, y + 16, side, "b" if k else "t2"),
                f'<path class="{cls}" d="M{x0 + 100:.1f},{y} h{length - 4:.1f} a4,4 0 0 1 4,4 v16 a4,4 0 0 1 -4,4 '
                f'h-{length - 4:.1f} z"/>',
                text(x0 + 100 + length + 9, y + 16.5, label, "h" if k else "t2"),
            ]
        body.append(text(x0, 204, held, "t2"))
        said.append(f"{name}, {what}: {was} without, {now} with, {held}")
    return svg(
        width,
        height,
        "The same work for a lower bill",
        f"Billed by the provider on {answers['model']}. {'. '.join(said)}.",
        body,
    )


# -- what one conversation sends ----------------------------------------------


def requests(data):
    """One column per message: the whole history in grey, what compache sends of it in blue."""
    turns = data["turns"]
    width, height = 880, 330
    left, right, top, bottom = 24, 88, 84, 50
    span = (width - left - right) / len(turns)
    plot, base = height - top - bottom, height - bottom
    ceiling = max(t["full"] for t in turns)
    thick = min(22, span * 0.7)

    def y(value):
        return base - value / ceiling * plot

    def column(cls, x0, value):
        h = value / ceiling * plot
        r = min(4, h)
        return (
            f'<path class="{cls}" d="M{x0:.1f},{base} v-{h - r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},-{r:.1f} '
            f'h{thick - 2 * r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} v{h - r:.1f} z"/>'
        )

    last = turns[-1]
    full, sent = (f"{last[key] / 1000:.0f}k" for key in ("full", "sent"))
    title = (
        f"By message {len(turns)}, your app sends {full} tokens. compache sends {sent}."
    )
    body = [
        text(24, 32, title, "big"),
        text(24, 51, "Tokens sent with each message of one chat.", "t2"),
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>',
    ]
    for i, t in enumerate(turns):
        middle = left + (i + 0.5) * span
        x0, shown = middle - thick / 2, min(t["sent"], t["full"])
        if shown < t["full"]:
            body.append(column("cell", x0, t["full"]))
        body.append(column("c", x0, shown))
        if t["turn"] == 1 or t["turn"] % 5 == 0:
            body.append(text(middle, base + 20, t["turn"], "tm", "middle"))
    end = width - right + 6
    body += [
        text(end, y(last["full"]) + 10, "everything", "t2"),
        text(end, y(last["sent"]) + 10, "compache", "h"),
        text(left + (width - left - right) / 2, base + 38, "Message", "t2", "middle"),
    ]
    return svg(
        width,
        height,
        title,
        f"A {len(turns)}-message chat. The whole history grows to {last['full']:,} tokens with the last message, "
        f"compache sends {last['sent']:,} there and never more than {max(t['sent'] for t in turns):,}.",
        body,
    )


# -- how long users are away ---------------------------------------------------


def grouped(title, subtitle, legend, groups, description, size):
    """A panel of columns per group, the saving beside them and what was picked under them."""
    width, height = 880, 400
    left, right, top, bottom, gap = 24, 24, 112, 52, 20
    panel = (width - left - right - gap * (len(groups) - 1)) / len(groups)
    plot, thick, space = height - top - bottom, 46, 14
    base = top + plot
    highest = max(value for group in groups for _, value in group["columns"])
    body = [
        "<style>.n2{fill:#c3c2b7}@media (prefers-color-scheme:dark){.n2{fill:#5a5954}}</style>",
        text(24, 32, title, "big"),
        text(24, 51, subtitle, "t2"),
    ]
    at = 24
    for cls, label in legend:
        body.append(
            f'<rect class="{cls}" x="{at}" y="66" width="10" height="10" rx="2"/>'
        )
        body.append(text(at + 16, 75.5, label, "t2"))
        at += 6.2 * len(label) + 44
    for i, group in enumerate(groups):
        x0, columns = left + i * (panel + gap), group["columns"]
        wide = len(columns) * thick + (len(columns) - 1) * space
        body += [
            text(x0, top - 4, group["heading"], "h"),
            f'<line class="axis" x1="{x0:.1f}" x2="{x0 + wide:.1f}" y1="{base}" y2="{base}"/>',
        ]
        # The tallest column leaves room for its own number under the heading.
        scale = (plot - 40) / highest
        for k, (cls, value) in enumerate(columns):
            x, tall = x0 + k * (thick + space), value * scale
            body.append(
                f'<path class="{cls}" d="M{x:.1f},{base} v-{tall - 4:.1f} a4,4 0 0 1 4,-4 h{thick - 8} '
                f'a4,4 0 0 1 4,4 v{tall - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x + thick / 2,
                    base - tall - 8,
                    size(value),
                    "h" if cls == "c" else "t2",
                    "middle",
                )
            )
        # The saving stands beside the column it belongs to.
        beside = x0 + wide + space
        for row, (saved, against) in enumerate(group["beside"]):
            body += [
                text(beside, base - 56 + 36 * row, saved, "h"),
                text(beside, base - 41 + 36 * row, against, "tm"),
            ]
        body += [
            text(x0, base + 20, "compache picked", "tm"),
            text(x0, base + 36, group["picked"], "t2"),
        ]
    return svg(width, height, title, description, body)


def breaks(data):
    """Three groups of columns, one per return pattern: sending everything on each cache, and compache."""
    clocks = list(data["clocks"].values())

    def everything(clock, seconds):
        return next(
            r["dollars_per_1000_turns"]
            for r in clock["candidates"]
            if r["recipe"] is None
            and r["max_pings"] == 0
            and r["cache_ttl_seconds"] == seconds
        )

    groups, summary = [], []
    for clock in clocks:
        fitted = next(r for r in clock["candidates"] if r["fitted"])
        ours = fitted["dollars_per_1000_turns"]
        short, long = everything(clock, 300), everything(clock, 3600)
        pings = fitted["max_pings"]
        cache = "1-hour" if fitted["cache_ttl_seconds"] == 3600 else "5-minute"
        chose = f"{cache} cache, " + (f"up to {pings} pings" if pings else "no pings")
        groups.append(
            {
                "heading": f"Users back after {away(clock['median_idle_seconds'])}",
                "columns": [("n", short), ("n2", long), ("c", ours)],
                "beside": [
                    (f"{1 - ours / short:.0%} less", "than 5-minute"),
                    (f"{1 - ours / long:.0%} less", "than 1-hour"),
                ],
                "picked": chose,
            }
        )
        summary.append(
            f"Users back after {away(clock['median_idle_seconds'])}: everything {dollars(short)} on the 5-minute cache, "
            f"{dollars(long)} on the 1-hour cache, compache {dollars(ours)} with {chose}."
        )
    return grouped(
        "The longer users are away, the more compache saves",
        f"Cost per 1,000 messages. Same document chats, {data['model_name']}, "
        f"{clocks[0]['test_turns']:,} messages compache had not seen.",
        (
            ("n", "Send everything, 5-minute cache"),
            ("n2", "Send everything, 1-hour cache"),
            ("c", "compache"),
        ),
        groups,
        "Cost per 1,000 messages of sending everything and of compache, for three return patterns. "
        + " ".join(summary),
        dollars,
    )


def agents(data):
    """Three groups of columns, one per way of working: an agent's calls as they are, and with compache."""
    groups, summary = [], []
    for row in data["figure"]:
        ours, whole, best = row["policy"], row["whole"], row["cached"]
        picked = f"{row['rounds']} rounds in full, "
        pings = row["user_pings"]
        if row["header_ttl"] > row["ttl"]:
            picked += "1-hour instructions"
        elif pings:
            picked += f"{pings} ping{'' if pings == 1 else 's'}"
        else:
            picked += f"{'1-hour' if row['ttl'] == 3600 else '5-minute'} cache"
        groups.append(
            {
                "heading": row["heading"],
                "columns": [("n", whole), ("n2", best), ("c", ours)],
                "beside": [
                    (f"{1 - ours / whole:.0%} less", "than everything"),
                    (f"{1 - ours / best:.0%} less", "than cached"),
                ],
                "picked": picked,
            }
        )
        summary.append(
            f"{row['heading']}: everything {dollars(whole)}, cached at its best {dollars(best)}, "
            f"compache {dollars(ours)} with {picked}."
        )
    return grouped(
        "What an agent saves follows from how it is worked with",
        f"Cost per 1,000 model calls on {data['model_name']}. {data['runs']} public coding-agent runs, "
        "the half compache had not seen.",
        (
            ("n", "Send everything"),
            ("n2", "Send everything, cached at its best"),
            ("c", "compache"),
        ),
        groups,
        "Cost per 1,000 model calls of sending everything and of compache, for three ways of working. "
        + " ".join(summary),
        dollars,
    )


def sessions(data):
    """A panel per long session, one line per tool: the bill so far after each prompt, averaged over its runs."""
    called = {
        "none": "alone",
        "headroom": "Headroom",
        "kompressor": "Kompressor",
        "compache": "compache",
    }
    tone = {"compache": "c", "none": "n"}
    width, height = 880, 400
    left, right, top, bottom, gap = 62, 134, 118, 82, 186
    panel = (width - left - right - gap) / 2
    plot, base = height - top - bottom, height - bottom
    body = [
        "<style>.n2s{stroke:#b5b3a9}.n2{fill:#b5b3a9}.band{opacity:.16}"
        "@media (prefers-color-scheme:dark){.n2s{stroke:#66655f}.n2{fill:#66655f}.band{opacity:.26}}</style>",
        text(24, 32, data["title"], "big"),
        text(24, 51, data["subtitle"], "t2"),
    ]

    def reach(arm):
        """The largest request of each run of one arm, from the smallest of them to the largest."""
        sizes = [
            r["largest"] for p in data["panels"] for r in p["runs"] if r["arm"] == arm
        ]
        return f"{min(sizes) / 1000:.0f}k to {max(sizes) / 1000:.0f}k"

    said = []
    for i, part in enumerate(data["panels"]):
        x0 = left + i * (panel + gap)
        runs = {
            arm: [r["curve"] for r in part["runs"] if r["arm"] == arm] for arm in called
        }
        # As far as both compache and the agent alone got in every run. A tool that stopped before that is left out.
        count = min(len(c) for arm in ("none", "compache") for c in runs[arm])
        early = [a for a, cs in runs.items() if cs and min(map(len, cs)) < count]
        lines = {}
        for arm, curves in runs.items():
            if curves and arm not in early:
                at = list(zip(*(c[:count] for c in curves), strict=True))
                lines[arm] = [(sum(v) / len(v), min(v), max(v)) for v in at]
        highest = max(v[-1][2] for v in lines.values())
        step = next(s for s in (0.1, 0.2, 0.5, 1.0, 2.0) if highest / s <= 4.5)
        ceiling = step * -(-highest // step)

        def x(k, x0=x0, count=count):
            return x0 + (k - 1) / (count - 1) * panel

        def y(value, ceiling=ceiling):
            return base - value / ceiling * plot

        body.append(text(x0 - 38, top - 34, part["heading"], "h"))
        for j in range(round(ceiling / step) + 1):
            at = y(j * step)
            body.append(
                f'<line class="{"grid" if j else "axis"}" x1="{x0:.1f}" x2="{x0 + panel:.1f}" '
                f'y1="{at:.1f}" y2="{at:.1f}"/>'
            )
            label = f"${j * step:.2f}" if j else "0"
            body.append(text(x0 - 9, at + 4, label, "tm", "end"))
        for k in (1, 5, 10, count):
            body.append(text(x(k), base + 20, k, "tm", "middle"))
        body.append(text(x0 + panel / 2, base + 38, "Prompt", "t2", "middle"))
        order = [
            a for a in ("none", "headroom", "kompressor", "compache") if a in lines
        ]
        for arm in order:
            values, cls = lines[arm], tone.get(arm, "n2")
            if len(runs[arm]) > 1:
                upper = [f"{x(k):.1f},{y(v[2]):.1f}" for k, v in enumerate(values, 1)]
                lower = [f"{x(k):.1f},{y(v[1]):.1f}" for k, v in enumerate(values, 1)]
                body.append(
                    f'<polygon class="band {cls}" points="{" ".join(upper + lower[::-1])}"/>'
                )
        for arm in order:
            values, cls = lines[arm], tone.get(arm, "n2")
            path = " ".join(
                f"{x(k):.1f},{y(v[0]):.1f}" for k, v in enumerate(values, 1)
            )
            body.append(f'<polyline class="line {cls}s" points="{path}"/>')
        alone, ours = lines["none"][-1][0], lines["compache"][-1][0]
        # A label at the end of each line, moved apart where two would touch.
        free = float("-inf")
        for arm in sorted(order, key=lambda a: -lines[a][-1][0]):
            cost, cls = lines[arm][-1][0], tone.get(arm, "n2")
            end = y(cost)
            body.append(
                f'<circle class="{cls} ring" cx="{x(count):.1f}" cy="{end:.1f}" r="5"/>'
            )
            place = max(end + 4, free + 17)
            free = place
            if place - end - 4 > 2:
                body.append(
                    f'<line class="axis" x1="{x(count) + 7:.1f}" y1="{end + 1:.1f}" '
                    f'x2="{x(count) + 12:.1f}" y2="{place - 4:.1f}"/>'
                )
            style = "h" if arm == "compache" else "t2"
            body.append(text(x(count) + 15, place, f"{called[arm]} ${cost:.2f}", style))
        body.append(text(x(count) + 15, free + 17, f"{1 - ours / alone:.0%} less", "h"))
        if early:
            names = " and ".join(called[a] for a in early)
            body.append(
                text(
                    x0 - 38,
                    top - 17,
                    f"{names} ran out of budget and are left out.",
                    "tm",
                )
            )
        said.append(
            f"{part['heading']}, after prompt {count}: "
            + ", ".join(f"{called[a]} ${lines[a][-1][0]:.2f}" for a in order)
        )
    body.append(
        text(
            24,
            height - 16,
            "Above 100k tokens a request costs five times as much on this model. "
            f"Largest requests: alone {reach('none')}, with compache {reach('compache')}.",
            "t2",
        )
    )
    return svg(
        width,
        height,
        data["title"],
        f"{data['subtitle']} {'. '.join(said)}.",
        body,
    )


# -- one thread -------------------------------------------------------------


def thread(data):
    """One conversation as three lanes, with a curve from each turn to the earlier parts it needed."""
    turns = data["turns"]
    count = len(turns)
    width = 880
    left, right, top = 178, 150, 104
    span = (width - left - right) / count
    lanes = {
        "input": (top + 44, 8),
        "retrieval": (top + 76, 30),
        "output": (top + 130, 10),
    }
    height = top + 190
    marker = top + 14
    names = {"input": "Questions", "retrieval": "Retrieved text", "output": "Answers"}
    recipe = data["recipe"]
    kept = {
        "input": recipe["input_runs"],
        "retrieval": recipe["retrieval_runs"],
        "output": recipe["output_runs"],
    }

    def x(turn):
        return left + (turn - 0.5) * span

    def median(kind):
        values = sorted(t["tokens"][kind] for t in turns)
        return values[len(values) // 2]

    needing = [t for t in turns if isinstance(t["needs"], dict)]
    in_view = sum(t["needs"]["in_view"] for t in needing)
    unknown = sum(t["needs"] == "unknown" for t in turns)
    body = [
        text(
            24,
            30,
            "What each turn of one conversation needed from earlier turns",
            "big",
        ),
        text(
            24,
            49,
            f"A {count}-turn DocChat thread about a medicine label. {len(needing)} turns need something earlier, "
            f"{count - len(needing) - unknown} need nothing, {unknown} could not be assessed.",
            "t2",
        ),
        '<path class="arc cs" d="M24,72 q9,-10 18,0"/>',
        text(48, 74, "needed part still in view", "t2"),
        '<path class="arc os" d="M222,72 q9,-10 18,0"/>',
        text(246, 74, "needed part archived, fetched back", "t2"),
        '<circle class="box" cx="478" cy="70" r="3.5"/>',
        text(488, 74, "needs nothing earlier", "t2"),
        text(634, 74, "?", "tm"),
        text(646, 74, "not assessed", "t2"),
        text(left - 14, marker + 4, "Turn", "t2", "end"),
        text(width - right + 14, marker + 4, "Kept in view for", "t2"),
    ]
    for kind, (y0, tall) in lanes.items():
        body.append(text(left - 14, y0 + tall / 2 + 1, names[kind], "b", "end"))
        body.append(
            text(
                left - 14,
                y0 + tall / 2 + 16,
                f"about {median(kind):,} tokens a turn",
                "tm",
                "end",
            )
        )
        window = kept[kind]
        note = (
            "the whole thread"
            if window is None
            else "1 turn"
            if window == 1
            else f"{window} turns"
        )
        body.append(
            text(
                width - right + 14,
                y0 + tall / 2 + 4,
                note,
                "b" if kind == "retrieval" else "",
            )
        )
        for t in turns:
            body.append(
                f'<rect class="cell" x="{x(t["turn"]) - span * 0.36:.1f}" y="{y0}" width="{span * 0.72:.1f}" height="{tall}" rx="2"/>'
            )
    for t in turns:
        number = t["turn"]
        if number == 1 or number % 5 == 0:
            body.append(text(x(number), marker - 14, number, "tm", "middle"))
        if t["needs"] == "unknown":
            body.append(text(x(number), marker + 4, "?", "tm", "middle"))
        elif isinstance(t["needs"], dict):
            body.append(
                f'<circle class="{"c" if t["needs"]["in_view"] else "o"}" cx="{x(number):.1f}" cy="{marker}" r="4.5"/>'
            )
        else:
            body.append(
                f'<circle class="box" cx="{x(number):.1f}" cy="{marker}" r="3.5"/>'
            )
    # Curves last, so they sit above the cells they point at.
    for t in needing:
        cls = "c" if t["needs"]["in_view"] else "o"
        for kind, source in t["needs"]["parts"]:
            y0, tall = lanes[kind]
            x1, x2, y2 = x(t["turn"]), x(source), y0 + tall / 2
            bend = min(60, 18 + 5 * (t["turn"] - source))
            body.append(
                f'<rect class="{cls}" x="{x2 - span * 0.36:.1f}" y="{y0}" width="{span * 0.72:.1f}" height="{tall}" rx="2"/>'
            )
            body.append(
                f'<path class="arc {cls}s" d="M{x1:.1f},{marker + 5} C{x1:.1f},{marker + 5 + bend} {x2 + span * 0.9:.1f},{y2:.1f} {x2 + span * 0.36:.1f},{y2:.1f}"/>'
            )
    return svg(
        width,
        height,
        "What each turn of one conversation needed from earlier turns",
        f"A {count}-turn conversation drawn as three lanes: questions, retrieved text and answers. "
        f"{len(needing)} turns need an earlier part. For {in_view} of them that part is still in view under the fitted "
        f"policy, which keeps questions and answers for a long time and retrieved text for one turn. "
        f"The other {len(needing) - in_view} fetch it back.",
        body,
    )


# -- against OpenClaw -------------------------------------------------------


def openclaw(data):
    """Two bars per thread length: OpenClaw's defaults in grey, compache in blue."""
    rows = data["rows"]
    width, height = 880, 96 + 62 * len(rows) + 30
    x0, full, bar, top, step = 190, 520, 18, 100, 62
    longest = max(r["openclaw"] for r in rows)
    body = [
        text(24, 30, f"Cost per 1,000 turns · {data['model']}", "big"),
        text(
            24,
            49,
            "The calculator's default workload at three thread lengths, against OpenClaw's defaults.",
            "t2",
        ),
        '<rect class="n" x="24" y="64" width="10" height="10" rx="2"/>',
        text(40, 73.5, "OpenClaw", "t2"),
        '<rect class="c" x="116" y="64" width="10" height="10" rx="2"/>',
        text(132, 73.5, "compache", "t2"),
        f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 10}" y2="{top + step * len(rows) - 14}"/>',
    ]
    described = []
    for i, row in enumerate(rows):
        y0 = top + i * step
        body.append(
            text(x0 - 14, y0 + bar + 5, f"{row['turns']}-turn threads", "b", "end")
        )
        for j, (key, cls) in enumerate((("openclaw", "n"), ("compache", "c"))):
            length = max(2, row[key] / longest * full)
            yy = y0 + j * (bar + 2)
            body.append(
                f'<path class="{cls}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x0 + length + 8,
                    yy + bar - 4.5,
                    dollars(row[key]),
                    "b" if key == "compache" else "t2",
                )
            )
        saving = 1 - row["compache"] / row["openclaw"]
        body.append(text(width - 24, y0 + bar + 5, f"{saving:.0%} less", "h", "end"))
        described.append(
            f"{row['turns']}-turn threads: OpenClaw {dollars(row['openclaw'])}, compache {dollars(row['compache'])}"
        )
    return svg(
        width,
        height,
        f"Cost per 1,000 turns on {data['model']}: OpenClaw against compache",
        "Modeled cost per 1,000 turns at three thread lengths. "
        + "; ".join(described)
        + ".",
        body,
    )


# -- the calculator's link preview ------------------------------------------------


def preview(data):
    """What a shared calculator link shows: the calculator's default workload, priced both ways.

    Link previews need a bitmap. `site/preview.png` is this figure, taken with a browser:

        chrome --headless --window-size=1200,630 --screenshot=site/preview.png docs/assets/preview.svg
    """
    row = data["rows"][0]
    drawn = (ROOT / "site" / "mark.svg").read_text("utf-8")
    mark = re.search(r' d="([^"]+)"', drawn)[1]
    width, height, x0, full, bar = 1200, 630, 80, 820, 74
    longest = max(row["openclaw"], row["compache"])
    style = (
        'text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:#0b0b0b}'
        ".two{fill:#52514e}.c{fill:#2a78d6}.n{fill:#898781}"
    )

    def line(x, y, size, content, cls="", weight=400):
        return (
            f'<text x="{x:.0f}" y="{y}" font-size="{size}" font-weight="{weight}"'
            f"{f' class={chr(34)}{cls}{chr(34)}' if cls else ''}>{escape(content)}</text>"
        )

    body = [
        f'<path class="c" transform="translate({x0 - 2.5} 59) scale(.65)" d="{mark}"/>',
        line(x0 + 50, 94, 36, "compache", weight=650),
        line(x0, 186, 58, "What would your long chats cost?", weight=650),
        line(
            x0,
            238,
            28,
            f"{data['model']} · {row['turns']}-turn threads · 25k retrieved tokens per turn",
            "two",
        ),
    ]
    for i, (key, name, cls) in enumerate(
        (("openclaw", "OpenClaw’s defaults", "n"), ("compache", "compache", "c"))
    ):
        top = 306 + i * 118
        length = row[key] / longest * full
        body += [
            line(x0, top - 10, 24, name, "two"),
            f'<rect class="{cls}" x="{x0}" y="{top}" width="{length:.1f}" height="{bar}" rx="8"/>',
            line(x0 + length + 22, top + 52, 44, dollars(row[key]), weight=650),
        ]
    saving = 1 - row["compache"] / row["openclaw"]
    body += [
        line(x0, 568, 32, f"{saving:.0%} less per 1,000 turns, modeled.", weight=650),
        line(x0 + 562, 568, 32, "Put in your own numbers.", "two"),
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t">\n'
        f'<title id="t">compache cost calculator: {escape(data["model"])}, {row["turns"]}-turn threads, '
        f"OpenClaw {dollars(row['openclaw'])}, compache {dollars(row['compache'])} per 1,000 turns</title>\n"
        f"<style>{style}</style>\n"
        f'<rect width="{width}" height="{height}" fill="#f9f9f7"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


# -- live model calls ---------------------------------------------------------


def live(data):
    """One group of bars per way in: what the provider billed under no policy and under each policy."""
    ways = data["ways"]
    rows = sum(len(way["arms"]) for way in ways)
    bar, pitch, gap, top = 14, 21, 26, 104
    width, height = 880, top + rows * pitch + len(ways) * gap + 10
    x0, full = 392, 356
    longest = max(arm["cost"] for way in ways for arm in way["arms"])
    body = [
        text(
            24,
            30,
            "What the provider billed per 1,000 turns, through every way in",
            "big",
        ),
        text(24, 49, data["subtitle"], "t2"),
        '<rect class="n" x="24" y="64" width="10" height="10" rx="2"/>',
        text(
            40, 73.5, "No policy: the application as it is, with its own caching", "t2"
        ),
        '<rect class="c" x="396" y="64" width="10" height="10" rx="2"/>',
        text(412, 73.5, "With a compache policy", "t2"),
    ]
    described, y0 = [], top
    for way in ways:
        arms, base = way["arms"], way["arms"][0]["cost"]
        body.append(text(24, y0 + bar - 2.5, way["name"], "b"))
        body += [
            text(24, y0 + bar + 15.5 + 16 * k, line, "tm")
            for k, line in enumerate(way["detail"])
        ]
        body.append(
            f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y0 - 5}" y2="{y0 + len(arms) * pitch - 2}"/>'
        )
        said = []
        for j, arm in enumerate(arms):
            yy, length = y0 + j * pitch, max(6, arm["cost"] / longest * full)
            body.append(text(x0 - 12, yy + bar - 2.5, arm["name"], "t2", "end"))
            body.append(
                f'<path class="{"c" if j else "n"}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x0 + length + 8,
                    yy + bar - 2.5,
                    dollars(arm["cost"]),
                    "b" if j else "t2",
                )
            )
            if j:
                change = f"{arm['cost'] / base - 1:+.0%}".replace("-", "\u2212")
                body.append(text(x0 + length + 54, yy + bar - 2.5, change, "t2"))
            said.append(f"{arm['name'].lower()} {dollars(arm['cost'])}")
        described.append(
            f"{way['name']}, {', '.join(way['detail'])}: " + ", ".join(said)
        )
        y0 += len(arms) * pitch + gap
    return svg(
        width,
        height,
        "What the provider billed per 1,000 turns through every way in, with and without a compache policy",
        f"{data['subtitle']} {'. '.join(described)}.",
        body,
    )


# -- other tools ----------------------------------------------------------------


def compare(data):
    """One group of bars per return pattern: what each tool's requests cost, and compache's."""
    clocks = data["clocks"]
    rows = sum(len(clock["arms"]) for clock in clocks)
    bar, pitch, gap, top = 14, 21, 26, 92
    width, height = 880, top + rows * pitch + len(clocks) * gap + 4
    x0, full = 388, 372
    longest = max(arm["cost"] for clock in clocks for arm in clock["arms"])
    body = [
        text(24, 30, "Cost per 1,000 turns of document chats, by tool", "big"),
        text(
            24,
            49,
            f"DocChat's held-out chats on {data['model']}, priced by the provider's cache rules. Headroom and Kompressor were",
            "t2",
        ),
        text(
            24,
            65,
            "run on the retrieved text. Cache kept warm is the cheapest of ten settings. compache's price includes fetching back.",
            "t2",
        ),
    ]
    described, y0 = [], top
    for clock in clocks:
        arms, base = clock["arms"], clock["arms"][0]["cost"]
        body.append(
            text(
                24,
                y0 + bar - 2.5,
                f"Users back after {away(clock['away_seconds'])}",
                "b",
            )
        )
        body.append(text(24, y0 + bar + 13.5, "at the median", "tm"))
        body.append(
            f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y0 - 5}" y2="{y0 + len(arms) * pitch - 2}"/>'
        )
        said = []
        for j, arm in enumerate(arms):
            ours = arm["name"] == "compache"
            yy, length = y0 + j * pitch, max(6, arm["cost"] / longest * full)
            body.append(
                text(x0 - 12, yy + bar - 2.5, arm["name"], "b" if ours else "t2", "end")
            )
            body.append(
                f'<path class="{"c" if ours else "n"}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>'
            )
            body.append(
                text(
                    x0 + length + 8,
                    yy + bar - 2.5,
                    dollars(arm["cost"]),
                    "b" if ours else "t2",
                )
            )
            if j:
                change = f"{arm['cost'] / base - 1:+.0%}".replace("-", "\u2212")
                body.append(text(x0 + length + 54, yy + bar - 2.5, change, "t2"))
            said.append(f"{arm['name']} {dollars(arm['cost'])}")
        described.append(
            f"Users back after {away(clock['away_seconds'])}: " + ", ".join(said)
        )
        y0 += len(arms) * pitch + gap
    return svg(
        width,
        height,
        "Cost per 1,000 turns of document chats: other tools against compache",
        f"DocChat's held-out chats on {data['model']}. {'. '.join(described)}.",
        body,
    )


def tools(data):
    """Two panels, one bar per tool in each: what the provider billed, and how many answers were right."""
    arms, base = data["arms"], data["arms"][0]
    bar, pitch, top = 16, 27, 106
    width, height = 880, top + len(arms) * pitch + 12
    panels = [
        (262, 232, "cost", "Bill per 1,000 turns"),
        (656, 150, "correct", f"Right answers, out of {base['graded']}"),
    ]
    title = "Live: the same conversations through each tool"
    body = [
        text(24, 30, title, "big"),
        text(
            24,
            49,
            f"{data['conversations']} held-out DocChat conversations, {base['turns']} turns on {data['model']}, "
            "at the users' own pace. Billed by the provider.",
            "t2",
        ),
    ]
    for x0, full, key, name in panels:
        longest = max(arm[key] for arm in arms)
        body += [
            text(x0, top - 18, name, "h"),
            f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 6}" y2="{top + len(arms) * pitch - 5}"/>',
        ]
        for j, arm in enumerate(arms):
            ours, with_ours = arm["name"] == "compache", "compache" in arm["name"]
            yy, length = top + j * pitch, max(6, arm[key] / longest * full)
            label = f"${arm[key]:.2f}" if key == "cost" else str(arm[key])
            body += [
                f'<path class="{"c" if with_ours else "n"}" d="M{x0},{yy} h{length - 4:.1f} a4,4 0 0 1 4,4 '
                f'v{bar - 8} a4,4 0 0 1 -4,4 h-{length - 4:.1f} z"/>',
                text(x0 + length + 8, yy + bar - 4, label, "b" if ours else "t2"),
            ]
            if key != "cost":
                continue
            body.append(
                text(x0 - 12, yy + bar - 4, arm["name"], "b" if ours else "t2", "end")
            )
            if j:
                change = f"{arm[key] / base[key] - 1:+.0%}".replace("-", "\u2212")
                body.append(text(x0 + length + 54, yy + bar - 4, change, "t2"))
    said = [
        f"{arm['name']} ${arm['cost']:.2f}, {arm['correct']} of {arm['graded']} answers right"
        for arm in arms
    ]
    return svg(width, height, title, ". ".join(said) + ".", body)


# -- how it works -----------------------------------------------------------


def workflow():
    """Three steps: record threads, fit a policy, apply it on every request."""
    width, height = 880, 208
    steps = [
        (
            "1  Record",
            "Your threads",
            ["each message with its time", "labels for a sample of turns"],
        ),
        (
            "2  Fit",
            "compache fit",
            ["replays every candidate policy", "against the cache at your timestamps"],
        ),
        (
            "3  Apply",
            "policy.json",
            [
                "what stays in view, per kind of part",
                "how many pings keep the cache warm",
            ],
        ),
    ]
    box, gap, top = 252, 38, 58
    body = [text(24, 32, "Fit once on recorded threads, apply on every request", "big")]
    for i, (step, name, lines) in enumerate(steps):
        x0 = 24 + i * (box + gap)
        body.append(
            f'<rect class="box" x="{x0}" y="{top}" width="{box}" height="118" rx="8"/>'
        )
        body.append(text(x0 + 18, top + 28, step, "tm b"))
        body.append(text(x0 + 18, top + 54, name, "big"))
        body += [
            text(x0 + 18, top + 78 + 17 * k, line, "t2") for k, line in enumerate(lines)
        ]
        if i < len(steps) - 1:
            ax = x0 + box + 8
            body.append(
                f'<path class="line cs" d="M{ax},{top + 59} h{gap - 16} m-6,-6 l6,6 l-6,6"/>'
            )
    body.append(
        text(
            24,
            height - 12,
            "compache decides what each request holds and when a ping is due. Your application sends the requests and the pings.",
            "tm",
        )
    )
    return svg(
        width,
        height,
        "How compache works: record, fit, apply",
        "Recorded threads with timestamps and labels for a sample of turns go into compache fit, which replays every "
        "candidate policy against the cache. The result is a policy file that says what stays in view and how many "
        "keep-warm pings to send, applied on every request without a model call.",
        body,
    )


FIGURES = {
    "headline": ("headline.svg", "live.json", headline),
    "requests": ("requests.svg", "requests.json", requests),
    "breaks": ("breaks-cost.svg", "policy-grid.json", breaks),
    "agents": ("agents-cost.svg", "agents.json", agents),
    "sessions": ("agents-live.svg", "agents-live.json", sessions),
    "thread": ("thread-dependencies.svg", "thread.json", thread),
    "openclaw": ("openclaw-cost.svg", "openclaw.json", openclaw),
    "preview": ("preview.svg", "openclaw.json", preview),
    "live": ("live-cost.svg", "live.json", live),
    "compare": ("compare-cost.svg", "compare.json", compare),
    "tools": ("compare-live.svg", "compare-live.json", tools),
    "workflow": ("workflow.svg", None, workflow),
}


def main(argv=None):
    names = (sys.argv[1:] if argv is None else argv) or list(FIGURES)
    for name in names:
        if name not in FIGURES:
            raise SystemExit(
                f"unknown figure {name!r}; choose from {', '.join(FIGURES)}"
            )
        target, source, draw = FIGURES[name]
        args = [json.loads((DATA / source).read_text("utf-8"))] if source else []
        (ASSETS / target).write_text(draw(*args), "utf-8")
        print(f"wrote {(ASSETS / target).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
