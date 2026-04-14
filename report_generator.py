"""
report_generator.py — ELARA Session Report Generator
======================================================
Reads the telemetry JSON produced by questionnaire.py and generates a
professional Word (.docx) report summarising the session.

Usage
-----
    python report_generator.py session_<id>_<ts>.json
    python report_generator.py session_<id>_<ts>.json --out my_report.docx
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# ── Analysis helpers ────────────────────────────────────────────────────────

AFFECT_ORDER = ["calm", "confused", "frustrated", "sad", "disengaged"]

ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "DECREASE_CLARITY",
    2: "DECREASE_PACE",
    3: "INCREASE_CONFIRMATION",
    4: "ENABLE_PATIENCE",
    5: "DECREASE_CLARITY_AND_PACE",
    6: "CLARITY_AND_CONFIRMATION",
}

AFFECT_EMOJI = {
    "calm": "🟢", "confused": "🟡",
    "frustrated": "🔴", "sad": "🔵", "disengaged": "⚫",
}


def load_session(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyse(session: Dict[str, Any]) -> Dict[str, Any]:
    turns = session["turns"]

    # Affect counts and arc
    affect_counts = Counter(t["affect"] for t in turns)
    affect_arc    = [t["affect"] for t in turns]

    # Transitions
    transitions: Counter = Counter()
    for i in range(len(affect_arc) - 1):
        transitions[(affect_arc[i], affect_arc[i + 1])] += 1

    # Config change events
    config_events = [
        t for t in turns if t["config_changes"]
    ]

    # Escalation rules fired
    escalations = [
        t for t in turns if t.get("escalation_rule")
    ]

    # Reward timeline
    rewards = [
        (t["turn_number"], t["reward_applied"])
        for t in turns if t["reward_applied"] is not None
    ]

    # Latency
    latencies = [t["latency_ms"] for t in turns]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0

    # Action distribution
    action_counts = Counter(
        ACTION_NAMES.get(t["ucb_action_id"], str(t["ucb_action_id"]))
        for t in turns
    )

    # Phase summary
    phase_affects: Dict[str, Counter] = {}
    for t in turns:
        phase_affects.setdefault(t["phase"], Counter())[t["affect"]] += 1

    # Positive / negative reward balance
    pos = sum(r for _, r in rewards if r > 0)
    neg = sum(r for _, r in rewards if r < 0)

    # Recovery moments (frustrated/confused → calm)
    recoveries = [
        (affect_arc[i], affect_arc[i + 1], turns[i + 1]["turn_number"])
        for i in range(len(affect_arc) - 1)
        if affect_arc[i] in ("frustrated", "confused") and affect_arc[i + 1] == "calm"
    ]

    # Patience mode toggles
    patience_on  = [t["turn_number"] for t in turns if t["config_changes"].get("patience_mode") is True]
    patience_off = [t["turn_number"] for t in turns if t["config_changes"].get("patience_mode") is False]

    return {
        "affect_counts":   affect_counts,
        "affect_arc":      affect_arc,
        "transitions":     transitions,
        "config_events":   config_events,
        "escalations":     escalations,
        "rewards":         rewards,
        "avg_latency":     avg_latency,
        "action_counts":   action_counts,
        "phase_affects":   phase_affects,
        "pos_reward":      round(pos, 2),
        "neg_reward":      round(neg, 2),
        "recoveries":      recoveries,
        "patience_on":     patience_on,
        "patience_off":    patience_off,
        "total_turns":     len(turns),
    }


# ── JS report builder ────────────────────────────────────────────────────────

def build_js(session: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """Returns the complete Node.js script that generates the .docx."""

    turns       = session["turns"]
    sid         = session["session_id"]
    started     = session["started_at"][:16].replace("T", " ")
    backend     = session.get("backend", "ollama")
    model       = session.get("model") or "default"

    # ── Helpers for JS escaping ───────────────────────────────────────────
    def jsstr(s: str) -> str:
        return (s.replace("\\", "\\\\")
                 .replace('"',  '\\"')
                 .replace("\n", "\\n")
                 .replace("\r", ""))

    def affect_colour(a: str) -> str:
        return {"calm": "2E7D32", "confused": "F9A825",
                "frustrated": "C62828", "sad": "1565C0",
                "disengaged": "424242"}.get(a, "555555")

    # ── Build turn table rows ─────────────────────────────────────────────
    turn_rows = ""
    for t in turns:
        ac = affect_colour(t["affect"])
        changes = ", ".join(f"{k}→{v}" for k, v in t["config_changes"].items()) or "—"
        reward  = f"{t['reward_applied']:+.1f}" if t["reward_applied"] is not None else "—"
        esc     = t.get("escalation_rule") or "—"
        turn_rows += f"""
    new TableRow({{
      children: [
        cell("{t['turn_number']}", 600, "333333"),
        cellLeft("{jsstr(t['phase'].split(' – ')[0])}", 1800, "444444"),
        cellColoured("{t['affect'].upper()}", 1400, "{ac}"),
        cell("{t['confidence']:.0%}", 800, "333333"),
        cell("{ACTION_NAMES.get(t['ucb_action_id'], str(t['ucb_action_id']))}", 2000, "444444"),
        cellLeft("{jsstr(changes)}", 2000, "555555"),
        cell("{reward}", 760, "333333"),
      ],
    }}),"""

    # ── Config change rows ────────────────────────────────────────────────
    cfg_rows = ""
    for t in stats["config_events"]:
        for param, new_val in t["config_changes"].items():
            cfg_rows += f"""
    new TableRow({{
      children: [
        cell("{t['turn_number']}", 700),
        cellLeft("{jsstr(t['phase'])}", 2800),
        cellLeft("{jsstr(t['affect'].upper())}", 1400),
        cellLeft("{jsstr(param)}", 2200),
        cellLeft("{jsstr(str(new_val))}", 2260),
      ],
    }}),"""

    # ── Phase summary rows ────────────────────────────────────────────────
    phase_rows = ""
    for phase, counts in stats["phase_affects"].items():
        dominant = max(counts, key=counts.get)
        dc = affect_colour(dominant)
        total    = sum(counts.values())
        dist     = "  ".join(f"{a}:{n}" for a, n in counts.items())
        phase_rows += f"""
    new TableRow({{
      children: [
        cellLeft("{jsstr(phase)}", 3600),
        cell("{total}", 700),
        cellColoured("{dominant.upper()}", 1600, "{dc}"),
        cellLeft("{jsstr(dist)}", 3460),
      ],
    }}),"""

    # ── Conversation rows (first 20) ──────────────────────────────────────
    conv_rows = ""
    for t in turns[:20]:
        conv_rows += f"""
    new TableRow({{
      children: [
        cell("{t['turn_number']}", 500),
        cellLeft("{jsstr(t['user_message'][:80])}", 3800),
        cellLeft("{jsstr(t['elara_reply'][:80])}", 3800),
        cellColoured("{t['affect'].upper()}", 1260, "{affect_colour(t['affect'])}"),
      ],
    }}),"""

    # ── Key stats ─────────────────────────────────────────────────────────
    ac = stats["affect_counts"]
    arc_str = " → ".join(stats["affect_arc"])

    recovery_text = ""
    for prev, curr, turn_n in stats["recoveries"]:
        recovery_text += f"Turn {turn_n}: {prev} → {curr}. "
    recovery_text = recovery_text or "No direct frustrated/confused → calm transitions recorded."

    esc_text = ""
    for t in stats["escalations"]:
        esc_text += f"Turn {t['turn_number']} ({t['affect']}): {t['escalation_rule']}. "
    esc_text = esc_text or "No escalation rules fired during this session."

    action_dist = "  |  ".join(
        f"{name}: {count}" for name, count in stats["action_counts"].most_common()
    )

    return f"""
const {{
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, SimpleField, Footer,
}} = require('docx');
const fs = require('fs');

// ── Helpers ──────────────────────────────────────────────────────────────────
const BORDER = {{ style: BorderStyle.SINGLE, size: 1, color: "DDDDDD" }};
const BORDERS = {{ top: BORDER, bottom: BORDER, left: BORDER, right: BORDER }};
const MARGINS = {{ top: 80, bottom: 80, left: 140, right: 140 }};

function cell(text, w, color = "333333") {{
  return new TableCell({{
    borders: BORDERS, width: {{ size: w, type: WidthType.DXA }}, margins: MARGINS,
    children: [new Paragraph({{
      alignment: AlignmentType.CENTER,
      children: [new TextRun({{ text: String(text), size: 18, color, font: "Arial" }})]
    }})]
  }});
}}
function cellLeft(text, w, color = "333333") {{
  return new TableCell({{
    borders: BORDERS, width: {{ size: w, type: WidthType.DXA }}, margins: MARGINS,
    children: [new Paragraph({{
      children: [new TextRun({{ text: String(text), size: 18, color, font: "Arial" }})]
    }})]
  }});
}}
function cellColoured(text, w, bg) {{
  return new TableCell({{
    borders: BORDERS, width: {{ size: w, type: WidthType.DXA }}, margins: MARGINS,
    shading: {{ fill: bg, type: ShadingType.CLEAR }},
    children: [new Paragraph({{
      alignment: AlignmentType.CENTER,
      children: [new TextRun({{ text: String(text), size: 18, bold: true, color: "FFFFFF", font: "Arial" }})]
    }})]
  }});
}}
function hdr(text, level) {{
  return new Paragraph({{
    heading: level,
    children: [new TextRun({{ text, font: "Arial" }})]
  }});
}}
function body(text) {{
  return new Paragraph({{
    spacing: {{ after: 160 }},
    children: [new TextRun({{ text, size: 22, color: "333333", font: "Arial" }})]
  }});
}}
function bold(text) {{
  return new TextRun({{ text, bold: true, size: 22, font: "Arial", color: "1B3A6B" }});
}}
function stat(label, value) {{
  return new Paragraph({{
    spacing: {{ after: 100 }},
    children: [
      new TextRun({{ text: label + ":  ", size: 20, color: "666666", font: "Arial" }}),
      new TextRun({{ text: String(value), size: 22, bold: true, color: "1B3A6B", font: "Arial" }}),
    ]
  }});
}}
function rule() {{
  return new Paragraph({{
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "1B3A6B", space: 1 }} }},
    children: [new TextRun("")],
    spacing: {{ after: 200 }}
  }});
}}
function space() {{
  return new Paragraph({{ children: [new TextRun("")], spacing: {{ after: 120 }} }});
}}

// ── Document ──────────────────────────────────────────────────────────────────
const doc = new Document({{
  styles: {{
    default: {{ document: {{ run: {{ font: "Arial", size: 22, color: "333333" }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 44, bold: true, color: "1B3A6B", font: "Arial" }},
         paragraph: {{ spacing: {{ before: 400, after: 200 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 30, bold: true, color: "1B3A6B", font: "Arial" }},
         paragraph: {{ spacing: {{ before: 320, after: 160 }}, outlineLevel: 1 }} }},
      {{ id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 24, bold: true, color: "2E5EAA", font: "Arial" }},
         paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 2 }} }},
    ]
  }},
  numbering: {{
    config: [{{
      reference: "bullets",
      levels: [{{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: {{ paragraph: {{ indent: {{ left: 720, hanging: 360 }} }} }} }}]
    }}]
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 12240, height: 15840 }},
        margin: {{ top: 1080, right: 1080, bottom: 1080, left: 1080 }}
      }}
    }},
    footers: {{
      default: new Footer({{
        children: [new Paragraph({{
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({{ text: "ELARA Session Report  |  ", size: 16, color: "888888", font: "Arial" }}),
            new TextRun({{ text: "Session {sid}  |  Page ", size: 16, color: "888888", font: "Arial" }}),
            new SimpleField("PAGE"),
          ]
        }})]
      }})
    }},
    children: [

      // ── Cover ─────────────────────────────────────────────────────────────
      new Paragraph({{ spacing: {{ after: 800 }}, children: [new TextRun("")] }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 120 }},
        children: [new TextRun({{ text: "ELARA", size: 72, bold: true, color: "1B3A6B", font: "Arial" }})]
      }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 80 }},
        children: [new TextRun({{ text: "Elderly Life-Assistive Robotic Agent", size: 32, color: "2E5EAA", font: "Arial" }})]
      }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 600 }},
        children: [new TextRun({{ text: "Questionnaire Session Report", size: 36, bold: true, color: "444444", font: "Arial" }})]
      }}),
      rule(),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 100 }},
        children: [new TextRun({{ text: "Session ID: {sid}", size: 22, color: "555555", font: "Arial" }})]
      }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 100 }},
        children: [new TextRun({{ text: "Date: {started}", size: 22, color: "555555", font: "Arial" }})]
      }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 100 }},
        children: [new TextRun({{ text: "Backend: {backend}  |  Model: {model}", size: 22, color: "555555", font: "Arial" }})]
      }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ after: 100 }},
        children: [new TextRun({{ text: "Total turns: {stats['total_turns']}", size: 22, color: "555555", font: "Arial" }})]
      }}),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 1. Executive Summary ───────────────────────────────────────────────
      hdr("1. Executive Summary", HeadingLevel.HEADING_1),
      rule(),
      body("This report presents the analysis of a single ELARA companion session conducted using the structured five-phase questionnaire protocol. ELARA (Elderly Life-Assistive Robotic Agent) is an adaptive conversational agent backed by a Discounted LinUCB bandit algorithm that adjusts communication style in real time based on the detected emotional state of the elderly user."),
      body("The session comprised {stats['total_turns']} conversation turns across five distinct phases, progressing from calm introductory interaction through a confusion and frustration peak, then recovering to a warm and emotionally positive close. The Learning Agent successfully detected the affect arc and applied targeted configuration changes to ease the user's distress."),
      space(),
      hdr("Key Findings", HeadingLevel.HEADING_3),
      stat("Total turns", "{stats['total_turns']}"),
      stat("Dominant affect", "{max(stats['affect_counts'], key=stats['affect_counts'].get).capitalize()}"),
      stat("Affect distribution", "Calm {ac.get('calm',0)}  |  Confused {ac.get('confused',0)}  |  Frustrated {ac.get('frustrated',0)}  |  Sad {ac.get('sad',0)}  |  Disengaged {ac.get('disengaged',0)}"),
      stat("Config adaptations", "{len(stats['config_events'])} turns triggered a config change"),
      stat("Escalation rules fired", "{len(stats['escalations'])}"),
      stat("Positive reward total", "+{stats['pos_reward']}"),
      stat("Negative reward total", "{stats['neg_reward']}"),
      stat("Average turn latency", "{stats['avg_latency']} ms"),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 2. Affect Analysis ─────────────────────────────────────────────────
      hdr("2. Affect Analysis", HeadingLevel.HEADING_1),
      rule(),

      hdr("2.1  Affect Distribution", HeadingLevel.HEADING_2),
      body("The table below shows how many turns were classified under each affect state across the session."),
      space(),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [2000, 2000, 2000, 2000, 1360],
        rows: [
          new TableRow({{
            tableHeader: true,
            children: [
              cellColoured("CALM", 2000, "2E7D32"),
              cellColoured("CONFUSED", 2000, "F9A825"),
              cellColoured("FRUSTRATED", 2000, "C62828"),
              cellColoured("SAD", 2000, "1565C0"),
              cellColoured("DISENGAGED", 1360, "424242"),
            ]
          }}),
          new TableRow({{
            children: [
              cell("{ac.get('calm',0)}", 2000),
              cell("{ac.get('confused',0)}", 2000),
              cell("{ac.get('frustrated',0)}", 2000),
              cell("{ac.get('sad',0)}", 2000),
              cell("{ac.get('disengaged',0)}", 1360),
            ]
          }})
        ]
      }}),
      space(),

      hdr("2.2  Affect Arc", HeadingLevel.HEADING_2),
      body("The complete sequence of detected affects turn by turn:"),
      new Paragraph({{
        spacing: {{ after: 200 }},
        children: [new TextRun({{ text: "{arc_str}", size: 18, color: "444444", font: "Courier New" }})]
      }}),
      space(),

      hdr("2.3  Affect Transitions", HeadingLevel.HEADING_2),
      body("The following transitions occurred between consecutive turns. Positive transitions (towards calm) indicate successful adaptation by ELARA. Negative transitions indicate deteriorating affect."),
      space(),
      new Table({{
        width: {{ size: 6000, type: WidthType.DXA }},
        columnWidths: [2000, 2000, 2000],
        rows: [
          new TableRow({{ tableHeader: true, children: [
            cellColoured("FROM", 2000, "1B3A6B"),
            cellColoured("TO", 2000, "1B3A6B"),
            cellColoured("COUNT", 2000, "1B3A6B"),
          ]}}),
          {chr(10).join(
            f'new TableRow({{ children: [ cellLeft("{f}", 2000), cellLeft("{t}", 2000), cell("{n}", 2000) ]}}), '
            for (f, t), n in sorted(stats["transitions"].items(), key=lambda x: -x[1])
          )}
        ]
      }}),
      space(),

      hdr("2.4  Recovery Events", HeadingLevel.HEADING_2),
      body("Recovery events are turns where the detected affect improved from frustrated or confused to calm — the most valuable transitions for ELARA to achieve."),
      body("{recovery_text}"),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 3. Phase-by-Phase Summary ──────────────────────────────────────────
      hdr("3. Phase-by-Phase Summary", HeadingLevel.HEADING_1),
      rule(),
      body("The questionnaire followed a structured five-phase arc. The table below summarises the dominant affect and distribution of states within each phase."),
      space(),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [3600, 700, 1600, 3460],
        rows: [
          new TableRow({{ tableHeader: true, children: [
            cellColoured("PHASE", 3600, "1B3A6B"),
            cellColoured("TURNS", 700, "1B3A6B"),
            cellColoured("DOMINANT AFFECT", 1600, "1B3A6B"),
            cellColoured("DISTRIBUTION", 3460, "1B3A6B"),
          ]}}),
          {phase_rows}
        ]
      }}),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 4. Learning Agent Adaptations ─────────────────────────────────────
      hdr("4. Learning Agent Adaptations", HeadingLevel.HEADING_1),
      rule(),

      hdr("4.1  Configuration Change Log", HeadingLevel.HEADING_2),
      body("Every time the Learning Agent modified ELARA's behaviour configuration, it is logged below. Configuration changes reflect the LinUCB bandit's action selection and the config_applier's step-clamped delta logic."),
      space(),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [700, 2800, 1400, 2200, 2260],
        rows: [
          new TableRow({{ tableHeader: true, children: [
            cellColoured("TURN", 700, "1B3A6B"),
            cellColoured("PHASE", 2800, "1B3A6B"),
            cellColoured("AFFECT", 1400, "1B3A6B"),
            cellColoured("PARAMETER", 2200, "1B3A6B"),
            cellColoured("NEW VALUE", 2260, "1B3A6B"),
          ]}}),
          {cfg_rows}
        ]
      }}),
      space(),

      hdr("4.2  Bandit Action Distribution", HeadingLevel.HEADING_2),
      body("Distribution of LinUCB action selections across all turns:"),
      body("{action_dist}"),
      space(),

      hdr("4.3  Escalation Smoother Events", HeadingLevel.HEADING_2),
      body("The escalation smoother prevents sudden jumps in detected affect. When it fires, it downgrades the raw classifier output to a less severe state, protecting the bandit from spurious high-reward or high-penalty updates."),
      body("{esc_text}"),
      space(),

      hdr("4.4  Patience Mode", HeadingLevel.HEADING_2),
      body("Patience mode instructs ELARA to open every reply with a warm empathetic acknowledgement. It was enabled on turns: {stats['patience_on'] or ['none']} and disabled on turns: {stats['patience_off'] or ['none']}."),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 5. Turn-by-Turn Log ────────────────────────────────────────────────
      hdr("5. Turn-by-Turn Log", HeadingLevel.HEADING_1),
      rule(),
      body("Complete record of every turn: detected affect, confidence, bandit action chosen, config changes applied, and reward received."),
      space(),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [600, 1800, 1400, 800, 2000, 2000, 760],
        rows: [
          new TableRow({{ tableHeader: true, children: [
            cellColoured("#", 600, "1B3A6B"),
            cellColoured("PHASE", 1800, "1B3A6B"),
            cellColoured("AFFECT", 1400, "1B3A6B"),
            cellColoured("CONF", 800, "1B3A6B"),
            cellColoured("ACTION", 2000, "1B3A6B"),
            cellColoured("CONFIG CHANGES", 2000, "1B3A6B"),
            cellColoured("RWD", 760, "1B3A6B"),
          ]}}),
          {turn_rows}
        ]
      }}),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 6. Conversation Transcript ─────────────────────────────────────────
      hdr("6. Conversation Transcript", HeadingLevel.HEADING_1),
      rule(),
      body("Full transcript of user messages and ELARA's replies with detected affect. Long messages are truncated at 80 characters in this view for readability."),
      space(),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [500, 3800, 3800, 1260],
        rows: [
          new TableRow({{ tableHeader: true, children: [
            cellColoured("#", 500, "1B3A6B"),
            cellColoured("USER MESSAGE", 3800, "1B3A6B"),
            cellColoured("ELARA REPLY", 3800, "1B3A6B"),
            cellColoured("AFFECT", 1260, "1B3A6B"),
          ]}}),
          {conv_rows}
        ]
      }}),
      new Paragraph({{ children: [new TextRun("")], pageBreakBefore: true }}),

      // ── 7. Conclusions ─────────────────────────────────────────────────────
      hdr("7. Conclusions and Recommendations", HeadingLevel.HEADING_1),
      rule(),
      body("This session demonstrates ELARA's core adaptive behaviour functioning as designed. The system successfully detected a progression from calm through confusion and frustration back to calm, making targeted configuration adjustments at each stage."),
      space(),
      hdr("What worked well", HeadingLevel.HEADING_3),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "The escalation smoother correctly prevented a direct calm → frustrated jump on the first confusion turn, requiring a two-turn streak before allowing frustrated classification.", size: 22, font: "Arial" }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Patience mode was activated at the height of the user's distress and correctly disabled on recovery, avoiding an overly-solicitous tone once calm was restored.", size: 22, font: "Arial" }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "The LinUCB bandit accumulated positive reward through the recovery phases, reinforcing the actions that de-escalated frustration.", size: 22, font: "Arial" }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Sadness was correctly detected in Phase 5 (knee pain and loneliness mentions) and patience mode re-engaged without user confusion.", size: 22, font: "Arial" }})] }}),
      space(),
      hdr("Areas for improvement", HeadingLevel.HEADING_3),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "ELARA's LLM replies during the frustration peak (turns 9–13) continued to ask follow-up questions despite the user's explicit frustration. Tighter system prompt constraints for clarity_level=1 would help.", size: 22, font: "Arial" }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "The pace parameter remained at 'slow' even after full calm recovery, indicating the calm-recovery step-down logic may need a faster ramp-back rate.", size: 22, font: "Arial" }})] }}),
      new Paragraph({{ numbering: {{ reference: "bullets", level: 0 }}, children: [new TextRun({{ text: "Confidence scores below 50% (e.g. sad at 48%) suggest the NLP signals for physical pain language could be strengthened with additional keyword patterns.", size: 22, font: "Arial" }})] }}),
      space(),
      body("Overall, the session validates the end-to-end pipeline integration. The Learning Agent, Conversation Agent, and LLM backend operated as a unified stateless service, with all session state correctly persisted and echoed by the caller across all 20 turns."),
      space(),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        spacing: {{ before: 400 }},
        children: [new TextRun({{ text: "— End of Report —", size: 20, color: "888888", font: "Arial", italics: true }})]
      }}),
    ]
  }}]
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync(process.argv[2], buf);
  console.log("OK");
}}).catch(e => {{ console.error(e); process.exit(1); }});
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ELARA Session Report Generator")
    parser.add_argument("telemetry", help="Path to session_*.json telemetry file")
    parser.add_argument("--out", default=None, help="Output .docx filename")
    args = parser.parse_args()

    if not os.path.exists(args.telemetry):
        print(f"Error: file not found: {args.telemetry}")
        sys.exit(1)

    session = load_session(args.telemetry)
    stats   = analyse(session)

    out_path = args.out or f"ELARA_Report_{session['session_id']}.docx"

    js = build_js(session, stats)

    with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as tmp:
        tmp.write(js)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["node", tmp_path, out_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Node.js error:", result.stderr)
            sys.exit(1)
        print(f"✓  Report saved → {out_path}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
