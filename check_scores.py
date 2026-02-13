#!/usr/bin/env python3
"""Check all trainX/testY holdout scores from experiments, ranked by total score.

Run as Flask app: python check_scores.py --serve [--port 5000]
Run as CLI: python check_scores.py [--include-all]
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Global config (set at startup)
ARCHIVE_ROOT = "output/experiments"
FILTER_AFTER_LEAK_FIX = True


def parse_exp_info(exp_dir: Path) -> dict | None:
    """Parse experiment info from directory name and manifest."""
    name = exp_dir.name

    # Parse timestamp from dirname (format: YYYYMMDD-HHMMSS-...)
    # The dirname is in UTC, convert to EST (UTC-5)
    try:
        timestamp_str = name[:15]
        timestamp_utc = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        timestamp_est = timestamp_utc - timedelta(hours=5)
        dt_formatted = timestamp_est.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return None

    # Extract commit hash (format: ...-HASH-... or ...-HASH)
    parts = name.split("-")
    commit = "unknown"
    if len(parts) >= 3:
        commit = parts[2][:8]

    # Load run_name from manifest if available
    run_name = ""
    manifest_path = exp_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                run_name = manifest.get("args", {}).get("run_name", "")
                if not commit or commit == "unknown":
                    commit = manifest.get("git_commit", "")[:8]
        except (json.JSONDecodeError, IOError):
            pass

    if not run_name and len(parts) >= 4:
        run_name = "-".join(parts[3:])

    return {
        "path": exp_dir,
        "name": name,
        "datetime": dt_formatted,
        "commit": commit,
        "run_name": run_name,
    }


def load_holdout_scores(exp_dir: Path) -> list[dict]:
    """Load all trainX/testY holdout scores from an experiment directory."""
    scores = []

    for lang_dir in exp_dir.glob("languages/*"):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name

        holdout_dir = lang_dir / "holdout"
        if not holdout_dir.exists():
            continue

        for metrics_file in holdout_dir.glob("*/metrics.json"):
            parts = metrics_file.parent.name.split("__")
            if len(parts) == 2 and parts[0].startswith("train_") and parts[1].startswith("test_"):
                train_id = parts[0].replace("train_", "")
                test_id = parts[1]

                try:
                    with open(metrics_file) as f:
                        data = json.load(f)

                    scores.append({
                        "language": lang,
                        "train": train_id,
                        "test": test_id,
                        "score": data.get("competition_score", 0),
                        "precision": data.get("precision", 0),
                        "recall": data.get("recall", 0),
                        "f1": data.get("f1", 0),
                        "tp": data.get("tp", 0),
                        "fp": data.get("fp", 0),
                        "fn": data.get("fn", 0),
                    })
                except (json.JSONDecodeError, IOError):
                    continue

    return scores


def is_experiment_after_leak_fix(exp_info: dict) -> bool:
    """Check if experiment was run after the test set leak fix.

    The fix was implemented around 2026-02-07 01:34 UTC = 2026-02-06 20:34 EST (commit 8db60deb).
    We filter out experiments before this date.
    """
    cutoff_date = datetime(2026, 2, 6, 20, 34, 0)  # EST
    exp_date = datetime.strptime(exp_info["datetime"], "%Y-%m-%d %H:%M")
    return exp_date >= cutoff_date


def find_all_experiments(archive_root: str, filter_after_leak_fix: bool = True) -> list[dict]:
    """Find all experiments with their scores, sorted by total score descending."""
    archive_path = Path(archive_root)
    if not archive_path.exists():
        return []

    experiments = []
    skipped_count = 0
    for item in archive_path.iterdir():
        if not item.is_dir():
            continue

        exp_info = parse_exp_info(item)
        if not exp_info:
            continue

        # Filter out experiments before the leak fix
        if filter_after_leak_fix and not is_experiment_after_leak_fix(exp_info):
            skipped_count += 1
            continue

        scores = load_holdout_scores(item)
        if not scores:
            continue

        total_score = sum(s["score"] for s in scores)
        avg_precision = sum(s["precision"] for s in scores) / len(scores)

        # Calculate per-language totals
        lang_totals = {}
        for s in scores:
            lang_totals[s["language"]] = lang_totals.get(s["language"], 0) + s["score"]

        exp_info["scores"] = scores
        exp_info["total_score"] = total_score
        exp_info["avg_precision"] = avg_precision
        exp_info["lang_totals"] = lang_totals
        experiments.append(exp_info)

    # Sort by total score descending, then by datetime descending
    experiments.sort(key=lambda x: (-x["total_score"], -datetime.strptime(x["datetime"], "%Y-%m-%d %H:%M").timestamp()))

    if filter_after_leak_fix and skipped_count > 0:
        print(f"  (Filtered out {skipped_count} experiment(s) from before the leak fix)\n")

    return experiments


def print_all_scores(experiments: list[dict]):
    """Print all experiments ranked by score."""
    if not experiments:
        print("No experiments with holdout scores found")
        return

    # Calculate max possible scores based on actual bot counts
    # EN datasets: 30 (66 bots) + 32 (63 bots) = 129 bots -> max 4*129 = 516
    # FR datasets: 31 (27 bots) + 33 (28 bots) = 55 bots -> max 4*55 = 220
    # Total: 184 bots -> max 4*184 = 736
    max_en = 4 * (66 + 63)  # 516
    max_fr = 4 * (27 + 28)  # 220
    max_total = max_en + max_fr  # 736

    print(f"\n{'='*100}")
    print(f"{'RANK':<5} {'DATE':<16} {'COMMIT':<10} {'RUN NAME':<25} {'EN':>8} {'FR':>8} {'TOTAL':>8} {'AVG PREC':>8}")
    print(f"{'='*100}")

    for rank, exp in enumerate(experiments, 1):
        en_score = exp["lang_totals"].get("en", 0)
        fr_score = exp["lang_totals"].get("fr", 0)
        run_name = exp["run_name"][:24] if exp["run_name"] else "-"
        print(f"{rank:<5} {exp['datetime']:<16} {exp['commit']:<10} {run_name:<25} {en_score:>8.0f} {fr_score:>8.0f} {exp['total_score']:>8.0f} {exp['avg_precision']:>8.3f}")

    print(f"{'='*100}")

    # Summary at bottom
    best = experiments[0]
    best_en = best["lang_totals"].get("en", 0)
    best_fr = best["lang_totals"].get("fr", 0)

    print(f"\n{'─'*100}")
    print(f"SUMMARY")
    print(f"{'─'*100}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Best total score:  {best['total_score']:.0f}")
    print(f"")
    print(f"  Score breakdown (best experiment):")
    print(f"    EN:  {best_en:>6.0f} / {max_en}  ({100*best_en/max_en:.1f}%)")
    print(f"    FR:  {best_fr:>6.0f} / {max_fr}  ({100*best_fr/max_fr:.1f}%)")
    print(f"    ALL: {best['total_score']:>6.0f} / {max_total}  ({100*best['total_score']/max_total:.1f}%)")
    print(f"{'─'*100}\n")

    # Print detailed breakdown for top 5
    for rank, exp in enumerate(experiments[:5], 1):
        print_exp_detail(exp, rank)


def print_exp_detail(exp: dict, rank: int):
    """Print detailed scores for a single experiment."""
    # Max scores based on actual bot counts per dataset:
    # EN: dataset 30 (66 bots), dataset 32 (63 bots)
    # FR: dataset 31 (27 bots), dataset 33 (28 bots)
    # Max score = 4*TP - 2*FP - 1*FN where ideally TP=all_bots, FP=0, FN=0
    bots_per_dataset = {"30": 66, "31": 27, "32": 63, "33": 28, "test_30": 66, "test_31": 27, "test_32": 63, "test_33": 28}

    print(f"\n{'─'*100}")
    print(f"#{rank} | {exp['datetime']} | commit: {exp['commit']} | {exp['run_name']}")
    print(f"{'─'*100}")

    by_lang: dict[str, list[dict]] = {}
    for s in exp["scores"]:
        by_lang.setdefault(s["language"], []).append(s)

    grand_total = 0
    grand_max = 0

    for lang in sorted(by_lang.keys()):
        lang_total = sum(s["score"] for s in by_lang[lang])
        # Calculate actual max for this language based on test datasets
        lang_max = sum(4 * bots_per_dataset.get(s["test"], 0) for s in by_lang[lang])
        grand_total += lang_total
        grand_max += lang_max

        print(f"\n  {lang.upper()} (subtotal: {lang_total:.0f} / {lang_max})")
        print(f"  {'-'*80}")
        print(f"  {'Train->Test':<12} {'Score':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
        print(f"  {'-'*80}")

        for s in sorted(by_lang[lang], key=lambda x: (x["train"], x["test"])):
            label = f"{s['train']}->{s['test']}"
            print(f"  {label:<12} {s['score']:>8.1f} {s['precision']:>8.3f} {s['recall']:>8.3f} {s['f1']:>8.3f} {s['tp']:>4.0f} {s['fp']:>4.0f} {s['fn']:>4.0f}")

    # Summary for this experiment
    print(f"\n  {'='*80}")
    if grand_max > 0:
        print(f"  TOTAL: {grand_total:.0f} / {grand_max} ({100*grand_total/grand_max:.1f}%)")
    else:
        print(f"  TOTAL: {grand_total:.0f} / {grand_max} (N/A)")
    print(f"  {'='*80}")


# HTML Templates
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot Detection Experiments Leaderboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #94a3b8; margin-bottom: 20px; }
        .stats-bar {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            padding: 15px;
            background: #1e293b;
            border-radius: 12px;
        }
        .stat {
            padding: 10px 20px;
            background: #334155;
            border-radius: 8px;
            min-width: 120px;
        }
        .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: #60a5fa; }
        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            align-items: center;
        }
        .controls select, .controls input {
            padding: 8px 12px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 0.9rem;
        }
        .controls label { color: #94a3b8; font-size: 0.9rem; }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }
        th {
            background: #0f172a;
            font-weight: 600;
            color: #94a3b8;
            cursor: pointer;
            user-select: none;
            position: relative;
        }
        th:hover { background: #1e293b; }
        th.sort-asc::after { content: " ▲"; color: #60a5fa; }
        th.sort-desc::after { content: " ▼"; color: #60a5fa; }
        tr:hover { background: #252f47; }
        tr.best { background: rgba(34, 197, 94, 0.1); }
        tr.best td { border-bottom-color: rgba(34, 197, 94, 0.3); }
        .rank { font-weight: 700; font-size: 1.1rem; }
        .rank-1 { color: #fbbf24; }
        .rank-2 { color: #94a3b8; }
        .rank-3 { color: #b45309; }
        .score { font-weight: 700; color: #60a5fa; }
        .score-high { color: #22c55e; }
        .score-med { color: #fbbf24; }
        .score-low { color: #ef4444; }
        .commit { font-family: monospace; font-size: 0.85rem; color: #94a3b8; }
        .date { color: #94a3b8; font-size: 0.9rem; }
        .run-name { max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .lang-score { font-family: monospace; }
        .precision { color: #a78bfa; }
        a { color: #60a5fa; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .detail-row { display: none; background: #0f172a; }
        .detail-row.active { display: table-row; }
        .detail-content {
            padding: 20px;
            background: #0f172a;
            border-left: 3px solid #60a5fa;
        }
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .detail-section {
            background: #1e293b;
            padding: 15px;
            border-radius: 8px;
        }
        .detail-section h4 {
            color: #60a5fa;
            margin-bottom: 10px;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        .detail-table {
            width: 100%;
            font-size: 0.85rem;
        }
        .detail-table th, .detail-table td {
            padding: 6px 10px;
            border-bottom: 1px solid #334155;
        }
        .detail-table th {
            background: transparent;
            font-weight: 500;
        }
        .toggle-btn {
            background: #334155;
            border: none;
            color: #60a5fa;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .toggle-btn:hover { background: #475569; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #94a3b8;
        }
        .progress-bar {
            height: 6px;
            background: #334155;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 4px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Bot Detection Experiments</h1>
        <p class="subtitle">Holdout evaluation leaderboard (trainX → testY)</p>
        
        <div class="stats-bar">
            <div class="stat">
                <div class="stat-label">Total Experiments</div>
                <div class="stat-value">{{ total_exps }}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Best Score</div>
                <div class="stat-value">{{ best_score }}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Best EN</div>
                <div class="stat-value">{{ best_en }}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Best FR</div>
                <div class="stat-value">{{ best_fr }}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Max Possible</div>
                <div class="stat-value">736</div>
            </div>
        </div>

        <div class="controls">
            <label>Sort by:</label>
            <select id="sortSelect" onchange="sortTable()">
                <option value="rank">Rank (Score Desc)</option>
                <option value="date">Date (Newest First)</option>
                <option value="date_asc">Date (Oldest First)</option>
                <option value="en">EN Score</option>
                <option value="fr">FR Score</option>
                <option value="precision">Avg Precision</option>
                <option value="name">Run Name</option>
            </select>
            <label>Filter:</label>
            <input type="text" id="filterInput" placeholder="Search run name or commit..." onkeyup="filterTable()">
            <label>Min Score:</label>
            <input type="number" id="minScore" placeholder="0" onchange="filterTable()" style="width: 80px;">
        </div>

        <table id="leaderboard">
            <thead>
                <tr>
                    <th onclick="sortBy('rank')">Rank</th>
                    <th onclick="sortBy('date')">Date (EST)</th>
                    <th onclick="sortBy('commit')">Commit</th>
                    <th onclick="sortBy('name')">Run Name</th>
                    <th onclick="sortBy('en')" style="text-align:right">EN Score</th>
                    <th onclick="sortBy('fr')" style="text-align:right">FR Score</th>
                    <th onclick="sortBy('total')" style="text-align:right">Total</th>
                    <th onclick="sortBy('precision')" style="text-align:right">Avg Prec</th>
                    <th style="text-align:center">Details</th>
                </tr>
            </thead>
            <tbody>
                {% for exp in experiments %}
                <tr class="exp-row {% if loop.index == 1 %}best{% endif %}" data-id="{{ exp.name }}" data-rank="{{ loop.index }}" data-date="{{ exp.timestamp }}" data-en="{{ exp.lang_totals.get('en', 0) }}" data-fr="{{ exp.lang_totals.get('fr', 0) }}" data-total="{{ exp.total_score }}" data-precision="{{ exp.avg_precision }}" data-name="{{ exp.run_name.lower() }}" data-commit="{{ exp.commit.lower() }}">
                    <td class="rank-cell"><span class="rank rank-{{ loop.index if loop.index <= 3 else '' }}">#{{ loop.index }}</span></td>
                    <td class="date">{{ exp.datetime }}</td>
                    <td class="commit"><a href="https://github.com/q/bot-or-not/commit/{{ exp.commit }}" target="_blank">{{ exp.commit }}</a></td>
                    <td class="run-name" title="{{ exp.run_name }}">{{ exp.run_name or '-' }}</td>
                    <td class="lang-score" style="text-align:right">{{ "%.0f"|format(exp.lang_totals.get('en', 0)) }}</td>
                    <td class="lang-score" style="text-align:right">{{ "%.0f"|format(exp.lang_totals.get('fr', 0)) }}</td>
                    <td class="score {% if exp.total_score >= 600 %}score-high{% elif exp.total_score >= 400 %}score-med{% else %}score-low{% endif %}" style="text-align:right">{{ "%.0f"|format(exp.total_score) }}</td>
                    <td class="precision" style="text-align:right">{{ "%.3f"|format(exp.avg_precision) }}</td>
                    <td style="text-align:center"><button class="toggle-btn" onclick="toggleDetail('{{ exp.name }}')">View</button></td>
                </tr>
                <tr id="detail-{{ exp.name }}" class="detail-row" data-parent="{{ exp.name }}">
                    <td colspan="9">
                        <div class="detail-content">
                            <div class="detail-grid">
                                {% for lang, scores in exp.by_lang.items() %}
                                <div class="detail-section">
                                    <h4>{{ lang.upper() }} Details</h4>
                                    <table class="detail-table">
                                        <thead>
                                            <tr>
                                                <th>Train→Test</th>
                                                <th style="text-align:right">Score</th>
                                                <th style="text-align:right">Prec</th>
                                                <th style="text-align:right">Rec</th>
                                                <th style="text-align:right">F1</th>
                                                <th style="text-align:right">TP</th>
                                                <th style="text-align:right">FP</th>
                                                <th style="text-align:right">FN</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for s in scores %}
                                            <tr>
                                                <td>{{ s.train }}→{{ s.test }}</td>
                                                <td style="text-align:right; font-weight:600">{{ "%.1f"|format(s.score) }}</td>
                                                <td style="text-align:right">{{ "%.3f"|format(s.precision) }}</td>
                                                <td style="text-align:right">{{ "%.3f"|format(s.recall) }}</td>
                                                <td style="text-align:right">{{ "%.3f"|format(s.f1) }}</td>
                                                <td style="text-align:right">{{ "%.0f"|format(s.tp) }}</td>
                                                <td style="text-align:right">{{ "%.0f"|format(s.fp) }}</td>
                                                <td style="text-align:right">{{ "%.0f"|format(s.fn) }}</td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if not experiments %}
        <div class="empty-state">
            <h3>No experiments found</h3>
            <p>Run some experiments first with: <code>uv run train.py --archive-root output/experiments</code></p>
        </div>
        {% endif %}
    </div>

    <script>
        // Sorting state
        let currentSort = 'rank';
        let sortDirection = 1;  // 1 = ascending, -1 = descending

        // Default direction for each column (1 = asc, -1 = desc)
        const defaultDirections = {
            'rank': -1,      // best rank first (#1 at top)
            'date': -1,      // newest first
            'date_asc': 1,   // oldest first
            'en': -1,        // highest score first
            'fr': -1,
            'total': -1,
            'precision': -1,
            'name': 1,
            'commit': 1
        };

        function sortTable() {
            const select = document.getElementById('sortSelect');
            sortBy(select.value);
        }

        function sortBy(column) {
            const tbody = document.querySelector('#leaderboard tbody');
            const rows = Array.from(tbody.querySelectorAll('tr.exp-row'));

            // Determine sort direction
            if (currentSort === column) {
                // Toggle direction if clicking same column
                sortDirection *= -1;
            } else {
                // New column - use default direction
                sortDirection = defaultDirections[column] || 1;
                currentSort = column;

                // Update select dropdown if called from header click
                const select = document.getElementById('sortSelect');
                if (select && select.value !== column) {
                    select.value = column;
                }
            }

            rows.sort((a, b) => {
                let aVal, bVal;
                switch(column) {
                    case 'rank':
                        aVal = parseInt(a.dataset.rank);
                        bVal = parseInt(b.dataset.rank);
                        break;
                    case 'date':
                    case 'date_asc':
                        aVal = parseInt(a.dataset.date);
                        bVal = parseInt(b.dataset.date);
                        break;
                    case 'en':
                        aVal = parseFloat(a.dataset.en);
                        bVal = parseFloat(b.dataset.en);
                        break;
                    case 'fr':
                        aVal = parseFloat(a.dataset.fr);
                        bVal = parseFloat(b.dataset.fr);
                        break;
                    case 'total':
                        aVal = parseFloat(a.dataset.total);
                        bVal = parseFloat(b.dataset.total);
                        break;
                    case 'precision':
                        aVal = parseFloat(a.dataset.precision);
                        bVal = parseFloat(b.dataset.precision);
                        break;
                    case 'name':
                        aVal = a.dataset.name;
                        bVal = b.dataset.name;
                        break;
                    case 'commit':
                        aVal = a.dataset.commit;
                        bVal = b.dataset.commit;
                        break;
                    default:
                        return 0;
                }

                if (typeof aVal === 'string') {
                    return sortDirection * aVal.localeCompare(bVal);
                }
                return sortDirection * (aVal - bVal);
            });

            // Reorder rows in DOM and update visual ranks
            rows.forEach((row, index) => {
                // Get the original rank (sticky to experiment)
                const originalRank = parseInt(row.dataset.rank);
                const rankCell = row.querySelector('.rank-cell span');

                // Keep the original rank display (sticky)
                rankCell.textContent = '#' + originalRank;
                rankCell.className = 'rank' + (originalRank <= 3 ? ' rank-' + originalRank : '');

                // Update best row highlighting based on original rank
                row.classList.remove('best');
                if (originalRank === 1) row.classList.add('best');

                // Move row to new position
                tbody.appendChild(row);

                // Move corresponding detail row right after
                const expId = row.dataset.id;
                const detailRow = document.getElementById('detail-' + expId);
                if (detailRow) {
                    detailRow.classList.remove('active');
                    tbody.appendChild(detailRow);
                }
            });

            // Update header indicators
            document.querySelectorAll('th').forEach(th => th.classList.remove('sort-asc', 'sort-desc'));
            const thIndex = {rank:0, date:1, commit:2, name:3, en:4, fr:5, total:6, precision:7}[column];
            if (thIndex !== undefined) {
                document.querySelectorAll('th')[thIndex].classList.add(sortDirection > 0 ? 'sort-asc' : 'sort-desc');
            }
        }

        function filterTable() {
            const filter = document.getElementById('filterInput').value.toLowerCase();
            const minScore = parseFloat(document.getElementById('minScore').value) || 0;
            const rows = document.querySelectorAll('#leaderboard tbody tr.exp-row');
            
            rows.forEach(row => {
                const name = row.dataset.name;
                const commit = row.dataset.commit;
                const total = parseFloat(row.dataset.total);
                const match = (name.includes(filter) || commit.includes(filter)) && total >= minScore;
                row.style.display = match ? '' : 'none';
                
                // Hide detail row when filtering
                const expId = row.dataset.id;
                const detailRow = document.getElementById('detail-' + expId);
                if (detailRow) {
                    detailRow.style.display = 'none';
                    detailRow.classList.remove('active');
                }
            });
        }

        function toggleDetail(expName) {
            const detailRow = document.getElementById('detail-' + expName);
            if (detailRow) {
                detailRow.classList.toggle('active');
            }
        }
    </script>
</body>
</html>
"""


def prepare_experiments_for_template(experiments):
    """Add computed fields for template rendering."""
    for exp in experiments:
        # Group scores by language
        by_lang = {}
        for s in exp.get('scores', []):
            by_lang.setdefault(s['language'], []).append(s)
        
        # Sort within each language
        for lang in by_lang:
            by_lang[lang].sort(key=lambda x: (x['train'], x['test']))
        
        exp['by_lang'] = by_lang
        
        # Parse timestamp for sorting
        try:
            exp['timestamp'] = int(datetime.strptime(exp['datetime'], "%Y-%m-%d %H:%M").timestamp())
        except:
            exp['timestamp'] = 0
    return experiments


@app.route('/')
def index():
    experiments = find_all_experiments(ARCHIVE_ROOT, filter_after_leak_fix=FILTER_AFTER_LEAK_FIX)
    experiments = prepare_experiments_for_template(experiments)
    
    best_score = experiments[0]['total_score'] if experiments else 0
    best_en = experiments[0]['lang_totals'].get('en', 0) if experiments else 0
    best_fr = experiments[0]['lang_totals'].get('fr', 0) if experiments else 0
    
    return render_template_string(INDEX_TEMPLATE,
        experiments=experiments,
        total_exps=len(experiments),
        best_score=int(best_score),
        best_en=int(best_en),
        best_fr=int(best_fr)
    )


@app.route('/api/experiments')
def api_experiments():
    """JSON API for experiments data."""
    experiments = find_all_experiments(ARCHIVE_ROOT, filter_after_leak_fix=FILTER_AFTER_LEAK_FIX)
    # Convert Path objects to strings for JSON serialization
    serializable_exps = []
    for exp in experiments:
        exp_copy = {k: (str(v) if isinstance(v, Path) else v) for k, v in exp.items()}
        serializable_exps.append(exp_copy)
    return jsonify({
        'experiments': serializable_exps,
        'total': len(serializable_exps),
        'best_score': serializable_exps[0]['total_score'] if serializable_exps else 0
    })


@app.route('/api/experiment/<path:exp_name>')
def api_experiment_detail(exp_name):
    """JSON API for single experiment details."""
    exp_path = Path(ARCHIVE_ROOT) / exp_name
    if not exp_path.exists():
        return jsonify({'error': 'Not found'}), 404
    
    exp_info = parse_exp_info(exp_path)
    if not exp_info:
        return jsonify({'error': 'Invalid experiment'}), 400
    
    scores = load_holdout_scores(exp_path)
    exp_info['scores'] = scores
    exp_info['total_score'] = sum(s['score'] for s in scores)
    exp_info['avg_precision'] = sum(s['precision'] for s in scores) / len(scores) if scores else 0
    
    # Convert Path objects to strings for JSON serialization
    exp_info = {k: (str(v) if isinstance(v, Path) else v) for k, v in exp_info.items()}
    return jsonify(exp_info)


def serve_flask(port=5000):
    """Run the Flask development server."""
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    print(f"Archive root: {ARCHIVE_ROOT}")
    print(f"Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=port, debug=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check trainX/testY holdout scores")
    parser.add_argument("--include-all", action="store_true", help="Include experiments from before the leak fix")
    parser.add_argument("--serve", action="store_true", help="Run as Flask web app")
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server (default: 5000)")
    args = parser.parse_args()

    global ARCHIVE_ROOT, FILTER_AFTER_LEAK_FIX
    ARCHIVE_ROOT = os.environ.get("ARCHIVE_ROOT", "output/experiments")
    FILTER_AFTER_LEAK_FIX = not args.include_all

    if args.serve:
        serve_flask(args.port)
    else:
        experiments = find_all_experiments(ARCHIVE_ROOT, filter_after_leak_fix=FILTER_AFTER_LEAK_FIX)
        print_all_scores(experiments)


if __name__ == "__main__":
    main()
