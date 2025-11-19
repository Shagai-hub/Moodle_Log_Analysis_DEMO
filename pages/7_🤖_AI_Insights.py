import re
import textwrap
from collections import Counter
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd
import streamlit as st

from assets.ui_components import apply_theme, divider, info_panel, page_header, section_header, nav_footer
from utils.config_manager import ConfigManager
from utils.session_data_manager import SessionDataManager
from utils.ui_steps import render_steps


render_steps(active="3 Interpret")
apply_theme()

if "data_manager" not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if "config" not in st.session_state:
    st.session_state.config = ConfigManager()


@st.cache_data(show_spinner=False)
def generate_model_summary(prompt, model_name, temperature):
    """Generate a summary with a lightweight local model."""
    if not model_name or str(model_name).lower() in {"manual", "none", "disabled"}:
        return {"summary": None, "error": "manual_mode"}

    try:
        from transformers import pipeline

        generator = pipeline("text2text-generation", model=model_name)
        generation_kwargs = {
            "max_length": 200,
            "min_length": 40,
        }
        if temperature and float(temperature) > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = float(temperature)
        else:
            generation_kwargs["do_sample"] = False
        outputs = generator(prompt, **generation_kwargs)
        text = outputs[0]["generated_text"].strip() if outputs else ""
        return {"summary": text, "error": None}
    except Exception as exc:  # pragma: no cover - defensive for missing models
        return {"summary": None, "error": str(exc)}


def resolve_column(df, column_name):
    if not column_name or df is None or df.empty:
        return None
    if column_name in df.columns:
        return column_name
    lower_map = {col.lower(): col for col in df.columns}
    return lower_map.get(column_name.lower())


def build_student_key(userid, userfullname, fallback=None):
    if userid is not None and not (isinstance(userid, float) and np.isnan(userid)):
        return f"id::{userid}"
    if isinstance(userfullname, str) and userfullname.strip():
        return f"name::{userfullname.strip().lower()}"
    return f"row::{fallback}" if fallback is not None else None


def extract_metrics(row, resolved_columns):
    metrics = {}
    if row is None:
        return metrics
    for slug, column in resolved_columns.items():
        if column and column in row.index:
            metrics[slug] = row[column]
    return metrics


def compute_metrics_snapshot(df):
    snapshot = {
        "total_students": int(len(df)),
        "generated_at": datetime.utcnow().isoformat(),
    }
    if df is None or df.empty:
        return snapshot

    col_posts = resolve_column(df, "total_posts")
    if col_posts:
        posts = pd.to_numeric(df[col_posts], errors="coerce")
        snapshot["avg_posts"] = posts.mean()
        snapshot["median_posts"] = posts.median()
        snapshot["low_post_count"] = int((posts < 1).sum())
    col_eng = resolve_column(df, "engagement_rate")
    if col_eng:
        engagement = pd.to_numeric(df[col_eng], errors="coerce")
        snapshot["avg_engagement"] = engagement.mean()
        snapshot["low_engagement_count"] = int((engagement < 0.2).sum())
    deadline_cols = [
        resolve_column(df, "deadline_exceeded_posts_Quasi_exam_I"),
        resolve_column(df, "deadline_exceeded_posts_Quasi_exam_II"),
        resolve_column(df, "deadline_exceeded_posts_Quasi_exam_III"),
    ]
    deadline_counts = []
    for col in deadline_cols:
        if col:
            deadline_counts.append(pd.to_numeric(df[col], errors="coerce").fillna(0))
    if deadline_counts:
        combined = sum(deadline_counts)
        snapshot["avg_deadline_misses"] = combined.mean()
    replies_col = resolve_column(df, "total_replies_to_professor")
    if replies_col:
        replies = pd.to_numeric(df[replies_col], errors="coerce")
        snapshot["low_professor_reply_count"] = int((replies <= 1).sum())
    return snapshot


def compute_social_connectedness(df):
    """Classify isolated vs highly connected students using interaction metrics."""
    result = {
        "available": False,
        "isolated": [],
        "connected": [],
        "thresholds": {},
        "missing_columns": ["unique_discussions", "unique_interactions", "total_posts", "engagement_rate"],
        "summary": {"total": 0, "isolated": 0, "connected": 0},
    }
    if df is None or df.empty:
        return result

    resolved = {
        "unique_discussions": resolve_column(df, "unique_discussions"),
        "unique_interactions": resolve_column(df, "unique_interactions"),
        "total_posts": resolve_column(df, "total_posts"),
        "engagement_rate": resolve_column(df, "engagement_rate"),
    }
    missing = [slug for slug, col in resolved.items() if not col]
    active_cols = {slug: col for slug, col in resolved.items() if col}
    if not active_cols:
        result["missing_columns"] = missing
        return result

    numeric_df = df[["userid", "userfullname"] + list(active_cols.values())].copy()
    for col in active_cols.values():
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    thresholds = {slug: float(numeric_df[col].median()) for slug, col in active_cols.items()}
    weights = {
        "unique_discussions": 1.6,
        "unique_interactions": 1.6,
        "total_posts": 1.0,
        "engagement_rate": 1.0,
    }
    available_weight = sum(weights.get(slug, 1.0) for slug in active_cols)
    connected_cutoff = max(2.4, available_weight * 0.55)
    isolated_cutoff = max(1.2, available_weight * 0.3)
    labels = {
        "unique_discussions": "unique discussions",
        "unique_interactions": "peer interactions",
        "total_posts": "posts",
        "engagement_rate": "engagement rate",
    }

    isolated_entries = []
    connected_entries = []

    for _, row in numeric_df.iterrows():
        warnings = []
        highlights = []
        metrics = {}
        network_score = 0.0

        for slug, col in active_cols.items():
            value = row.get(col)
            metrics[slug] = value
            weight = weights.get(slug, 1.0)
            threshold = thresholds.get(slug)
            if threshold is None or pd.isna(threshold) or pd.isna(value):
                continue
            if value >= threshold:
                network_score += weight
                if slug in {"unique_discussions", "unique_interactions"}:
                    highlights.append(f"strong {labels[slug]}")
            else:
                label = labels.get(slug, slug.replace("_", " "))
                if slug in {"unique_discussions", "unique_interactions"}:
                    count_repr = format_metric_value(value, 0)
                    warnings.append(f"only {count_repr} {label}")
                elif slug == "engagement_rate":
                    warnings.append("engagement below cohort median")
                elif slug == "total_posts":
                    warnings.append("low post volume")

        entry = {
            "userfullname": row.get("userfullname") or (f"ID {int(row['userid'])}" if row.get("userid") else "Unknown"),
            "userid": row.get("userid"),
            "metrics": metrics,
            "network_score": round(network_score, 2),
        }

        if network_score >= connected_cutoff:
            entry["highlights"] = highlights or ["Consistent participation footprint"]
            connected_entries.append(entry)
        elif network_score <= isolated_cutoff or len(warnings) >= 2:
            entry["warnings"] = warnings or ["Sparse participation signals"]
            isolated_entries.append(entry)

    connected_entries = sorted(connected_entries, key=lambda row: row["network_score"], reverse=True)[:20]
    isolated_entries = sorted(isolated_entries, key=lambda row: row["network_score"])[:20]

    result.update({
        "available": True,
        "connected": connected_entries,
        "isolated": isolated_entries,
        "thresholds": thresholds,
        "missing_columns": missing,
        "summary": {
            "total": int(len(numeric_df)),
            "isolated": len(isolated_entries),
            "connected": len(connected_entries),
        },
    })
    return result

def detect_anomalies(df, settings):
    anomalies = []
    missing_columns = set()
    if df is None or df.empty:
        return anomalies, missing_columns

    low_threshold = float(settings.get("zscore_low_threshold", -1.5))
    high_threshold = float(settings.get("zscore_high_threshold", 2.5))

    columns_to_check = [
        {"column": "engagement_rate", "label": "Engagement rate", "direction": "low"},
        {"column": "total_posts", "label": "Total posts", "direction": "low"},
        {"column": "deadline_exceeded_posts_Quasi_exam_I", "label": "Deadline misses (Exam I)", "direction": "high"},
        {"column": "deadline_exceeded_posts_Quasi_exam_II", "label": "Deadline misses (Exam II)", "direction": "high"},
    ]

    for item in columns_to_check:
        resolved = resolve_column(df, item["column"])
        if not resolved:
            missing_columns.add(item["column"])
            continue
        series = pd.to_numeric(df[resolved], errors="coerce")
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        zscores = (series - mean) / std
        if item["direction"] == "low":
            mask = zscores <= low_threshold
        else:
            mask = zscores >= high_threshold
        for idx in df.index[mask]:
            row = df.loc[idx]
            z_value = zscores.loc[idx]
            anomalies.append({
                "userid": row.get("userid"),
                "userfullname": row.get("userfullname"),
                "row_index": idx,
                "column": resolved,
                "label": item["label"],
                "value": row.get(resolved),
                "zscore": float(z_value),
                "direction": item["direction"],
                "description": f"{item['label']} {'low' if item['direction']=='low' else 'high'} ({z_value:+.1f} σ)",
            })
    return anomalies, missing_columns


def assemble_watchlist(df, rule_results, anomalies, metric_columns):
    severity_rank = {"high": 3, "medium": 2, "low": 1}
    row_lookup = {}
    for idx, row in df.iterrows():
        key = build_student_key(row.get("userid"), row.get("userfullname"), idx)
        row_lookup[key] = row

    resolved_columns = {slug: resolve_column(df, slug) for slug in metric_columns}
    watch_map = {}

    def ensure_entry(userid, userfullname, fallback=None):
        key = build_student_key(userid, userfullname, fallback)
        if key is None:
            key = f"adhoc::{len(watch_map)}"
        if key not in watch_map:
            row = row_lookup.get(key)
            metrics = extract_metrics(row, resolved_columns)
            watch_map[key] = {
                "userid": userid if not (isinstance(userid, float) and np.isnan(userid)) else None,
                "userfullname": userfullname,
                "flags": [],
                "messages": [],
                "playbooks": [],
                "severity": "low",
                "severity_rank": 0,
                "anomalies": [],
                "metrics": metrics,
            }
        return watch_map[key]

    for entry in rule_results.get("per_student", []):
        student = ensure_entry(entry.get("userid"), entry.get("userfullname"), entry.get("row_index"))
        for rule in entry.get("rules", []):
            label = rule.get("label")
            if label and label not in student["flags"]:
                student["flags"].append(label)
            message = rule.get("message")
            if message and message not in student["messages"]:
                student["messages"].append(message)
            playbook = rule.get("playbook") or []
            for step in playbook:
                if step and step not in student["playbooks"]:
                    student["playbooks"].append(step)
            severity = rule.get("severity", "medium")
            rank = severity_rank.get(severity, 1)
            if rank > student["severity_rank"]:
                student["severity_rank"] = rank
                student["severity"] = severity

    for anomaly in anomalies:
        student = ensure_entry(anomaly.get("userid"), anomaly.get("userfullname"), anomaly.get("row_index"))
        desc = anomaly.get("description")
        if desc and desc not in student["anomalies"]:
            student["anomalies"].append(desc)
        # An anomaly without explicit rule should still mark severity to medium
        if student["severity_rank"] < 2:
            student["severity_rank"] = 2
            student["severity"] = "medium"

    watchlist = [entry for entry in watch_map.values() if entry["flags"] or entry["anomalies"]]
    watchlist.sort(key=lambda item: (-item["severity_rank"], -(len(item["flags"]) + len(item["anomalies"])) ))
    return watchlist


def build_summary_prompt(snapshot, watchlist, anomalies):
    lines = [
        "Summarize the following cohort metrics and flag notable risks.",
        f"Total students: {snapshot.get('total_students', 0)}",
    ]
    if "avg_engagement" in snapshot:
        lines.append(f"Average engagement rate: {snapshot['avg_engagement']:.2f}")
    if "avg_posts" in snapshot:
        lines.append(f"Average posts: {snapshot['avg_posts']:.1f}")
    if "avg_deadline_misses" in snapshot:
        lines.append(f"Average deadline misses: {snapshot['avg_deadline_misses']:.1f}")
    lines.append(f"Students flagged: {len(watchlist)}")
    flagged_names = [entry["userfullname"] or f"ID {entry['userid']}" for entry in watchlist][:5]
    if flagged_names:
        lines.append("Flagged students: " + ", ".join(flagged_names))
    if anomalies:
        anomaly_text = ", ".join(anomaly["description"] for anomaly in anomalies[:5])
        lines.append(f"Detected anomalies: {anomaly_text}")
    return "\n".join(lines)


def build_manual_summary(snapshot, watchlist, anomalies):
    total = snapshot.get("total_students", 0) or 0
    flagged = len(watchlist)
    parts = []
    parts.append(f"{flagged} student{'s' if flagged != 1 else ''} on the watchlist out of {total}.")
    avg_eng = snapshot.get("avg_engagement")
    if avg_eng is not None:
        parts.append(f"Average engagement rate is {avg_eng:.2f}.")
    avg_posts = snapshot.get("avg_posts")
    if avg_posts is not None:
        parts.append(f"Learners post {avg_posts:.1f} messages on average.")
    if anomalies:
        key_anomaly = anomalies[0]
        parts.append(f"Notable anomaly: {key_anomaly['description']}.")
    if flagged:
        names = ", ".join((entry["userfullname"] or f"ID {entry['userid']}") for entry in watchlist[:3])
        parts.append(f"Focus attention on: {names}.")
    return " ".join(parts)


def generate_sample_notifications(watchlist, metrics_snapshot, max_notifications=3):
    notifications = []
    if not watchlist:
        return notifications
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    for entry in watchlist[:max_notifications]:
        name = entry["userfullname"] or (f"ID {entry['userid']}" if entry.get("userid") else "Unnamed student")
        headline = entry["flags"][0] if entry["flags"] else entry["anomalies"][0]
        metrics = entry.get("metrics", {})
        metrics_bits = []
        if "total_posts" in metrics and metrics["total_posts"] is not None:
            metrics_bits.append(f"posts={metrics['total_posts']}")
        if "engagement_rate" in metrics and metrics["engagement_rate"] is not None:
            metrics_bits.append(f"engagement={float(metrics['engagement_rate']):.2f}")
        if "total_replies_to_professor" in metrics and metrics["total_replies_to_professor"] is not None:
            metrics_bits.append(f"replies_to_prof={metrics['total_replies_to_professor']}")
        if "avg_reply_time" in metrics and metrics["avg_reply_time"] is not None:
            metrics_bits.append(f"avg_reply={metrics['avg_reply_time']}")
        metric_summary = ", ".join(metrics_bits) if metrics_bits else "limited metrics available"
        playbook = entry.get("playbooks") or ["Review activity log and reach out with a supportive message."]
        body = textwrap.dedent(f"""
        Hello Instructor,

        Student {name} triggered the '{headline}' indicator. Recent metrics: {metric_summary}.

        Suggested next steps:
        - {playbook[0]}
        """).strip()
        body += f"\n\nGenerated automatically on {timestamp}."
        notifications.append({
            "student": name,
            "subject": f"Watchlist indicator for {name}",
            "body": body,
        })
    return notifications


def format_watchlist_dataframe(watchlist):
    if not watchlist:
        return pd.DataFrame()
    rows = []
    for entry in watchlist:
        metrics = entry.get("metrics", {})
        rows.append({
            "Student": entry["userfullname"] or (f"ID {entry['userid']}" if entry.get("userid") else "Unknown"),
            "Severity": entry.get("severity", "medium").title(),
            "Flags": ", ".join(entry["flags"]) if entry["flags"] else "—",
            "Anomalies": ", ".join(entry["anomalies"]) if entry["anomalies"] else "—",
            "Posts": metrics.get("total_posts"),
            "Engagement": f"{float(metrics['engagement_rate']):.2f}" if metrics.get("engagement_rate") is not None else None,
            "Deadline Misses": metrics.get("deadline_exceeded_posts_Quasi_exam_I"),
            "Replies to Prof": metrics.get("total_replies_to_professor"),
            "Avg Reply (h)": metrics.get("avg_reply_time"),
        })
    df = pd.DataFrame(rows)
    return df



def build_focus_points(metrics_snapshot, watchlist, anomalies, social_graph=None):
    points = []
    total = metrics_snapshot.get("total_students") or 0
    watch_count = len(watchlist)
    if watch_count:
        ratio = f" (~{(watch_count / total) * 100:.0f}% of the cohort)" if total else ""
        first_entry = watchlist[0] if watchlist else {}
        top_flag = None
        for bucket in (first_entry.get("flags") or [], first_entry.get("anomalies") or []):
            if bucket:
                top_flag = bucket[0]
                break
        if top_flag:
            points.append(f"{watch_count} learners{ratio} triggered '{top_flag}'.")
        else:
            points.append(f"{watch_count} learners{ratio} are on the watchlist.")
    low_eng = metrics_snapshot.get("low_engagement_count")
    if low_eng:
        points.append(f"{low_eng} students are operating below the engagement threshold.")
    low_posts = metrics_snapshot.get("low_post_count")
    if low_posts:
        points.append(f"{low_posts} students have not contributed a post yet.")
    avg_deadlines = metrics_snapshot.get("avg_deadline_misses")
    if avg_deadlines and avg_deadlines > 0.5:
        points.append(f"Average deadline misses sit at {avg_deadlines:.1f} per student.")
    if anomalies:
        labels = sorted({a.get("label") for a in anomalies if a.get("label")})
        if labels:
            joined = ", ".join(labels[:2])
            points.append(f"Outliers detected for {joined}.")
    if social_graph:
        iso_count = len(social_graph.get("isolated", []))
        if iso_count:
            points.append(f"{iso_count} learners show isolation risk (limited peer touchpoints).")
    if not points:
        points.append("No major risks detected—use the chat below to inspect particular learners.")
    return points[:3]


def build_severity_dataframe(severity_counts):
    if not severity_counts:
        return None
    rows = [
        {"Severity": key.title(), "Students": value}
        for key, value in severity_counts.items()
        if value
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("Severity")
    return df


def render_watchlist_cards(watchlist):
    top_entries = watchlist[: min(3, len(watchlist))]
    if not top_entries:
        return
    cols = st.columns(len(top_entries))
    severity_styles = {
        "high": ("rgba(248,113,113,0.2)", "#f87171"),
        "medium": ("rgba(251,191,36,0.18)", "#facc15"),
        "low": ("rgba(52,211,153,0.18)", "#34d399"),
    }
    for entry, col in zip(top_entries, cols):
        severity = (entry.get("severity") or "medium").lower()
        bg, accent = severity_styles.get(severity, ("rgba(148,163,184,0.18)", "#94a3b8"))
        name = entry.get("userfullname") or (f"ID {entry['userid']}" if entry.get("userid") else "Unknown")
        reasons = entry.get("flags") or entry.get("anomalies") or []
        metrics = entry.get("metrics", {})
        stat_bits = []
        posts = metrics.get("total_posts")
        if posts is not None:
            stat_bits.append(f"{int(posts)} posts")
        engagement = metrics.get("engagement_rate")
        if engagement is not None:
            stat_bits.append(f"{float(engagement):.2f} engagement")
        misses = metrics.get("deadline_exceeded_posts_Quasi_exam_I")
        if misses:
            stat_bits.append(f"{misses} deadline misses")
        stats_line = " · ".join(stat_bits) if stat_bits else "Limited metrics"
        reason_block = "<br>".join(reasons[:2]) if reasons else "General anomaly trigger"
        col.markdown(
            f"""
            <div style="border:1px solid rgba(148,163,184,0.25); border-radius:18px; padding:1rem; background:{bg};">
              <div style="font-size:0.8rem; text-transform:uppercase; font-weight:600; color:{accent};">{severity.title()}</div>
              <div style="font-size:1.1rem; font-weight:600; margin-top:0.2rem;">{name}</div>
              <div style="font-size:0.9rem; color:var(--muted,#94a3b8); margin:.3rem 0;">{stats_line}</div>
              <div style="font-size:0.85rem; line-height:1.3;">{reason_block}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def generate_ai_insights(student_df, config, data_manager):
    settings = config.ai_insights_settings if hasattr(config, "ai_insights_settings") else {}
    rules = config.ai_insight_rules if hasattr(config, "ai_insight_rules") else []

    rule_results = data_manager.evaluate_rules(student_df, rules, settings)
    anomalies, anomaly_missing = detect_anomalies(student_df, settings)

    metric_columns = [
        "total_posts",
        "engagement_rate",
        "total_replies_to_professor",
        "avg_reply_time",
        "deadline_exceeded_posts_Quasi_exam_I",
        "deadline_exceeded_posts_Quasi_exam_II",
        "deadline_exceeded_posts_Quasi_exam_III",
    ]
    watchlist = assemble_watchlist(student_df, rule_results, anomalies, metric_columns)
    metrics_snapshot = compute_metrics_snapshot(student_df)
    social_graph = compute_social_connectedness(student_df)
    metrics_snapshot["isolated_count"] = len(social_graph.get("isolated", []))

    prompt = build_summary_prompt(metrics_snapshot, watchlist, anomalies)
    prompt_trimmed = prompt[:1800]
    summary_result = generate_model_summary(
        prompt_trimmed,
        settings.get("summary_model", "google/flan-t5-small"),
        settings.get("summary_temperature", 0.0),
    )
    manual = build_manual_summary(metrics_snapshot, watchlist, anomalies)
    summary_text = summary_result["summary"] or manual

    notifications = generate_sample_notifications(watchlist, metrics_snapshot)
    severity_counts = Counter(entry.get("severity", "medium") for entry in watchlist if entry)
    focus_points = build_focus_points(metrics_snapshot, watchlist, anomalies, social_graph)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary_text,
        "fallback_summary": manual,
        "model_summary": summary_result["summary"],
        "model_error": summary_result["error"],
        "students_to_watch": watchlist,
        "notifications": notifications,
        "rule_counts": rule_results.get("rule_counts", {}),
        "missing_columns": sorted(set(rule_results.get("missing_columns", [])) | set(anomaly_missing)),
        "metrics_snapshot": metrics_snapshot,
        "anomalies": anomalies,
        "settings": settings,
        "summary_prompt": prompt_trimmed,
        "severity_counts": dict(severity_counts),
        "summary_points": focus_points,
        "social_graph": social_graph,
    }


def display_summary_section(insights):
    section_header("Summary / Risk Overview", tight=True)
    info_panel(insights.get("summary", "No summary available."), icon="📌")

    if insights.get("model_summary") is None and insights.get("model_error"):
        st.caption("Showing heuristic overview.")
    elif insights.get("model_summary"):
        st.caption("Summary generated with {model}.".format(model=insights["settings"].get("summary_model", "")))
        st.caption(f"Heuristic snapshot: {insights.get('fallback_summary', '')}")

    metrics = insights.get("metrics_snapshot", {})
    watch_count = len(insights.get("students_to_watch", []))
    total = metrics.get("total_students", 0) or 0
    avg_engagement = metrics.get("avg_engagement")
    avg_posts = metrics.get("avg_posts")
    avg_deadlines = metrics.get("avg_deadline_misses")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta = f"{(watch_count / total) * 100:.0f}%" if total else None
        st.metric("Watchlist size", watch_count, delta=delta, help="Share of the cohort currently flagged.")
    with col2:
        st.metric("Avg engagement", f"{avg_engagement:.2f}" if avg_engagement is not None else "N/A")
    with col3:
        st.metric("Avg posts", f"{avg_posts:.1f}" if avg_posts is not None else "N/A")
    with col4:
        st.metric("Avg deadline misses", f"{avg_deadlines:.1f}" if avg_deadlines is not None else "N/A")

    focus_points = insights.get("summary_points") or []
    if focus_points:
        st.markdown("**Focus areas**")
        for point in focus_points:
            st.markdown(f"- {point}")

    severity_df = build_severity_dataframe(insights.get("severity_counts"))
    if severity_df is not None:
        st.markdown("**Watchlist mix**")
        st.bar_chart(severity_df)

    rule_counts = insights.get("rule_counts", {})
    if rule_counts:
        st.markdown("**Rule triggers**")
        rule_df = (
            pd.DataFrame([{"Rule": key, "Hits": value} for key, value in rule_counts.items()])
            .sort_values("Hits", ascending=False)
        )
        st.dataframe(rule_df, use_container_width=True, hide_index=True)

    if insights.get("missing_columns"):
        missing = ", ".join(insights["missing_columns"])
        st.warning(f"Some rules or detectors skipped missing columns: {missing}")

    social_graph = insights.get("social_graph") or {}
    isolated_count = len(social_graph.get("isolated", []))
    if isolated_count:
        st.warning(f"{isolated_count} students show isolation risk. Review the Connectivity tab for outreach ideas.")


def display_watchlist_section(insights):
    section_header("Students to Watch", icon="⚠️", tight=True)
    watchlist = insights.get("students_to_watch", [])
    if not watchlist:
        info_panel("All quiet for now - no students matched the current watch conditions.")
        return
    st.write("---")
    render_watchlist_cards(watchlist)
    st.write("---")
    watch_df = format_watchlist_dataframe(watchlist)
    st.dataframe(watch_df, use_container_width=True, hide_index=True)
    
    st.write("---")

    unique_playbooks = []
    for entry in watchlist:
        for step in entry.get("playbooks", []):
            if step not in unique_playbooks:
                unique_playbooks.append(step)
    if unique_playbooks:
        st.markdown("**Suggested playbook actions:**")
        for step in unique_playbooks[:5]:
            st.markdown(f"- {step}")


def display_notifications_section(insights):
    section_header("Sample Notifications", icon="📬", tight=True)
    notifications = insights.get("notifications", [])
    if not notifications:
        info_panel("No alerts generated because the watchlist is empty.", icon="ℹ️")
        return
    for idx, note in enumerate(notifications, start=1):
        st.markdown(f"**{idx}. {note['subject']}**")
        st.markdown(f"```\n{note['body']}\n```")


def build_social_table(entries, positive=False):
    if not entries:
        return None
    rows = []
    note_key = "highlights" if positive else "warnings"
    fallback_note = "Consistent participation" if positive else "Needs nudges to participate"
    for entry in entries:
        metrics = entry.get("metrics", {})
        rows.append(
            {
                "Student": entry.get("userfullname") or (f"ID {entry.get('userid')}" if entry.get("userid") else "Unknown"),
                "Network score": entry.get("network_score"),
                "Unique discussions": format_metric_value(metrics.get("unique_discussions"), 0),
                "Peer interactions": format_metric_value(metrics.get("unique_interactions"), 0),
                "Posts": format_metric_value(metrics.get("total_posts"), 0),
                "Engagement": format_metric_value(metrics.get("engagement_rate"), 2),
                "Notes": ", ".join(entry.get(note_key, [])[:2]) or fallback_note,
            }
        )
    return pd.DataFrame(rows)


def display_social_connectedness_section(insights):
    section_header("Isolation & Connectedness", icon="🧩", tight=True)
    social_graph = insights.get("social_graph") or {}
    if not social_graph.get("available"):
        missing = ", ".join(social_graph.get("missing_columns", []))
        info_panel(
            "Need the unique discussion, interaction, post, or engagement columns to compute connectivity."
            + (f" Missing: {missing}" if missing else ""),
            icon="ℹ️",
        )
        return

    summary = social_graph.get("summary", {})
    isolated = social_graph.get("isolated", [])
    connected = social_graph.get("connected", [])

    col_total, col_isolated, col_connected = st.columns(3)
    with col_total:
        st.metric("Students analysed", summary.get("total", 0))
    with col_isolated:
        st.metric(
            "Isolation risks",
            len(isolated),
            help="Flagged when unique discussions, peer interactions, and engagement trail the cohort median.",
        )
    with col_connected:
        st.metric("Highly connected", len(connected))

    if isolated:
        sample_names = ", ".join(entry.get("userfullname") for entry in isolated[:4])
        st.warning(f"{len(isolated)} students show isolation risk: {sample_names}")
    else:
        st.success("No isolation warnings at the moment—peer interactions look balanced.")

    iso_df = build_social_table(isolated, positive=False)
    if iso_df is not None:
        st.markdown("**Isolated students (needs outreach)**")
        st.dataframe(iso_df, use_container_width=True, hide_index=True)
    else:
        st.info("Everyone currently meets the minimum peer interaction thresholds.")

    if connected:
        st.markdown("**Highly connected students (potential ambassadors)**")
        conn_df = build_social_table(connected, positive=True)
        st.dataframe(conn_df, use_container_width=True, hide_index=True)

    thresholds = social_graph.get("thresholds", {})
    threshold_bits = []
    for slug, value in thresholds.items():
        if value is None or pd.isna(value):
            continue
        label = slug.replace("_", " ").title()
        threshold_bits.append(f"{label}: {value:.2f}")
    if threshold_bits:
        st.caption("Median thresholds used for the classification → " + " · ".join(threshold_bits))



def get_student_metric_columns(df):
    return {
        "total_posts": resolve_column(df, "total_posts"),
        "engagement_rate": resolve_column(df, "engagement_rate"),
        "total_replies_to_professor": resolve_column(df, "total_replies_to_professor"),
        "avg_reply_time": resolve_column(df, "avg_reply_time"),
        "deadline_exceeded_posts_Quasi_exam_I": resolve_column(df, "deadline_exceeded_posts_Quasi_exam_I"),
        "deadline_exceeded_posts_Quasi_exam_II": resolve_column(df, "deadline_exceeded_posts_Quasi_exam_II"),
        "deadline_exceeded_posts_Quasi_exam_III": resolve_column(df, "deadline_exceeded_posts_Quasi_exam_III"),
        "unique_discussions": resolve_column(df, "unique_discussions"),
        "unique_interactions": resolve_column(df, "unique_interactions"),
    }


def handle_general_chat_request(prompt, normalized, student_df, insights, resolved_cols):
    social_graph = insights.get("social_graph") or {}
    if ("isolat" in normalized) or ("disconnected" in normalized):
        isolated = social_graph.get("isolated") or []
        if not isolated:
            return "No isolated students are flagged at the moment."
        lines = [f"{len(isolated)} students show isolation risk:"]
        for entry in isolated[:5]:
            metrics = entry.get("metrics", {})
            discussions = format_metric_value(metrics.get("unique_discussions"), 0)
            peers = format_metric_value(metrics.get("unique_interactions"), 0)
            lines.append(f"- {entry.get('userfullname')}: {discussions} discussions · {peers} peer interactions")
        return "\n".join(lines)

    if "connected" in normalized and "disconnected" not in normalized:
        connected = social_graph.get("connected") or []
        if not connected:
            return "I need more interaction metrics before highlighting well-connected students."
        lines = ["Highly connected students:"]
        for entry in connected[:5]:
            metrics = entry.get("metrics", {})
            discussions = format_metric_value(metrics.get("unique_discussions"), 0)
            peers = format_metric_value(metrics.get("unique_interactions"), 0)
            lines.append(f"- {entry.get('userfullname')}: {discussions} discussions · {peers} peer interactions")
        return "\n".join(lines)

    if any(word in normalized for word in ["watchlist", "alert", "warning"]):
        watchlist = insights.get("students_to_watch", [])
        if not watchlist:
            return "The watchlist is empty right now."
        lines = ["Current watchlist:"]
        for entry in watchlist[:5]:
            label = entry.get("severity", "medium").title()
            reason = ", ".join((entry.get("flags") or entry.get("anomalies") or [])[:2])
            lines.append(f"- {entry.get('userfullname')}: {label} ({reason or 'general anomaly'})")
        return "\n".join(lines)

    ranking_response = describe_metric_ranking_from_prompt(normalized, student_df, resolved_cols)
    if ranking_response:
        return ranking_response
    return None


def describe_metric_ranking_from_prompt(normalized, student_df, resolved_cols):
    wants_high = any(
        token in normalized
        for token in ["top", "best", "high", "highest", "most", "above average", "strong", "star", "excellent"]
    )
    wants_low = any(
        token in normalized
        for token in ["low", "lowest", "least", "fewest", "below average", "underperform", "struggling", "bottom"]
    )
    if not wants_high and not wants_low:
        return None
    ascending = wants_low and not wants_high

    metric_map = [
        (["engagement", "motivation"], resolved_cols.get("engagement_rate"), "engagement rate", 2),
        (["post", "activity", "contribution"], resolved_cols.get("total_posts"), "total posts", 0),
        (["reply", "response"], resolved_cols.get("total_replies_to_professor"), "replies to the professor", 0),
        (["discussion", "thread"], resolved_cols.get("unique_discussions"), "unique discussions", 0),
        (["interaction", "peer"], resolved_cols.get("unique_interactions"), "peer interactions", 0),
    ]

    for keywords, column_name, label, decimals in metric_map:
        if column_name and any(keyword in normalized for keyword in keywords):
            return describe_metric_ranking(student_df, column_name, label, ascending=ascending, decimals=decimals)

    # Handle deadline-specific queries separately
    if "deadline" in normalized or "late" in normalized or "overdue" in normalized:
        deadline_cols = [
            resolved_cols.get("deadline_exceeded_posts_Quasi_exam_I"),
            resolved_cols.get("deadline_exceeded_posts_Quasi_exam_II"),
            resolved_cols.get("deadline_exceeded_posts_Quasi_exam_III"),
        ]
        deadline_cols = [col for col in deadline_cols if col]
        if deadline_cols:
            deadline_df = student_df.copy()
            totals = deadline_df[deadline_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            deadline_df["_deadline_total"] = totals
            return describe_metric_ranking(
                deadline_df,
                "_deadline_total",
                "deadline misses",
                ascending=ascending,
                decimals=0,
            )
    return None


def describe_metric_ranking(df, column_name, label, ascending=True, decimals=2):
    if column_name not in df.columns:
        return None
    working = df[["userfullname", column_name]].copy()
    working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    ranked = working.dropna(subset=[column_name]).sort_values(column_name, ascending=ascending).head(5)
    if ranked.empty:
        return None
    direction_label = "lowest" if ascending else "highest"
    lines = [f"{label.title()} ({direction_label}):"]
    for _, row in ranked.iterrows():
        value = row[column_name]
        lines.append(f"- {row['userfullname']}: {format_metric_value(value, decimals)}")
    return "\n".join(lines)


def extract_student_mentions(prompt, available_names):
    normalized = prompt.lower()
    matched = []
    for name in available_names:
        lower = name.lower()
        if lower and lower in normalized:
            matched.append(name)
    if matched:
        seen = set()
        ordered = []
        for name in matched:
            low = name.lower()
            if low not in seen:
                seen.add(low)
                ordered.append(name)
        return ordered
    lower_names = [name.lower() for name in available_names]
    chunks = re.split(r"\b(?:and|vs|versus|,|/|&|with|compare)\b", normalized)
    guesses = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        best = get_close_matches(chunk, lower_names, n=1, cutoff=0.68)
        if best:
            idx = lower_names.index(best[0])
            guesses.append(available_names[idx])
    seen = set()
    ordered = []
    for name in guesses:
        low = name.lower()
        if low not in seen:
            seen.add(low)
            ordered.append(name)
    return ordered


def lookup_watchlist_entry(name, watchlist):
    target = name.lower()
    for entry in watchlist:
        entry_name = (entry.get("userfullname") or "").lower()
        if entry_name == target:
            return entry
    return None


def format_metric_value(value, decimals=1):
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "-"
        return f"{value:.{decimals}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        numeric = float(value)
        return f"{numeric:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def answer_student_question(prompt, student_df, insights):
    if student_df is None or student_df.empty:
        return "I need the attribute matrix before answering questions."
    if "userfullname" not in student_df.columns:
        return "Student names are missing from the attribute matrix."
    available_names = [n for n in student_df["userfullname"].dropna().astype(str).unique() if n.strip()]
    if not available_names:
        return "Student names are missing from the attribute matrix."
    resolved_cols = get_student_metric_columns(student_df)
    matches = extract_student_mentions(prompt, available_names)
    normalized = prompt.lower()
    watchlist = insights.get("students_to_watch", [])

    rows = []
    for name in matches[:2]:
        matching = student_df[student_df["userfullname"].astype(str).str.lower() == name.lower()]
        if matching.empty:
            continue
        rows.append((name, matching.iloc[0]))
    if not rows:
        general_response = handle_general_chat_request(prompt, normalized, student_df, insights, resolved_cols)
        if general_response:
            return general_response
        return "I couldn't resolve those names inside the attribute matrix."

    if len(rows) == 1:
        name, row = rows[0]
        metrics = extract_metrics(row, resolved_cols)
        watch_entry = lookup_watchlist_entry(name, watchlist)
        lines = [f"**{name}** snapshot:"]
        lines.append(f"- Posts: {format_metric_value(metrics.get('total_posts'), 0)}")
        lines.append(f"- Engagement: {format_metric_value(metrics.get('engagement_rate'), 2)}")
        lines.append(f"- Replies to professor: {format_metric_value(metrics.get('total_replies_to_professor'), 0)}")
        lines.append(f"- Avg reply time (h): {format_metric_value(metrics.get('avg_reply_time'), 1)}")
        deadline_i = metrics.get('deadline_exceeded_posts_Quasi_exam_I')
        if deadline_i is not None:
            lines.append(f"- Deadline misses (Exam I): {format_metric_value(deadline_i, 0)}")
        deadline_ii = metrics.get('deadline_exceeded_posts_Quasi_exam_II')
        if deadline_ii is not None:
            lines.append(f"- Deadline misses (Exam II): {format_metric_value(deadline_ii, 0)}")
        if watch_entry:
            reasons = watch_entry.get("flags") or watch_entry.get("anomalies") or []
            detail = ", ".join(reasons[:2]) if reasons else "general risk pattern"
            lines.append(f"- Watchlist: {watch_entry.get('severity', 'medium').title()} ({detail}).")
        return "\n".join(lines)

    (name_a, row_a), (name_b, row_b) = rows[:2]
    metrics_a = extract_metrics(row_a, resolved_cols)
    metrics_b = extract_metrics(row_b, resolved_cols)
    table_rows = [
        ("Posts", format_metric_value(metrics_a.get("total_posts"), 0), format_metric_value(metrics_b.get("total_posts"), 0)),
        ("Engagement", format_metric_value(metrics_a.get("engagement_rate"), 2), format_metric_value(metrics_b.get("engagement_rate"), 2)),
        ("Replies to professor", format_metric_value(metrics_a.get("total_replies_to_professor"), 0), format_metric_value(metrics_b.get("total_replies_to_professor"), 0)),
        ("Avg reply time (h)", format_metric_value(metrics_a.get("avg_reply_time"), 1), format_metric_value(metrics_b.get("avg_reply_time"), 1)),
    ]
    table_md = [f"| Metric | {name_a} | {name_b} |", "| --- | --- | --- |"]
    for label, val_a, val_b in table_rows:
        table_md.append(f"| {label} | {val_a} | {val_b} |")
    lines = ["Comparison overview:", "\n".join(table_md)]
    watch_a = lookup_watchlist_entry(name_a, watchlist)
    watch_b = lookup_watchlist_entry(name_b, watchlist)
    watch_notes = []
    if watch_a:
        watch_notes.append(f"{name_a}: {watch_a.get('severity', 'medium').title()} risk")
    if watch_b:
        watch_notes.append(f"{name_b}: {watch_b.get('severity', 'medium').title()} risk")
    if watch_notes:
        lines.append("Watchlist notes: " + "; ".join(watch_notes))
    return "\n".join(lines)


def render_student_chatbot(student_df, insights):
    section_header("Student Copilot", icon="✨", tight=True)
    st.caption("Ask about any student, compare two learners, or type prompts like 'isolated students' or 'top engagement'.")
    chat_key = "ai_student_chat_history"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []


    for message in st.session_state[chat_key]:
        st.chat_message(message["role"]).write(message["content"])

    prompt = st.chat_input("Ask about a student or compare two")
    if prompt:
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        response = answer_student_question(prompt, student_df, insights)
        st.session_state[chat_key].append({"role": "assistant", "content": response})
        st.rerun()
        
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear chat", key="clear_ai_chat", use_container_width=True):
            st.session_state[chat_key] = []
            st.rerun()

def main():
    page_header(
        "AI Insights",
        icon="✨",
        align="left",
        compact=True,
    )

    data_manager = st.session_state.data_manager
    config = st.session_state.config

    student_attributes = data_manager.get_student_attributes()
    if student_attributes is None or student_attributes.empty:
        info_panel(
            "No attribute matrix found. Run the Attribute Analysis step to compute student features first.",
            icon="ℹ️",
        )
        divider()
        nav_footer(
            back={
                "label": "⬅️ Back to Visualizations",
                "page": "pages/6_📊_Visualizations.py",
                "key": "nav_back_to_visualizations_from_ai",
                "fallback": "📊 Visualizations",
            }
        )
        return

    col_refresh, col_placeholder = st.columns([1, 3])
    with col_refresh:
        refresh_requested = st.button("🔁 Refresh Insights", use_container_width=True)

    if refresh_requested:
        data_manager.mark_ai_insights_dirty()

    insights = data_manager.get_ai_insights()
    if refresh_requested or data_manager.is_ai_insights_dirty() or not insights:
        with st.spinner("Synthesizing AI insights..."):
            insights = generate_ai_insights(student_attributes, config, data_manager)
        data_manager.store_ai_insights(insights)
        st.success("AI insights updated.")


    divider()
    tab_summary, tab_watchlist, tab_network, tab_notifications, tab_chat = st.tabs(
        ["Summary", "Watchlist", "Connectivity", "Notifications", "Student Q&A"]
    )

    with tab_summary:
        display_summary_section(insights)
    with tab_watchlist:
        display_watchlist_section(insights)
    with tab_network:
        display_social_connectedness_section(insights)
    with tab_notifications:
        display_notifications_section(insights)
    with tab_chat:
        render_student_chatbot(student_attributes, insights)

    divider()
    nav_footer(
        back={
            "label": "⬅️ Back to Visualizations",
            "page": "pages/6_📊_Visualizations.py",
            "key": "nav_back_to_visualizations_from_ai",
            "fallback": "📊 Visualizations",
        }
    )
    


if __name__ == "__main__":
    main()


