import textwrap
from datetime import datetime

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
    }


def display_summary_section(insights):
    section_header("Summary / Risk Overview", icon="🧠", tight=True)
    info_panel(insights.get("summary", "No summary available."), icon="💡")

    if insights.get("model_summary") is None and insights.get("model_error"):
        st.caption(f"Model summary unavailable ({insights['model_error']}). Showing heuristic overview.")
    elif insights.get("model_summary"):
        st.caption("Summary generated with {model}.".format(model=insights["settings"].get("summary_model", "")))
        st.caption(f"Heuristic snapshot: {insights.get('fallback_summary', '')}")

    metrics = insights.get("metrics_snapshot", {})
    watch_count = len(insights.get("students_to_watch", []))
    total = metrics.get("total_students", 0) or 0
    avg_engagement = metrics.get("avg_engagement")
    avg_posts = metrics.get("avg_posts")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Watchlist size", watch_count, help="Students currently flagged by rules or anomalies.")
    with col2:
        if avg_engagement is not None:
            st.metric("Avg engagement", f"{avg_engagement:.2f}")
        else:
            st.metric("Avg engagement", "N/A")
    with col3:
        if avg_posts is not None:
            st.metric("Avg posts", f"{avg_posts:.1f}")
        else:
            st.metric("Avg posts", "N/A")

    if insights.get("missing_columns"):
        missing = ", ".join(insights["missing_columns"])
        st.warning(f"Some rules or detectors skipped missing columns: {missing}")


def display_watchlist_section(insights):
    section_header("Students to Watch", icon="⚠️", tight=True)
    watchlist = insights.get("students_to_watch", [])
    if not watchlist:
        info_panel("All quiet for now — no students matched the current watch conditions.", icon="✅")
        return

    watch_df = format_watchlist_dataframe(watchlist)
    st.dataframe(watch_df, use_container_width=True, hide_index=True)

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


def main():
    page_header(
        "AI Insights",
        "Leverage local-friendly intelligence to surface risks and suggested interventions.",
        icon="🤖",
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
    else:
        generated_at = insights.get("generated_at")
        if generated_at:
            st.caption(f"Using cached insights generated at {generated_at}.")

    divider()
    display_summary_section(insights)
    divider()
    display_watchlist_section(insights)
    divider()
    display_notifications_section(insights)

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
