import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

from utils.session_data_manager import SessionDataManager
from utils.config_manager import ConfigManager
from utils.attribute_calculations import (
    activity_attrs,
    engagement_attrs,
    content_attrs,
)
from utils.coco_utils import clean_coco_dataframe
from assets.ui_components import apply_theme, divider, info_panel, page_header, section_header, nav_footer
from utils.ui_steps import render_steps
# Safe initialization
if "data_manager" not in st.session_state:
    st.session_state.data_manager = SessionDataManager()
if "config" not in st.session_state:
    st.session_state.config = ConfigManager()

render_steps(active="2 Visualize")
apply_theme()

EXAM_PREFIX = "deadline_exceeded_posts_"


def get_exam_columns(df):
    return [col for col in df.columns if col.startswith(EXAM_PREFIX)]


def main():
    data_manager = st.session_state.data_manager
    _ = st.session_state.config

    page_header(
        "Visual Analytics",
        "Explore computed metrics and rankings without re-running the heavy pipelines.",
        icon="📊",
        align="left",
        compact=True,
    )

    student_attributes = data_manager.get_student_attributes()
    ranked_results = data_manager.get_ranked_results()
    coco_results = data_manager.get_coco_results()
    validation_results = data_manager.get_validation_results()

    if student_attributes is None:
        st.warning("Please compute student attributes first on the Attribute Analysis page.")
        divider()
        nav_footer(
            back={
                "label": "⬅️ Back to Attribute Analysis",
                "page": "pages/3_📈_Attribute_Analysis.py",
                "key": "viz_back_to_attributes",
                "fallback": "📈 Attribute Analysis",
            },
            message="Visualisations become available once attributes are computed.",
        )
        return

    attribute_cols = [col for col in student_attributes.columns if col not in ["userid", "userfullname"]]
    if not attribute_cols:
        st.error("No attributes available. Re-run the Attribute Analysis with at least one metric selected.")
        return

    status_lines = [
        f"Attributes: {'✅ Ready' if student_attributes is not None else '⚠️ Missing'}",
        f"Ranking: {'✅ Ready' if ranked_results is not None else '⚠️ Run the Ranking page'}",
        f"COCO: {'✅ Ready' if coco_results else '⚠️ Run the COCO analysis'}",
        f"Validation: {'✅ Ready' if validation_results is not None else '⚠️ Available after COCO run'}",
    ]
    info_panel("<br>".join(status_lines), icon="ℹ️")

    attr_tab, student_tab, ranking_tab, coco_tab = st.tabs(
        ["📊 Attribute Insights", "🧭 Student Explorer", "🏆 Ranking Insights", "🔍 COCO & Validation"]
    )

    with attr_tab:
        render_attribute_distribution(student_attributes, attribute_cols)
        divider()
        render_student_comparison(student_attributes, attribute_cols)
        divider()
        render_above_below_matrix(student_attributes, attribute_cols)
        divider()
        render_top_performers(student_attributes, attribute_cols)
        divider()
        render_category_analysis(student_attributes)
        divider()
        render_exam_focus(student_attributes)

    with student_tab:
        render_student_profile(student_attributes, attribute_cols)
        divider()
        render_student_overview(student_attributes, ranked_results, validation_results)

    with ranking_tab:
        if ranked_results is None:
            info_panel(
                "Ranking results are not available yet. Run the Ranking page to populate this section.",
                icon="⚠️",
            )
        else:
            render_ranking_insights(ranked_results)

    with coco_tab:
        if not coco_results:
            info_panel("COCO results are not available yet. Run the COCO analysis to view this section.", icon="⚠️")
        else:
            render_coco_overview(coco_results, validation_results)
            divider()
            if validation_results is not None and not validation_results.empty:
                render_validation_dashboard(validation_results)
            else:
                info_panel("Validation results will appear automatically after the combined COCO + validation run.", icon="ℹ️")

    divider()
    nav_footer(
        back={
            "label": "⬅️ Back to Ranking",
            "page": "pages/4_🏆_Ranking.py",
            "key": "viz_back_to_ranking",
            "fallback": "🏆 Ranking",
        },
        forward={
            "label": "✨ Go to AI Insights",
            "page": "pages/7_🤖_AI_Insights.py",
            "key": "pulse",
            "fallback": "🤖 AI Insights",
        },
        message="Visualisations use cached data—re-run computation pages to refresh the figures.",
    )


def render_attribute_distribution(oam_combined, attribute_cols):
    section_header("Attribute Distribution Analysis", icon="📊")

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_attribute = st.selectbox(
            "Select attribute",
            attribute_cols,
            key="viz_attr_dist_select",
            
        )

        if selected_attribute:
            attr_data = oam_combined[selected_attribute]
            st.metric("Average", f"{attr_data.mean():.2f}")
            st.metric("Median", f"{attr_data.median():.2f}")
            st.metric("Std Deviation", f"{attr_data.std():.2f}")

    with col2:
        if selected_attribute:
            hist_fig = px.histogram(
                oam_combined,
                x=selected_attribute,
                nbins=50,
                title=f"Distribution of {selected_attribute.replace('_', ' ').title()}",
            )
            hist_fig.update_layout(
                xaxis_title=selected_attribute.replace("_", " ").title(),
                yaxis_title="Number of Students",
                showlegend=False,
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            box_fig = px.box(
                oam_combined,
                y=selected_attribute,
                title=f"Box Plot - {selected_attribute.replace('_', ' ').title()}",
            )
            st.plotly_chart(box_fig, use_container_width=True)


def render_student_comparison(oam_combined, attribute_cols):
    section_header("Student Performance Comparison", icon="👥")

    col1, col2 = st.columns(2)
    with col1:
        selected_students = st.multiselect(
            "Select students to compare",
            options=oam_combined["userfullname"].tolist(),
            default=oam_combined["userfullname"].head(5).tolist(),
            key="viz_student_compare_select",
        )

    with col2:
        default_attrs = attribute_cols[:5] if len(attribute_cols) >= 3 else attribute_cols
        selected_attributes = st.multiselect(
            "Select attributes",
            options=attribute_cols,
            default=default_attrs,
            key="viz_attribute_compare_select",
        )

    if selected_students and selected_attributes:
        comparison_data = oam_combined[oam_combined["userfullname"].isin(selected_students)].copy()

        if len(selected_attributes) >= 3:
            radar_fig = create_radar_chart(comparison_data, selected_students, selected_attributes)
            st.plotly_chart(radar_fig, use_container_width=True)

        bar_fig = create_attribute_comparison_bar(comparison_data, selected_students, selected_attributes)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption("Use the heatmap below to spot who sits above or below the cohort averages.")


def render_above_below_matrix(oam_combined, attribute_cols):
    section_header("Above vs Below Cohort Average", icon="🎯")
    if not attribute_cols:
        st.info("Compute attributes to compare students against cohort averages.")
        return

    default_attrs = attribute_cols[:6] if len(attribute_cols) >= 6 else attribute_cols
    selected_attrs = st.multiselect(
        "Attributes to highlight",
        options=attribute_cols,
        default=default_attrs,
        key="viz_above_below_select",
    )
    if not selected_attrs:
        st.info("Pick at least one attribute to render the heatmap.")
        return

    subset = oam_combined[["userfullname"] + selected_attrs].copy()
    cohort_means = oam_combined[selected_attrs].mean()

    def _style_column(series):
        mean_val = cohort_means.get(series.name)
        styled = []
        for value in series:
            if pd.isna(value) or mean_val is None or pd.isna(mean_val):
                styled.append("")
                continue
            if value >= mean_val:
                styled.append("background-color:#dcfce7;color:#14532d;font-weight:600;")
            else:
                styled.append("background-color:#fee2e2;color:#991b1b;font-weight:600;")
        return styled

    styled = (
        subset.set_index("userfullname")
        .style.format("{:.2f}")
        .apply(_style_column, axis=0)
    )
    st.dataframe(styled, use_container_width=True, height=min(600, 40 * len(subset) + 120))
    st.caption("Green cells = above cohort average · Red cells = below cohort average.")


def create_radar_chart(comparison_data, students, attributes):
    fig = go.Figure()
    normalized = comparison_data.copy()

    for attr in attributes:
        max_val = normalized[attr].max()
        if max_val > 0:
            normalized[attr] = normalized[attr] / max_val

    for student in students:
        student_data = normalized[normalized["userfullname"] == student]
        values = student_data[attributes].iloc[0].tolist()
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=attributes + [attributes[0]],
                fill="toself",
                name=student,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Student Comparison Radar Chart",
    )
    return fig


def create_attribute_comparison_bar(comparison_data, students, attributes):
    melt_data = comparison_data.melt(
        id_vars=["userfullname"],
        value_vars=attributes,
        var_name="Attribute",
        value_name="Value",
    )

    fig = px.bar(
        melt_data,
        x="userfullname",
        y="Value",
        color="Attribute",
        barmode="group",
        title="Student Attribute Comparison",
    )
    fig.update_layout(
        xaxis_title="Students",
        yaxis_title="Attribute Value",
        showlegend=True,
    )
    return fig


def render_top_performers(oam_combined, attribute_cols):
    section_header("Top Performers by Attribute", icon="🔥")

    selected_attribute = st.selectbox(
        "Select attribute for ranking",
        attribute_cols,
        key="viz_top_attr_select",
    )
    top_n = st.slider("Number of students to show", 5, min(30, len(oam_combined)), 10, key="viz_top_n_slider")

    if selected_attribute:
        rankings = oam_combined[["userfullname", selected_attribute]].dropna().copy()
        top_students = rankings.nlargest(top_n, selected_attribute)

        fig = px.bar(
            top_students,
            y="userfullname",
            x=selected_attribute,
            orientation="h",
            title=f"Top {top_n} Students - {selected_attribute.replace('_', ' ').title()}",
            color=selected_attribute,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            yaxis_title="Student",
            xaxis_title=selected_attribute.replace("_", " ").title(),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_students, use_container_width=True, hide_index=True)


def render_student_profile(oam_combined, attribute_cols):
    section_header("Student Attribute Profile", icon="📈")

    selected_student = st.selectbox(
        "Select student",
        oam_combined["userfullname"].tolist(),
        key="viz_student_profile_select",
    )

    if selected_student:
        student_row = oam_combined[oam_combined["userfullname"] == selected_student].iloc[0]
        profile_data = []
        for attr in attribute_cols:
            profile_data.append(
                {
                    "Attribute": attr.replace("_", " ").title(),
                    "Score": student_row[attr],
                    "Class Average": oam_combined[attr].mean(),
                    "Status": "✅ Above Avg" if student_row[attr] > oam_combined[attr].mean() else "⚠️ Below Avg",
                }
            )

        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df.style.format({"Score": "{:.2f}", "Class Average": "{:.2f}"}), use_container_width=True, height=480)

        above_avg = sum(student_row[attr] > oam_combined[attr].mean() for attr in attribute_cols)
        st.info(f"**Summary:** {above_avg} of {len(attribute_cols)} attributes are above the cohort average.")


def render_category_analysis(oam_combined):
    section_header("Category-wise Attribute Analysis", icon="📋")

    categories = {
        "Activity": [col for col in oam_combined.columns if col in activity_attrs],
        "Engagement": [col for col in oam_combined.columns if col in engagement_attrs],
        "Content": [col for col in oam_combined.columns if col in content_attrs],
        "Exam": get_exam_columns(oam_combined),
    }
    categories = {key: value for key, value in categories.items() if value}

    if not categories:
        st.info("No categorized attributes available.")
        return

    selected_category = st.selectbox(
        "Select category",
        list(categories.keys()),
        key="viz_category_select",
    )

    if selected_category:
        category_cols = categories[selected_category]
        category_avg = oam_combined[category_cols].mean()

        fig = px.bar(
            x=[col.replace("_", " ").title() for col in category_cols],
            y=category_avg.values,
            title=f"{selected_category} Category - Average Scores",
            labels={"x": "Attributes", "y": "Average Score"},
            color=category_avg.values,
            color_continuous_scale="Greens",
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Top Performers - {selected_category} Category")
        scores = oam_combined[["userfullname"]].copy()
        scores["Category Total"] = oam_combined[category_cols].sum(axis=1)
        top_students = scores.nlargest(10, "Category Total")

        fig_top = px.bar(
            top_students,
            y="userfullname",
            x="Category Total",
            orientation="h",
            title="Top 10 Students",
            color="Category Total",
            color_continuous_scale="Plasma",
        )
        st.plotly_chart(fig_top, use_container_width=True)


def render_exam_focus(oam_combined):
    section_header("Exam Attribute Focus", icon="📝")

    exam_columns = get_exam_columns(oam_combined)
    if not exam_columns:
        st.info("No exam-related attributes available.")
        return

    selected_exam = st.selectbox(
        "Select exam attribute",
        exam_columns,
        key="viz_exam_select",
    )

    if selected_exam:
        metric_data = oam_combined[["userfullname", selected_exam]].dropna()
        st.metric("Average", f"{metric_data[selected_exam].mean():.2f}")
        st.metric("Median", f"{metric_data[selected_exam].median():.2f}")

        col1, col2 = st.columns(2)
        with col1:
            top_students = metric_data.nlargest(10, selected_exam)
            fig_top = px.bar(
                top_students,
                y="userfullname",
                x=selected_exam,
                orientation="h",
                title=f"Top 10 Students - {selected_exam.replace('_', ' ').title()}",
                color=selected_exam,
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            bottom_students = metric_data.nsmallest(10, selected_exam)
            fig_bottom = px.bar(
                bottom_students,
                y="userfullname",
                x=selected_exam,
                orientation="h",
                title=f"Needs Attention - {selected_exam.replace('_', ' ').title()}",
                color=selected_exam,
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_bottom, use_container_width=True)


def render_student_overview(oam_combined, ranked_results=None, validation_results=None):
    section_header("Student Overview Card", icon="🧭")

    selected_student = st.selectbox(
        "Choose a student",
        oam_combined["userfullname"].tolist(),
        key="viz_student_overview_select",
    )

    if selected_student:
        student_row = oam_combined[oam_combined["userfullname"] == selected_student].iloc[0]
        attribute_cols = [col for col in oam_combined.columns if col not in ["userid", "userfullname"]]
        above_avg = sum(student_row[attr] > oam_combined[attr].mean() for attr in attribute_cols)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Attributes Above Avg", above_avg)
        with col2:
            st.metric("Attributes Below Avg", len(attribute_cols) - above_avg)

        if ranked_results is not None:
            ranking_copy = ranked_results.copy()
            rank_cols = [col for col in ranking_copy.columns if col.endswith("_rank")]
            if rank_cols and selected_student in ranking_copy["userfullname"].values:
                ranking_copy["Average_Rank"] = ranking_copy[rank_cols].mean(axis=1)
                ranking_copy["Overall_Rank"] = ranking_copy["Average_Rank"].rank(method="min").astype(int)
                student_rank = ranking_copy[ranking_copy["userfullname"] == selected_student].iloc[0]
                avg_rank = student_rank["Average_Rank"]
                overall_rank = int(student_rank["Overall_Rank"])
                with col3:
                    st.metric("Overall Rank", overall_rank, help=f"Average rank: {avg_rank:.1f}")
            else:
                with col3:
                    st.metric("Overall Rank", "—", help="Ranking data not available for this student.")
        else:
            with col3:
                st.metric("Overall Rank", "—", help="Run the Ranking page to populate this value.")

        if validation_results is not None and "Validation_Result" in validation_results.columns:
            val_row = validation_results[validation_results["userfullname"] == selected_student]
            if not val_row.empty:
                status = val_row["Validation_Result"].iloc[0]
                st.info(f"Validation status: **{status}**")


def render_ranking_insights(ranked_results):
    section_header("Ranking Insights", icon="🏆")

    rank_cols = [col for col in ranked_results.columns if col.endswith("_rank")]
    if not rank_cols:
        st.info("No ranking columns were generated. Re-run the Ranking page with at least one attribute.")
        return

    ranking_copy = ranked_results.copy()
    ranking_copy["Average_Rank"] = ranking_copy[rank_cols].mean(axis=1)
    ranking_copy["Overall_Rank"] = ranking_copy["Average_Rank"].rank(method="min").astype(int)

    max_students = min(50, len(ranking_copy))
    top_n = st.slider(
        "Number of students to display",
        5,
        max_students,
        min(10, max_students),
        key="viz_ranking_top_slider",
    )
    top_students = ranking_copy.head(top_n)

    fig = px.bar(
        top_students,
        x="userfullname",
        y="Average_Rank",
        color="Average_Rank",
        color_continuous_scale="Blues_r",
        title="Average Rank (lower is better)",
    )
    fig.update_layout(xaxis_title="Student", yaxis_title="Average Rank")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_students[["userfullname", "Overall_Rank", "Average_Rank"] + rank_cols],
        use_container_width=True,
        hide_index=True,
    )

    selected_students = st.multiselect(
        "Highlight students for attribute-wise ranking heatmap",
        options=ranking_copy["userfullname"].tolist(),
        default=ranking_copy["userfullname"].head(min(5, len(ranking_copy))).tolist(),
        key="viz_ranking_heatmap_select",
    )

    if selected_students:
        heatmap_df = ranking_copy[ranking_copy["userfullname"].isin(selected_students)]
        heatmap_data = heatmap_df.set_index("userfullname")[rank_cols]
        heatmap_fig = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale="Viridis_r",
            title="Ranking Positions Across Attributes (lower is better)",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)


def render_coco_overview(coco_results, validation_results):
    section_header("COCO Result Overview", icon="🔍")

    score_table = None
    for name, df in coco_results.items():
        if "Becslés" in df.columns or any("score" in str(col).lower() for col in df.columns):
            temp_df = clean_coco_dataframe(df.copy())
            if "Becslés" in temp_df.columns:
                temp_df["Becslés_numeric"] = pd.to_numeric(temp_df["Becslés"], errors="coerce")
            elif any("score" in str(col).lower() for col in temp_df.columns):
                score_col = next(col for col in temp_df.columns if "score" in str(col).lower())
                temp_df["Becslés_numeric"] = pd.to_numeric(temp_df[score_col], errors="coerce")
            else:
                continue
            score_table = temp_df.dropna(subset=["Becslés_numeric"])
            break

    if score_table is None or score_table.empty:
        info_panel("Could not identify a score table in the COCO output.", icon="ℹ️")
        return

    top_scores = score_table.nlargest(30, "Becslés_numeric")
    if "Alternatives" in top_scores.columns:
        name_col = "Alternatives"
    elif "userfullname" in top_scores.columns:
        name_col = "userfullname"
    else:
        name_col = top_scores.columns[0]

    fig = px.bar(
        top_scores,
        x=name_col,
        y="Becslés_numeric",
        color="Becslés_numeric",
        color_continuous_scale="Teal",
        title="Top COCO Scores",
    )
    fig.update_layout(xaxis_title="Student", yaxis_title="COCO Score")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_scores[[name_col, "Becslés_numeric"]].rename(columns={name_col: "Student", "Becslés_numeric": "Score"}),
        use_container_width=True,
        hide_index=True,
    )

    if validation_results is not None and not validation_results.empty:
        validity_rate = (
            validation_results["Is_Valid"].mean() * 100 if "Is_Valid" in validation_results.columns else np.nan
        )
        if not np.isnan(validity_rate):
            st.metric("Validation Success Rate", f"{validity_rate:.1f}%")


def render_validation_dashboard(validation_results):
    section_header("Validation Results Dashboard", icon="✅")

    if validation_results.empty:
        st.info("Validation results are empty.")
        return

    results = validation_results.copy()
    valid_count = results["Is_Valid"].sum()
    total_count = len(results)
    validity_percentage = (valid_count / total_count) * 100 if total_count else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", total_count, help="Number of students included in validation")
    with col2:
        st.metric("Valid Results", f"{valid_count}", delta=f"{validity_percentage:.1f}% success rate")
    with col3:
        if validity_percentage >= 80:
            icon = "✅"
            delta_color = "normal"
        elif validity_percentage >= 50:
            icon = "⚠️"
            delta_color = "off"
        else:
            icon = "❌"
            delta_color = "inverse"
        st.metric("Validity Rate", f"{validity_percentage:.1f}%", delta=icon, delta_color=delta_color)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Score Distribution", "🔍 Validation Analysis", "📋 Detailed Results", "⚠️ Review Cases"]
    )

    with tab1:
        section_header("Final Score Distribution by Validation Status", tight=True)
        fig = px.histogram(
            results,
            x="Becslés",
            color="Validation_Result",
            nbins=20,
            color_discrete_map={"Valid": "#00CC96", "Invalid": "#EF553B"},
            opacity=0.7,
            barmode="overlay",
        )
        fig.update_layout(
            title="Distribution of Final Scores",
            xaxis_title="Final Score (Becslés)",
            yaxis_title="Number of Students",
            legend_title="Validation Result",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_header("Delta Comparison: Original vs Inverted", tight=True)
        scatter_fig = px.scatter(
            results,
            x="Original_Delta",
            y="Inverted_Delta",
            color="Validation_Result",
            color_discrete_map={"Valid": "#00CC96", "Invalid": "#EF553B"},
            hover_data=["userfullname", "Final_Rank", "Becslés"],
            size="Becslés",
            size_max=15,
            opacity=0.7,
        )
        scatter_fig.add_hline(y=0, line_dash="dash", line_color="grey")
        scatter_fig.add_vline(x=0, line_dash="dash", line_color="grey")
        scatter_fig.update_layout(
            title="Original Delta vs Inverted Delta",
            xaxis_title="Original Delta",
            yaxis_title="Inverted Delta",
            legend_title="Validation Result",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        section_header("Validation Insights", icon="📊", tight=True)
        insights_col1, insights_col2 = st.columns(2)
        if validity_percentage >= 80:
            status_icon = "✅"
            status_message = "**Strong Consistency** — results show high reliability."
        elif validity_percentage >= 50:
            status_icon = "⚠️"
            status_message = "**Moderate Consistency** — some results need review."
        else:
            status_icon = "❌"
            status_message = "**Low Consistency** — significant review required."

        with insights_col1:
            info_panel(
                f"{status_message}<br><br>"
                f"<strong>Success Rate</strong>: {validity_percentage:.1f}%<br>"
                f"<strong>Valid Cases</strong>: {valid_count}/{total_count}",
                icon=status_icon,
            )

        delta_correlation = results["Original_Delta"].corr(results["Inverted_Delta"])
        correlation_hint = (
            "Low correlation expected for valid inversion" if abs(delta_correlation) < 0.3 else "Unexpected correlation pattern detected"
        )
        with insights_col2:
            info_panel(
                f"<strong>Delta Correlation</strong>: {delta_correlation:.3f}<br>"
                f"{correlation_hint}<br><br>"
                f"<strong>Mean Delta (Original)</strong>: {results['Original_Delta'].mean():.3f}<br>"
                f"<strong>Mean Delta (Inverted)</strong>: {results['Inverted_Delta'].mean():.3f}",
                icon="📐",
            )

    with tab3:
        section_header("Detailed Validation Results", icon="📋", tight=True)
        display_columns = [
            "userfullname",
            "Final_Rank",
            "Becslés",
            "Validation_Result",
            "Original_Delta",
            "Inverted_Delta",
        ]
        display_df = results[display_columns].copy()
        display_df = display_df.rename(
            columns={
                "userfullname": "Student Name",
                "Final_Rank": "Rank",
                "Becslés": "Final Score",
                "Validation_Result": "Validation",
                "Original_Delta": "Original Delta",
                "Inverted_Delta": "Inverted Delta",
            }
        ).sort_values("Rank")
        st.dataframe(display_df, use_container_width=True, height=400)

        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Detailed Results (CSV)",
            csv_data,
            f"validation_detailed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True,
        )

    with tab4:
        invalid_cases = results[~results["Is_Valid"]]
        if not invalid_cases.empty:
            section_header("Cases Requiring Review", icon="⚠️", tight=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Invalid Cases", len(invalid_cases))
            with col2:
                st.metric("Percentage of Total", f"{(len(invalid_cases) / total_count) * 100:.1f}%")
            with col3:
                avg_invalid_score = invalid_cases["Becslés"].mean()
                st.metric("Avg Score (Invalid)", f"{avg_invalid_score:.3f}")

            invalid_display = invalid_cases[
                ["userfullname", "Final_Rank", "Becslés", "Original_Delta", "Inverted_Delta", "Validation_Product"]
            ].copy()
            invalid_display = invalid_display.rename(
                columns={
                    "userfullname": "Student Name",
                    "Final_Rank": "Rank",
                    "Becslés": "Final Score",
                    "Original_Delta": "Original Delta",
                    "Inverted_Delta": "Inverted Delta",
                    "Validation_Product": "Product",
                }
            ).sort_values("Rank")
            invalid_display = invalid_display.round({"Final Score": 3, "Original Delta": 3, "Inverted Delta": 3, "Product": 3})
            st.dataframe(invalid_display, use_container_width=True)

            with st.expander("🔍 **Invalid Cases Analysis**"):
                st.write("**Pattern Analysis:**")
                score_bins = pd.cut(invalid_cases["Becslés"], bins=5)
                bin_counts = score_bins.value_counts().sort_index()
                st.write("**Score Distribution of Invalid Cases:**")
                for bin_range, count in bin_counts.items():
                    st.write(f"• {bin_range}: {count} cases")

                st.write("**Delta Analysis:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        f"Original Delta Range: {invalid_cases['Original_Delta'].min():.3f} "
                        f"to {invalid_cases['Original_Delta'].max():.3f}"
                    )
                with col2:
                    st.write(
                        f"Inverted Delta Range: {invalid_cases['Inverted_Delta'].min():.3f} "
                        f"to {invalid_cases['Inverted_Delta'].max():.3f}"
                    )
        else:
            st.success("🎉 **Excellent!** No invalid cases found requiring review.")

    divider()
    chart_df = results[["userfullname", "Becslés", "Validation_Result", "Final_Rank"]].copy()
    chart_df = chart_df.rename(
        columns={
            "userfullname": "Student",
            "Becslés": "Score",
            "Validation_Result": "Validation",
            "Final_Rank": "Rank",
        }
    )
    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Student:N", sort=None, title="Student", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Score:Q", title="Final Score (Becslés)"),
            color=alt.Color(
                "Validation:N",
                scale=alt.Scale(domain=["Valid", "Invalid"], range=["#00ff00", "#ff0000"]),
                legend=alt.Legend(title="Validation Result"),
            ),
            tooltip=["Student", "Score", "Validation", "Rank"],
        )
        .properties(width=700, height=400, title="Student Final Scores with Validation Results")
    )
    st.altair_chart(bar_chart, use_container_width=True)

    section_header("Export Options", icon="💾", tight=True)
    summary_data = {
        "Metric": ["Total Students", "Valid Cases", "Invalid Cases", "Validity Rate", "Average Score"],
        "Value": [
            total_count,
            valid_count,
            len(results[~results["Is_Valid"]]),
            f"{validity_percentage:.1f}%",
            f"{results['Becslés'].mean():.3f}",
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False)
    st.download_button(
        "📄 Download Summary Report (CSV)",
        summary_csv,
        f"validation_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
        help="High-level validation metrics and summary",
    )


if __name__ == "__main__":
    main()
