#config manager for the app#
import json
import copy
import streamlit as st
import pandas as pd
from datetime import datetime

class ConfigManager:
    def __init__(self):
        self.load_defaults()
    
    def load_defaults(self):
        """Load default configuration"""
        self.professors = []
        self.deadlines = {}
        self.analysis_settings = {
            'enable_ml_attributes': True,
            'auto_compute_attributes': False,
            'y_value': 1000
        }
        self.ai_insights_settings = self._default_ai_settings()
        self.ai_insight_rules = self._default_ai_rules()
    
    def render_sidebar_config(self):
        """Render configuration UI in sidebar"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Professor configuration
            with st.expander("👨‍🏫 Professors", expanded=False):
                st.markdown("Professors will be excluded from student analysis")
                raw_info = st.session_state.get("raw_data")
                raw_df = raw_info.get("dataframe") if isinstance(raw_info, dict) else None
                detected_names = []
                if raw_df is not None and "userfullname" in raw_df.columns:
                    detected_names = (
                        raw_df["userfullname"].dropna().astype(str).value_counts().index.tolist()
                    )
                selected_names = st.multiselect(
                    "Select professor names from data",
                    options=detected_names,
                    default=[name for name in self.professors if name in detected_names],
                    help="Choose from detected user names to tag professors.",
                )
                prof_input = st.text_area(
                    "Professor names (one per line)", 
                    value="\n".join(self.professors),
                    height=100,
                    help="Enter one professor name per line. These users will be excluded from student analysis."
                )
                manual_names = [p.strip() for p in prof_input.split('\n') if p.strip()]
                combined = list(dict.fromkeys(selected_names + manual_names))
                self.professors = combined
                st.write(f"**Currently configured:** {len(self.professors)} professors")
            
            # Deadline configuration
            with st.expander("📅 Exam Deadlines", expanded=False):
                st.markdown("Set deadlines for each exam")
                
                # Dynamic deadline management
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_exam = st.text_input("Add new exam name")
                with col2:
                    if st.button("Add Exam") and new_exam:
                        if new_exam not in self.deadlines:
                            self.deadlines[new_exam] = pd.Timestamp.now()
                            st.rerun()
                
                # Display and edit existing deadlines
                for exam_name in list(self.deadlines.keys()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_deadline = st.date_input(
                            f"{exam_name} Deadline",
                            value=self.deadlines[exam_name].date()
                        )
                        self.deadlines[exam_name] = pd.to_datetime(new_deadline)
                    with col2:
                        if st.button("🗑️", key=f"del_{exam_name}") and len(self.deadlines) > 1:
                            del self.deadlines[exam_name]
                            st.rerun()
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings", expanded=False):
                st.subheader("Analysis Settings")
                self.analysis_settings['enable_ml_attributes'] = st.checkbox(
                    "Enable ML-based Attributes", 
                    value=self.analysis_settings['enable_ml_attributes'],
                    help="Compute topic relevance and AI detection (slower but more advanced)"
                )
                self.analysis_settings['y_value'] = st.number_input(
                    "Y Value for Ranking",
                    value=self.analysis_settings['y_value'],
                    help="Reference value for COCO analysis"
                )

            with st.expander("🤖 AI Insights", expanded=False):
                st.subheader("Model & Thresholds")
                model_options = [
                    ("manual", "Manual narrative (fastest)"),
                    ("google/flan-t5-small", "FLAN-T5 small (Hugging Face)"),
                    ("facebook/bart-base", "BART base"),
                    ("facebook/bart-large-cnn", "BART large CNN"),
                    ("sshleifer/distilbart-cnn-12-6", "DistilBART CNN"),
                ]
                option_values = [value for value, _ in model_options]
                labels_map = {value: label for value, label in model_options}
                current_model = self.ai_insights_settings.get("summary_model", option_values[0])
                selected_model = st.selectbox(
                    "Summary Model",
                    option_values,
                    index=option_values.index(current_model) if current_model in option_values else 0,
                    format_func=lambda value: labels_map.get(value, value),
                    help="Manual mode skips heavy ML downloads. Model options require the Transformers package and download time.",
                )
                self.ai_insights_settings["summary_model"] = selected_model
                self.ai_insights_settings["summary_temperature"] = st.slider(
                    "Summary Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(self.ai_insights_settings.get("summary_temperature", 0.0)),
                    step=0.05,
                    help="Higher values add variability to generated summaries.",
                )
                self.ai_insights_settings["low_engagement_threshold"] = st.number_input(
                    "Low Engagement Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(self.ai_insights_settings.get("low_engagement_threshold", 0.25)),
                    step=0.05,
                    help="Flag students whose engagement rate drops below this value.",
                    format="%.2f",
                )
                self.ai_insights_settings["low_posts_threshold"] = st.number_input(
                    "Low Posts Threshold",
                    min_value=0,
                    max_value=100,
                    value=int(self.ai_insights_settings.get("low_posts_threshold", 5)),
                    step=1,
                    help="Flag students whose total posts fall below this count.",
                )
                self.ai_insights_settings["deadline_miss_threshold"] = st.number_input(
                    "Deadline Miss Threshold",
                    min_value=0,
                    max_value=10,
                    value=int(self.ai_insights_settings.get("deadline_miss_threshold", 1)),
                    step=1,
                    help="Flag students exceeding this number of deadline misses per exam.",
                )
                self.ai_insights_settings["low_professor_reply_threshold"] = st.number_input(
                    "Replies to Professor Threshold",
                    min_value=0,
                    max_value=20,
                    value=int(self.ai_insights_settings.get("low_professor_reply_threshold", 1)),
                    step=1,
                    help="Flag students at or below this count of replies to the professor.",
                )
                self.ai_insights_settings["slow_reply_threshold_hours"] = st.number_input(
                    "Slow Reply Threshold (hours)",
                    min_value=0.0,
                    max_value=168.0,
                    value=float(self.ai_insights_settings.get("slow_reply_threshold_hours", 24)),
                    step=1.0,
                    help="Flag students whose average reply time exceeds this duration.",
                    format="%.0f",
                )
                self.ai_insights_settings["zscore_low_threshold"] = st.slider(
                    "Low Activity Z-score Threshold",
                    min_value=-3.0,
                    max_value=-0.5,
                    value=float(self.ai_insights_settings.get("zscore_low_threshold", -1.5)),
                    step=0.1,
                    help="Values below this z-score will be flagged as low outliers.",
                )
                self.ai_insights_settings["zscore_high_threshold"] = st.slider(
                    "High Activity Z-score Threshold",
                    min_value=0.5,
                    max_value=3.0,
                    value=float(self.ai_insights_settings.get("zscore_high_threshold", 2.5)),
                    step=0.1,
                    help="Values above this z-score will be flagged as unusually high.",
                )

                st.markdown("### Rule Definitions")
                st.caption("Edit JSON to fine-tune the AI playbooks that tag students. Each rule supports simple AND conditions.")

                rules_editor_key = "ai_rules_text_area"
                default_rules_text = json.dumps(self.ai_insight_rules, indent=2)
                if rules_editor_key not in st.session_state:
                    st.session_state[rules_editor_key] = default_rules_text
                elif st.session_state[rules_editor_key] == "" and self.ai_insight_rules:
                    st.session_state[rules_editor_key] = default_rules_text

                col_rules_btn, _ = st.columns([1, 3])
                with col_rules_btn:
                    if st.button("↩️ Restore Default Rules"):
                        self.ai_insight_rules = self._default_ai_rules()
                        st.session_state[rules_editor_key] = json.dumps(self.ai_insight_rules, indent=2)

                rules_text = st.text_area(
                    "AI Insight Rules (JSON)",
                    value=st.session_state[rules_editor_key],
                    height=220,
                    key=rules_editor_key,
                )
                try:
                    parsed_rules = json.loads(rules_text) if rules_text.strip() else []
                    if isinstance(parsed_rules, list):
                        self.ai_insight_rules = parsed_rules
                        st.session_state[rules_editor_key] = rules_text
                    else:
                        st.warning("Rules JSON should be a list. Keeping previous configuration.")
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON for AI insight rules: {exc}. Using last valid configuration.")
            
            # Configuration actions
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Config", use_container_width=True):
                    self.save_to_session()
                    st.success("Configuration saved!")
            with col2:
                if st.button("🔄 Reset Defaults", use_container_width=True):
                    self.load_defaults()
                    st.rerun()
    
    def save_to_session(self):
        """Save configuration to session state"""
        st.session_state.config = self
    
    def get_config_summary(self):
        """Get a summary of current configuration"""
        return {
            'professors_count': len(self.professors),
            'exams_count': len(self.deadlines),
            'exam_names': list(self.deadlines.keys()),
            'ml_enabled': self.analysis_settings['enable_ml_attributes'],
            'y_value': self.analysis_settings['y_value']
        }
    
    def to_dict(self):
        """Convert configuration to dictionary for storage"""
        return {
            'professors': self.professors,
            'deadlines': {k: v.isoformat() for k, v in self.deadlines.items()},
            'analysis_settings': self.analysis_settings,
            'ai_insights_settings': self.ai_insights_settings,
            'ai_insight_rules': self.ai_insight_rules,
        }
    
    def from_dict(self, config_dict):
        """Load configuration from dictionary"""
        self.professors = config_dict.get('professors', self.professors)
        
        # Convert string dates back to datetime
        deadlines_dict = config_dict.get('deadlines', {})
        self.deadlines = {}
        for k, v in deadlines_dict.items():
            if isinstance(v, str):
                self.deadlines[k] = pd.to_datetime(v)
            else:
                self.deadlines[k] = v
        
        self.analysis_settings = config_dict.get('analysis_settings', self.analysis_settings)
        self.ai_insights_settings = config_dict.get('ai_insights_settings', self.ai_insights_settings)
        self.ai_insight_rules = config_dict.get('ai_insight_rules', self.ai_insight_rules)

    def _default_ai_settings(self):
        return {
            "summary_model": "manual",
            "summary_temperature": 0.0,
            "low_engagement_threshold": 0.25,
            "low_posts_threshold": 5,
            "deadline_miss_threshold": 1,
            "low_professor_reply_threshold": 1,
            "slow_reply_threshold_hours": 24,
            "zscore_low_threshold": -1.5,
            "zscore_high_threshold": 2.5,
        }

    def _default_ai_rules(self):
        return copy.deepcopy([
            {
                "id": "low_engagement_combo",
                "label": "Low Engagement",
                "severity": "high",
                "message": "Engagement rate and posting activity fall below configured thresholds.",
                "conditions": [
                    {"column": "engagement_rate", "op": "lt", "setting": "low_engagement_threshold"},
                    {"column": "total_posts", "op": "lt", "setting": "low_posts_threshold"},
                ],
                "playbook": [
                    "Send a nudge summarizing missed interactions.",
                    "Offer a short 1:1 check-in slot.",
                ],
            },
            {
                "id": "few_professor_replies",
                "label": "No replies to professor",
                "severity": "medium",
                "message": "Replies back to the professor are below the configured threshold.",
                "conditions": [
                    {"column": "total_replies_to_professor", "op": "le", "setting": "low_professor_reply_threshold"}
                ],
            },
            {
                "id": "slow_responder",
                "label": "Slow response times",
                "severity": "medium",
                "message": "Average reply time exceeds configured hours.",
                "conditions": [
                    {"column": "avg_reply_time", "op": "gt", "setting": "slow_reply_threshold_hours"}
                ],
            },
        ])
