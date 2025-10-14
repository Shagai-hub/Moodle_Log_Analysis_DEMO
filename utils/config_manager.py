import streamlit as st
import pandas as pd
from datetime import datetime

class ConfigManager:
    def __init__(self):
        self.load_defaults()
    
    def load_defaults(self):
        """Load default configuration"""
        self.professors = ["professor_1", "professor_2"]
        self.deadlines = {
            "Quasi Exam I": pd.to_datetime("2024-11-09 00:00:00"),
            "Quasi Exam II": pd.to_datetime("2024-11-09 00:00:00"),
            "Quasi Exam III": pd.to_datetime("2024-11-16 00:00:00"),
        }
        self.parent_ids_pattern = [163486]
        self.analysis_settings = {
            'enable_ml_attributes': True,
            'auto_compute_attributes': False,
            'y_value': 1000
        }
    
    def render_sidebar_config(self):
        """Render configuration UI in sidebar"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Professor configuration
            with st.expander("👨‍🏫 Professors", expanded=False):
                st.markdown("Professors will be excluded from student analysis")
                prof_input = st.text_area(
                    "Professor names (one per line)", 
                    value="\n".join(self.professors),
                    height=100,
                    help="Enter one professor name per line. These users will be excluded from student analysis."
                )
                self.professors = [p.strip() for p in prof_input.split('\n') if p.strip()]
                st.write(f"**Currently configured:** {len(self.professors)} professors")
            
            # Deadline configuration
            with st.expander("📅 Exam Deadlines", expanded=False):
                st.markdown("Set deadlines for each exam")
                
                # Dynamic deadline management
                col1, col2 = st.columns([3, 1])
                with col1:
                    exam_names = list(self.deadlines.keys())
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
                st.subheader("Pattern Matching")
                pattern_input = st.text_input(
                    "Parent IDs for Pattern Matching",
                    value=", ".join(map(str, self.parent_ids_pattern)),
                    help="Comma-separated list of parent IDs used for pattern detection"
                )
                try:
                    self.parent_ids_pattern = [int(x.strip()) for x in pattern_input.split(",") if x.strip()]
                except ValueError:
                    st.error("Please enter valid integer IDs")
                
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
            'pattern_ids_count': len(self.parent_ids_pattern),
            'ml_enabled': self.analysis_settings['enable_ml_attributes'],
            'y_value': self.analysis_settings['y_value']
        }
    
    def to_dict(self):
        """Convert configuration to dictionary for storage"""
        return {
            'professors': self.professors,
            'deadlines': {k: v.isoformat() for k, v in self.deadlines.items()},
            'parent_ids_pattern': self.parent_ids_pattern,
            'analysis_settings': self.analysis_settings
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
        
        self.parent_ids_pattern = config_dict.get('parent_ids_pattern', self.parent_ids_pattern)
        self.analysis_settings = config_dict.get('analysis_settings', self.analysis_settings)