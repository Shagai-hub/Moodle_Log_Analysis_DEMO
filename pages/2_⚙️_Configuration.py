# pages/2_⚙️_Configuration.py
import streamlit as st
from utils.config_manager import ConfigManager
from utils.session_data_manager import SessionDataManager
import pandas as pd

# Safe initialization
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = SessionDataManager()

def main():
    config = st.session_state.config
    data_manager = st.session_state.data_manager
    
    st.title("⚙️ Configuration Settings")
    
    # Configuration summary
    
    # Main configuration interface
    st.subheader("Configure Analysis Parameters")
    
    tab1, tab2, tab3 = st.tabs(["👨‍🏫 Professors & Exams", "📊 Analysis Settings", "💾 Export/Import"])
    
    with tab1:
        # Professor configuration
        st.markdown("### Professor Settings")
        
        prof_input = st.text_area(
            "Professor names (one per line)", 
            value="\n".join(config.professors),
            height=150,
            help="Enter one professor name per line. These names will be used to identify professor-related activities in the logs."
        )
        config.professors = [p.strip() for p in prof_input.split('\n') if p.strip()]
        
        # Deadline configuration
        st.markdown("### Exam Deadline Settings")
        st.markdown("Set deadlines for each exam (posts after these dates will be flagged)")
        
        # Add new exam
        col1, col2 = st.columns([3, 1])
        with col1:
            new_exam = st.text_input("Add new exam name", placeholder="e.g., Final Exam")
        with col2:
            if st.button("Add Exam", use_container_width=True) and new_exam:
                if new_exam not in config.deadlines:
                    config.deadlines[new_exam] = pd.Timestamp.now()
                    st.success(f"Added {new_exam}")
                    st.rerun()
        
        # Existing deadlines
        for exam_name in list(config.deadlines.keys()):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.text_input("Exam name", value=exam_name, key=f"name_{exam_name}", disabled=True)
            with col2:
                new_date = st.date_input(
                    "Deadline",
                    value=config.deadlines[exam_name].date(),
                    key=f"date_{exam_name}"
                )
                config.deadlines[exam_name] = pd.to_datetime(new_date)
            with col3:
                if st.button("🗑️", key=f"delete_{exam_name}") and len(config.deadlines) > 1:
                    del config.deadlines[exam_name]
                    st.rerun()
    
    with tab2:
        st.markdown("### Advanced Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pattern Matching")
            pattern_input = st.text_input(
                "Parent IDs for pattern analysis",
                value=", ".join(map(str, config.parent_ids_pattern)),
                help="Comma-separated list of parent IDs used for pattern_followed attribute"
            )
            try:
                config.parent_ids_pattern = [int(x.strip()) for x in pattern_input.split(",") if x.strip()]
            except ValueError:
                st.error("Please enter valid integer IDs")
        
        with col2:
            st.markdown("#### COCO Analysis Settings")
            config.analysis_settings['y_value'] = st.number_input(
                "Y value for COCO analysis",
                value=config.analysis_settings['y_value'],
                min_value=0,
                max_value=100000,
                help="Reference value used in ranking and COCO analysis"
            )
            
    with tab3:
        st.markdown("### Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Save/Load")
            if st.button("💾 Save Current Configuration", use_container_width=True):
                config.save_to_session()
                data_manager._add_to_history("Configuration updated", f"{summary['exams_count']} exams, {summary['professors_count']} professors")
                st.success("Configuration saved to session!")
            
            # Export configuration as JSON
            import json
            config_dict = config.to_dict()
            config_json = json.dumps(config_dict, indent=2)
            st.download_button(
                "📥 Export Configuration as JSON",
                config_json,
                "moodle_analyzer_config.json",
                "application/json",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Reset")
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                config.load_defaults()
                st.success("Configuration reset to defaults!")
                st.rerun()
            
            # Import configuration
            uploaded_config = st.file_uploader("Import configuration JSON", type=['json'])
            if uploaded_config is not None:
                try:
                    import json
                    config_dict = json.load(uploaded_config)
                    config.from_dict(config_dict)
                    st.success("Configuration imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing configuration: {e}")
    
    # Configuration preview
    with st.expander("🔍 Configuration Preview", expanded=False):
        st.json(config.to_dict())
        
        st.subheader("📊 Current Configuration")
    summary = config.get_config_summary()

if __name__ == "__main__":
    main()