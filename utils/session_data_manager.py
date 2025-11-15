#session_data_manager.py - manage session state for data storage and retrieval
import pandas as pd
import streamlit as st
from datetime import datetime

class SessionDataManager:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize all data storage"""
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'student_attributes' not in st.session_state:
            st.session_state.student_attributes = None
        if 'ranked_results' not in st.session_state:
            st.session_state.ranked_results = None
        if 'coco_results' not in st.session_state:
            st.session_state.coco_results = {}
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'ai_insights' not in st.session_state:
            st.session_state.ai_insights = None
        if 'ai_insights_dirty' not in st.session_state:
            st.session_state.ai_insights_dirty = True
    
    def store_raw_data(self, df, source_info=None):
        """Store raw data in session state"""
        st.session_state.raw_data = {
            'dataframe': df,
            'upload_time': pd.Timestamp.now(),
            'source': source_info,
            'shape': df.shape,
            'columns': list(df.columns)
        }
        self._add_to_history("Raw data uploaded", f"Shape: {df.shape}")
    
    def get_raw_data(self):
        """Get raw data DataFrame from session state"""
        if st.session_state.raw_data and 'dataframe' in st.session_state.raw_data:
            return st.session_state.raw_data['dataframe']
        return None
    
    def get_raw_data_info(self):
        """Get raw data metadata"""
        return st.session_state.raw_data
    
    def get_analysis_history(self):
        """Get the analysis history"""
        return st.session_state.analysis_history
    
    def _add_to_history(self, action, details):
        """Track analysis steps"""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        st.session_state.analysis_history.append({
            'timestamp': pd.Timestamp.now(),
            'action': action,
            'details': str(details)
        })
    
    # Add methods for other data types you'll need
    def store_student_attributes(self, df):
        st.session_state.student_attributes = df
        self._add_to_history("Student attributes computed", f"Shape: {df.shape}")
        self.mark_ai_insights_dirty()
    
    def get_student_attributes(self):
        return st.session_state.student_attributes
    
    def store_ranked_results(self, df):
        st.session_state.ranked_results = df
        self._add_to_history("Ranking completed", f"Shape: {df.shape}")
        self.mark_ai_insights_dirty()
    
    def get_ranked_results(self):
        return st.session_state.ranked_results
    
    def store_coco_results(self, tables):
        """Store COCO analysis results"""
        st.session_state.coco_results = tables
        self._add_to_history("COCO analysis completed", f"{len(tables)} tables generated")
    
    def get_coco_results(self, table_name=None):
        """Get COCO analysis results"""
        if table_name:
            return st.session_state.coco_results.get(table_name)
        return st.session_state.coco_results
    
    def store_validation_results(self, df):
        """Store validation results"""
        st.session_state.validation_results = df
        self._add_to_history("Validation completed", f"Results for {len(df)} students")
    
    def get_validation_results(self):
        """Get validation results"""
        return st.session_state.validation_results
    
    def clear_session(self):
        """Clear all session data"""
        self.init_session_state()  # Reset to initial state

    def mark_ai_insights_dirty(self):
        st.session_state.ai_insights_dirty = True

    def is_ai_insights_dirty(self):
        return st.session_state.get("ai_insights_dirty", True)

    def store_ai_insights(self, insights):
        st.session_state.ai_insights = insights
        st.session_state.ai_insights_dirty = False
        summary = (insights or {}).get("summary", "")
        snippet = (summary[:75] + "...") if summary and len(summary) > 78 else summary
        self._add_to_history("AI insights updated", snippet)

    def get_ai_insights(self):
        return st.session_state.ai_insights

    def evaluate_rules(self, df, rules, settings=None):
        """Evaluate configured rules against the provided dataframe."""
        if df is None or df.empty or not rules:
            return {
                "per_student": [],
                "missing_columns": [],
                "rule_counts": {},
            }
        settings = settings or {}
        missing_columns = set()
        rule_counts = {rule.get("id", idx): 0 for idx, rule in enumerate(rules)}
        per_student = {}
        
        for idx, row in df.iterrows():
            student_key = row.get("userid")
            if pd.isna(student_key):
                student_key = f"row_{idx}"
            entry = {
                "userid": row.get("userid"),
                "userfullname": row.get("userfullname"),
                "row_index": idx,
                "rules": [],
            }
            per_student.setdefault(student_key, entry)
            
            for rule_index, rule in enumerate(rules):
                rule_id = rule.get("id", f"rule_{rule_index}")
                resolved = self._rule_matches(row, rule, df, settings, missing_columns)
                if resolved:
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
                    per_student[student_key]["rules"].append({
                        "id": rule_id,
                        "label": rule.get("label", rule_id),
                        "message": rule.get("message", ""),
                        "severity": rule.get("severity", "medium"),
                        "playbook": rule.get("playbook"),
                    })
        
        per_student_list = [entry for entry in per_student.values() if entry["rules"]]
        return {
            "per_student": per_student_list,
            "missing_columns": sorted(missing_columns),
            "rule_counts": rule_counts,
        }

    @staticmethod
    def _resolve_column_name(df, column_name):
        if not column_name:
            return None
        if column_name in df.columns:
            return column_name
        lower_map = {col.lower(): col for col in df.columns}
        return lower_map.get(column_name.lower())

    def _rule_matches(self, row, rule, df, settings, missing_columns):
        """Check whether a single rule matches a row."""
        conditions = rule.get("conditions", [])
        if not conditions:
            return False
        for condition in conditions:
            column = condition.get("column")
            resolved_column = self._resolve_column_name(df, column)
            if not resolved_column:
                missing_columns.add(column)
                return False
            value = row.get(resolved_column)
            if not self._evaluate_condition(value, condition, settings):
                return False
        return True

    @staticmethod
    def _coerce_numeric(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                stripped = value.strip()
                if stripped == "":
                    return None
                return float(stripped)
            except (TypeError, ValueError):
                return None
        return None

    def _evaluate_condition(self, value, condition, settings):
        op = (condition.get("op") or "eq").lower()
        threshold = self._condition_value(condition, settings)

        if op in {"lt", "less_than"}:
            value_num = self._coerce_numeric(value)
            threshold = self._coerce_numeric(threshold)
            if value_num is None or threshold is None:
                return False
            return value_num < threshold
        if op in {"le", "lte", "less_or_equal"}:
            value_num = self._coerce_numeric(value)
            threshold = self._coerce_numeric(threshold)
            if value_num is None or threshold is None:
                return False
            return value_num <= threshold
        if op in {"gt", "greater_than"}:
            value_num = self._coerce_numeric(value)
            threshold = self._coerce_numeric(threshold)
            if value_num is None or threshold is None:
                return False
            return value_num > threshold
        if op in {"ge", "gte", "greater_or_equal"}:
            value_num = self._coerce_numeric(value)
            threshold = self._coerce_numeric(threshold)
            if value_num is None or threshold is None:
                return False
            return value_num >= threshold
        if op in {"eq", "equals"}:
            if pd.isna(value) and pd.isna(threshold):
                return True
            return value == threshold
        if op in {"ne", "not_equals"}:
            if pd.isna(value) and pd.isna(threshold):
                return False
            return value != threshold
        if op == "between":
            lower = condition.get("min")
            upper = condition.get("max")
            if condition.get("min_setting"):
                lower = settings.get(condition.get("min_setting"))
            if condition.get("max_setting"):
                upper = settings.get(condition.get("max_setting"))
            lower = self._coerce_numeric(lower)
            upper = self._coerce_numeric(upper)
            value_num = self._coerce_numeric(value)
            if None in (lower, upper, value_num):
                return False
            return lower <= value_num <= upper
        if op == "is_true":
            return bool(value)
        if op == "is_false":
            return not bool(value)
        if op == "missing":
            return pd.isna(value)
        if op == "not_missing":
            return not pd.isna(value)
        if op == "contains":
            if value is None:
                return False
            target = threshold
            if target is None:
                return False
            return str(target).lower() in str(value).lower()
        return False

    @staticmethod
    def _condition_value(condition, settings):
        if "setting" in condition:
            return settings.get(condition["setting"])
        if "value" in condition:
            return condition.get("value")
        return None
