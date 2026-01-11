Moodle Log Analyzer DEMO
⚠️ DEMO VERSION NOTICE
This is a DEMONSTRATION VERSION of the Moodle Log Analyzer. Operate with sample data.

---Overview---
This demo Streamlit application processes Moodle discussion forum exports to compute various student engagement attributes, rank students based on their performance as well as on the direction-rules/codes defined by the particular users, and perform antidiscriminative optimization based on similarity analysis using COCO(component-based object comparison for objectivity) analysis. It's particularly useful for educational researchers and instructors who want to analyze student participation patterns in online discussions in an objective way across different course structures and cohort types.

---Demo Version---
-Professor names are selectable from detected user names (or manual entry)
-Exam names and deadlines are configured from detected subject values
-Maximum dataset size restrictions may apply
-Some advanced ML features may operate with reduced accuracy

Features
-Upload CSV or Excel files from Moodle discussion forums which can be downloaded directly from the data upload page
-Automatic data validation and cleaning
-Student ranking based on selected attributes
-COCO (component-based object comparison for objectivity) analysis for multi-dimensional optimization
-Validation through function-symmetry-based evaluation
-Interactive Visualizations & Dashboards
-AI summaries and rule based detection
-Student Q&A Copilot
-Comprehensive Export Functionality: Export every stage of the pipeline


---attribute Analysis---
Compute core engagement metrics across four categories (plus per-exam deadline metrics):
  1. Activity Metrics
-total_posts
-active_days
-average_posts_per_day
-max_streak
-modification_count

  2. Engagement Metrics
-total_replies_to_professor
-unique_interactions
-unique_discussions
-engagement_rate
-avg_reply_time
-valid_response

  3. Content Analysis
-total_characters
-total_words
-citation_count
-topic_relevance_score
-avg_AI_involvedMsg_score

  4. Exam Performance
-deadline_exceeded_posts_<exam_name> (generated for each configured exam)


This VERSION is provided for demonstration purposes only.
