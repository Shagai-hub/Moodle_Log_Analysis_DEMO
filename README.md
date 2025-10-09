Moodle Log Analyzer DEMO
⚠️ DEMO VERSION NOTICE
This is a DEMONSTRATION VERSION of the Moodle Log Analyzer. Operate with sample data.

---Overview---
This demo Streamlit application processes Moodle discussion forum exports to compute various student engagement attributes, rank students based on their performance as well as on the direction-rules/codes defined by the particular users, and perform antidiscriminative optimization based on similarity analysis using COCO(component-based object comparison for objectivity) analysis. It's particularly useful for educational researchers and instructors who want to analyze student participation patterns in online discussions in an objective way.

---Demo Version---
Fixed Y-value (1000) for antidiscrimination analysis
Sample professor names ("professor_1", "professor_2") pre-configured
Limited to predefined exam patterns and deadlines
Maximum dataset size restrictions may apply
Some advanced ML features may operate with reduced accuracy

Features
---Data Processing---
Upload CSV or Excel files from Moodle discussion forums
Automatic data validation and cleaning
SQLite database storage for persistent data

---attribute Analysis---
Compute 19 different engagement metrics across four categories:
  1. Activity Metrics
1.total_posts
2.active_days
3.total_characters
4.total_words
5.unique_discussions
6.max streak

  2. Engagement Metrics
7.total_replies_to_professor
8.unique_interactions
9.engagement_rate
10.avg_reply_time
11.modification_count

  3. Content Analysis
12.valid_response
13.citation_count
14.topic_relevance_score
15.avg_AI_involvedMsg_score

  4. Exam Performance
16.Pattern_followed_quasi_exam_i
17.deadline_exceeded_posts_Quasi_exam_I
18.deadline_exceeded_posts_Quasi_exam_II
19.deadline_exceeded_posts_Quasi_exam_III

---Advanced Analytics---
Student ranking based on selected attributes
COCO (component-based object comparison for objectivity) analysis for multi-dimensional optimization
Validation through function-symmetry-based evaluation
Interactive visualizations


This VERSION is provided for demonstration purposes only.
