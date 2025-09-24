Moodle Log Analyzer DEMO
⚠️ DEMO VERSION NOTICE
This is a DEMONSTRATION VERSION of the Moodle Log Analyzer with limited functionality for evaluation purposes. Some features may be restricted or operate with sample data.

A Streamlit application for analyzing Moodle discussion forum data to compute student engagement metrics, perform ranking analysis, and validate results using COCO (Complexity Constraint) methodology.


---Demo Version Limitations---
This demo version includes the following limitations:
Fixed Y-value (1000) for ranking analysis
Sample professor names ("professor_1", "professor_2") pre-configured
Limited to predefined exam patterns and deadlines
COCO analysis may use simulated data in offline mode
Maximum dataset size restrictions may apply
Some advanced ML features may operate with reduced accuracy

---Overview---
This demo application processes Moodle discussion forum exports to compute various student engagement attributes, rank students based on their performance, and perform sophisticated validation using COCO analysis. It's particularly useful for educational researchers and instructors who want to analyze student participation patterns in online discussions.

Features
---Data Processing---
Upload CSV or Excel files from Moodle discussion forums
Automatic data validation and cleaning
SQLite database storage for persistent data

---attribute Analysis---
Compute 19 different engagement metrics across four categories:
Activity Metrics
Engagement Metrics
Content Analysis
Exam Performance

---Advanced Analytics---
Student ranking based on selected attributes
COCO (Complexity Constraint) analysis for multi-objective optimization
Validation through inverted ranking analysis
Interactive visualizations

------------Demo Configuration Notes---------
Pre-configured Settings:
Professors: "professor_1", "professor_2" (excluded from analysis
Y-value: Fixed at 1000 for ranking
Exam Patterns: Predefined parent IDs for pattern recognition
Deadlines: Sample exam dates configured

Demo Limitations:
Cannot modify professor exclusion list
Fixed ranking parameters
Limited to predefined exam patterns
COCO service may have usage restrictions

Output Limitations
Demo version generates:
Object Attribute Matrix (OAM) with sample data
Basic ranking results with fixed Y-value
Limited COCO analysis results
Basic validation metrics

--License--
This DEMO VERSION is provided for evaluation purposes only. Commercial use, redistribution, or modification is prohibited. All rights reserved.
