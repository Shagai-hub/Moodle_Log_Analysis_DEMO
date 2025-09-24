import os
import re
import requests
from datetime import timedelta
import streamlit as st
import pandas as pd
import sqlite3
from collections import Counter
from textstat import flesch_kincaid_grade, gunning_fog, smog_index
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from bs4 import BeautifulSoup
from io import StringIO
import numpy as np
import altair as alt

st.title("Moodle Log Analyzer DEMO")

DB_NAME = "logs.db"
PROFESSORS = ["professor_1", "professor_2"]  # exclude these from OAM
PARENT_IDS_PATTERN = [163486]  # used for pattern_followed (adjust if needed)

# Deadlines (use next-day midnight for '24:00:00' semantics)
DEADLINES = {
    "Quasi Exam I": pd.to_datetime("2024-11-09 00:00:00"),
    "Quasi Exam II": pd.to_datetime("2024-11-09 00:00:00"),
    "Quasi Exam III": pd.to_datetime("2024-11-16 00:00:00"),
}

# ---------- Utilities ----------
def read_table(conn, table_name):
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        return pd.DataFrame()

def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


# Connect or create DB
conn = sqlite3.connect(DB_NAME, check_same_thread=False)

st.header("📥 Upload Discussion Data")

uploaded_file = st.file_uploader(
    "Upload your discussion data file (CSV or XLSX):", 
    type=["csv", "xlsx"], 
    help="File should contain columns like userid, userfullname, message, created, etc."
)

current_table_name = None

if uploaded_file:
    # Process the uploaded file
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    df = None

    if file_ext == ".csv":
        encodings = ["utf-8", "latin1", "cp1250", "utf-8-sig"]
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                # Try different separators
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=',')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep=';')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
                break
            except Exception as e:
                continue

    elif file_ext == ".xlsx":
        try:
            import openpyxl
        except ImportError:
            st.error("Missing 'openpyxl' library. Install: `pip install openpyxl`")
            st.stop()
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Failed to read '{uploaded_file.name}': {e}")
            st.stop()

    if df is None:
        st.error(f"❌ Failed to read '{uploaded_file.name}'. Please check the file format.")
        st.stop()

    # Validate required columns
    required_columns = ['userid', 'userfullname', 'message']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("💡 Your file should contain these columns: userid, userfullname, message")
        st.stop()

    # Convert date columns if present
    if 'created' in df.columns:
        try:
            df['created'] = pd.to_datetime(df['created'], errors='coerce')
            df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.warning(f"Could not parse 'created' dates: {e}")

    if 'modified' in df.columns:
        try:
            df['modified'] = pd.to_datetime(df['modified'], errors='coerce')
            df['modified'] = df['modified'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.warning(f"Could not parse 'modified' dates: {e}")

    # Clean column names for SQLite
    df.columns = [col.replace(" ", "_").replace(".", "_").replace("-", "_").replace("(", "").replace(")", "") 
                 for col in df.columns]

    # Create table name from filename
    table_name = os.path.splitext(uploaded_file.name)[0]
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
    
    # Ensure table name is valid
    if not table_name or table_name[0].isdigit():
        table_name = "discussion_data_" + table_name

    try:
        # Save to database
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        
        current_table_name = table_name
        
        st.success(f"✅ Successfully loaded '{uploaded_file.name}' as '{table_name}'")
        
        # Show dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Unique Users", df['userfullname'].nunique())
        
        # Show data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(10))

    except Exception as e:
        st.error(f"❌ Failed to save data to database: {e}")
        st.stop()


# ---------- OAM Analyzer ----------
st.header("Select attributes to compute")

if uploaded_file and current_table_name:
    st.header("🎯 Analysis")
    
    df_all = read_table(conn, current_table_name)
    
    if df_all.empty:
        st.error(f"❌ Cannot read table '{current_table_name}'. Please re-upload your data.")
        st.stop()
    
    # Exclude professors
    df_all["userfullname"] = df_all["userfullname"].astype(str)
    df = df_all[~df_all["userfullname"].isin(PROFESSORS)].copy()
    
    st.write(f"📊 Analyzing dataset: **{current_table_name}**")


# ---------- Attribute functions ----------

def active_days(df):
    df2 = df.copy()
    df2["date_only"] = to_dt(df2["created"]).dt.date
    res = df2.groupby(["userid", "userfullname"], as_index=False)["date_only"].nunique()
    res = res.rename(columns={"date_only": "active_days"})
    return res
def total_replies_to_professor(df, df_all, prof_name="professor_1"):
    # Use df_all (which includes professors) to find professor post IDs#
    prof_ids = df_all[df_all["userfullname"] == prof_name]["id"].unique().tolist()
    if len(prof_ids) == 0:
     return pd.DataFrame(columns=["userid", "userfullname", "total_replies_to_professor"])
        # Use df (which excludes professors) to count replies to professor
    res = df[df["parent"].isin(prof_ids)].groupby(["userid", "userfullname"], as_index=False).size().rename(columns={"size": "total_replies_to_professor"})
    return res
def total_characters(df):
    if "charcount" in df.columns:
        return df.groupby(["userid", "userfullname"], as_index=False)["charcount"].sum().rename(columns={"charcount": "total_characters"})
    df2 = df.copy()
    df2["charcount_calc"] = df2["message"].fillna("").str.len()
    return df2.groupby(["userid", "userfullname"], as_index=False)["charcount_calc"].sum().rename(columns={"charcount_calc": "total_characters"})
def total_words(df):
    if "wordcount" in df.columns:
        return df.groupby(["userid", "userfullname"], as_index=False)["wordcount"].sum().rename(columns={"wordcount": "total_words"})
    df2 = df.copy()
    df2["wordcount_calc"] = df2["message"].fillna("").str.split().map(len)
    return df2.groupby(["userid", "userfullname"], as_index=False)["wordcount_calc"].sum().rename(columns={"wordcount_calc": "total_words"})
def unique_interactions(df):
    edges = df[["id", "parent", "userid", "userfullname"]].copy()
    parent_map = df.set_index("id")[["userid", "userfullname"]]
    edges["other_userid"] = edges.apply(lambda r: parent_map["userid"].get(r["parent"]) if r["parent"] in parent_map.index else None, axis=1)
    edges = edges.dropna(subset=["other_userid"])
    edges = edges[edges["other_userid"] != edges["userid"]]
    res = edges.groupby(["userid", "userfullname"], as_index=False)["other_userid"].nunique().rename(columns={"other_userid": "unique_interactions"})
    return res
def unique_discussions(df):
    if "subject" in df.columns:
        res = df.groupby(["userid", "userfullname"], as_index=False)["subject"].nunique().rename(columns={"subject": "unique_discussions"})
        return res
    if "discussionid" in df.columns:
        res = df.groupby(["userid", "userfullname"], as_index=False)["discussionid"].nunique().rename(columns={"discussionid": "unique_discussions"})
        return res
    df2 = df.copy()
    df2["thread_root"] = df2["parent"].fillna(0)
    res = df2.groupby(["userid", "userfullname"], as_index=False)["thread_root"].nunique().rename(columns={"thread_root": "unique_discussions"})
    return res
def engagement_rate(df, df_all):
    prof_total = len(df_all[df_all["userfullname"] == "professor_1"])
    if prof_total == 0:
        prof_total = 1
    replies = total_replies_to_professor(df, df_all)
    replies["engagement_rate"] = replies["total_replies_to_professor"] / prof_total
    return replies[["userid","userfullname","engagement_rate"]]
def avg_reply_time(df, df_all):
    prof_ids = df_all[df_all["userfullname"] == "professor_1"]["id"].unique().tolist()
    if len(prof_ids) == 0:
        # Return 0 for all users instead of empty DataFrame
        all_users = df[["userid", "userfullname"]].drop_duplicates()
        all_users["avg_reply_time"] = 0
        return all_users
        
    repls = df[df["parent"].isin(prof_ids)].copy()
    if len(repls) == 0:
        # Return 0 for all users instead of empty DataFrame
        all_users = df[["userid", "userfullname"]].drop_duplicates()
        all_users["avg_reply_time"] = 0
        return all_users
    
    # Rest of your calculation code...
    repls["created_dt"] = to_dt(repls["created"])
    parent_created = df_all.set_index("id")["created"].to_dict()
    repls["parent_created"] = repls["parent"].map(parent_created)
    repls["parent_created_dt"] = to_dt(repls["parent_created"])
    repls["reply_hours"] = (repls["created_dt"] - repls["parent_created_dt"]).dt.total_seconds() / 3600.0
    
    # Calculate average and ensure all users are included
    avg_times = repls.groupby(["userid","userfullname"], as_index=False)["reply_hours"].mean()
    avg_times = avg_times.rename(columns={"reply_hours": "avg_reply_time"})
    
    # Merge with all users to ensure everyone is included
    all_users = df[["userid", "userfullname"]].drop_duplicates()
    result = all_users.merge(avg_times, on=["userid", "userfullname"], how="left")
    result["avg_reply_time"] = result["avg_reply_time"].fillna(0)
    
    return result

def modification_count(df):
    df2 = df.copy()
    df2["created_dt"] = to_dt(df2["created"])
    df2["modified_dt"] = to_dt(df2["modified"])
    mod = df2[df2["created_dt"].notna() & df2["modified_dt"].notna() & (df2["modified_dt"] != df2["created_dt"])].copy()
    if mod.empty:
        return pd.DataFrame(columns=["userid","userfullname","modification_count","avg_modified_time_minutes"])
    mod["modified_seconds"] = (mod["modified_dt"] - mod["created_dt"]).dt.total_seconds()
    res = mod.groupby(["userid","userfullname"], as_index=False).agg(
        modification_count=("id","count")
    )
    return res
def average_posts_per_day(df):
    df2 = df.copy()
    df2["post_date"] = to_dt(df2["created"]).dt.date
    perday = df2.groupby(["userid","userfullname","post_date"], as_index=False).size().rename(columns={"size":"posts_per_day"})
    res = perday.groupby(["userid","userfullname"], as_index=False)["posts_per_day"].mean().rename(columns={"posts_per_day":"average_posts_per_day"})
    return res
def valid_response(df):
    cond = df.groupby(["userid","userfullname"]).apply(
        lambda g: int((g[(g["parent"]==163483) & (g["message"].astype(str).str.strip()=='37')].shape[0] > 0) and
                      (g[(g["parent"]==163486) & (g["message"].astype(str).str.strip()=='5')].shape[0] > 0))
    ).reset_index(name='valid_response')
    return cond
def citation_count(df):
    df2 = df.copy()
    df2["message_fill"] = df2["message"].fillna("").astype(str)
    pattern = re.compile(r'(http[s]?://|www\.|doi:)', re.IGNORECASE)
    df2["has_citation"] = df2["message_fill"].apply(lambda t: 1 if pattern.search(t) else 0)
    res = df2.groupby(["userid","userfullname"], as_index=False)["has_citation"].sum().rename(columns={"has_citation":"citation_count"})
    return res
def max_streak(df):
    df2 = df.copy()
    df2["date_only"] = to_dt(df2["created"]).dt.date
    out = []
    for (uid, uname), g in df2.groupby(["userid","userfullname"]):
        dates = sorted(g["date_only"].dropna().unique())
        if not dates:
            out.append({"userid":uid, "userfullname":uname, "max_streak":0})
            continue
        max_st = cur = 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]) == timedelta(days=1):
                cur += 1
            else:
                cur = 1
            if cur > max_st:
                max_st = cur
        out.append({"userid":uid, "userfullname":uname, "max_streak":int(max_st)})
    return pd.DataFrame(out)
def pattern_followed(df):
    df2 = df.copy()
    dfp = df2[df2["parent"].isin(PARENT_IDS_PATTERN)].copy()
    dfp["is_pattern"] = dfp["message"].fillna("").astype(str).str.match(r'^[0-9]+$')
    res = dfp.groupby(["userid","userfullname"], as_index=False)["is_pattern"].sum().rename(columns={"is_pattern":"Pattern_followed_quasi_exam_i"})
    return res
# Hooks for ML-based attributes (you will provide code)
def compute_topic_relevance(df, df_all, professor_name="professor_1"):
    """
    Compute topic relevance scores for each user based on similarity between
    their replies and the professor's original posts.
    
    Returns DataFrame with columns: ["userid", "userfullname", "topic_relevance_score"]
    """
    try:
        # Find professor's user ID dynamically
        professor_users = df_all[df_all["userfullname"] == professor_name]["userid"].unique()
        
        if len(professor_users) == 0:
            print(f"Professor '{professor_name}' not found in data")
            # Return 0 for all users
            all_users = df[["userid", "userfullname"]].drop_duplicates()
            all_users["topic_relevance_score"] = 0
            return all_users
        
        # Use the first professor ID found (assuming there's only one professor with this name)
        professor_userid = professor_users[0]
        print(f"Using professor ID: {professor_userid} for '{professor_name}'")
        
        # Get professor posts from the full dataset (includes professors)
        prof_posts = df_all[df_all["userid"] == professor_userid][["id", "message"]].copy()
        prof_posts = prof_posts.rename(columns={"id": "prof_id", "message": "prof_message"})
        
        # Handle missing or null messages
        prof_posts['prof_message'] = prof_posts['prof_message'].fillna('')
        prof_posts = prof_posts[prof_posts['prof_message'] != '']
        
        if prof_posts.empty:
            print("No professor posts found")
            return pd.DataFrame(columns=["userid", "userfullname", "topic_relevance_score"])
        
        # Get student replies from the filtered dataset (excludes professors)
        student_replies = df[df["parent"].isin(prof_posts["prof_id"])][["id", "parent", "userid", "userfullname", "message"]].copy()
        student_replies = student_replies.rename(columns={"message": "student_message"})
        
        if student_replies.empty:
            print("No student replies to professor posts found")
            # Return 0 for all users
            all_users = df[["userid", "userfullname"]].drop_duplicates()
            all_users["topic_relevance_score"] = 0
            return all_users
        
        # Load pre-trained model (cache this to avoid reloading each time)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for professor posts
        prof_posts['embedding'] = prof_posts['prof_message'].apply(
            lambda x: model.encode(x) if isinstance(x, str) and x.strip() else None
        )
        
        # Filter out any professor posts without embeddings
        prof_posts = prof_posts[prof_posts['embedding'].notna()]
        
        # Function to calculate similarity
        def calculate_similarity(reply_row, prof_posts_df):
            prof_id = reply_row['parent']
            matching_prof = prof_posts_df[prof_posts_df['prof_id'] == prof_id]
            
            if matching_prof.empty:
                return None
            
            prof_embedding = matching_prof['embedding'].values[0]
            reply_embedding = model.encode(reply_row['student_message'])
            similarity = cosine_similarity([prof_embedding], [reply_embedding])[0][0]
            return similarity
        
        # Compute similarity for each student reply
        student_replies['topic_relevance_score'] = student_replies.apply(
            lambda x: calculate_similarity(x, prof_posts) if x['student_message'] else None, axis=1
        )
        
        # Fill NA values with 0
        student_replies['topic_relevance_score'] = student_replies['topic_relevance_score'].fillna(0)
        
        # Group by user and calculate average relevance score
        result = student_replies.groupby(['userid', 'userfullname'])['topic_relevance_score'].mean().reset_index()
        
        # Ensure all users are included (even those with no replies to professor)
        all_users = df[["userid", "userfullname"]].drop_duplicates()
        result = all_users.merge(result, on=["userid", "userfullname"], how="left")
        result["topic_relevance_score"] = result["topic_relevance_score"].fillna(0)
        
        return result
        
    except Exception as e:
        print(f"Error computing topic relevance: {e}")
        # Return 0 for all users in case of error
        all_users = df[["userid", "userfullname"]].drop_duplicates()
        all_users["topic_relevance_score"] = 0
        return all_users
    
def clean_text(text):
      """Clean text for analysis."""
      text = re.sub(r'[^\w\s]', '', str(text))  # Remove punctuation
      text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
      return text.lower()

def compute_avg_ai_involved(df):
  """
  Compute average AI involvement score for each user.
  Returns DataFrame with columns: ["userid", "userfullname", "avg_AI_involvedMsg_score"]
  """
  try:
      df_clean = df.copy()
      df_clean['cleaned_message'] = df_clean['message'].apply(clean_text)
      
      # Add readability metrics
      df_clean['flesch_kincaid'] = df_clean['cleaned_message'].apply(flesch_kincaid_grade)
      df_clean['gunning_fog'] = df_clean['cleaned_message'].apply(gunning_fog)
      df_clean['smog_index'] = df_clean['cleaned_message'].apply(smog_index)
      df_clean['normalized_readability'] = (df_clean['flesch_kincaid'] + 
                                           df_clean['gunning_fog'] + 
                                           df_clean['smog_index']) / 3

      # Add repetition score
      def repetition_score(text):
          words = text.split()
          word_counts = Counter(words)
          unique_words = len(word_counts)
          total_words = len(words)
          return 1 - (unique_words / total_words) if total_words > 0 else 0

      df_clean['repetition_score'] = df_clean['cleaned_message'].apply(repetition_score)

      # Load AI detector model
      try:
          tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
          model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
          classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
          
          def detect(message):
              try:
                  prediction = classifier(message, truncation=True, max_length=512)
                  return prediction[0]['score']
              except Exception as e:
                  print(f"Error processing message: {e}")
                  return 0.0

          df_clean['ai_probability'] = df_clean['cleaned_message'].apply(detect)
      except Exception as e:
          print(f"Error loading AI detection model: {e}")
          df_clean['ai_probability'] = 0.0

      # Combine metrics into a single AI score
      df_clean['ai_score'] = (df_clean['normalized_readability'] + 
                             df_clean['repetition_score'] + 
                             df_clean['ai_probability']) / 3

      # Scale AI score to a range of 1 to 10
      df_clean['ai_rating'] = df_clean['ai_score'].apply(lambda x: max(1, min(10, round(x * 10))))
      
      # Group by user and compute average AI rating
      result = df_clean.groupby(['userid', 'userfullname'])['ai_rating'].mean().reset_index()
      result = result.rename(columns={'ai_rating': 'avg_AI_involvedMsg_score'})
      
      return result
  except Exception as e:
      print(f"Error computing AI involvement: {e}")
      return pd.DataFrame(columns=["userid", "userfullname", "avg_AI_involvedMsg_score"])
def total_posts(df):
    return df.groupby(["userid", "userfullname"], as_index=False).size().rename(columns={"size": "total_posts"})

# ---------- Attribute registry ----------
ATTRIBUTE_FUNCS ={
        "total_posts": total_posts,
        "active_days": active_days,
        "total_replies_to_professor": total_replies_to_professor,
        "total_characters": total_characters,
        "total_words": total_words,
        "unique_interactions": unique_interactions,
        "unique_discussions": unique_discussions,
        "engagement_rate": engagement_rate,
        "deadline_exceeded_posts_Quasi_exam_I": lambda d: (d[d["subject"].fillna("").str.contains("Quasi Exam I", na=False) & (to_dt(d["created"]) > DEADLINES["Quasi Exam I"])].groupby(["userid","userfullname"], as_index=False).size().rename(columns={"size":"deadline_exceeded_posts_Quasi_exam_I"})),
        "deadline_exceeded_posts_Quasi_exam_II": lambda d: (d[d["subject"].fillna("").str.contains("Quasi Exam II", na=False) & (to_dt(d["created"]) > DEADLINES["Quasi Exam II"])].groupby(["userid","userfullname"], as_index=False).size().rename(columns={"size":"deadline_exceeded_posts_Quasi_exam_II"})),
        "deadline_exceeded_posts_Quasi_exam_III": lambda d: (d[d["subject"].fillna("").str.contains("Quasi Exam III", na=False) & (to_dt(d["created"]) > DEADLINES["Quasi Exam III"])].groupby(["userid","userfullname"], as_index=False).size().rename(columns={"size":"deadline_exceeded_posts_Quasi_exam_III"})),
        "Pattern_followed_quasi_exam_i": pattern_followed,
        "avg_reply_time": avg_reply_time,
        "modification_count": modification_count,
        "valid_response": valid_response,
        "citation_count": citation_count,
        "max_streak": max_streak,
        "topic_relevance_score": compute_topic_relevance,
        "avg_AI_involvedMsg_score": compute_avg_ai_involved,
        }

# Attribute groups
activity_attrs = [
    "total_posts", 
    "active_days", 
    "average_posts_per_day",
    "max_streak",
    "modification_count"
]

engagement_attrs = [
    "total_replies_to_professor", 
    "unique_interactions", 
    "unique_discussions",
    "engagement_rate",
    "avg_reply_time",
    "valid_response"
]

content_attrs = [
    "total_characters", 
    "total_words", 
    "citation_count",
    "topic_relevance_score",
    "avg_AI_involvedMsg_score"
]

exam_attrs = [
    "deadline_exceeded_exam1", 
    "deadline_exceeded_exam2", 
    "deadline_exceeded_exam3",
    "Pattern_followed_quasi_exam_i"
]

available_attributes = list(ATTRIBUTE_FUNCS.keys())
# --- initialize session state (only once) ---
if "selected_attributes" not in st.session_state:
    st.session_state.selected_attributes = []

# --- map attributes -> widget keys (so we can programmatically toggle widgets) ---
attr_key_map = {}
for attr in activity_attrs:
    attr_key_map[attr] = f"activity_{attr}"
for attr in engagement_attrs:
    attr_key_map[attr] = f"engagement_{attr}"
for attr in content_attrs:
    attr_key_map[attr] = f"content_{attr}"
for attr in exam_attrs:
    attr_key_map[attr] = f"exam_{attr}"

# Helper callbacks for buttons
def select_all():
    st.session_state.selected_attributes = available_attributes.copy()
    # set each widget key to True if it's a known checkbox key
    for attr in available_attributes:
        key = attr_key_map.get(attr)
        if key is not None:
            st.session_state[key] = True

def clear_all():
    st.session_state.selected_attributes = []
    # set known checkbox keys to False
    for key in attr_key_map.values():
        st.session_state[key] = False

# --- UI: info / expanders (use the same descriptions you had) ---
with st.expander("ℹ️ Attribute Descriptions", expanded=True):
    st.markdown("""
    **Activity Metrics:**
    - `total_posts`: Total number of posts made by user  
    - `active_days`: Number of days user was active  
    - `average_posts_per_day`: Average posts per active day  
    - `max_streak`: Longest consecutive days of activity  
    - `modification_count`: How many times a student modified posts  
    - `avg_modified_time_minutes`: Average minutes until first modification  

    **Engagement Metrics:**
    - `total_replies_to_professor`: Direct replies to instructor  
    - `unique_interactions`: Number of unique users interacted with  
    - `unique_discussions`: Number of unique discussions participated in  
    - `engagement_rate`: Overall engagement level  
    - `avg_reply_time`: Average time to respond to discussions  
    - `valid_response`: Whether exam answers were valid  

    **Content Analysis:**
    - `total_characters`: Total characters written  
    - `total_words`: Total words written  
    - `citation_count`: Number of external citations/links used  
    - `topic_relevance_score`: Semantic relevance to discussion topic  
    - `avg_AI_involvedMsg_score`: Average AI involvement score (1–10)  

    **Exam Performance:**
    - `deadline_exceeded_exam1/2/3`: Posts made after exam deadlines  
    - `Pattern_followed_quasi_exam_i`: Adherence to exam posting patterns  
    """)

# --- checkboxes inside expanders ---
with st.expander("📊 Activity Metrics", expanded=False):
    st.markdown("*Measures posting frequency, consistency, and engagement patterns*")
    for attr in activity_attrs:
        if attr in available_attributes:
            key = attr_key_map[attr]
            # prefer current widget state if exists; otherwise fallback to list membership
            initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
            checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
            # update selected_attributes list based on widget state change
            if checked and attr not in st.session_state.selected_attributes:
                st.session_state.selected_attributes.append(attr)
            if not checked and attr in st.session_state.selected_attributes:
                st.session_state.selected_attributes.remove(attr)

with st.expander("💬 Engagement Metrics", expanded=False):
    st.markdown("*Measures interaction quality and response patterns*")
    for attr in engagement_attrs:
        if attr in available_attributes:
            key = attr_key_map[attr]
            initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
            checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
            if checked and attr not in st.session_state.selected_attributes:
                st.session_state.selected_attributes.append(attr)
            if not checked and attr in st.session_state.selected_attributes:
                st.session_state.selected_attributes.remove(attr)

with st.expander("📝 Content Analysis", expanded=False):
    st.markdown("*Analyzes content quality, length, and relevance*")
    st.warning("⚠️ Computing these attributes may take some time. Please be patient.")
    
    for attr in content_attrs:
        if attr in available_attributes:
            key = attr_key_map[attr]
            initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
            checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
            if checked and attr not in st.session_state.selected_attributes:
                st.session_state.selected_attributes.append(attr)
            if not checked and attr in st.session_state.selected_attributes:
                st.session_state.selected_attributes.remove(attr)

with st.expander("📋 Exam Performance", expanded=False):
    st.markdown("*Tracks exam-related posting behavior and deadline compliance*")
    for attr in exam_attrs:
        if attr in available_attributes:
            key = attr_key_map[attr]
            initial = st.session_state.get(key, attr in st.session_state.selected_attributes)
            checked = st.checkbox(attr.replace("_", " ").title(), key=key, value=initial)
            if checked and attr not in st.session_state.selected_attributes:
                st.session_state.selected_attributes.append(attr)
            if not checked and attr in st.session_state.selected_attributes:
                st.session_state.selected_attributes.remove(attr)

st.markdown("---")

# Buttons using callbacks
col1, col2 = st.columns(2)
with col1:
    st.button("✅ Select All", on_click=select_all, use_container_width=True)
with col2:
    st.button("❌ Clear All", on_click=clear_all, use_container_width=True)

# display count
st.markdown(
    f"""
    <div style="
        background-color:#000000;
        padding:1px;
        margin:10px;
        border-radius:10px;
        text-align:center;
        font-weight:bold;
        font-size:22px;
        border:1px solid #ddd;
    ">
    📊 Selected<br>{len(st.session_state.selected_attributes)}
    </div>
    """,
    unsafe_allow_html=True
)
        
if st.button("Compute selected attributes"):
    if "df" not in locals():
        st.error("Please upload data first.")
    else:
        students = df[["userid","userfullname"]].drop_duplicates().sort_values("userfullname").reset_index(drop=True)
        oam = students.copy()
        
        # Use st.session_state.selected_attributes instead of selected_attributes
        selected_attributes = st.session_state.selected_attributes
        st.info(f"Computing {len(selected_attributes)} attribute(s) for {len(students)} students...")
        
        progress = st.progress(0)
        for i, attr in enumerate(selected_attributes):  # Fixed this line
            progress.progress(int((i/len(selected_attributes))*100))
            try:
                func = ATTRIBUTE_FUNCS[attr]
                if attr in ["total_replies_to_professor", "avg_reply_time", "engagement_rate", "topic_relevance_score"]:
                    result = func(df, df_all)
                else:
                    result = func(df)
                if result is None or result.empty:
                    oam[attr] = pd.NA
                else:
                    key_cols = ["userid", "userfullname"]
                    result_cols = [c for c in result.columns if c not in key_cols]
                    if not result_cols:
                        st.warning(f"Attribute {attr} produced no value column; skipping")
                        oam[attr] = pd.NA
                    else:
                        oam = oam.merge(result.drop(columns=["userfullname"], errors='ignore'), on="userid", how="left")
            except Exception as e:
                st.error(f"Error computing {attr}: {e}")
            oam = oam.fillna(0)
        progress.progress(100)

        # reorder and store OAM
        fixed = ["userid","userfullname"]
        attr_cols = [c for c in oam.columns if c not in fixed]
        attr_cols_sorted = sorted(attr_cols)
        oam = oam[fixed + attr_cols_sorted]
        oam = oam.sort_values("userfullname")
        oam.to_sql("student_attributes", conn, if_exists="replace", index=False)
        st.success("Attributes computed and stored into table 'student_attributes'.")
        st.subheader("Object Attribute Matrix (OAM)")
        st.dataframe(oam)

def rank_students(df_oam, selected_attrs, attr_directions):
    rdf = df_oam.copy()
    for attr in selected_attrs:
        if attr not in rdf.columns:
            continue
        direction = attr_directions.get(attr, 1)  # default higher is better
        ascending = True if direction == 0 else False
        rdf[attr] = rdf[attr].rank(method="min", ascending=ascending).astype(int)
    return rdf

def send_coco_request(matrix_data, job_name="MyTest", stair="", object_names="", attribute_names="", keep_files=False, timeout=120):
    """
    Send a request to the COCO Y0 service with some robustness for cloud deployments.
    Returns the requests.Response object (so caller can inspect status, headers and raw bytes).
    """
    url = "https://miau.my-x.hu/myx-free/coco/engine3.php"
    session = requests.Session()
    # Use a browser-like User-Agent so the remote site doesn't return a "bot" or reduced page.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://miau.my-x.hu/",
    }

    data = {
        'job': job_name,
        'matrix': matrix_data.replace("\n", "\r\n"),
        'stair': stair,
        'modell': 'Y0',
        'object': object_names,
        'attribute': attribute_names
    }
    if keep_files:
        data['fajl_megtart'] = '1'

    resp = session.post(url, data=data, headers=headers, timeout=timeout, allow_redirects=True)
    # Ensure correct encoding decoding later from bytes
    # Do not change resp.encoding here — decoding will be explicit in parse function
    return resp

def parse_coco_html(resp_or_html):
    """
    Parse the HTML response from COCO and extract all tables robustly.
    Accepts either a requests.Response or a raw HTML string.
    Returns dict of { 'table_0': DataFrame, ... }.
    """
    # Prepare html string
    try:
        if hasattr(resp_or_html, "content") and hasattr(resp_or_html, "status_code"):
            # requests.Response
            raw_bytes = resp_or_html.content
            # site uses ISO-8859-2; explicitly decode with replace to avoid crashes
            html = raw_bytes.decode('iso-8859-2', errors='replace')
        else:
            html = str(resp_or_html)
    except Exception as e:
        st.error(f"Could not decode HTML response: {e}")
        return {}

    # Try multiple BeautifulSoup parsers depending on availability
    soup = None
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            soup = BeautifulSoup(html, parser)
            break
        except Exception:
            continue
    if soup is None:
        # ultimate fallback
        soup = BeautifulSoup(html, "html.parser")

    # Find all <table> tags first
    tables = soup.find_all('table')
    table_dataframes = {}

    # If we found tables via BS, try to parse each
    if tables:
        for i, table in enumerate(tables):
            table_html = str(table)
            try:
                # For first table we may have header-less structure; use header=None as fallback
                try:
                    df_list = pd.read_html(StringIO(table_html))
                    df = df_list[0]
                except Exception as e_read:
                    # fallback: try header=None
                    df = pd.read_html(StringIO(table_html), header=None)[0]
            except Exception as e:
                # Last-resort parse: manual row/td extraction via BeautifulSoup
                try:
                    rows = []
                    for tr in table.find_all("tr"):
                        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                        if cols:
                            rows.append(cols)
                    df = pd.DataFrame(rows)
                except Exception as e_manual:
                    print(f"Could not parse table {i}: {e} / {e_manual}")
                    continue

            # Make columns unique & safe
            cols = list(df.columns)
            unique_cols = []
            for idx, col in enumerate(cols):
                base = str(col) if not pd.isna(col) else f"col_{idx}"
                # collapse whitespace and replace special chars
                clean = re.sub(r'[^A-Za-z0-9_]', '_', base.strip())
                # ensure uniqueness
                if clean in unique_cols:
                    suffix = 1
                    while f"{clean}_{suffix}" in unique_cols:
                        suffix += 1
                    clean = f"{clean}_{suffix}"
                unique_cols.append(clean)
            df.columns = unique_cols

            table_dataframes[f"table_{i}"] = df

    else:
        # No <table> tags found — try pd.read_html on full page (handles some malformed HTML)
        try:
            dfs = pd.read_html(StringIO(html))
            for i, df in enumerate(dfs):
                # clean column names similar to above
                cols = list(df.columns)
                unique_cols = []
                for idx, col in enumerate(cols):
                    base = str(col) if not pd.isna(col) else f"col_{idx}"
                    clean = re.sub(r'[^A-Za-z0-9_]', '_', base.strip())
                    if clean in unique_cols:
                        suffix = 1
                        while f"{clean}_{suffix}" in unique_cols:
                            suffix += 1
                        clean = f"{clean}_{suffix}"
                    unique_cols.append(clean)
                df.columns = unique_cols
                table_dataframes[f"table_{i}"] = df
        except Exception as e_full:
            # nothing found — return empty dict (caller will handle debug saving)
            print(f"pd.read_html on full page failed: {e_full}")

    return table_dataframes

def save_coco_debug_html(conn, html, resp=None):
    try:
        debug_df = pd.DataFrame({
            'timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")],
            'status_code': [getattr(resp, "status_code", None)],
            'url': [getattr(resp, "url", None)],
            'html_snippet': [html[:4000]]  # store first chunk to keep table size reasonable
        })
        debug_df.to_sql('coco_run_debug', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"Failed to save debug HTML to DB: {e}")

def clean_column_name(name):
    """
    Clean column names for SQL compatibility
    """
    # Remove special characters and replace spaces with underscores
    if isinstance(name, str):
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return f"column_{name}"

def invert_ranking(matrix_df):
    """
    Invert the ranking: NumOfObjects - RankedValue + 1
    """
    num_objects = len(matrix_df)
    inverted_df = matrix_df.copy()
    
    # Invert all numeric columns except the last one (Y_value)
    for col in inverted_df.columns[:-1]:  # All columns except the last one
        if inverted_df[col].dtype in [np.int64, np.float64]:
            inverted_df[col] = num_objects - inverted_df[col] + 1
    
    return inverted_df

st.subheader("Ranking")

# Add Y value selection
selected_y = 1000
st.write("Y value is 1000 on Demo version")

attr_directions = {} 
with st.expander("Ranking Directions", expanded=False):
    # Use st.session_state.selected_attributes instead of selected_attributes
    if st.session_state.selected_attributes: 
        st.markdown("**Predefined Ranking Directions:**")
        for attr in st.session_state.selected_attributes:  # Fixed this line
            # Set direction: 0 = Lower is better, 1 = Higher is better
            if attr.startswith("deadline_exceeded_posts_Quasi_exam_"):
                direction = 0  # Lower is better 
                direction_text = "🔻 Lower is better"
            else:
                direction = 1  # Higher is better 
                direction_text = "🔺 Higher is better"
            
            attr_directions[attr] = direction
            
            # Display the attribute with its predefined direction
            st.write(f"**{attr.replace('_', ' ').title()}:** {direction_text}")
    else:
        st.info("No attributes selected. Select attributes above to see their ranking directions.")


if st.button("▶ Run Ranking and Show Results", use_container_width=True):
    # Use st.session_state.selected_attributes here as well
    if not st.session_state.selected_attributes: 
        st.warning("No attributes selected for ranking.")
    else: 
        oam_db = pd.read_sql_query("SELECT * FROM student_attributes", conn)
        ranked = rank_students(oam_db, st.session_state.selected_attributes, attr_directions)  # Fixed this line
        
        # Add selected Y value as the last column
        ranked["Y_value"] = selected_y
        ranked.loc[ranked.index[-1], "Y_value"] = 100000
        
        # Store ranked results in database
        ranked.to_sql("ranked_student_results", conn, if_exists="replace", index=False)
        st.success("Ranked results stored in 'ranked_student_results' table!")
        
        st.subheader("📊 Ranked Results") 
        st.dataframe(ranked) 
        
        csv_data = ranked.to_csv(index=False).encode("utf-8") 
        st.download_button( 
            "⬇ Download Ranked CSV", 
            csv_data, 
            "ranked_results.csv", 
            "text/csv", 
            use_container_width=True 
        )

# COCO Analysis Section 
st.subheader("COCO Analysis")
try:
    ranked_db = pd.read_sql_query("SELECT * FROM ranked_student_results", conn)
    has_ranked_data = True
except:
    has_ranked_data = False

if has_ranked_data:
    # Extract just the numeric values (without userid and userfullname) for COCO
    matrix_df = ranked_db.drop(columns=["userid", "userfullname"])
    
    # Convert to string with tab separation - exactly like input2.txt format
    # Create a string with tab-separated values, one row per line
    matrix_lines = []
    for _, row in matrix_df.iterrows():
        # Convert each value to string and join with tabs
        row_str = "\t".join(str(val) for val in row)
        matrix_lines.append(row_str)
    
    # Join all lines with newline characters (no trailing newline)
    matrix_data = "\n".join(matrix_lines)
    
    # Send empty object and attribute names
    object_names = ""
    attribute_names = ""
    
try:
    ranked_db = pd.read_sql_query("SELECT * FROM ranked_student_results", conn)
    has_ranked_data = True
except Exception:
    has_ranked_data = False

if not has_ranked_data:
    st.info("Run ranking first to enable COCO analysis")
else:
    try:
        stair_value = int(ranked_db['userid'].nunique())
    except Exception:
        stair_value = int(ranked_db.shape[0])

    st.info(f"Stair value automatically set to number of student objects: {stair_value}")

if st.button("Run COCO Analysis", use_container_width=True):
    if 'matrix_data' not in locals():
        st.error("No ranked data found. Please run ranking first to enable COCO analysis.")
    else:
        st.info("Sending request to COCO service...")
        try:
            resp = send_coco_request(
                matrix_data,
                job_name="StudentRanking",
                stair=str(stair_value),
                object_names=object_names,
                attribute_names=attribute_names,
                keep_files=False
            )

            st.info(f"COCO HTTP status: {resp.status_code}")
            html_response = None
            try:
                # parse_coco_html will decode bytes using ISO-8859-2
                tables = parse_coco_html(resp)
            except Exception as e_parse_top:
                st.error(f"Parsing raised exception: {e_parse_top}")
                tables = {}

            # If no tables were found, save debug HTML and show snippet for investigation
            if not tables:
                try:
                    raw_html = resp.content.decode('iso-8859-2', errors='replace')
                except Exception:
                    raw_html = resp.text if hasattr(resp, 'text') else "<no html>"
                save_coco_debug_html(conn, raw_html, resp=resp)
                st.warning("No tables found in COCO response. A debug copy of the HTML was saved to 'coco_run_debug'.")
                # show a snippet so you can inspect in UI quickly
                st.text_area("COCO response snippet (first 4000 chars):", raw_html[:4000], height=300)
            else:
                # Store tables in DB
                for table_name, df in tables.items():
                    if not df.empty:
                        # if first row contains header names (non-numeric) move it to header
                        first_row = df.iloc[0]
                        if any(isinstance(val, str) and not val.replace('.', '').isdigit() for val in first_row):
                            df.columns = first_row
                            df = df.iloc[1:].reset_index(drop=True)

                        # clean columns
                        clean_columns = []
                        for idx, col in enumerate(df.columns):
                            if isinstance(col, str) and not col.startswith('Unnamed:'):
                                clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                                clean_columns.append(clean_col)
                            else:
                                clean_columns.append(f"column_{idx}")
                        df.columns = clean_columns

                    df.to_sql(table_name, conn, if_exists='replace', index=False)

                try:
                    y_val = selected_y
                except NameError:
                    y_val = None

                metadata_df = pd.DataFrame({
                    'run_timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")],
                    'stair_value': [stair_value],
                    'y_value': [y_val],
                    'num_tables': [len(tables)]
                })
                metadata_df.to_sql('coco_run_metadata', conn, if_exists='append', index=False)

                st.success(f"COCO analysis completed! {len(tables)} tables stored in the database.")
                st.write("Tables extracted from COCO analysis:")
                for name, df in tables.items():
                    st.write(f"  {name}: {df.shape[0]} rows x {df.shape[1]} columns")

                if 'table_4' in tables:
                    st.subheader("Table 4 Preview (Becsl_s values)")
                    st.dataframe(tables['table_4'])
                else:
                    st.warning("Table 4 not found in COCO response")

        except requests.exceptions.RequestException as e:
            st.error(f"Network error while contacting COCO service: {e}")
        except Exception as e:
            st.error(f"An error occurred during COCO analysis: {e}")
            st.error("Please check your internet connection and try again.")


# Validation Section
st.subheader("Validation")
try:
    ranked_db = pd.read_sql_query("SELECT * FROM ranked_student_results", conn)
    coco_table4 = pd.read_sql_query("SELECT * FROM table_4", conn)
    has_validation_data = True
except:
    has_validation_data = False
    st.info("Run COCO analysis first to enable validation")

if has_validation_data and st.button("Run Validation", use_container_width=True):
    st.info("Running validation...")
    
    try:
        # Step 1: Extract the numeric values for inversion
        matrix_df = ranked_db.drop(columns=["userid", "userfullname"])
        
        # Step 2: Invert the ranking
        inverted_matrix_df = invert_ranking(matrix_df)
        
        # Step 3: Convert inverted matrix to string format for COCO
        inverted_matrix_lines = []
        for _, row in inverted_matrix_df.iterrows():
            row_str = "\t".join(str(val) for val in row)
            inverted_matrix_lines.append(row_str)
        
        inverted_matrix_data = "\n".join(inverted_matrix_lines)
        
        # Step 4: Send inverted matrix to COCO
        html_response_inverted = send_coco_request(
            inverted_matrix_data, 
            job_name="StudentRankingInverted", 
            stair=str(stair_value),
            object_names=object_names,
            attribute_names=attribute_names,
            keep_files=False
        )
        
        # Step 5: Parse the inverted COCO response to get all tables
        tables_inverted = parse_coco_html(html_response_inverted)
        
        # Store all inverted tables in the database
        for table_name, df in tables_inverted.items():
            inverted_table_name = f"inverted_{table_name}"
            
            # Clean the table
            if not df.empty:
                first_row = df.iloc[0]
                if any(isinstance(val, str) and not val.replace('.', '').isdigit() for val in first_row):
                    df.columns = first_row
                    df = df.iloc[1:].reset_index(drop=True)
                
                # Clean column names
                clean_columns = []
                for idx, col in enumerate(df.columns):
                    if isinstance(col, str) and not col.startswith('Unnamed:'):
                        clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                        clean_columns.append(clean_col)
                    else:
                        clean_columns.append(f"column_{idx}")
                
                df.columns = clean_columns
            
            # Store inverted DataFrame in SQLite
            df.to_sql(inverted_table_name, conn, if_exists='replace', index=False)
        
        if 'table_4' in tables_inverted:
            df_table4_inverted = tables_inverted['table_4']
            
            # Step 6: Perform validation
            # Find the Delta_T_ny columns in both tables

            delta_col_original = None
            delta_col_inverted = None
            
            # Look for columns containing 'Delta_T_ny' or similar patterns
            for col in coco_table4.columns:
                if 'Delta_T_ny' in col or 'Delta' in col and 'ny' in col:
                    delta_col_original = col
                    break
                # Also check for common variations
                elif 'Delta' in col and ('T_ny' in col or 't_ny' in col):
                    delta_col_original = col
                    break
            
            for col in df_table4_inverted.columns:
                if 'Delta_T_ny' in col or 'Delta' in col and 'ny' in col:
                    delta_col_inverted = col
                    break
                # Also check for common variations
                elif 'Delta' in col and ('T_ny' in col or 't_ny' in col):
                    delta_col_inverted = col
                    break
            
            # If still not found, try to find by position or data pattern
            if not delta_col_original or not delta_col_inverted:
                st.warning("Delta_T_ny columns not found by name. Trying alternative detection...")
                
                # Show available columns for debugging
                st.write("Available columns in original table:", list(coco_table4.columns))
                st.write("Available columns in inverted table:", list(df_table4_inverted.columns))
                
                # Try to identify Delta columns by data pattern (they should contain numeric values)
                for col in coco_table4.columns:
                    # Check if column contains numeric data that could be delta values
                    try:
                        # Try to convert to numeric
                        pd.to_numeric(coco_table4[col], errors='raise')
                        # If successful and column name suggests it's a delta column
                        if 'delta' in col.lower() or 'diff' in col.lower() or 'error' in col.lower():
                            delta_col_original = col
                            break
                    except:
                        continue
                
                for col in df_table4_inverted.columns:
                    try:
                        pd.to_numeric(df_table4_inverted[col], errors='raise')
                        if 'delta' in col.lower() or 'diff' in col.lower() or 'error' in col.lower():
                            delta_col_inverted = col
                            break
                    except:
                        continue
            
            if delta_col_original and delta_col_inverted:
                # Convert to numeric
                original_delta = pd.to_numeric(coco_table4[delta_col_original], errors='coerce')
                inverted_delta = pd.to_numeric(df_table4_inverted[delta_col_inverted], errors='coerce')
                
                # Calculate validation: original_delta * inverted_delta <= 0 is valid
                validation_product = original_delta * inverted_delta
                is_valid = validation_product <= 0
                
                # Step 7: Add results to ranked table
                ranked_validated = ranked_db.copy()
                
                # Find Becsl_s column in original table (similar flexible search)
                becsl_col = None
                for col in coco_table4.columns:
                    if 'Becsl_s' in col or 'Becsl' in col or 'Estimate' in col or 'Estimated' in col:
                        becsl_col = col
                        break
                
                # If still not found, look for numeric columns that could be estimates
                if not becsl_col:
                    for col in coco_table4.columns:
                        try:
                            pd.to_numeric(coco_table4[col], errors='raise')
                            # If it's a numeric column and not a delta column, it might be the estimate
                            if col != delta_col_original and ('value' in col.lower() or 'est' in col.lower() or 'becsl' in col.lower()):
                                becsl_col = col
                                break
                        except:
                            continue
                
                if becsl_col:
                    ranked_validated['Becsl_s'] = pd.to_numeric(coco_table4[becsl_col], errors='coerce')
                    ranked_validated['Validation_Result'] = is_valid.map({True: 'Valid', False: 'Invalid'})
                    
                    # Store validated results
                    ranked_validated.to_sql("ranked_student_results_validated", conn, if_exists='replace', index=False)
                    
                    st.success("Validation completed! Results stored in 'ranked_student_results_validated' table.")
                    
                    # Show validation summary
                    st.subheader("Validation Summary")
                    valid_count = is_valid.sum()
                    total_count = len(is_valid)
                    st.write(f"Valid results: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                    
                    st.subheader("Validated Results")
                    st.dataframe(ranked_validated)

                    chart_df = ranked_validated[["userfullname", "Becsl_s", "Validation_Result"]].copy()
                    
                    # Build bar chart with color coding
                    bar_chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("userfullname:N", sort=None, title="Student"),
                            y=alt.Y("Becsl_s:Q", title="Estimation"),
                            color=alt.Color("Validation_Result:N",
                                            scale=alt.Scale(domain=["Valid", "Invalid"],
                                                            range=["green", "red"]))
                        )
                        .properties(
                            width=700,
                            height=400,
                            title="Student Estimation (Becsl_s)"
                        )
                    )
                    
                    st.altair_chart(bar_chart, use_container_width=True)
                    
                    # Download button for validated results
                    csv_validated = ranked_validated.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download Validated CSV",
                        csv_validated,
                        "validated_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Becsl_s column not found in Table 4. Available columns: " + str(list(coco_table4.columns)))
            else:
                st.error(f"Delta_T_ny columns not found in Table 4. Original columns: {list(coco_table4.columns)}, Inverted columns: {list(df_table4_inverted.columns)}")
    except Exception as e:
        st.error(f"An error occurred during validation: {e}")
conn.close()









