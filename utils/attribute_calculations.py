import pandas as pd
import re
from datetime import timedelta
from collections import Counter
import streamlit as st
from textstat import flesch_kincaid_grade, gunning_fog, smog_index
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Utility Functions ----------
def to_dt(series):
    return pd.to_datetime(series, errors="coerce")

def clean_text(text):
    """Clean text for analysis."""
    text = re.sub(r'[^\w\s]', '', str(text))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.lower()

# ---------- Attribute Functions (Configuration-Aware) ----------
def active_days(df):
    df2 = df.copy()
    df2["date_only"] = to_dt(df2["created"]).dt.date
    res = df2.groupby(["userid", "userfullname"], as_index=False)["date_only"].nunique()
    res = res.rename(columns={"date_only": "active_days"})
    return res

def total_replies_to_professor(df, df_all, prof_name=None):
    """Count replies to professor using configured professor names"""
    config = st.session_state.config
    
    # Use first professor from config if not specified
    if prof_name is None and config.professors:
        prof_name = config.professors[0]
    elif not prof_name:
        prof_name = "professor_1"
    
    # Use df_all (which includes professors) to find professor post IDs
    prof_ids = df_all[df_all["userfullname"] == prof_name]["id"].unique().tolist()
    
    if len(prof_ids) == 0:
        return pd.DataFrame(columns=["userid", "userfullname", "total_replies_to_professor"])
    
    # Use df (which excludes professors) to count replies to professor
    res = df[df["parent"].isin(prof_ids)].groupby(["userid", "userfullname"], as_index=False).size()
    res = res.rename(columns={"size": "total_replies_to_professor"})
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
    """Calculate engagement rate using configured professor"""
    config = st.session_state.config
    prof_name = config.professors[0] if config.professors else "professor_1"
    
    prof_total = len(df_all[df_all["userfullname"] == prof_name])
    if prof_total == 0:
        prof_total = 1
    
    replies = total_replies_to_professor(df, df_all, prof_name)
    replies["engagement_rate"] = replies["total_replies_to_professor"] / prof_total
    return replies[["userid","userfullname","engagement_rate"]]

def avg_reply_time(df, df_all):
    """Calculate average reply time using configured professor"""
    config = st.session_state.config
    prof_name = config.professors[0] if config.professors else "professor_1"
    
    prof_ids = df_all[df_all["userfullname"] == prof_name]["id"].unique().tolist()
    
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
    """Valid response function - consider making parent IDs configurable in future"""
    # Currently using hardcoded values, but you could make these configurable
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
    """Pattern following using configured parent IDs"""
    config = st.session_state.config
    parent_ids_pattern = config.parent_ids_pattern
    
    df2 = df.copy()
    dfp = df2[df2["parent"].isin(parent_ids_pattern)].copy()
    dfp["is_pattern"] = dfp["message"].fillna("").astype(str).str.match(r'^[0-9]+$')
    res = dfp.groupby(["userid","userfullname"], as_index=False)["is_pattern"].sum()
    res = res.rename(columns={"is_pattern":"Pattern_followed_quasi_exam_i"})
    return res

def deadline_exceeded_posts_generic(df, exam_name):
    """Generic deadline exceeded function using configured deadlines"""
    config = st.session_state.config
    
    if exam_name not in config.deadlines:
        return pd.DataFrame(columns=["userid", "userfullname", f"deadline_exceeded_posts_{exam_name}"])
    
    deadline = config.deadlines[exam_name]
    
    # Look for posts related to this exam after the deadline
    exam_posts = df[df["subject"].fillna("").str.contains(exam_name, na=False) & (to_dt(df["created"]) > deadline)]
    result = exam_posts.groupby(["userid", "userfullname"], as_index=False).size()
    result = result.rename(columns={"size": f"deadline_exceeded_posts_{exam_name.replace(' ', '_')}"})
    return result

def deadline_exceeded_posts_Quasi_exam_I(df):
    return deadline_exceeded_posts_generic(df, "Quasi_exam_I")

def deadline_exceeded_posts_Quasi_exam_II(df):
    return deadline_exceeded_posts_generic(df, "Quasi_exam_II")

def deadline_exceeded_posts_Quasi_exam_III(df):
    return deadline_exceeded_posts_generic(df, "Quasi_exam_III")

def compute_topic_relevance(df, df_all, professor_name=None):
    """
    Compute topic relevance scores using configured professor
    """
    config = st.session_state.config
    
    # Use configured professor if not specified
    if professor_name is None and config.professors:
        professor_name = config.professors[0]
    elif not professor_name:
        professor_name = "professor_1"
    
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

# ---------- Attribute Registry ----------
ATTRIBUTE_FUNCS = {
    "total_posts": total_posts,
    "active_days": active_days,
    "total_replies_to_professor": total_replies_to_professor,
    "total_characters": total_characters,
    "total_words": total_words,
    "unique_interactions": unique_interactions,
    "unique_discussions": unique_discussions,
    "engagement_rate": engagement_rate,
    "deadline_exceeded_posts_Quasi_exam_I": deadline_exceeded_posts_Quasi_exam_I,
    "deadline_exceeded_posts_Quasi_exam_II": deadline_exceeded_posts_Quasi_exam_II,
    "deadline_exceeded_posts_Quasi_exam_III": deadline_exceeded_posts_Quasi_exam_III,
    "Pattern_followed_quasi_exam_i": pattern_followed,
    "avg_reply_time": avg_reply_time,
    "modification_count": modification_count,
    "valid_response": valid_response,
    "citation_count": citation_count,
    "max_streak": max_streak,
    "topic_relevance_score": compute_topic_relevance,
    "avg_AI_involvedMsg_score": compute_avg_ai_involved,
    "average_posts_per_day" : average_posts_per_day
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
    "deadline_exceeded_posts_Quasi_exam_I", 
    "deadline_exceeded_posts_Quasi_exam_II", 
    "deadline_exceeded_posts_Quasi_exam_III",
    "Pattern_followed_quasi_exam_i"
]

available_attributes = list(ATTRIBUTE_FUNCS.keys())