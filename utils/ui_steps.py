import streamlit as st  

def render_steps(active: str):
    st.markdown("""
    <style>
      .stepbar{position:sticky; top:0; z-index:100; padding:10px 0; 
               background: rgba(0,0,0,0); backdrop-filter:saturate(1.2) blur(2px);}
      .steps{display:flex; gap:10px; align-items:center; justify-content:center; flex-wrap:wrap;}
      .step{padding:8px 14px; border-radius:999px; border:1px solid rgba(128,128,128,.3);
            font-weight:600; opacity:.7;}
      .step.active{opacity:1; box-shadow:0 0 0 2px rgba(124,58,237,.35) inset;}
      .sep{opacity:.5; font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

    def pill(label): 
        cls = "step active" if label.lower()==active.lower() else "step"
        return f"<span class='{cls}'>{label}</span>"

    st.markdown(
        f"<div class='stepbar'><div class='steps'>"
        f"{pill('1 Analyze')}<span class='sep'>→</span>"
        f"{pill('2 Visualize')}<span class='sep'>→</span>"
        f"{pill('3 Interpret')}"
        f"</div></div>", unsafe_allow_html=True
    )