import streamlit as st  

def render_steps(active: str):
    st.markdown("""
    <style>
      .stepbar{position:sticky; top:0; z-index:100; padding:10px 0; 
               background: rgba(0,0,0,0); margin-top: 24px; backdrop-filter:saturate(1.2) blur(2px);}
      .steps{display:flex; gap:10px; align-items:center; justify-content:center; flex-wrap:wrap;}
      .step{padding:8px 14px; border-radius:999px; border:1px solid rgba(23,33,43,.14);
            background:rgba(255,255,255,.7); font-weight:600; opacity:.78;}
      .step.active{opacity:1; color:#173f4f; box-shadow:0 0 0 2px rgba(23,63,79,.16) inset;}
      .sep{opacity:.5; font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

    def pill(label): 
        cls = "step active" if label.lower()==active.lower() else "step"
        return f"<span class='{cls}'>{label}</span>"

    st.markdown(
        f"<div class='stepbar'><div class='steps'>"
        f"{pill('1 Analyze')}<span class='sep'>/</span>"
        f"{pill('2 Visualize')}<span class='sep'>/</span>"
        f"{pill('3 Interpret')}"
        f"</div></div>", unsafe_allow_html=True
    )
