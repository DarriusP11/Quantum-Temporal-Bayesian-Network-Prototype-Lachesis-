# -------------------------------------------------------------------
# Streamlit integration helper (optional)
# -------------------------------------------------------------------
def render_sentiment_tab(ss_get):
    """
    Render a Streamlit tab for sentiment analysis inside the QTBN app.
    Expects:
      - ss_get: function to get session_state values (from main app)
    """
    import streamlit as st

    st.subheader("Market Sentiment Analysis")

    # Pull tickers from session state
    tickers = [t.strip() for t in ss_get("tickers", "").split(",") if t.strip()]

    # Optional: extra keywords from user
    extra_kw = st.text_input("Extra keywords (comma‑separated)", "")

    try:
        mult, avg, df = compute_news_sentiment_multiplier(
            tickers=tickers,
            keywords=[k.strip() for k in extra_kw.split(",") if k.strip()],
            max_items=50,
            curve="smooth"
        )

        c1, c2 = st.columns(2)
        c1.metric("Average Sentiment", f"{avg:.3f}")
        c2.metric("Volatility Multiplier", f"{mult:.3f}")

        st.dataframe(df, use_container_width=True)
        st.caption("Apply this multiplier to VaR/CVaR or other risk metrics.")

    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
