import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS


# ==========================================
# 1. CREATING THE SAMPLE DATA
# ==========================================

reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality."
]

df = pd.DataFrame({
    "id": [f"R{i:02}" for i in range(1, 19)],
    "text": reviews
})


# ==========================================
# 2. SENTIMENT ANALYSIS
# ==========================================

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)


# ==========================================
# 3. ASSIGNING MAIN TOPICS
# ==========================================

topics = {
    "R01": "Insulation",
    "R02": "Size",
    "R03": "Design",
    "R04": "Lid",
    "R05": "Lid",
    "R06": "Cleaning",
    "R07": "Insulation",
    "R08": "Overall",
    "R09": "Insulation",
    "R10": "Cleaning",
    "R11": "Size",
    "R12": "Overall",
    "R13": "Lid",
    "R14": "Delivery",
    "R15": "Insulation",
    "R16": "Price",
    "R17": "Handle",
    "R18": "Overall"
}

df["topic"] = df["id"].map(topics)


# ==========================================
# 4. STREAMLIT DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Lab 8 | Sentiment Analysis",
    layout="wide"
)

st.title("Product Feedback: A Text Summary")

st.caption(
    "Practice data only. Automated sentiment labels are estimates "
    "and should be reviewed by humans."
)

st.subheader("Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Comments", len(df))
col2.metric("Positive Comments", (df["sentiment"] == "Positive").sum())
col3.metric("Negative Comments", (df["sentiment"] == "Negative").sum())


# ==========================================
# 5. SENTIMENT COUNT CHART
# ==========================================

st.subheader("Comment Counts by Sentiment")

order = ["Positive", "Neutral", "Negative"]

counts = df["sentiment"].value_counts().reindex(
    order, fill_value=0
)

st.bar_chart(counts, horizontal=True)


# ==========================================
# 6. IMPROVED WORD CLOUD
# ==========================================

st.subheader("Words Used in the Comments")

# Added "very" because it is a frequent but
# relatively uninformative descriptive word.
#
# Retained "cold" because it represents an
# important product feature: insulation.

stop_words = set(STOPWORDS) | {
    "bottle",
    "drinks",
    "drink",
    "very"
}

cloud = WordCloud(
    width=900,
    height=500,
    background_color="white",
    colormap="viridis",
    stopwords=stop_words,
    collocations=False
).generate(" ".join(df["text"]))

fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(cloud, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig, clear_figure=True)

st.caption(
    "Larger words appear more frequently after stop words are removed. "
    "The word 'cold' is retained because it highlights the product's "
    "insulation performance. Word size indicates frequency, not "
    "importance, satisfaction, or sentiment."
)

st.write(
    "**Stop-word improvement:** The word 'very' was added to the "
    "stop-word list because it frequently acts as an intensifier "
    "without identifying a specific product feature. The word 'cold' "
    "was retained because it helps readers identify insulation as "
    "a recurring topic in customer feedback."
)


# ==========================================
# 7. REVIEW ORIGINAL COMMENTS
# ==========================================

st.subheader("Review the Original Comments")

st.dataframe(
    df[["id", "text", "topic", "sentiment", "compound"]],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "The compound score is a sentiment model output, "
    "not a confidence rating."
)


# ==========================================
# 8. MANUAL SENTIMENT AUDIT
# ==========================================

st.subheader("Manual Sentiment Audit")

st.write(
    "Two comments from each sentiment category are selected "
    "for manual review. Compare the automated label with the "
    "actual meaning of each comment."
)

audit = pd.concat([
    df[df["sentiment"] == "Positive"].head(2),
    df[df["sentiment"] == "Neutral"].head(2),
    df[df["sentiment"] == "Negative"].head(2)
])

audit = audit[["id", "text", "sentiment", "compound"]].copy()

# Initial manual assessment based on the meaning
# of the selected comments.
audit["Manual Assessment"] = "Review required"
audit["Reason"] = "Compare the label with the original comment."

st.dataframe(
    audit,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Check whether each automated label is reasonable. "
    "Record any borderline or incorrect classifications "
    "in your assignment discussion."
)


# ==========================================
# 9. TESTING NEGATION
# ==========================================

st.subheader("Testing Negation")

original_comment = "Good value, but cleaning the narrow opening is difficult."

modified_comment = "The bottle is not easy to clean."

original_sentiment, original_score = label_sentiment(
    original_comment
)

modified_sentiment, modified_score = label_sentiment(
    modified_comment
)

negation_df = pd.DataFrame({
    "Comment": [
        original_comment,
        modified_comment
    ],
    "Sentiment": [
        original_sentiment,
        modified_sentiment
    ],
    "Compound Score": [
        original_score,
        modified_score
    ]
})

st.dataframe(
    negation_df,
    use_container_width=True,
    hide_index=True
)

st.write(
    "Negation can change the meaning of a sentence. "
    "However, automated sentiment analysis may not always "
    "interpret negation correctly, especially when sentence "
    "structure and context differ."
)


# ==========================================
# 10. SENTIMENT COMPARISON BY TOPIC
# ==========================================

st.subheader("Sentiment by Product Topic")

topic_counts = pd.crosstab(
    df["topic"],
    df["sentiment"]
).reindex(columns=order, fill_value=0)

st.bar_chart(topic_counts)

st.caption(
    "The chart compares sentiment counts across product topics. "
    "Some topics have very few comments, so the results should "
    "not be treated as representative comparisons."
)


# ==========================================
# 11. PRODUCT TEAM SUMMARY
# ==========================================

st.subheader("Summary for the Product Team")

st.write("""
The analysis of 18 synthetic customer comments provides an overview
of customer experiences with the water bottle. The sentiment count
chart identifies the most common sentiment category, while the word
cloud highlights frequently mentioned terms related to product
features. Insulation, lid quality, size and cleaning are among the
main topics appearing in the feedback.

However, automated sentiment analysis has limitations. VADER may
misinterpret mixed opinions, negation and context. For example, a
customer may appreciate the bottle's affordability while expressing
dissatisfaction with its cleaning difficulty. Assigning only one
sentiment label cannot fully represent both opinions. Furthermore,
the small synthetic dataset does not represent actual customer
experiences or the wider market.

The product team should investigate recurring complaints about lid
durability, leakage, insulation and cleaning. Improving these areas
could address several issues identified in the comments. For better
insights, the company should collect a larger dataset of genuine
customer reviews through surveys and online feedback. Manual
validation and regular evaluation of sentiment labels should also
be performed before using the results to support product decisions.
""")


# ==========================================
# 12. REFLECTION
# ==========================================

st.subheader("Reflection")

st.write("""
The visualisations revealed common product-related topics and
provided an overview of the distribution of customer sentiments.
The word cloud made frequently mentioned terms easier to identify,
while the sentiment chart allowed quick comparison between
categories.

However, the visuals could mislead readers if word frequency is
mistaken for importance or if automated sentiment labels are
assumed to represent the complete meaning of a comment. A single
sentiment label may also hide mixed opinions.

To improve reliability, sentiment labels should be manually
validated using a larger sample of genuine customer feedback.
Collecting reviews from different customer groups and using
additional contextual analysis would provide more representative
and meaningful results.
""")