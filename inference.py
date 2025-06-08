import streamlit as st
import requests

API_BASE_URL = "https://ml-api-tr6c3pgfyq-as.a.run.app"

st.title("Mental Health Text Analysis")
user_input = st.text_area("How are you feeling today? Describe it as detailed as possible!", height=200)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(f"{API_BASE_URL}/predict/analyze", json={"text": user_input})
            if response.status_code == 200:
                result = response.json()

                st.subheader("Predicted Stress & Emotion")
                st.markdown(f"- **Stress Level:** {result['predicted_stress']['label']}")
                st.markdown(f"- **Emotion:** {result['predicted_emotion']['label']}")
                st.markdown(f"- **Stress Score:** {result['stress_level']['stress_level']:.2f}")

                st.subheader("Explanation & Suggestion")
                st.text(result.get("analysis", "No explanation provided."))

                st.subheader("Recommended Videos")
                for link in result.get("recommended_videos", {}).get("recommendations", []):
                    st.video(link)

            else:
                st.error("API Error: " + response.text)
