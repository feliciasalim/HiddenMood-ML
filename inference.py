import streamlit as st
import requests

API_BASE_URL = "https://ml-api-tr6c3pgfyq-as.a.run.app" 

st.title("Mental Health Text Analysis")
user_input = st.text_area("Enter your mental health text:", height=200)

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
                st.markdown(f"- **Stress Score:** {result['stress_level']['stress_level']}")

                st.subheader("Explanation & Suggestion")
                gemini_response = requests.post(f"{API_BASE_URL}/predict/analyze", json={
                    "stress_level": result['predicted_stress']['label'],
                    "text": user_input,
                    "emotion": result['predicted_emotion']['label']
                })
                if gemini_response.status_code == 200:
                    st.text(gemini_response.json().get("analysis", "No explanation provided."))

                st.subheader("Recommended Videos")
                vids = requests.post(f"{API_BASE_URL}/predict/analyze", json={"text": user_input})
                for link in vids.json().get(["recommended_videos"]["recommendations"], []):
                    st.markdown(f"[Watch Video]({link})")
            else:
                st.error("API Error: " + response.text)
