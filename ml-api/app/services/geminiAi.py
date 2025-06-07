from vertexai.preview.generative_models import GenerativeModel
import vertexai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Initializing Vertex AI with project shining-wharf-460613-f0")
    vertexai.init(project="shining-wharf-460613-f0", location="us-central1")
    vertex_model = GenerativeModel("gemini-2.5-flash-preview-05-20")
    logger.info("Vertex AI initialized successfully")
except Exception as e:
    logger.error("Failed to initialize Vertex AI: %s", str(e), exc_info=True)
    raise

def analyze_with_vertex(stress_level, text, emotion):
    try:
        prompt = f"""
        Analyze this text and list the words contributing to the stress level generated,
        explain it (in max 3 points only and make sure the explanation is adjusted with the stress level),
        and give suggestions (only in paragraph (not longer than 30 words) & dont give useless/offensive suggestions, if there's like low/med
        stress you can advise some activities they can do, for high stress level or any text with
        suicidal thoughts suggest them to reach medical help):
        Stress Level: "{stress_level}"
        Text: "{text}"
        Emotion: "{emotion}"

        Format the result as:
        - Why you're getting this result:
        - Suggestions:
        """
        logger.info("Sending prompt to Vertex AI")
        response = vertex_model.generate_content(prompt)
        logger.info("Received response from Vertex AI")
        return response.text
    except Exception as e:
        logger.error("Vertex AI analysis failed: %s", str(e), exc_info=True)
        raise