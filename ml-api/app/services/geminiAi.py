from google.cloud import secretmanager
from vertexai.preview.generative_models import GenerativeModel
import vertexai
import os
import logging
import tempfile

def get_secret():
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/shining-wharf-460613-f0/secrets/ml-api-service-account-key/versions/1"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
    temp_file.write(get_secret())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

vertexai.init(project="shining-wharf-460613-f0", location="us-central1")
vertex_model = GenerativeModel("gemini-2.5-flash-preview-05-20")

def analyze_with_vertex(stress_level, text, emotion):
    prompt = f"""
    Analyze this text and list the words contributing to the stress level generated,
    explain it (in max 5 points only and make sure the explanation is adjusted with the stress level),
    and give suggestions (only in paragraph & dont give useless/offensive suggestions, if there's like low/med
    stress you can advise some activities they can do, for high stress level or any text with
    suicidal thoughts suggest them to reach medical help):
    Stress Level: "{stress_level}"
    Text: "{text}"
    Emotion: "{emotion}"

    Format the result as:
    - Why you're getting this result:
    - Suggestions:
    """
    response = vertex_model.generate_content(prompt)
    return response.text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_secret():
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/shining-wharf-460613-f0/secrets/ml-api-service-account-key/versions/1"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Failed to access secret: {e}")
        raise