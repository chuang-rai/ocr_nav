from rai_ai_core_library.llm.gemini import GeminiQueryInterface
from google.genai import types


class GeminiPlusQueryInterface(GeminiQueryInterface):
    def query(
        self, prompt: str, system_prompt: str, image_bytes: bytes | None = None, image_format: str = "jpeg"
    ) -> str:
        """Perform a Gemini query with a text answer."""

        if image_bytes is not None:
            image_bytes = bytes(image_bytes)

            ros_format = image_format.lower()
            if "png" in ros_format:
                mime_type = "image/png"
            elif "jpeg" in ros_format or "jpg" in ros_format:
                mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg"

            prompt_content = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ]
        else:
            prompt_content = prompt

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt, response_mime_type="application/json"
                ),
                contents=prompt_content,
            )
            return response
        except Exception as e:
            print(f"Error occurred while querying Gemini: {e}")
            raise
