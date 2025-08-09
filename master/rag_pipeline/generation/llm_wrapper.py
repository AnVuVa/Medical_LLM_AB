from openai import OpenAI
import backoff

import os

_base_url_ ={
    "ollama": "http://localhost:11434/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openai": "https://api.openai.com/v1",
}

_api_key_ = {
    "ollama": "ollama",
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
}

class ChatAssistant:
    def __init__(self, model_name:str, provider:str = "ollama"):
        """
        Args:
            model_name: The name of the model to use.
            provider: The provider of the model. Can be "ollama", "mistral", or "openai".
        """
        self.model_name = model_name
        self.client = OpenAI(
            base_url=_base_url_[provider],
            api_key=_api_key_[provider],
        )
    
    @backoff.on_exception(backoff.expo, Exception)
    def get_response(self, user: str, sys: str = ""):
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ]
            )
        return response.choices[0].message.content
    
    @backoff.on_exception(backoff.expo, Exception)
    def get_streaming_response(self, user: str, sys: str = ""):
        """Yields the response token by token (streaming)."""
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            stream=True
        )
        
        # Iterate over the stream of chunks
        for chunk in response_stream:
            # The actual token is in chunk.choices[0].delta.content
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token