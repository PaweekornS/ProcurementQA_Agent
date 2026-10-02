"""Base class for OpenAI type models"""
import os
from typing import Optional
from openai import OpenAI
from .base import BaseModel


class OpenAIBaseModel(BaseModel):
    """Base class for models based on OpenAI API"""

    # Automatic SDK-level retries; subclasses with their own retry loop set this to 0 so the
    # two layers do not multiply (worst case = (loop retries) x (SDK retries) x timeout).
    sdk_max_retries = 2
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        env_api_key_name: str = "OPENAI_API_KEY",
        default_base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI type model
        
        Args:
            model_name: Model name
            device: Device (not used for OpenAI type, kept for interface consistency)
            api_key: API key, if None then get from environment variable
            base_url: API base URL
            env_api_key_name: API key name in environment variable
            default_base_url: Default base_url
        """
        super().__init__(model_name, device)
        
        # Get API key from environment variable, if not available use passed key
        self.api_key = api_key or os.getenv(env_api_key_name)
        if not self.api_key:
            raise ValueError(
                f"{env_api_key_name} API key not provided, please set {env_api_key_name} environment variable or pass api_key parameter"
            )
        
        self.base_url = base_url or default_base_url
        if not self.base_url:
            raise ValueError("base_url not provided")
        
        # Bound every LLM call: the SDK default (600 s) lets one stalled request blow the
        # whole /qa latency budget.
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=self.sdk_max_retries,
        )
    
    def generate_response(self, user_input: str, max_length: int = 4096, temperature: float = 0.1) -> str:
        """
        Generate response
        
        Args:
            user_input: User input
            max_length: Maximum number of tokens
            temperature: Temperature parameter
            
        Returns:
            Model generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=max_length,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"API call error: {e}")
            return "API call failed"

