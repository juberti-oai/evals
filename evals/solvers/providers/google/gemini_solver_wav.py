import base64
import copy
import io
import os
from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Union

import google.api_core.exceptions
import google.generativeai as genai
from google.generativeai.client import get_default_generative_client
import librosa
import numpy as np

from evals.record import record_sampling
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState
from evals.utils.api_utils import create_retrying

# Load API key from environment variable
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
GEMINI_RETRY_EXCEPTIONS = (
    google.api_core.exceptions.RetryError,
    google.api_core.exceptions.TooManyRequests,
    google.api_core.exceptions.ResourceExhausted,
)

def _data_url_to_wav(url):
    if not url.startswith("data:"):
        raise ValueError("Not a data URL")
    _, raw_data = url.split(",", 1)
    return base64.b64decode(raw_data)


# TODO: Could we just use google's own types?
# e.g. google.generativeai.types.content_types.ContentType
@dataclass
class GoogleMessage:
    role: str
    parts: list[str]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_evals_message(msg: Message):
        valid_roles = {"user", "model"}
        to_google_role = {
            "system": "user",  # Google doesn't have a "system" role
            "user": "user",
            "assistant": "model",
        }
        gmsg = GoogleMessage(
            role=to_google_role.get(msg.role, msg.role),
            parts=[msg.content] if isinstance(msg.content, str) else msg.content,
        )
        assert gmsg.role in valid_roles, f"Invalid role: {gmsg.role}"
        return gmsg
    
    

class GeminiSolverWav(Solver):
    """
    A solver class that uses Google's Gemini API to generate responses.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, Any] = {},
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)

        self.model_name = model_name
        self.gen_config = genai.GenerationConfig(**generation_config)

        # We manually define the client. This is normally defined automatically when calling
        # the API, but it isn't thread-safe, so we anticipate its creation here
        self.glm_client = get_default_generative_client()

    @property
    def model(self) -> str:
        return self.model_name
    
    # Remove timestamp from the message
    # Seems like Gemini is trained on ASR data with timestamp XX:XX before the translation
    def _post_process(self, message):
        # Remove timestamps in format XX:XX (e.g., 00:00, 01:12)
        return re.sub(r'\d{2}:\d{2}', '', message).strip()

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        msgs = task_state.messages
        gmsgs = self._convert_msgs_to_google_format(msgs)
        try:
            glm_model = genai.GenerativeModel(model_name=self.model_name)
            glm_model._client = self.glm_client
            gen_content_resp = create_retrying(
                glm_model.generate_content,
                retry_exceptions=GEMINI_RETRY_EXCEPTIONS,
                **{
                    "contents": gmsgs,
                    "generation_config": self.gen_config,
                    "safety_settings": SAFETY_SETTINGS,
                },
            )
            print("Gen content resp:", gen_content_resp.text)
            if gen_content_resp.prompt_feedback.block_reason:
                # Blocked by safety filters
                solver_result = SolverResult(
                    str(gen_content_resp.prompt_feedback),
                    error=gen_content_resp.prompt_feedback,
                )
            else:
                # Get text response
                solver_result = SolverResult(
                    gen_content_resp.text,
                    error=gen_content_resp.prompt_feedback,
                )
        except (google.api_core.exceptions.GoogleAPIError,) as e:
            solver_result = SolverResult(
                e.message,
                error=e,
            )
        except ValueError as e:
            # TODO: Why does this error ever occur and how can we handle it better?
            # (See google/generativeai/types/generation_types.py for the triggers)
            known_errors = [
                "The `response.text` quick accessor",
                "The `response.parts` quick accessor",
            ]
            if any(err in str(e) for err in known_errors):
                solver_result = SolverResult(
                    str(e),
                    error=e,
                )
            else:
                raise e

        return solver_result

    @staticmethod
    def _convert_msgs_to_google_format(msgs: list[Message]) -> list[GoogleMessage]:
        """
        Gemini API requires that the message list has
        - Roles as 'user' or 'model'
        - Alternating 'user' and 'model' messages
        - Ends with a 'user' message
        """
        
        # Enforce valid roles
        gmsgs = []
        for msg in msgs:
            gmsg = GoogleMessage.from_evals_message(msg)
            gmsgs.append(gmsg)
            assert gmsg.role in {"user", "model"}, f"Invalid role: {gmsg.role}"
        
        # Enforce alternating messages
        # e.g. [user1, user2, model1, user3] -> [user12, model1, user3]
        std_msgs = []
        for msg in gmsgs:
            if len(std_msgs) > 0 and msg.role == std_msgs[-1].role:
                # handle the case where msg.parts is a string or a dict. 
                # The assumption is that std_msgs[-1].parts is a string
                if isinstance(msg.parts[0], str):
                    std_msgs[-1].parts = ["\n".join(std_msgs[-1].parts + msg.parts)]
                elif isinstance(msg.parts[0], dict):
                    std_msgs[-1].parts = [{
                        "type": "text",
                        "text": std_msgs[-1].parts[0] + "\n" + msg.parts[0]["text"]
                    }, {
                        "type": "audio_url",
                        "audio_url": msg.parts[1]["audio_url"]
                    }]

            else:
                # Proceed as normal
                std_msgs.append(msg)

        final_messages = []
        std_msgs[-1].parts[1]["mime_type"] = "audio/wav"
        wav_bytes = _data_url_to_wav(std_msgs[-1].parts[1]["audio_url"]["url"])
        std_msgs[-1].parts[1]["data"] = wav_bytes
        std_msgs[-1].parts[1].pop("audio_url")
        std_msgs[-1].parts[1].pop("type")
        final_messages = [
            std_msgs[-1].parts[0]['text'], 
            std_msgs[-1].parts[1]
        ]
        return final_messages 


    @property
    def name(self) -> str:
        return self.model

    @property
    def model_version(self) -> Union[str, dict]:
        return self.model

    def __deepcopy__(self, memo):
        """
        Deepcopy everything except for self.glm_client, which is instead shared across all copies
        """
        cls = self.__class__
        result = cls.__new__(cls)

        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "glm_client":
                setattr(result, k, copy.deepcopy(v, memo))

        result.glm_client = self.glm_client
        return result
