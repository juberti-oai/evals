import copy
import os
import base64
import replicate
from typing import Any, Dict, Union
from dataclasses import dataclass

from evals.record import record_sampling
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState
from evals.utils.api_utils import create_retrying

# Load API key from environment variable
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if REPLICATE_API_TOKEN is None:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")

MODEL_VERSION = "36c9bcf70a56f40d9a27445c30c769308b18180296749f86ec9b682baf7ad351"

@dataclass
class ReplicateInput:
    input_audio: str  # base64 encoded audio data URI
    prompt: str = ""  # optional text prompt

    def to_dict(self):
        return {
            "input_audio": self.input_audio,
            "prompt": self.prompt if self.prompt else None
        }


class LlamaOmniReplicateSolver(Solver):
    """
    A solver class that uses Replicate's API to run LlamaOmni model.
    """

    def __init__(
        self,
        model_name: str = "ictnlp/llama-omni",
        model_version: str = MODEL_VERSION,
        generation_config: Dict[str, Any] = {},
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        print("args", model_name, model_version, generation_config)
        self._model_name = model_name
        self._model_version = model_version
        self.gen_config = generation_config

    @property
    def model_version(self) -> str:
        return self._model_version
    
    @model_version.setter
    def model_version(self, value: str):
        self._model_version = value

    @property
    def model_name(self) -> str:
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value

    def _process_audio_content(self, content: list) -> tuple[str, str]:
        """Process audio content from message parts."""
        print("Processing audio content:", type(content))
        audio_uri = None
        prompt = None
        
        for part in content:
            print("Processing part:", type(part))
            if isinstance(part, dict):  # Handle dict format
                if part.get("type") == "audio_url":
                    audio_uri = part["audio_url"]["url"]
                elif part.get("type") == "text":
                    prompt = part["text"]
            elif hasattr(part, "type"):  # Handle Message object format
                if part.type == "audio_url":
                    audio_uri = part.audio_url.url
                elif part.type == "text":
                    prompt = part.text
                
        return audio_uri, prompt

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        print("\nSolving task with last message:", type(task_state.messages[-1]))
        
        # Process the last message if it contains audio
        last_message = task_state.messages[-1]
        if hasattr(last_message, "content") and not isinstance(last_message.content, str):
            audio_uri, prompt = self._process_audio_content(last_message.content)
            print("Extracted audio_uri:", audio_uri is not None, "prompt:", prompt is not None)
            if audio_uri is None:
                return SolverResult("No audio content found", error="No audio content")
        else:
            return SolverResult("No audio content found", error="No audio content")

        # Prepare input for Replicate API
        replicate_input = ReplicateInput(
            input_audio=audio_uri,
            prompt=prompt
        ).to_dict()

        try:
            # Call Replicate API
            output = replicate.run(
                f"{self.model_name}:{self.model_version}",
                input=replicate_input
            )

            # Extract text response
            if isinstance(output, dict) and "text" in output:
                solver_result = SolverResult(output["text"])
            else:
                solver_result = SolverResult(str(output))

        except Exception as e:
            solver_result = SolverResult(
                str(e),
                error=e,
            )

        print("completion", solver_result.output)

        # # Record the sampling
        # record_sampling(
        #     prompt=task_state.messages,
        #     sampled=[solver_result.output],
        #     model=self.model,
        # )
        return solver_result

    @property
    def name(self) -> str:
        return f"{self.model_name}:{self.model_version}"

    @property
    def model(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        """Create a deep copy of the solver."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
            
        return result


        