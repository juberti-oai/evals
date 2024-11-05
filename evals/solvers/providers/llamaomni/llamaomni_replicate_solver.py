import base64
import replicate
import os
from typing import Any, Dict, Optional, List
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

class LlamaOmniReplicateSolver(Solver):
    """
    A solver class for running LlamaOmni model using Replicate deployment.

    Args:
        deployment_owner: str - Owner of the deployment (e.g. "lipatrick")
        deployment_name: str - Name of the deployment (e.g. "llama-omni")
        api_token: Optional[str] - Replicate API token. If not provided, will use REPLICATE_API_TOKEN env var
        extra_options: Optional[Dict[str, Any]] - Additional options for model generation
        postprocessors: list[str] - List of postprocessors to apply
        registry: Any - Registry object for the solver
    """

    def __init__(
        self,
        deployment_owner: str,
        deployment_name: str,
        api_token: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)
        
        self.deployment_owner = deployment_owner
        self.deployment_name = deployment_name
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("Replicate API token must be provided either through api_token parameter or REPLICATE_API_TOKEN environment variable")
        
        self.extra_options = extra_options or {}
        
        # Get deployment directly
        self.deployment = replicate.deployments.get(f"{deployment_owner}/{deployment_name}")

    def _process_audio_content(self, content: list) -> tuple[str, str]:
        """Process audio content from message parts."""
        audio_data = None
        prompt = None
        
        for part in content:
            if part["type"] == "audio_url":
                # Get base64 encoded audio
                audio_data = part["audio_url"]["url"]
            elif part["type"] == "text":
                prompt = part["text"]
                
        return audio_data, prompt

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        # Process the last message if it contains audio
        if not isinstance(task_state.messages[-1].content, str):
            audio_data, prompt = self._process_audio_content(task_state.messages[-1].content)
            
            if not audio_data:
                raise ValueError("No audio data found in the message")

            # Create input dictionary with all parameters
            input_data = {
                "input_audio": audio_data,
                "prompt": prompt or "",
                **self.extra_options
            }

            # Create prediction using deployment
            prediction = self.deployment.predictions.create(
                input=input_data
            )
            
            # Wait for prediction to complete
            prediction.wait()
            print("prediction:", prediction)
            
            return SolverResult(prediction.output)
        
        return SolverResult("")


        