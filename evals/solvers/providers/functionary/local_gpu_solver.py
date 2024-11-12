import logging
import torch
import torch.multiprocessing as mp
import transformers
from typing import Any, Dict, Optional

from evals.solvers.solver import Solver, SolverResult
from evals.solvers.utils import BatchedProcessPoolExecutor
from evals.task_state import TaskState

DEFAULT_MAX_BATCH_SIZE = 32

class FunctionaryGPUSolver(Solver):
    """
    A solver class for locally running the Functionary model on multiple GPUs using a ProcessPoolExecutor.
    The model is employed using a Hugging Face Transformers pipeline.

    The way that it works is that:
    1. Initialization:
        a. We create `num_gpus` processes, each with a copy of the model.
        b. Process 0 is run first to download the model and avoid race conditions.
        c. The other processes are started after the model is downloaded.

    2. Processing:
        a. We batch process requests using BatchedProcessPoolExecutor
        b. Each batch is processed using the model's generate() function
        c. Results are returned to the appropriate threads

    Parameters:
    - model: str - The model name/path to use for the pipeline
    - num_gpus: int - The number of GPUs to use for parallel processing (default: all available GPUs)
    - max_batch_size: int - The maximum batch size to use for inference (default: 32)
    - extra_options: Dict[str, Any] - Extra options to pass to the pipeline
        - max_new_tokens: int - The maximum number of new tokens to generate (default: 256)
        - temperature: float - The temperature for sampling (default: 1.0)
        - repetition_penalty: float - The repetition penalty for sampling (default: 1.0)
    """

    def __init__(
        self,
        model: str,
        num_gpus: int = torch.cuda.device_count(),
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)

        if extra_options is None:
            extra_options = {}

        # Set the start method for multiprocessing
        mp.set_start_method("spawn", force=True)

        rank_queue = mp.Queue()
        
        # Start with the primary process/GPU to download the model
        rank_queue.put(0)

        if "max_new_tokens" not in extra_options:
            extra_options["max_new_tokens"] = 256

        self.executor = BatchedProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            max_batch_size=int(max_batch_size),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model, extra_options),
            batch_worker_fn=solver_worker,
        )

    def copy(self):
        # The queue objects must not be copied
        return self

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        # Process the messages into Functionary's expected format
        formatted_prompt = format_messages(msgs)
        
        # Submit the prompt for processing
        completion_output = self.executor.submit({"prompt": formatted_prompt}).result()
        
        return SolverResult(completion_output)

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


def format_messages(messages):
    """Format messages into a single prompt."""
    formatted = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            formatted.append(content)
    return "\n".join(formatted)


def solver_initializer(
    rank_queue: mp.Queue, world_size: int, model: str, extra_options: Dict[str, Any]
):
    """Initializes the model and tokenizer on the specified GPU."""
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    global tokenizer, model_instance

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model_instance = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if rank == 0:
        # Let other initializers start after model download
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(inputs: Dict[str, Any]):
    """Process a batch of inputs using the model."""
    prompts = [item["prompt"] for item in inputs]
    
    results = []
    with torch.inference_mode():
        for prompt in prompts:
            # Apply the chat template without tools
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and generate
            model_inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt"
            ).to(model_instance.device)
            
            outputs = model_instance.generate(
                **model_inputs,
            )
            
            # Decode and clean up the response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the generated text
            response = generated_text[len(formatted_prompt):].strip()
            results.append(response)
            
    return results
