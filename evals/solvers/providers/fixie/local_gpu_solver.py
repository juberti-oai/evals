import base64
import io
import queue
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import process as futures_process
from typing import Any, Dict

import librosa
import torch
import torch.distributed
import torch.multiprocessing as mp
import transformers

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

SAMPLE_RATE = 16000


class LocalGPUSolver(Solver):
    """
    A solver class for the locally running a model on multiple GPUs using a ProcessPoolExecutor.
    The model is employed using a Hugging Face Transformers pipeline. We assume that the pipeline
    is an UltravoxPipeline class and the inputs are text/audio messages.

    Completion function parameters:
    - model: str - The model name/path to use for the pipeline
    - num_gpus: int - The number of GPUs to use for parallel processing (default: all available GPUs)
    """

    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        self.completion_fn_options = completion_fn_options

        num_gpus = completion_fn_options.get("num_gpus", torch.cuda.device_count())
        completion_fn_options.get("max_num_tokens", 64)

        # Set the start method for the entire script
        mp.set_start_method("spawn")

        rank_queue = mp.Queue()

        # Only start the primary to let it download the model first
        rank_queue.put(0)

        if "model" not in completion_fn_options:
            raise ValueError("LocalGPUSolver requires a model to be specified.")

        model = completion_fn_options["model"]
        extra_options = completion_fn_options.get("extra_options", {})
        if "max_new_tokens" not in extra_options:
            extra_options["max_new_tokens"] = 256

        # TODO: handle num_gpus=0 differently
        self.executor = ProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model, extra_options),
        )

    def copy(self):
        # The queue objects (in self.executor) must not be copied
        return self

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        inputs = {}
        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        if not isinstance(msgs[-1]["content"], str):
            # This means the last message is an audio message
            parts = msgs[-1]["content"]
            parts_str = [x["text"] if x["type"] == "text" else "<|audio|>" for x in parts]
            # Concatenate all text parts into a single string
            msgs[-1]["content"] = "".join(parts_str)
            data_parts = [x["image_url"] for x in parts if x["type"] == "image_url"]
            assert len(data_parts) == 1
            # Extract the audio data from the last message
            audio_data = data_parts[0]["url"].split(",")[1]
            audio_stream = io.BytesIO(base64.b64decode(audio_data))

            # Read the audio data using soundfile and enforce the expected sample rate
            inputs["audio"] = librosa.load(audio_stream, sr=SAMPLE_RATE)[0]
            inputs["sampling_rate"] = SAMPLE_RATE

        inputs["turns"] = msgs

        # This is where the magic happens: we send the inputs to be processed by the model
        completion_output = self.executor.submit(solver_worker, inputs).result()

        solver_result = SolverResult(completion_output)

        return solver_result

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


def solver_initializer(
    rank_queue: mp.Queue, world_size: int, model: str, extra_options: Dict[str, Any]
):
    """Initializes the pipeline and the underlying model on the specified GPU."""
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    global pipe

    pipe = transformers.pipeline(
        model=model,
        trust_remote_code=True,
        device=device,
        torch_dtype=torch.bfloat16,
        **extra_options,
    )

    # Let the other initializers start now that the download has finished
    for i in range(1, world_size):
        rank_queue.put(i)


def solver_worker(inputs: Dict[str, Any]):
    print("got inputs of type", type(inputs))
    if isinstance(inputs, list):
        print("inputs len", len(inputs), "inputs[0] type is", type(inputs[0]))

    prepped = [pipe.preprocess(item) for item in inputs]
    batch = {}

    for key in prepped[0]:
        padding_side = "right" if key == "audio_values" else "left"
        padding_value = pipe.tokenizer.pad_token_id if key == "input_ids" else 0
        batch[key] = transformers.pipelines.base._pad(
            prepped, key, padding_value, padding_side=padding_side
        ).to(pipe.model.device)

    with torch.inference_mode():
        terminators = [pipe.tokenizer.eos_token_id]
        if "<|eot_id|>" in pipe.tokenizer.added_tokens_encoder:
            terminators.append(pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        input_len = batch["input_ids"].shape[1]

        outputs = pipe.model.generate(
            **batch,
            eos_token_id=terminators,
            **pipe._forward_params,
        )
        out_texts = [
            pipe.tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs
        ]
        return out_texts


class _BatchedExectuorManagerThread(futures_process._ExecutorManagerThread):
    MAX_BATCH_SIZE = 64

    def add_call_item_to_queue(self):
        # Fills call_queue with _WorkItems from pending_work_items.
        # This function never blocks.
        while True:
            if self.call_queue.full():
                return

            work_ids = []
            while len(work_ids) < self.MAX_BATCH_SIZE:
                try:
                    work_id = self.work_ids_queue.get(block=False)
                    work_ids.append(work_id)
                except queue.Empty:
                    break
            if not work_ids:
                return

            work_items = [self.pending_work_items[work_id] for work_id in work_ids]
            is_not_cancelled = [item.future.set_running_or_notify_cancel() for item in work_items]
            work_items = [
                item
                for item, is_not_cancelled in zip(work_items, is_not_cancelled)
                if is_not_cancelled
            ]

            if work_items:
                self.call_queue.put(
                    futures_process._CallItem(
                        work_ids,
                        work_items[0].fn,
                        ([work_item.args[0] for work_item in work_items],),
                        {},
                    ),
                    block=True,
                )

    def process_result_item(self, result_item):
        # Process the received a result_item. This can be either the PID of a
        # worker that exited gracefully or a _ResultItem

        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert self.is_shutting_down()
            p = self.processes.pop(result_item)
            p.join()
            if not self.processes:
                self.join_executor_internals()
                return
        else:
            # Received a _ResultItem so mark the future as completed.
            result_work_ids = result_item.work_id
            work_items = [self.pending_work_items.pop(work_id, None) for work_id in result_work_ids]
            # work_item can be None if another process terminated (see above)
            for i, work_item in enumerate(work_items):
                if work_item is not None:
                    if result_item.exception:
                        work_item.future.set_exception(result_item.exception)
                    else:
                        work_item.future.set_result(result_item.result[i])


futures_process._ExecutorManagerThread = _BatchedExectuorManagerThread