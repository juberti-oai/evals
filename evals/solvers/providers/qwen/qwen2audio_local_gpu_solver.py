import base64
import dataclasses
import io
import logging
import queue
import threading
import time
import traceback
import torch
import torch.distributed
import torch.multiprocessing as mp
import librosa
import transformers
from urllib.request import urlopen
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

SAMPLE_RATE = 16000
DEFAULT_MAX_BATCH_SIZE = 32

class Qwen2AudioLocalGPUSolver(Solver):
    """
    A solver class for running the Qwen2-Audio model in parallel across multiple GPUs.
    Uses BatchedProcessPoolExecutor for efficient batch processing.

    Args:
        model_name: str - The model name/path (default: "Qwen/Qwen2-Audio-7B-Instruct")
        num_gpus: int - Number of GPUs to use (default: all available)
        max_batch_size: int - Maximum batch size for inference
        extra_options: Dict[str, Any] - Additional options for model generation
        postprocessors: list[str] - List of postprocessors to apply
        registry: Any - Registry object for the solver
    """

    def __init__(
        self,
        model_name: str,
        num_gpus: int = torch.cuda.device_count(),
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)

        self.model_name = model_name
        if extra_options is None:
            extra_options = {}

        # Set up multiprocessing
        mp.set_start_method("spawn", force=True)
        rank_queue = mp.Queue()
        rank_queue.put(0)  # Start with primary GPU

        self.executor = BatchedProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            max_batch_size=int(max_batch_size),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model_name),
            batch_worker_fn=solver_worker,
        )

    def copy(self):
        return self

    def _process_audio_content(self, content: list) -> tuple[torch.Tensor, str]:
        """Process audio content from message parts."""
        audios = []
        text_parts = []
        
        for i, part in enumerate(content):
            if part["type"] == "audio_url":
                if isinstance(part["audio_url"], dict) and "url" in part["audio_url"]:
                    audio_data = part["audio_url"]["url"].split(",")[1]
                    audio_stream = io.BytesIO(base64.b64decode(audio_data))
                else:
                    audio_stream = io.BytesIO(urlopen(part["audio_url"]).read())
                
                audio = librosa.load(audio_stream, sr=16000)[0]
                audios.append(audio)
                
                text_part = {
                    "type": "audio",
                    "audio_url": part["audio_url"]
                }
                text_parts.append(text_part)
                
            elif part["type"] == "text":
                text_part = {
                    "type": "text", 
                    "text": part["text"]
                }
                text_parts.append(text_part)
        
        return audios, list(reversed(text_parts))

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        inputs = {"conversation": [], "audios": []}
        
        # Process messages into conversation format
        conversation = [msg.to_dict() for msg in task_state.messages]
        
        # Process the last message if it contains audio
        if not isinstance(conversation[-1]["content"], str):
            audios, text_content = self._process_audio_content(conversation[-1]["content"])
            inputs["audios"].extend(audios)
            conversation[-1]["content"] = text_content
            
        inputs["conversation"] = conversation
        
        # Submit to executor and get result
        completion_output = self.executor.submit(inputs).result()
        print("completion_output: \n", completion_output, "\n\n")
        return SolverResult(completion_output)

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


def solver_initializer(
    rank_queue: mp.Queue,
    world_size: int,
    model_name: str,
):
    """Initialize the model and processor on the specified GPU."""
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    global processor, model
    
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    print("processor sampling rate: ", processor.feature_extractor.sampling_rate)
    model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16,
    )

    if device.type == "cpu":
        model = model.to(device)

    if rank == 0:
        # Let other initializers start after model is downloaded
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(inputs: List[Dict[str, Any]]) -> List[str]:
    """Process a batch of inputs using the model."""
    batch_text = []
    batch_audios = []

    # Process each input in the batch
    for input_item in inputs:
        # Apply chat template to conversation
        # text = processor.apply_chat_template(
        #     input_item["conversation"],
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        text = "<|audio_bos|><|AUDIO|><|audio_eos|>"+input_item["conversation"][-1]["content"][-1]["text"]+"<|en|>"
        batch_text.append(text)
        batch_audios.extend(input_item["audios"])

    model_inputs = processor(
        text=batch_text,
        audios=batch_audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=SAMPLE_RATE
    )    

    # Move to appropriate device
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    
    # Generate responses
    with torch.inference_mode():
        generate_ids = model.generate(
            **model_inputs,
            max_length=256, 
            do_sample=False,
            min_new_tokens=1
        )
    
    # Process only the new tokens
    generate_ids = generate_ids[:, model_inputs["input_ids"].size(1):]
    
    # Decode responses
    responses = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return responses



T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


@dataclasses.dataclass
class BatchableWorkItem:
    request: T_In
    future: futures.Future


class BatchedProcessPoolExecutor:
    def __init__(
        self,
        *args,
        batch_worker_fn: Callable[[List[T_In]], List[T_Out]],
        max_batch_size: int,
        max_workers: int = 1,
        **kwargs
    ):
        self.max_batch_size = max_batch_size
        self.batch_worker_fn = batch_worker_fn
        self._batch_queue = queue.Queue()
        self.available_workers = threading.Semaphore(value=max_workers + 1)
        self.process_pool_executor = ProcessPoolExecutor(
            *args, max_workers=max_workers, **kwargs
        )
        self._batch_thread = threading.Thread(target=self.batch_requests)
        self._batch_thread.start()

    def submit(self, request: T_In) -> futures.Future:
        item = BatchableWorkItem(request, futures.Future())
        self._batch_queue.put(item)
        return item.future

    def shutdown(self):
        # shut down the process pool executor
        self.process_pool_executor.shutdown()

        while not self._batch_queue.empty():
            try:
                item = self._batch_queue.get(block=False)
                if item is not None:
                    item.future.set_exception(Exception("The pool has already shut down."))
            except queue.Empty:
                pass

        # signal the batch thread to stop
        self._batch_queue.put(None)
        print("batched process pool executor has shut down.")

    def batch_requests(self):
        """
        Batches requests and dispatches them to the process pool executor

        Steps:
        1. greedily grab items
        2. dispatch them to ProcessPoolExecutor
        3. set the results back on the source future
        """
        # Wait a bit for the GPUs to be ready and allow the batch_queue to fill up a bit
        time.sleep(1)

        while True:
            # We don't wait to rush ahead too fast and fill up the queue with
            # batch_size=1 requests, but we also don't want any GPUs to be idle.
            self.available_workers.acquire()

            # greedily grab items
            work_items: List[BatchableWorkItem] = [self._batch_queue.get()]
            while len(work_items) < self.max_batch_size:
                try:
                    item = self._batch_queue.get(block=False)
                    work_items.append(item)
                except queue.Empty:
                    break

            # When we're done, a None item is added to the queue to signal the end of requests.
            # We will ignore any existing work_items since the process pool executor
            # is already shutting down.
            if work_items[-1] is None:
                if len(work_items) > 1:
                    logging.warn(
                        "There remained work items in the queue when shutting down. The items will be ignored."
                    )
                return

            requests = [item.request for item in work_items]
            task_futures = [item.future for item in work_items]

            # dispatch to the process pool
            try:
                result_future = self.process_pool_executor.submit(self.batch_worker_fn, requests)
            except Exception as e:
                self._handle_exception(e, task_futures)
                return

            # add callback for when the result is ready
            result_future.add_done_callback(_set_results_cb(task_futures, self._handle_exception))
            result_future.add_done_callback(lambda _: self.available_workers.release())

    def _handle_exception(self, e: Exception, task_futures: List[futures.Future]):
        """
        Handles exceptions by simply panicking:
         * prints the traceback to allow debugging
         * sets exception on all the task futures
         * shuts down the executor
        """
        print(traceback.format_exc())
        for f in task_futures:
            if not f.done():
                f.set_exception(e)
        self.shutdown()


def _set_results_cb(task_futures: List[futures.Future], handle_exception_cb: Callable):
    def cb(batch_future: futures.Future):
        try:
            for f, r in zip(task_futures, batch_future.result()):
                f.set_result(r)
        except Exception as e:
            handle_exception_cb(e, task_futures)

    return cb
