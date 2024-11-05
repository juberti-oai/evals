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
import whisper
from urllib.request import urlopen
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState
import numpy as np
import tempfile
import os
from .llamaomni.omni_speech.model.builder import load_pretrained_model
from .llamaomni.omni_speech.datasets.preprocess import tokenizer_speech_token
from .llamaomni.omni_speech.conversation import conv_templates

SAMPLE_RATE = 16000
DEFAULT_MAX_BATCH_SIZE = 1

class LlamaOmniLocalGPUSolver(Solver):
    """
    A solver class for running the LlamaOmni model on CPU or multiple GPUs.
    Uses BatchedProcessPoolExecutor for efficient batch processing.

    Args:
        model_name: str - The model name/path
        device: str - Device to run on ('cpu' or 'cuda')
        num_gpus: int - Number of GPUs to use (default: all available if device='cuda')
        max_batch_size: int - Maximum batch size for inference
        extra_options: Dict[str, Any] - Additional options for model generation
        postprocessors: list[str] - List of postprocessors to apply
        registry: Any - Registry object for the solver
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_gpus: int = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)

        self.model_name = model_name
        self.device = device
        self.extra_options = extra_options or {}

        # Set number of workers based on device
        if device == "cuda":
            num_gpus = num_gpus or torch.cuda.device_count()
            num_workers = max(1, num_gpus)
        else:
            num_workers = 1

        # Set up multiprocessing
        mp.set_start_method("spawn", force=True)
        rank_queue = mp.Queue()
        rank_queue.put(0)  # Start with primary worker

        self.executor = BatchedProcessPoolExecutor(
            max_workers=num_workers,
            max_batch_size=int(max_batch_size),
            initializer=solver_initializer,
            initargs=(rank_queue, num_workers, model_name, device),
            batch_worker_fn=solver_worker,
        )

    def copy(self):
        return self

    def _process_audio_content(self, content: list) -> tuple[list, list]:
        """Process audio content from message parts."""
        audios = []
        prompts = []
        
        for part in content:
            if part["type"] == "audio_url":
                # Handle base64 encoded audio
                audio_data = part["audio_url"]["url"].split(",")[1]
                audio_bytes = base64.b64decode(audio_data)
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_path = temp_file.name
                
                try:
                    # Load using whisper's load_audio function
                    audio_array = whisper.load_audio(temp_path)
                    audios.append(audio_array)
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_path)
                    
            elif part["type"] == "text":
                prompts.append(part["text"])
                
        return audios, prompts

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        inputs = {"audios": [], "prompts": []}
        
        # Process the last message if it contains audio
        if not isinstance(task_state.messages[-1].content, str):
            audios, prompts = self._process_audio_content(task_state.messages[-1].content)
            inputs["audios"].extend(audios)
            inputs["prompts"].extend(prompts) if prompts else inputs["prompts"].extend([None] * len(audios))
            
        # Submit to executor and get result
        completion_output = self.executor.submit(inputs).result()
        return SolverResult(completion_output)

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


def solver_initializer(
    rank_queue: mp.Queue,
    world_size: int,
    model_name: str,
    device: str,
):
    """Initialize the LlamaOmni model on the specified device."""
    rank = rank_queue.get()
    
    # Set device based on configuration
    if device == "cuda":
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    global model, tokenizer, context_len
    
    # Initialize model using load_pretrained_model
    tokenizer, model, context_len = load_pretrained_model(
        model_path=model_name,
        model_base=None,  # Can be made configurable through extra_options
        is_lora=False,    # Can be made configurable through extra_options
        s2s=False,         # Can be made configurable through extra_options
        device=device
    )

    # Move model to appropriate device if needed
    if device != "cuda:0":  # load_pretrained_model typically uses cuda:0 by default
        model = model.to(device)

    if rank == 0:
        # Let other initializers start after model is downloaded
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(inputs: List[Dict[str, Any]]) -> List[str]:
    """Process a batch of inputs using the LlamaOmni model."""
    print("inputs", inputs)
    batch_audios = []
    batch_prompts = []
    
    # Process each input in the batch
    for input_item in inputs:
        batch_audios.extend(input_item["audios"])
        batch_prompts.extend(input_item["prompts"])

    # Create conversation prompts like in the reference code
    processed_prompts = []
    for prompt in batch_prompts:
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        processed_prompts.append(conv.get_prompt())

    # Process input_ids with the processed prompts
    input_ids_list = []
    for prompt in processed_prompts:
        ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt')
        input_ids_list.append(ids)

    # Process audio similarly to reference code
    speech_list = []
    for audio in batch_audios:
        speech = whisper.pad_or_trim(audio.astype(np.float32))  # Ensure float32
        mel = whisper.log_mel_spectrogram(speech, n_mels=128)
        # Convert to float32 tensor and permute dimensions
        speech_tensor = torch.from_numpy(mel.numpy()).float().permute(1, 0)
        speech_list.append(speech_tensor)
        
    speech_lengths = [torch.LongTensor([audio.shape[0]]) for audio in batch_audios]
    
    input_ids = torch.stack(input_ids_list, dim=0)
    speech_tensors = torch.stack(speech_list, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
    speech_tensors = speech_tensors.to(device=device, dtype=torch.float32, non_blocking=True)
    speech_lengths = speech_lengths.to(device=device, dtype=torch.long, non_blocking=True)
    
    print("input_ids", input_ids)
    print("speech_tensors", speech_tensors)
    print("speech_lengths", speech_lengths)

    print("input ids shape", input_ids.shape)
    print("speech tensors shape", speech_tensors.shape)
    print("speech lengths shape", speech_lengths.shape)

    # Generate responses using LlamaOmni's interface
    outputs = model.generate(
        input_ids,
        speech=speech_tensors,
        speech_lengths=speech_lengths,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True,
        pad_token_id=128004,
    )
    
    # Decode responses
    decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_responses


# Reuse the BatchedProcessPoolExecutor implementation from DiVA
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
        self.process_pool_executor.shutdown()
        while not self._batch_queue.empty():
            try:
                item = self._batch_queue.get(block=False)
                if item is not None:
                    item.future.set_exception(Exception("The pool has already shut down."))
            except queue.Empty:
                pass
        self._batch_queue.put(None)

    def batch_requests(self):
        time.sleep(1)
        while True:
            self.available_workers.acquire()
            work_items: List[BatchableWorkItem] = [self._batch_queue.get()]
            
            while len(work_items) < self.max_batch_size:
                try:
                    item = self._batch_queue.get(block=False)
                    work_items.append(item)
                except queue.Empty:
                    break

            if work_items[-1] is None:
                if len(work_items) > 1:
                    logging.warn(
                        "There remained work items in the queue when shutting down. The items will be ignored."
                    )
                return

            requests = [item.request for item in work_items]
            task_futures = [item.future for item in work_items]

            try:
                result_future = self.process_pool_executor.submit(self.batch_worker_fn, requests)
            except Exception as e:
                self._handle_exception(e, task_futures)
                return

            result_future.add_done_callback(_set_results_cb(task_futures, self._handle_exception))
            result_future.add_done_callback(lambda _: self.available_workers.release())

    def _handle_exception(self, e: Exception, task_futures: List[futures.Future]):
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
