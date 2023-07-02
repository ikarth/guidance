import os
import time
import requests
import copy
import time
import asyncio
import types
import collections
import json
import re
from ._llm import LLM, LLMSession, SyncSession

import requests
import warnings

class TextGenerationWebUIRestClient:
    def __init__(self, prompt_url="http://0.0.0.0:5000/api/v1/generate", parameters=None):
        self.prompt_url = prompt_url
        self.parameters =  {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.001,
            "top_p": 0.1,
            "typical_p": 1,
            "repetition_penalty": 1.3,
            "top_k": 1,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "early_stopping": False,
            "seed": -1,
            "add_bos_token": True,
            "truncation_length": 2048,
            "ban_eos_token": False,
            "skip_special_tokens": True,
            "stopping_strings": []
        }
        
    def generate(self, prompt, parameters):
        if parameters:
            params = parameters
        else:
            params = self.parameters

        j = {
                "prompt": prompt,
                **params,
        }

        # print("Sending request: ", j)
        response = requests.post(
            self.prompt_url,
            headers={"Content-Type": "application/json"},
            json=j
        )

        response.raise_for_status()
        json_response = response.json()
        # print("Generate response: ", json_response)
        json_response["choices"] = json_response["results"]
        return json_response

class TextGenerationWebUI(LLM):
    #cache = LLM._open_cache("_text_generation_web_ui.diskcache")
    llm_name: str = "text-generation-webui"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60, token=None, endpoint=None, temperature=0.0, chat_mode="auto", organization=None):
        super().__init__()

        # fill in default endpoint value
        if endpoint is None:
            endpoint = os.environ.get("TEXT_GENERATION_WEB_UI_ENDPOINT", "http://0.0.0.0:5000/api/generate")

        cls_name = self.__class__.__name__
        if chat_mode:
            warnings.warn(f"Chat mode not supported for {cls_name}")
        if organization:
            warnings.warn(f"Organization not supported for {cls_name}")
        if caching:
            warnings.warn(f"Caching not supported for {cls_name}")
        if max_retries:
            warnings.warn(f"max_retries not supported for {cls_name}")
        if max_calls_per_min:
            warnings.warn(f"max_calls_per_min not supported for {cls_name}")

        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        self.token = token
        self.endpoint = endpoint
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self._text_generation_web_ui_client = TextGenerationWebUIRestClient() 
        self.caller = self._text_generation_web_ui_client.generate


    def session(self, asynchronous=False):
        if asynchronous:
            TextGenerationWebUISession(self)
        return SyncSession(TextGenerationWebUISession(self))


    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)


    def encode(self, string, fragment=True):
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(string)
    
    def decode(self, tokens, fragment=True):
        return self._tokenizer.decode(tokens)


# Define a deque to store the timestamps of the calls
class TextGenerationWebUISession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False, cache_seed=0, caching=None):
        """ Generate a completion of the given prompt.
        """

        assert token_healing is None or token_healing is False, "The TextGenerationWebUISession does not support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The TextGenerationWebUISession does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
        assert not stop_regex, "The TextGenerationWebUISession does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        if not stop:
            stop = []
        else:
            stop = [stop]

        # Temperature of 0.0 does not work with text generation web ui
        temperature = max(0.001, temperature)
        data = {
            "prompt": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "typical_p": 1,
                "repetition_penalty": 1.3,
                "top_k": 1,
                "min_length": 32,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "penalty_alpha": 0,
                "length_penalty": 1,
                "early_stopping": False,
                "seed": -1,
                "add_bos_token": True,
                "truncation_length": 2048,
                "ban_eos_token": False,
                "skip_special_tokens": True,
                'stopping_strings': stop
            }
        }
        out = self.llm.caller(**data)

        future = asyncio.Future()
        future.set_result(out)
        # print("Returning ", out)
        return future