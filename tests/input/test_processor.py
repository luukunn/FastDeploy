"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from fastdeploy.input.processor import Processor, _SAMPLING_EPS


# ===========================================================================
# Helpers
# ===========================================================================

def _make_processor(**overrides):
    """Create a Processor instance with __init__ bypassed for unit testing."""
    with patch.object(Processor, "__init__", return_value=None):
        proc = Processor.__new__(Processor)

    # Set sensible defaults
    proc.tokenizer = MagicMock()
    proc.tokenizer.eos_token_id = 2
    proc.tokenizer.pad_token_id = 0
    proc.tokenizer.vocab_size = 256
    proc.tokenizer.chat_template = "dummy"
    proc.eos_token_ids = [2]
    proc.eos_token_id_len = 1
    proc.pad_token_id = 0
    proc.generation_config = MagicMock()
    proc.decode_status = {}
    proc.model_status_dict = {}
    proc.tool_parser_dict = {}
    proc._tokenize_cache = OrderedDict()
    proc._tokenize_cache_capacity = 128
    proc.reasoning_parser = None
    proc.tool_parser_obj = None
    proc.mm_processor = None
    proc.tokenizer_type = "auto"

    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


# ===========================================================================
# Tests: _tokenize_text_request
# ===========================================================================

class TestTokenizeTextRequest(unittest.TestCase):
    def test_prompt_token_ids_direct(self):
        proc = _make_processor()
        request = {"prompt_token_ids": [1, 2, 3]}
        proc._tokenize_text_request(request, max_model_len=100)
        self.assertEqual(request["prompt_token_ids"], [1, 2, 3])

    def test_prompt_string(self):
        proc = _make_processor()
        proc.text2ids = Mock(return_value=np.array([10, 20, 30]))
        request = {"prompt": "hello", "prompt_token_ids": None}
        proc._tokenize_text_request(request, max_model_len=100)
        self.assertEqual(request["prompt_token_ids"], [10, 20, 30])
        self.assertEqual(request["prompt_tokens"], "hello")

    def test_prompt_list_of_ints(self):
        proc = _make_processor()
        request = {"prompt": [10, 20, 30], "prompt_token_ids": None}
        proc._tokenize_text_request(request, max_model_len=100)
        self.assertEqual(request["prompt_token_ids"], [10, 20, 30])

    def test_messages_path(self):
        proc = _make_processor()
        proc.messages2ids = Mock(return_value=[5, 6, 7])
        request = {"prompt_token_ids": None, "prompt": None, "messages": [{"role": "user", "content": "hi"}]}
        proc._tokenize_text_request(request, max_model_len=100)
        self.assertEqual(request["prompt_token_ids"], [5, 6, 7])

    def test_no_input_raises(self):
        proc = _make_processor()
        request = {"prompt_token_ids": None, "prompt": None, "messages": None}
        with self.assertRaises(ValueError):
            proc._tokenize_text_request(request, max_model_len=100)

    def test_empty_token_ids_raises(self):
        proc = _make_processor()
        request = {"prompt_token_ids": []}
        with self.assertRaises(ValueError):
            proc._tokenize_text_request(request, max_model_len=100)

    def test_completion_token_ids_appended(self):
        proc = _make_processor()
        request = {"prompt_token_ids": [1, 2], "completion_token_ids": [3, 4]}
        proc._tokenize_text_request(request, max_model_len=100)
        self.assertEqual(request["prompt_token_ids"], [1, 2, 3, 4])

    def test_messages_with_chat_template_kwargs(self):
        proc = _make_processor()
        proc.messages2ids = Mock(return_value=[5, 6])
        request = {
            "prompt_token_ids": None,
            "prompt": None,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"tools": [{"name": "foo"}]},
        }
        proc._tokenize_text_request(request, max_model_len=100)
        # chat_template_kwargs should be applied to request
        self.assertEqual(request["tools"], [{"name": "foo"}])

    def test_messages_invalid_chat_template_kwargs_raises(self):
        proc = _make_processor()
        proc.messages2ids = Mock(return_value=[5, 6])
        request = {
            "prompt_token_ids": None,
            "prompt": None,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": "invalid",
        }
        with self.assertRaises(ValueError):
            proc._tokenize_text_request(request, max_model_len=100)


# ===========================================================================
# Tests: process_messages
# ===========================================================================

class TestProcessMessages(unittest.TestCase):
    def test_extracts_images(self):
        proc = _make_processor()
        proc.tokenizer.apply_chat_template = Mock(return_value="prompt with <img>")
        request = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "http://img.png"}},
                ]}
            ]
        }
        proc.process_messages(request)
        self.assertIn("multimodal_data", request)
        self.assertEqual(len(request["multimodal_data"]["image"]), 1)
        self.assertEqual(request["prompt"], "prompt with <img>")

    def test_extracts_videos(self):
        proc = _make_processor()
        proc.tokenizer.apply_chat_template = Mock(return_value="prompt with <vid>")
        request = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": "http://vid.mp4"}},
                ]}
            ]
        }
        proc.process_messages(request)
        self.assertIn("multimodal_data", request)
        self.assertEqual(len(request["multimodal_data"]["video"]), 1)

    def test_text_only_messages(self):
        proc = _make_processor()
        proc.tokenizer.apply_chat_template = Mock(return_value="hello prompt")
        request = {
            "messages": [{"role": "user", "content": "hi"}]
        }
        proc.process_messages(request)
        self.assertNotIn("multimodal_data", request)
        self.assertEqual(request["prompt"], "hello prompt")

    def test_empty_messages_returns_early(self):
        proc = _make_processor()
        request = {"messages": None}
        proc.process_messages(request)
        self.assertNotIn("prompt", request)

    def test_chat_template_kwargs_applied(self):
        proc = _make_processor()
        proc.tokenizer.apply_chat_template = Mock(return_value="prompted")
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"tools": [{"name": "foo"}]},
        }
        proc.process_messages(request)
        self.assertEqual(request["tools"], [{"name": "foo"}])

    def test_invalid_chat_template_kwargs_raises(self):
        proc = _make_processor()
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": "invalid",
        }
        with self.assertRaises(ValueError):
            proc.process_messages(request)


# ===========================================================================
# Tests: process_request_dict
# ===========================================================================

class TestProcessRequestDict(unittest.TestCase):
    def _make_base_request(self, **overrides):
        request = {
            "request_id": "req1",
            "prompt_token_ids": [1, 2, 3],
            "prompt": None,
            "messages": None,
            "eos_token_ids": None,
            "bad_words": None,
            "bad_words_token_ids": None,
            "logits_processors_args": {},
            "max_tokens": None,
            "temperature": 1.0,
            "top_p": 0.9,
        }
        request.update(overrides)
        return request

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_text_only_flow(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request()
        result = proc.process_request_dict(request, max_model_len=100)
        self.assertEqual(result["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(result["prompt_token_ids_len"], 3)
        self.assertEqual(result["max_tokens"], 97)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_max_tokens_capped(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request(max_tokens=200)
        result = proc.process_request_dict(request, max_model_len=100)
        # max_tokens cannot exceed max_model_len - prompt_len
        self.assertEqual(result["max_tokens"], 97)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_max_tokens_default(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request()
        result = proc.process_request_dict(request, max_model_len=100)
        self.assertEqual(result["max_tokens"], 97)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_low_temperature_forces_top_k_1(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request(temperature=0.0)
        result = proc.process_request_dict(request, max_model_len=100)
        self.assertEqual(result["temperature"], 1)
        self.assertEqual(result["top_k"], 1)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_low_top_p_forces_top_k_1(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request(top_p=0.0)
        result = proc.process_request_dict(request, max_model_len=100)
        self.assertAlmostEqual(result["top_p"], _SAMPLING_EPS)
        self.assertEqual(result["top_k"], 1)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_truncation(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request(prompt_token_ids=list(range(50)))
        result = proc.process_request_dict(request, max_model_len=10)
        self.assertEqual(len(result["prompt_token_ids"]), 9)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_eos_token_ids_not_overwritten(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request(eos_token_ids=[99])
        result = proc.process_request_dict(request, max_model_len=100)
        self.assertEqual(result["eos_token_ids"], [99])

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_reasoning_max_tokens_default(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request()
        result = proc.process_request_dict(request, max_model_len=100)
        # reasoning_max_tokens defaults to 80% of max_tokens
        expected = max(int(result["max_tokens"] * 0.8), 1)
        self.assertEqual(result["reasoning_max_tokens"], expected)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_response_max_tokens_with_thinking_disabled(self, mock_stop):
        proc = _make_processor()
        request = self._make_base_request(response_max_tokens=5, enable_thinking=False)
        result = proc.process_request_dict(request, max_model_len=100)
        self.assertEqual(result["max_tokens"], 5)

    @patch("fastdeploy.input.processor.process_stop_token_ids")
    def test_multimodal_dispatch_calls_mm_process(self, mock_stop):
        """Verify that mm_processor.process(request) is called for multimodal requests."""
        proc = _make_processor()
        proc.mm_processor = MagicMock()

        def mock_process(req):
            req["prompt_token_ids"] = [1, 2, 3]
            req["multimodal_inputs"] = {}

        proc.mm_processor.process = Mock(side_effect=mock_process)

        request = self._make_base_request(
            prompt_token_ids=None,
            prompt="hello <img>",
            multimodal_data={"image": ["img_data"]},
        )
        result = proc.process_request_dict(request, max_model_len=100)
        proc.mm_processor.process.assert_called_once_with(request)


# ===========================================================================
# Tests: _apply_default_parameters
# ===========================================================================

class TestApplyDefaultParameters(unittest.TestCase):
    def test_fills_missing_params(self):
        proc = _make_processor()
        proc.generation_config = SimpleNamespace(top_p=0.9, temperature=0.8, repetition_penalty=1.1,
                                                 frequency_penalty=0.1, presence_penalty=0.2)
        request = {}
        result = proc._apply_default_parameters(request)
        self.assertEqual(result["top_p"], 0.9)
        self.assertEqual(result["temperature"], 0.8)

    def test_does_not_overwrite_existing(self):
        proc = _make_processor()
        proc.generation_config = SimpleNamespace(top_p=0.9, temperature=0.8, repetition_penalty=1.1,
                                                 frequency_penalty=0.1, presence_penalty=0.2)
        request = {"top_p": 0.5, "temperature": 0.3}
        result = proc._apply_default_parameters(request)
        self.assertEqual(result["top_p"], 0.5)
        self.assertEqual(result["temperature"], 0.3)


# ===========================================================================
# Tests: _encode_literal_text_with_cache
# ===========================================================================

class TestEncodeLiteralTextWithCache(unittest.TestCase):
    def test_basic_encode(self):
        proc = _make_processor()
        proc.tokenizer.tokenize = Mock(return_value=["tok_a", "tok_b"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[71, 72])
        result = proc._encode_literal_text_with_cache("hello")
        self.assertEqual(result, [71, 72])

    def test_cache_hit(self):
        proc = _make_processor()
        proc.tokenizer.tokenize = Mock(return_value=["tok"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[10])
        proc._encode_literal_text_with_cache("test")
        # Second call should use cache (not call tokenize again beyond first)
        proc.tokenizer.tokenize.reset_mock()
        result = proc._encode_literal_text_with_cache("test")
        self.assertEqual(result, [10])
        proc.tokenizer.tokenize.assert_not_called()


# ===========================================================================
# Tests: _get_think_token_ids
# ===========================================================================

class TestGetThinkTokenIds(unittest.TestCase):
    def test_basic(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={"<think>": 100, "</think>": 101})
        result = proc._get_think_token_ids()
        self.assertEqual(result, (100, 101))

    def test_cached(self):
        proc = _make_processor()
        proc._think_token_ids = (50, 51)
        result = proc._get_think_token_ids()
        self.assertEqual(result, (50, 51))

    def test_missing_tokens_returns_negative(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={})
        result = proc._get_think_token_ids()
        self.assertEqual(result, (-1, -1))


# ===========================================================================
# Tests: _update_thinking_prompt_state
# ===========================================================================

class TestUpdateThinkingPromptState(unittest.TestCase):
    def test_no_thinking_budget(self):
        proc = _make_processor()
        args = {}
        result = proc._update_thinking_prompt_state([1, 2, 3], args)
        self.assertEqual(result, {})

    def test_negative_thinking_budget(self):
        proc = _make_processor()
        args = {"thinking_budget": -1}
        result = proc._update_thinking_prompt_state([1, 2, 3], args)
        self.assertEqual(result, {"thinking_budget": -1})

    def test_already_checked(self):
        proc = _make_processor()
        args = {"thinking_budget": 5, "think_prompt_checked": True}
        result = proc._update_thinking_prompt_state([1, 2, 3], args)
        self.assertEqual(result, args)

    def test_with_think_start(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={"<think>": 100, "</think>": 101})
        args = {"thinking_budget": 5}
        prompt = [1, 100, 2, 3]
        result = proc._update_thinking_prompt_state(prompt, args)
        self.assertTrue(result["think_prompt_checked"])
        self.assertTrue(result["think_prompt_started"])
        self.assertFalse(result["think_prompt_ended"])
        self.assertEqual(result["think_prompt_last_token_id"], 3)

    def test_with_think_start_and_end(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={"<think>": 100, "</think>": 101})
        args = {"thinking_budget": 5}
        prompt = [1, 100, 2, 101, 3]
        result = proc._update_thinking_prompt_state(prompt, args)
        self.assertTrue(result["think_prompt_started"])
        self.assertTrue(result["think_prompt_ended"])

    def test_empty_prompt(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={"<think>": 100, "</think>": 101})
        args = {"thinking_budget": 5}
        result = proc._update_thinking_prompt_state([], args)
        self.assertEqual(result, args)

    def test_none_prompt(self):
        proc = _make_processor()
        args = {"thinking_budget": 5}
        result = proc._update_thinking_prompt_state(None, args)
        self.assertEqual(result, args)

    def test_numpy_prompt(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={"<think>": 100, "</think>": 101})
        args = {"thinking_budget": 5}
        prompt = np.array([1, 100, 2, 3], dtype=np.int64)
        result = proc._update_thinking_prompt_state(prompt, args)
        self.assertTrue(result["think_prompt_started"])

    def test_not_dict_returns_unchanged(self):
        proc = _make_processor()
        result = proc._update_thinking_prompt_state([1, 2], "not-dict")
        self.assertEqual(result, "not-dict")

    def test_without_start(self):
        proc = _make_processor()
        proc._think_token_ids = None
        proc.tokenizer.get_vocab = Mock(return_value={"<think>": 100, "</think>": 101})
        args = {"thinking_budget": 5}
        prompt = [1, 2, 3]
        result = proc._update_thinking_prompt_state(prompt, args)
        self.assertTrue(result["think_prompt_checked"])
        self.assertFalse(result["think_prompt_started"])


# ===========================================================================
# Tests: _prepare_think_stop_sentence
# ===========================================================================

class TestPrepareThinkStopSentence(unittest.TestCase):
    def test_encodes_stop_sentence(self):
        proc = _make_processor()
        proc._encode_literal_text_with_cache = Mock(return_value=[201, 202])
        args = {"thinking_budget": 10, "think_stop_sentence": "done"}
        result = proc._prepare_think_stop_sentence(args)
        self.assertEqual(result["think_stop_sentence_token_ids"], [201, 202])
        self.assertNotIn("think_stop_sentence", result)

    def test_no_stop_sentence_unchanged(self):
        proc = _make_processor()
        args = {"thinking_budget": 10}
        result = proc._prepare_think_stop_sentence(args)
        self.assertNotIn("think_stop_sentence_token_ids", result)

    def test_not_dict_returns_unchanged(self):
        proc = _make_processor()
        result = proc._prepare_think_stop_sentence("not-dict")
        self.assertEqual(result, "not-dict")


# ===========================================================================
# Tests: update_bad_words
# ===========================================================================

class TestUpdateBadWords(unittest.TestCase):
    def test_single_token_bad_word(self):
        proc = _make_processor()
        proc.tokenizer.tokenize = Mock(return_value=["bad"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[50])
        proc.tokenizer.vocab_size = 256
        result = proc.update_bad_words(["bad"], None)
        self.assertIn(50, result)

    def test_multi_token_bad_word_skipped(self):
        proc = _make_processor()
        proc.tokenizer.tokenize = Mock(return_value=["bad", "word"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[50, 51])
        proc.tokenizer.vocab_size = 256
        result = proc.update_bad_words(["bad word"], None)
        self.assertEqual(result, [])

    def test_existing_token_ids_merged(self):
        proc = _make_processor()
        proc.tokenizer.tokenize = Mock(return_value=["bad"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[50])
        proc.tokenizer.vocab_size = 256
        result = proc.update_bad_words(["bad"], [100])
        self.assertIn(100, result)
        self.assertIn(50, result)


# ===========================================================================
# Tests: update_stop_seq
# ===========================================================================

class TestUpdateStopSeq(unittest.TestCase):
    def test_basic(self):
        proc = _make_processor()
        proc.tokenizer.eos_token_id = 2
        proc.tokenizer.tokenize = Mock(return_value=["stop"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[99])
        result = proc.update_stop_seq(["stop"])
        # Returns (padded_seqs, seq_lens)
        self.assertEqual(len(result), 2)

    def test_string_input(self):
        proc = _make_processor()
        proc.tokenizer.eos_token_id = 2
        proc.tokenizer.tokenize = Mock(return_value=["s"])
        proc.tokenizer.convert_tokens_to_ids = Mock(return_value=[88])
        result = proc.update_stop_seq("stop")
        self.assertEqual(len(result), 2)


# ===========================================================================
# Tests: pad_batch_data
# ===========================================================================

class TestPadBatchData(unittest.TestCase):
    def test_basic_right_pad(self):
        proc = _make_processor()
        result = proc.pad_batch_data([[1, 2], [3]], pad_id=0)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result[1][1], 0)

    def test_left_pad(self):
        proc = _make_processor()
        result = proc.pad_batch_data([[1, 2], [3]], pad_id=0, pad_style="left")
        self.assertEqual(result[1][0], 0)
        self.assertEqual(result[1][1], 3)

    def test_return_seq_len(self):
        proc = _make_processor()
        padded, seq_len = proc.pad_batch_data([[1, 2, 3], [4, 5]], pad_id=-1, return_seq_len=True)
        np.testing.assert_array_equal(seq_len.flatten(), [3, 2])

    def test_empty_input(self):
        proc = _make_processor()
        result = proc.pad_batch_data([], pad_id=0)
        self.assertEqual(result.shape[1], 0)


# ===========================================================================
# Tests: process_response_dict (dispatch)
# ===========================================================================

class TestProcessResponseDictDispatch(unittest.TestCase):
    def test_stream_true_dispatches_to_streaming(self):
        proc = _make_processor()
        proc.process_response_dict_streaming = Mock(return_value="streaming")
        proc.model_status_dict["req1"] = {}
        result = proc.process_response_dict(
            {"outputs": {"token_ids": [1]}, "error_code": 200, "request_id": "req1", "finished": False},
            stream=True,
        )
        self.assertEqual(result, "streaming")

    def test_stream_false_dispatches_to_normal(self):
        proc = _make_processor()
        proc.process_response_dict_normal = Mock(return_value="normal")
        result = proc.process_response_dict(
            {"outputs": {"token_ids": [1]}, "error_code": 200, "request_id": "req1", "finished": False},
            stream=False,
        )
        self.assertEqual(result, "normal")

    def test_default_stream_is_true(self):
        proc = _make_processor()
        proc.process_response_dict_streaming = Mock(return_value="streaming")
        proc.model_status_dict["req1"] = {}
        result = proc.process_response_dict(
            {"outputs": {"token_ids": [1]}, "error_code": 200, "request_id": "req1", "finished": False},
        )
        self.assertEqual(result, "streaming")

    def test_error_code_returns_unchanged(self):
        proc = _make_processor()
        resp = {"outputs": {"token_ids": [1]}, "error_code": 500}
        result = proc.process_response_dict(resp)
        self.assertIs(result, resp)


# ===========================================================================
# Tests: process_response_dict_normal
# ===========================================================================

class TestProcessResponseDictNormal(unittest.TestCase):
    def _make_response(self, token_ids, finished=True, req_id="req1"):
        return {
            "request_id": req_id,
            "finished": finished,
            "outputs": {"token_ids": token_ids},
        }

    def test_basic_finished(self):
        proc = _make_processor()
        proc.decode_status["req1"] = [0, 0, [], ""]
        proc.tokenizer.decode_token = Mock(return_value=("hello", 0, 0))
        resp = self._make_response([1, 2, 3])
        result = proc.process_response_dict_normal(resp)
        self.assertIn("text", result["outputs"])
        self.assertNotIn("req1", proc.decode_status)

    def test_eos_stripped_when_finished(self):
        proc = _make_processor()
        proc.eos_token_ids = [2]
        proc.decode_status["req1"] = [0, 0, [], ""]
        proc.tokenizer.decode_token = Mock(return_value=("hi", 0, 0))
        resp = self._make_response([1, 2])
        result = proc.process_response_dict_normal(resp)
        # eos token (2) should be stripped before decoding
        proc.tokenizer.decode_token.assert_called()

    def test_eos_kept_when_include_stop_str(self):
        proc = _make_processor()
        proc.eos_token_ids = [2]
        proc.decode_status["req1"] = [0, 0, [], ""]
        proc.tokenizer.decode_token = Mock(return_value=("a", 0, 0))
        resp = self._make_response([1, 2])
        proc.process_response_dict_normal(resp, include_stop_str_in_output=True)
        # EOS should NOT be stripped


# ===========================================================================
# Tests: process_response_dict_streaming
# ===========================================================================

class TestProcessResponseDictStreaming(unittest.TestCase):
    def _make_response(self, token_ids, finished=False, req_id="req1"):
        return {
            "request_id": req_id,
            "finished": finished,
            "outputs": {"token_ids": token_ids},
        }

    def test_basic_non_finished(self):
        proc = _make_processor()
        proc.model_status_dict["req1"] = {}
        proc.decode_status["req1"] = [0, 0, [], ""]
        proc.tokenizer.decode_token = Mock(return_value=("delta", 2, 3))
        resp = self._make_response([10])
        result = proc.process_response_dict_streaming(resp)
        self.assertEqual(result["outputs"]["text"], "delta")
        self.assertFalse(result["outputs"]["skipped"])

    def test_finished_cleans_up_status(self):
        proc = _make_processor()
        proc.model_status_dict["req1"] = {}
        proc.decode_status["req1"] = [0, 0, [], ""]
        proc.tokenizer.decode_token = Mock(return_value=("done", 0, 0))
        resp = self._make_response([1], finished=True)
        proc.process_response_dict_streaming(resp)
        self.assertNotIn("req1", proc.decode_status)
        self.assertNotIn("req1", proc.model_status_dict)

    def test_finished_cleans_up_tool_parser_dict(self):
        proc = _make_processor()
        proc.model_status_dict["req1"] = {}
        proc.decode_status["req1"] = [0, 0, [], ""]
        proc.tool_parser_dict["req1"] = MagicMock()
        proc.tokenizer.decode_token = Mock(return_value=("done", 0, 0))
        resp = self._make_response([1], finished=True)
        proc.process_response_dict_streaming(resp)
        self.assertNotIn("req1", proc.tool_parser_dict)


# ===========================================================================
# Tests: process_logprob_response
# ===========================================================================

class TestProcessLogprobResponse(unittest.TestCase):
    def test_method_exists(self):
        proc = _make_processor()
        self.assertTrue(hasattr(proc, "process_logprob_response"))

    def test_decodes_token_ids(self):
        proc = _make_processor()
        proc.tokenizer.decode = Mock(return_value="hello")
        result = proc.process_logprob_response([1, 2, 3])
        self.assertEqual(result, "hello")
        proc.tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_passes_kwargs(self):
        proc = _make_processor()
        proc.tokenizer.decode = Mock(return_value="hi")
        proc.process_logprob_response([1], clean_up_tokenization_spaces=True)
        proc.tokenizer.decode.assert_called_once_with([1], clean_up_tokenization_spaces=True)


# ===========================================================================
# Tests: _apply_reasoning_parser
# ===========================================================================

class TestApplyReasoningParser(unittest.TestCase):
    def test_basic_request_id(self):
        proc = _make_processor()
        proc.reasoning_parser = MagicMock()
        proc.reasoning_parser.get_model_status = Mock(return_value="think_start")
        request = {"request_id": "req1", "prompt_token_ids": [1, 2, 3]}
        proc._apply_reasoning_parser(request)
        self.assertEqual(proc.model_status_dict["req1"], "think_start")
        self.assertTrue(request["enable_thinking"])

    def test_compound_request_id(self):
        proc = _make_processor()
        proc.reasoning_parser = MagicMock()
        proc.reasoning_parser.get_model_status = Mock(return_value="content")
        request = {"request_id": "req1_0", "prompt_token_ids": [1, 2, 3], "n": 2}
        proc._apply_reasoning_parser(request)
        self.assertEqual(proc.model_status_dict["req1_0"], "content")
        self.assertEqual(proc.model_status_dict["req1_1"], "content")
        self.assertFalse(request["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
