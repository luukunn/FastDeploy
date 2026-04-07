# vLLM Token 后处理流程：从 Token ID 到字符串的完整链路

## 概述

本文档梳理了 vLLM（V1 架构）中，模型产出 Token ID 后，后处理（Post-Processing）的完整流程。重点关注 **Token ID 如何转变为字符串**，以及数据在各模块之间的流转路径。

整体链路如下：

```
Model Forward → Sampler → SamplerOutput(GPU Tensor)
    → ModelRunnerOutput(CPU list)
        → Scheduler.update_from_output → EngineCoreOutput
            → OutputProcessor.process_outputs → Detokenizer → CompletionOutput → RequestOutput
                → OpenAI API Entrypoint → JSON Response
```

---

## 1. 采样阶段：Token ID 的产生

### 1.1 Sampler

**文件**: `vllm/v1/sample/sampler.py`, `vllm/v1/worker/gpu/sample/sampler.py`

模型前向推理（forward）产生 logits 后，由 `Sampler` 对 logits 进行采样（greedy / top-k / top-p 等），生成每个请求的 token id。

```python
class Sampler(nn.Module):
    def forward(self, logits, sampling_metadata, ...) -> SamplerOutput:
        # ... 采样逻辑 ...
        sampled = sampled.to(torch.int32)
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1),  # [num_requests, 1]
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output
```

### 1.2 SamplerOutput

**文件**: `vllm/v1/outputs.py`, `vllm/v1/worker/gpu/sample/output.py`

```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor  # GPU tensor, [num_reqs, max_num_generated_tokens]
    logprobs_tensors: LogprobsTensors | None
```

此时 token ids 仍为 **GPU 上的 Tensor**，尚未转移到 CPU。

---

## 2. GPU → CPU 转移：ModelRunnerOutput

### 2.1 GPUModelRunner.sample_tokens

**文件**: `vllm/v1/worker/gpu_model_runner.py`

`GPUModelRunner` 调用 `Sampler` 后，将采样结果从 GPU 拷贝至 CPU，并封装为 `ModelRunnerOutput`：

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]                           # [num_reqs]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]           # num_reqs x num_generated_tokens（CPU list）
    logprobs: LogprobsLists | None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None]
    pooler_output: list[torch.Tensor | None] | None
    ...
```

关键变化：`sampled_token_ids` 从 GPU Tensor → **CPU 的 Python list[list[int]]**。

---

## 3. Scheduler 处理：生成 EngineCoreOutput

### 3.1 Scheduler.update_from_output

**文件**: `vllm/v1/core/sched/scheduler.py`

Scheduler 接收 `ModelRunnerOutput`，遍历每个请求的新 token ids，执行以下操作：

1. **追加到请求的 output_token_ids**：`request.append_output_token_ids(output_token_id)`
2. **Stop 检查**：检查是否命中 EOS token、stop token、max_tokens 等终止条件
3. **构造 EngineCoreOutput**

```python
def update_from_output(self, scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]:
    for req_id, new_token_ids in ...:
        new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=new_token_ids,   # list[int]
                finish_reason=finish_reason,
                stop_reason=request.stop_reason,
                ...
            )
        )
```

### 3.2 EngineCoreOutput

**文件**: `vllm/v1/engine/__init__.py`

```python
class EngineCoreOutput(msgspec.Struct):
    request_id: str
    new_token_ids: list[int]       # 本次迭代新产生的 token id 列表
    finish_reason: FinishReason | None
    stop_reason: int | str | None
    ...
```

`EngineCoreOutput` 通过 IPC（msgspec 序列化）从 EngineCore 进程发送给 Engine 前端进程。

---

## 4. OutputProcessor：Detokenize 核心环节

### 4.1 OutputProcessor.process_outputs

**文件**: `vllm/v1/engine/output_processor.py`

这是整个后处理链路最核心的函数。它接收 `EngineCoreOutput` 列表，完成：

1. **统计更新**
2. **Detokenization（Token ID → 字符串）**
3. **构造 RequestOutput**

```python
def process_outputs(self, engine_core_outputs, ...) -> OutputProcessorOutput:
    for engine_core_output in engine_core_outputs:
        req_state = self.request_states[engine_core_output.request_id]
        new_token_ids = engine_core_output.new_token_ids

        # ★ 核心步骤：调用 detokenizer.update() 进行增量解码
        stop_string = req_state.detokenizer.update(
            new_token_ids,
            finish_reason == FinishReason.STOP
        )

        # 构造 RequestOutput
        request_output = req_state.make_request_output(
            new_token_ids, pooling_output, finish_reason, stop_reason, ...
        )
```

### 4.2 RequestState._new_completion_output

同在 `output_processor.py` 中，此方法负责将 detokenizer 产出的文本组装进 `CompletionOutput`：

```python
def _new_completion_output(self, token_ids, finish_reason, stop_reason, ...) -> CompletionOutput:
    finished = finish_reason is not None
    delta = self.output_kind == RequestOutputKind.DELTA

    # ★ 从 detokenizer 获取输出文本
    text = self.detokenizer.get_next_output_text(finished, delta)
    if not delta:
        token_ids = self.detokenizer.output_token_ids

    return CompletionOutput(
        index=self.request_index,
        text=text,                    # 解码后的字符串
        token_ids=token_ids,          # 对应的 token id 列表
        logprobs=logprobs,
        cumulative_logprob=...,
        finish_reason=...,
        stop_reason=...,
    )
```

---

## 5. Detokenizer 详解：Token ID → 字符串

### 5.1 类继承结构

**文件**: `vllm/v1/engine/detokenizer.py`

```
IncrementalDetokenizer          # 基类（无 tokenizer 时跳过解码）
├── BaseIncrementalDetokenizer  # 抽象基类，实现 update() 和 stop 检查
│   ├── FastIncrementalDetokenizer   # 使用 HuggingFace tokenizers 库的 DecodeStream
│   └── SlowIncrementalDetokenizer   # 使用 Python 逐步增量解码
```

### 5.2 工厂方法：选择 Detokenizer

```python
@classmethod
def from_new_request(cls, tokenizer, request) -> IncrementalDetokenizer:
    if tokenizer is None:
        return IncrementalDetokenizer()         # 跳过解码

    if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
        return FastIncrementalDetokenizer(tokenizer, request)  # 快速路径

    return SlowIncrementalDetokenizer(tokenizer, request)      # 慢速路径
```

- **快速路径**：需要 `tokenizers >= 0.22.0` 且 tokenizer 是 `PreTrainedTokenizerFast` 类型
- **慢速路径**：兜底方案，适用于所有 tokenizer

### 5.3 BaseIncrementalDetokenizer.update()

这是增量解码的核心方法：

```python
def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
    # 1) 增量解码每个新 token
    for new_token_id in new_token_ids:
        self.token_ids.append(new_token_id)
        self.output_text += self.decode_next(new_token_id)  # ★ 子类实现

    # 2) Stop string 检查
    if self.stop:
        stop = check_stop_strings(self.output_text, ...)
        if stop is not None:
            stop_string, truncate_to = stop
            self.output_text = self.output_text[:truncate_to]  # 截断

    return stop_string  # 返回匹配的 stop string 或 None
```

### 5.4 FastIncrementalDetokenizer（快速路径）

利用 HuggingFace `tokenizers` 库的 `DecodeStream` 进行流式增量解码：

```python
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer, request):
        # 使用 DecodeStream，以 prompt token ids 作为前缀初始化
        self.stream = DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

    def decode_next(self, next_token_id: int) -> str:
        # 调用 Rust 实现的 DecodeStream.step()
        token = self.stream.step(self.tokenizer, next_token_id)
        return token or ""
```

**优势**：
- `DecodeStream.step()` 是 Rust 实现，性能远优于 Python 循环
- 内部自动处理 UTF-8 byte fallback、special token 等边界情况

### 5.5 SlowIncrementalDetokenizer（慢速路径）

**文件**: `vllm/v1/engine/detokenizer.py` + `vllm/tokenizers/detokenizer_utils.py`

```python
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer, request):
        # 将 prompt token ids 转换为 token 字符串列表
        self.tokens, self.prefix_offset, self.read_offset = 
            convert_prompt_ids_to_tokens(tokenizer, request.prompt_token_ids, ...)

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = 
            detokenize_incrementally(
                tokenizer=self.tokenizer,
                all_input_ids=self.token_ids,
                prev_tokens=self.tokens,
                prefix_offset=self.prefix_offset,
                read_offset=self.read_offset,
                skip_special_tokens=self.skip_special_tokens,
                spaces_between_special_tokens=self.spaces_between_special_tokens,
            )
        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset
        return decoded_text
```

### 5.6 detokenize_incrementally() 算法详解

**文件**: `vllm/tokenizers/detokenizer_utils.py`

这是增量解码的底层实现，核心思路是利用 **prefix_offset / read_offset** 来对抗 tokenizer 的 cleanup 算法（如根据上下文决定是否添加空格）：

```python
def detokenize_incrementally(tokenizer, all_input_ids, prev_tokens,
                              prefix_offset, read_offset, ...):
    new_token_id = all_input_ids[-1]

    # 第一次迭代：将 prompt ids 转为 tokens
    if prev_tokens is None:
        prev_tokens, prefix_offset, read_offset = 
            convert_prompt_ids_to_tokens(tokenizer, all_input_ids[:-1], ...)

    # 将新 token id 转为 token 字符串
    new_tokens = tokenizer.convert_ids_to_tokens([new_token_id], ...)
    output_tokens = prev_tokens + new_tokens

    # 利用 prefix_text 消除 tokenizer 的 cleanup 副作用
    prefix_text = tokenizer.convert_tokens_to_string(
        output_tokens[prefix_offset:read_offset]
    )
    new_text = tokenizer.convert_tokens_to_string(
        output_tokens[prefix_offset:]
    )

    # 处理不完整的 UTF-8 序列
    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        return new_tokens, "", prefix_offset, read_offset

    # 返回增量文本
    new_text = new_text[len(prefix_text):]
    return new_tokens, new_text, read_offset, len(output_tokens)
```

**关键概念**：
- **prefix_offset**: 用于生成 prefix_text 的起始位置
- **read_offset**: 上一次解码结束的位置
- **prefix_text**: 作为上下文前缀，确保新 token 的空格/拼接行为正确
- **UTF-8 byte fallback**: 如果解码结果以 `�` 结尾，说明 byte 序列尚未完成，暂不输出

### 5.7 get_next_output_text()：输出文本的获取

```python
def get_next_output_text(self, finished: bool, delta: bool) -> str:
    buffer_length = 0 if finished else self.stop_buffer_length
    if not delta:
        # 非 delta 模式：返回完整的 output_text
        return self.output_text if not buffer_length else self.output_text[:-buffer_length]

    # delta 模式（流式）：只返回自上次调用以来的新增文本
    length = len(self.output_text) - buffer_length
    last_offset = self._last_output_text_offset
    if last_offset < length:
        self._last_output_text_offset = length
        return self.output_text[last_offset:length]
    return ""
```

**stop_buffer_length** 的作用：当设置了 stop string 但不包含在输出中时，需要缓冲末尾若干字符，防止提前输出可能属于 stop string 的内容。

---

## 6. 输出数据结构

### 6.1 CompletionOutput

**文件**: `vllm/outputs.py`

```python
@dataclass
class CompletionOutput:
    index: int                               # 在 n > 1 时区分不同的 completion
    text: str                                # ★ 解码后的文本字符串
    token_ids: Sequence[int]                 # ★ 对应的 token id 序列
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    finish_reason: str | None
    stop_reason: int | str | None
```

### 6.2 RequestOutput

**文件**: `vllm/outputs.py`

```python
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    prompt_logprobs: PromptLogprobs | None
    outputs: list[CompletionOutput]          # 一个或多个 completion 结果
    finished: bool
    metrics: RequestStateStats | None
    num_cached_tokens: int | None
```

---

## 7. API 入口层：格式化为 JSON 响应

### 7.1 OpenAI Chat Completion

**文件**: `vllm/entrypoints/openai/chat_completion/serving.py`

- **非流式**：等待所有 token 生成完毕，将 `RequestOutput` 转换为 `ChatCompletionResponse`
- **流式**：通过 `chat_completion_stream_generator()` 逐步将每个 `RequestOutput` 转换为 `ChatCompletionStreamResponse`（SSE 格式）

关键代码片段（流式）：

```python
async def chat_completion_stream_generator(self, request, result_generator, ...):
    async for res in result_generator:
        for output in res.outputs:
            # output.text 即为 detokenizer 产出的增量文本
            delta_text = output.text
            # 构造 ChatCompletionStreamResponse
            chunk = ChatCompletionStreamResponse(
                choices=[ChatCompletionResponseStreamChoice(
                    delta=DeltaMessage(content=delta_text),
                    ...
                )],
                ...
            )
            yield f"data: {chunk.model_dump_json()}

"
```

### 7.2 OpenAI Completion

**文件**: `vllm/entrypoints/openai/completion/serving.py`

类似地，`request_output_to_completion_response()` 方法将 `RequestOutput` 转换为 OpenAI Completion API 格式的 JSON 响应。

---

## 8. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Model Forward Pass                          │
│                  logits: [batch_size, vocab_size]                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Sampler.forward()                                │
│              GPU Tensor → sampled_token_ids                        │
│  输出: SamplerOutput(sampled_token_ids: Tensor[num_reqs, 1])       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               GPUModelRunner.sample_tokens()                       │
│           GPU Tensor → CPU list[list[int]]                         │
│  输出: ModelRunnerOutput(sampled_token_ids: list[list[int]])        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  (IPC / 进程间通信)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│             Scheduler.update_from_output()                         │
│  • 追加 token 到请求状态                                             │
│  • 检查 stop 条件 (EOS, stop_token, max_tokens)                    │
│  输出: EngineCoreOutput(new_token_ids: list[int])                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  (msgspec 序列化 / IPC)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│           OutputProcessor.process_outputs()                        │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │          IncrementalDetokenizer.update()                   │    │
│  │                                                            │    │
│  │   for each new_token_id:                                   │    │
│  │     token_ids.append(new_token_id)                         │    │
│  │     output_text += decode_next(new_token_id)  ← 核心解码   │    │
│  │                                                            │    │
│  │   Fast Path: DecodeStream.step() (Rust)                    │    │
│  │   Slow Path: detokenize_incrementally() (Python)           │    │
│  │                                                            │    │
│  │   → check_stop_strings() → 截断 output_text               │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  detokenizer.get_next_output_text(finished, delta)                 │
│      → text (增量或完整字符串)                                       │
│                                                                     │
│  输出: CompletionOutput(text=str, token_ids=list[int])             │
│      → RequestOutput(outputs=[CompletionOutput, ...])              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│           OpenAI API Entrypoint (Serving Layer)                    │
│                                                                     │
│  • ChatCompletionResponse / ChatCompletionStreamResponse           │
│  • CompletionResponse                                              │
│  • RequestOutput.outputs[i].text → JSON response.choices[i].text   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键源文件索引

| 模块 | 文件路径 | 职责 |
|------|----------|------|
| Sampler | `vllm/v1/sample/sampler.py` | logits → sampled token ids (GPU) |
| SamplerOutput | `vllm/v1/outputs.py` | 采样结果数据结构 |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` | GPU→CPU 转换, 封装 ModelRunnerOutput |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | Stop 检查, 构造 EngineCoreOutput |
| EngineCoreOutput | `vllm/v1/engine/__init__.py` | EngineCore→Engine 前端的通信数据结构 |
| OutputProcessor | `vllm/v1/engine/output_processor.py` | 调度 detokenize, 构造 RequestOutput |
| Detokenizer | `vllm/v1/engine/detokenizer.py` | 增量解码: token id → 字符串 |
| detokenizer_utils | `vllm/tokenizers/detokenizer_utils.py` | 底层增量解码算法实现 |
| CompletionOutput | `vllm/outputs.py` | 最终输出数据结构 (text + token_ids) |
| Chat Serving | `vllm/entrypoints/openai/chat_completion/serving.py` | OpenAI Chat API 格式化 |
| Completion Serving | `vllm/entrypoints/openai/completion/serving.py` | OpenAI Completion API 格式化 |

---

## 10. 总结

vLLM V1 架构中，Token ID 到字符串的转换流程设计具有以下特点：

1. **增量解码（Incremental Detokenization）**：每产生一个新 token，只解码该 token 对应的增量文本，而非重新解码全部 token。这对流式输出至关重要。

2. **双路径策略**：
   - **Fast Path**：利用 HuggingFace tokenizers 的 Rust 实现 `DecodeStream`，性能最优
   - **Slow Path**：纯 Python 实现，作为兜底方案，兼容所有 tokenizer 类型

3. **UTF-8 安全**：通过检测 `�` 字符来识别不完整的 byte 序列，避免输出乱码

4. **Stop String 缓冲**：在流式输出时，缓冲末尾字符以确保 stop string 被正确检测和截断

5. **多进程架构**：EngineCore（采样+调度）和 Engine 前端（解码+API）运行在不同进程，通过 msgspec 序列化的 `EngineCoreOutput` 通信，解码工作在前端进程完成，不占用 GPU 计算资源
