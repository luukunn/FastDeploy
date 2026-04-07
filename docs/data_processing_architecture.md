# vLLM 数据处理架构调研文档

> 基于 commit `34d317dcec3935e588d6c8ee8a7a57abb7a3e731` 分析，以 `/v1/chat/completions` 接口为主线。

## 1. 概述

vLLM 的数据处理架构负责将用户通过 OpenAI 兼容 API 发送的请求（包含文本、图片、音频等多模态内容），转换为引擎可直接消费的 token 化输入。整个流程分为 **API 层**、**Render 服务层**、**Renderer 渲染层**、**多模态处理层** 和 **引擎提交** 五大阶段。

关键设计：`OpenAIServingChat` 并不直接处理消息渲染和 tokenization，而是委托给 `OpenAIServingRender`；后者再调用 `BaseRenderer` 完成底层的模板渲染、tokenization 和多模态处理。

---

## 2. 整体架构概览

```
                           ┌─────────────────────────┐
                           │   用户 HTTP 请求            │
                           │   POST /v1/chat/completions│
                           └────────────┬────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │        HTTP Server (FastAPI)               │
                    │  ┌─────────────────────────────────────┐  │
                    │  │ OpenAI Protocol 解析                  │  │
                    │  │ (protocol.py → ChatCompletionRequest) │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ① API 层                             │  │
                    │  │ OpenAIServingChat                    │  │
                    │  │   .create_chat_completion()          │  │
                    │  │ ├── 初始化 ReasoningParser            │  │
                    │  │ ├── render_chat_request(request)     │  │
                    │  │ │     → 委托 OpenAIServingRender     │  │
                    │  │ ├── 构建 SamplingParams               │  │
                    │  │ └── → EngineCoreRequest              │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ② Render 服务层                      │  │
                    │  │ OpenAIServingRender.render_chat()    │  │
                    │  │ ├── Mistral tokenizer 特殊处理       │  │
                    │  │ ├── tool_choice / tool_parser 校验   │  │
                    │  │ ├── validate_chat_template()         │  │
                    │  │ └── preprocess_chat(request, ...)    │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ③ Renderer 渲染层                    │  │
                    │  │ BaseRenderer.render_chat_async()     │  │
                    │  │ ├── Step1: render_messages_async()   │  │
                    │  │ │   → 消息解析+chat template+mm提取  │  │
                    │  │ ├── Step2: tokenize_prompts_async()  │  │
                    │  │ │   → 文本 tokenization              │  │
                    │  │ ├── Step3: _apply_prompt_extras()    │  │
                    │  │ │   → 附加 mm_processor_kwargs       │  │
                    │  │ └── Step4: process_for_engine_async()│  │
                    │  │     → 多模态处理 + 组装引擎输入       │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ④ 多模态处理层 (可选)                │  │
                    │  │ BaseMultiModalProcessor.apply()      │  │
                    │  │ ├── _call_hf_processor()             │  │
                    │  │ │   → transformers ProcessorMixin    │  │
                    │  │ ├── 计算 mm_placeholders             │  │
                    │  │ └── → MultiModalInputs               │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ⑤ 引擎前端: AsyncLLM (EngineClient) │  │
                    │  │ AsyncLLM.generate()                  │  │
                    │  │ ├── input_processor.process_inputs() │  │
                    │  │ │   → EngineInput → EngineCoreRequest│  │
                    │  │ ├── output_processor.add_request()   │  │
                    │  │ │   → 注册 RequestState + Detokenizer│  │
                    │  │ ├── engine_core.add_request_async()  │  │
                    │  │ │   → ZMQ 发送到 EngineCore 进程     │  │
                    │  │ └── 等待 RequestOutputCollector 队列  │  │
                    │  └──────────────────┬──────────────────┘  │
                    └───────────────────┬─┘──────────────────────┘
                                        │ ZMQ IPC
                    ┌───────────────────▼───────────────────────┐
                    │        EngineCore (后台进程)                │
                    │  ┌─────────────────────────────────────┐  │
                    │  │ 接收请求 → 创建 Request 对象          │  │
                    │  │ (core.py: _handle_client_request)    │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ⑥ Scheduler 调度                     │  │
                    │  │ Scheduler.schedule()                 │  │
                    │  │ ├── 调度 RUNNING 请求 (decode)        │  │
                    │  │ ├── 恢复 PREEMPTED 请求 (swap-in)     │  │
                    │  │ ├── 调度 WAITING 新请求 (prefill)     │  │
                    │  │ │   ├── KV Cache Block 分配           │  │
                    │  │ │   ├── 前缀缓存匹配                  │  │
                    │  │ │   └── Token Budget 控制             │  │
                    │  │ └── → SchedulerOutput                │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ⑦ Executor 执行                      │  │
                    │  │ Executor.execute_model()             │  │
                    │  │ → GPUWorker.execute_model()          │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ⑧ GPUModelRunner 模型推理             │  │
                    │  │ GPUModelRunner.execute_model()       │  │
                    │  │ ├── _update_states()                 │  │
                    │  │ │   → 更新 InputBatch 持久状态        │  │
                    │  │ ├── _prepare_inputs()                │  │
                    │  │ │   → 构建 input_ids, positions,     │  │
                    │  │ │     attention_metadata 等 GPU 张量  │  │
                    │  │ ├── _model_forward()                 │  │
                    │  │ │   → model(input_ids, positions, ..)│  │
                    │  │ │   → hidden_states                  │  │
                    │  │ └── compute_logits()                 │  │
                    │  │     → hidden_states → logits          │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ⑨ 采样 & 输出处理                    │  │
                    │  │ GPUModelRunner.sample_tokens()       │  │
                    │  │ ├── _sample(logits)                  │  │
                    │  │ │   → Sampler: top-p/top-k/temp/...  │  │
                    │  │ │   → sampled_token_ids              │  │
                    │  │ ├── logprobs 计算 (可选)              │  │
                    │  │ └── → ModelRunnerOutput              │  │
                    │  └──────────────────┬──────────────────┘  │
                    │  ┌──────────────────▼──────────────────┐  │
                    │  │ ⑩ Scheduler 状态更新                  │  │
                    │  │ Scheduler.update_from_output()       │  │
                    │  │ ├── 处理 sampled_token_ids            │  │
                    │  │ ├── 判断 finish_reason               │  │
                    │  │ │   (stop/length/abort/repetition)   │  │
                    │  │ ├── 更新 Request 状态                 │  │
                    │  │ └── → EngineCoreOutputs              │  │
                    │  │     (每个请求一个 EngineCoreOutput)    │  │
                    │  └──────────────────┬──────────────────┘  │
                    └───────────────────┬─┘──────────────────────┘
                                        │ ZMQ IPC
                    ┌───────────────────▼───────────────────────┐
                    │    AsyncLLM 输出处理 (主进程)               │
                    │  ┌─────────────────────────────────────┐  │
                    │  │ ⑪ OutputProcessor (output_handler)   │  │
                    │  │ output_processor.process_outputs()   │  │
                    │  │ ├── 增量 Detokenize                  │  │
                    │  │ │   token_ids → 文本                  │  │
                    │  │ ├── Stop String 检测                  │  │
                    │  │ ├── LogProbs 处理                     │  │
                    │  │ ├── 构建 RequestOutput                │  │
                    │  │ │   (CompletionOutput + usage + ...)  │  │
                    │  │ └── queue.put(RequestOutput)          │  │
                    │  │     → 推送到 RequestOutputCollector    │  │
                    │  └──────────────────┬──────────────────┘  │
                    └───────────────────┬─┘──────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │    API 层输出后处理 (OpenAIServingChat)     │
                    │  ┌─────────────────────────────────────┐  │
                    │  │ ⑫ 流式/非流式响应构建                 │  │
                    │  │ ├── [流式] stream_generator()         │  │
                    │  │ │   ├── ReasoningParser 推理链解析    │  │
                    │  │ │   ├── ToolParser 工具调用解析        │  │
                    │  │ │   └── SSE: data: {choices: [...]}  │  │
                    │  │ ├── [非流式] full_generator()          │  │
                    │  │ │   └── 聚合全部输出 → 完整 JSON       │  │
                    │  │ └── LogProbs / Usage 构建             │  │
                    │  └──────────────────┬──────────────────┘  │
                    └───────────────────┬─┘──────────────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │   用户 HTTP 响应            │
                           │   ChatCompletionResponse   │
                           └─────────────────────────┘
```

---

## 3. 各阶段详细流程

### 3.1 API 层：OpenAIServingChat

**入口文件**：`vllm/entrypoints/openai/chat_completion/serving.py`

**核心类**：`OpenAIServingChat`（继承自 `OpenAIServing`）

#### 3.1.1 初始化阶段

`OpenAIServingChat.__init__()` 中的关键配置：

```python
class OpenAIServingChat(OpenAIServing):
    def __init__(self, engine_client, models, response_role, *,
                 openai_serving_render, ...):
        self.openai_serving_render = openai_serving_render  # ★ 持有 Render 服务的引用
        self.reasoning_parser_cls = ParserManager.get_reasoning_parser(reasoning_parser)
        self.tool_parser = ParserManager.get_tool_parser(tool_parser, ...)
        self.default_sampling_params = model_config.get_diff_sampling_param()
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.tool_call_id_type = get_tool_call_id_type(model_config)
```

| 配置项 | 来源 | 作用 |
|--------|------|------|
| `openai_serving_render` | 构造注入 | 持有 `OpenAIServingRender` 实例，**所有数据预处理委托给它** |
| `reasoning_parser_cls` | `ParserManager.get_reasoning_parser()` | 推理/思考链解析（如 QwQ、DeepSeek-R1） |
| `tool_parser` | `ParserManager.get_tool_parser()` | 工具调用解析（Function Calling） |
| `default_sampling_params` | `model_config.get_diff_sampling_param()` | 从 generation_config 获取的默认采样参数 |
| `use_harmony` | `model_config.hf_config.model_type == "gpt_oss"` | GPT-OSS 模型特殊处理标志 |
| `tool_call_id_type` | `get_tool_call_id_type(model_config)` | tool_call_id 生成策略（kimi_k2 / random） |

#### 3.1.2 `create_chat_completion()` 主流程

```python
async def create_chat_completion(self, request, raw_request):
    # 1. 初始化推理解析器
    reasoning_parser = self.reasoning_parser_cls(tokenizer, ...)

    # 2. ★ 渲染请求 — 委托给 OpenAIServingRender
    result = await self.render_chat_request(request)
    conversation, engine_prompts = result

    # 3. 构建请求元数据
    request_id = f"chatcmpl-{self._base_request_id(...)}"

    # 4. LoRA 适配器
    lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
    model_name = self.models.model_name(lora_request)

    # 5. 数据并行 rank
    data_parallel_rank = self._get_data_parallel_rank(raw_request)

    # 6. 构建采样参数
    max_tokens = get_max_tokens(max_model_len, request.max_tokens, ...)
    sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)

    # 7. 推理状态判断
    reasoning_ended = reasoning_parser.is_reasoning_end(prompt_token_ids)

    # 8. 提交引擎生成
    generator = self.engine_client.generate(
        engine_prompt, sampling_params, request_id,
        reasoning_ended=reasoning_ended, ...
    )
```

#### 3.1.3 `render_chat_request()` — 委托链的关键

```python
async def render_chat_request(self, request):
    # 1. 模型校验（LoRA 等）
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    # 2. 引擎健康检查
    if self.engine_client.errored:
        raise self.engine_client.dead_error

    # 3. ★★★ 委托给 OpenAIServingRender.render_chat()
    return await self.openai_serving_render.render_chat(request)
```

**关键点**：`OpenAIServingChat.render_chat_request()` 本身只做模型校验和引擎健康检查，真正的数据预处理全部委托给 `self.openai_serving_render.render_chat(request)`。

---

### 3.2 Render 服务层：OpenAIServingRender

**入口文件**：`vllm/entrypoints/serve/render/serving.py`

**核心类**：`OpenAIServingRender`

`OpenAIServingRender` 是数据预处理的**协调中心**，负责请求校验、工具调用配置、参数构建，然后调用底层 `BaseRenderer` 执行实际的渲染和处理。

#### 3.2.1初始化

```python
class OpenAIServingRender:
    def __init__(self, model_config, renderer, io_processor, model_registry, ...):
        self.renderer = renderer              # ★ 持有 BaseRenderer 实例
        self.model_config = model_config
        self.chat_template = chat_template
        self.tool_parser = ParserManager.get_tool_parser(tool_parser, ...)
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.default_sampling_params = model_config.get_diff_sampling_param()
```

#### 3.2.2 `render_chat()` — 核心预处理入口

这是数据预处理的**核心调度方法**：

```python
async def render_chat(self, request):
    tokenizer = self.renderer.tokenizer
    tool_parser = self.tool_parser

    # 1. Mistral tokenizer 特殊预处理
    if is_mistral_tokenizer(tokenizer):
        _mt.maybe_serialize_tool_calls(request)
        _mt.truncate_tool_call_ids(request)
        _mt.validate_request_params(request)

    # 2. tool_choice 校验
    tool_parsing_unavailable = (
        tool_parser is None
        and not is_mistral_tokenizer(tokenizer)
        and not self.use_harmony
    )
    if tool_parsing_unavailable and request.tool_choice not in (None, "none"):
        return self.create_error_response(...)

    # 3. 准备工具定义
    tool_dicts = [tool.model_dump() for tool in request.tools] if request.tools else None

    # 4. 分支处理
    if not self.use_harmony:
        # ★ 常规路径：校验 chat template → preprocess_chat()
        error_check_ret = self.validate_chat_template(...)
        if error_check_ret is not None:
            return error_check_ret

        conversation, engine_prompts = await self.preprocess_chat(
            request, request.messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=self.default_chat_template_kwargs,
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
        )
    else:
        # Harmony (GPT-OSS) 特殊路径
        conversation, engine_prompts = self._make_request_with_harmony(request, ...)

    return conversation, engine_prompts
```

#### 3.2.3 `preprocess_chat()` — 参数构建并调用 Renderer

```python
async def preprocess_chat(self, request, messages, default_template, ...):
    renderer = self.renderer
    mm_config = self.model_config.multimodal_config

    # 1. 合并模板 kwargs（注入 tools、tokenize 标志）
    default_template_kwargs = merge_kwargs(
        default_template_kwargs,
        dict(tools=tool_dicts, tokenize=is_mistral_tokenizer(renderer.tokenizer)),
    )

    # 2. 构建 TokenizeParams 和 ChatParams
    tok_params = request.build_tok_params(self.model_config)
    chat_params = request.build_chat_params(
        default_template, default_template_content_format
    ).with_defaults(
        default_template_kwargs,
        default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
        default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
    )

    # 3. ★★★ 调用 BaseRenderer.render_chat_async() — 真正的底层处理
    (conversation,), (engine_prompt,) = await renderer.render_chat_async(
        [messages],
        chat_params,
        tok_params,
        prompt_extras={
            k: v
            for k in ("mm_processor_kwargs", "cache_salt")
            if (v := getattr(request, k, None)) is not None
        },
    )

    # 4. 工具解析器调整请求（如有）
    if tool_parser is not None:
        tool_choice = getattr(request, "tool_choice", "none")
        if tool_choice != "none":
            tokenizer = renderer.get_tokenizer()
            request = tool_parser(tokenizer).adjust_request(request=request)

    return conversation, [engine_prompt]
```

---

### 3.3 Renderer 渲染层：BaseRenderer

**入口文件**：`vllm/renderers/base.py`

**核心类**：`BaseRenderer`（抽象基类），具体实现包括 `HfRenderer`、`MistralRenderer`、`Grok2Renderer`

`BaseRenderer.render_chat_async()` 是底层的**四步处理流水线**，由 `OpenAIServingRender.preprocess_chat()` 调用：

#### 3.3.1 `render_chat_async()` 四步流程

```python
async def render_chat_async(self, conversations, chat_params, tok_params=None,
                             *, prompt_extras=None):
    arrival_time = time.time()

    if tok_params is None:
        tok_params = self.default_chat_tok_params

    # Step 1: 消息渲染（并行处理多个 conversation）
    rendered = [
        self.render_messages_async(conversation, chat_params)
        for conversation in conversations
    ]
    out_conversations = []
    dict_prompts = []
    for conv, prompt in await asyncio.gather(*rendered):
        out_conversations.append(conv)
        dict_prompts.append(prompt)

    # Step 2: Tokenization（并行处理多个 prompt）
    tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

    # Step 3: 附加额外参数
    self._apply_prompt_extras(tok_prompts, prompt_extras)

    # Step 4: 引擎输入构建（并行处理，多模态处理在此触发）
    eng_prompts = await asyncio.gather(
        *(self.process_for_engine_async(p, arrival_time) for p in tok_prompts)
    )

    return out_conversations, eng_prompts
```

#### 3.3.2 Step 1 详解：消息渲染 (`render_messages`)

以 `HfRenderer` 为例（标准 HuggingFace 路径）：

```
render_messages(messages, params)
    │
    ├── parse_chat_messages(messages, model_config, ...)
    │   ├── 遍历 messages 中的 content parts
    │   ├── 识别 type: "text" / "image_url" / "input_audio" 等
    │   ├── 下载/解码图片 (URL → PIL.Image / base64 → PIL.Image)
    │   ├── 构建 mm_data: {"image": [PIL.Image, ...], "audio": [...]}
    │   └── 构建 mm_uuids: 用于缓存标识
    │
    ├── apply_chat_template(tokenizer, conversation, ...)
    │   └── Jinja2 模板渲染，生成带占位符的 prompt 文本
    │
    └── 组装 DictPrompt:
        {
            "prompt": "<|im_start|>user\n<image>\nWhat is this?\n...",
            "multi_modal_data": {"image": [PIL.Image]},
            "multi_modal_uuids": {"image": ["hash1"]},
        }
```

**关键模块**：

| 模块 | 作用 |
|------|------|
| `vllm/entrypoints/chat_utils.py` | `parse_chat_messages()` — 从 OpenAI 格式消息中提取多模态数据 |
| `vllm/renderers/hf.py` | `HfRenderer` — 标准 HuggingFace tokenizer 的渲染实现 |
| `vllm/renderers/mistral.py` | `MistralRenderer` — Mistral tokenizer 特殊处理 |
| `vllm/renderers/grok2.py` | `Grok2Renderer` — Grok2 模型特殊处理 |

**chat template 的来源优先级**：
1. 用户请求中指定（需开启 `trust_request_chat_template`）
2. 服务启动时 `--chat-template` 参数
3. `transformers.ProcessorMixin.chat_template`（多模态模型）
4. `tokenizer.chat_template`

#### 3.3.3 Step 2 详解：Tokenization

```python
async def _tokenize_singleton_prompt_async(self, prompt, params):
    if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
        # 应用 pre-tokenization（如 truncation）
        prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
        # 调用 tokenizer.encode()（异步微批处理）
        prompt = await self._tokenize_prompt_async(prompt, params)

    # 需要时反向 detokenize
    if params.needs_detokenization and "prompt" not in prompt:
        prompt = await self._detokenize_prompt_async(prompt)

    return params.apply_post_tokenization(self.tokenizer, prompt)
```

> **注意**：此阶段 `multi_modal_data` 原封不动地附着在 prompt dict 上传递，不做任何处理。

#### 3.3.4 Step 4 详解：引擎输入构建 (`process_for_engine_async`)

```
process_for_engine_async(prompt, arrival_time)
    │
    ├── _process_singleton(prompt) / _process_singleton_async(prompt)
    │   │
    │   └── _process_tokens(prompt)
    │       │
    │       ├── 检查 prompt.get("multi_modal_data")
    │       │
    │       ├── [有 mm_data] → _process_multimodal()  ← ★ 触发多模态处理
    │       │
    │       └── [无 mm_data] → token_inputs(prompt_token_ids)  ← 纯文本路径
    │
    └── engine_prompt["arrival_time"] = arrival_time
```

---

### 3.4 多模态处理层：HF Processor 集成

**核心文件**：
- `vllm/renderers/base.py` — `_process_multimodal()` 调度入口
- `vllm/multimodal/processing/processor.py` — `BaseMultiModalProcessor` 抽象处理器
- `vllm/multimodal/processing/context.py` — `InputProcessingContext` 上下文，封装 HF Processor 调用
- `vllm/model_executor/models/transformers/multimodal.py` — Transformers 通用多模态处理器

#### 3.4.1 多模态处理器的创建

在 `BaseRenderer.__init__()` 中：

```python
if config.model_config.is_multimodal_model:
    from vllm.multimodal import MULTIMODAL_REGISTRY as mm_registry

    mm_processor_cache = mm_registry.processor_cache_from_config(config)

    # 深拷贝 tokenizer 避免 Rust tokenizer 并发冲突
    mm_tokenizer = copy.deepcopy(tokenizer)

    with set_default_torch_num_threads():
        self.mm_processor = mm_registry.create_processor(
            config.model_config,
            tokenizer=mm_tokenizer,
            cache=mm_processor_cache,
        )
```

`MULTIMODAL_REGISTRY` 根据模型类型分发到不同的 Processor 实现：
- 多数 HF 模型 → 各自注册的 `BaseMultiModalProcessor` 子类
- `model_impl="transformers"` 的通用路径 → `MultiModalProcessor`（`transformers/multimodal.py`）

#### 3.4.2 `_process_multimodal()` 调度流程

```python
def _process_multimodal(self, prompt, mm_data, mm_uuids, mm_processor_kwargs, ...):
    # 1. 解析原始多模态数据为结构化 items
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)

    # 2. 处理 UUIDs（用于缓存去重）
    mm_uuid_items = parse_mm_uuids(mm_uuids)
    mm_uuid_items = self._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, ...)

    # 3. 构建处理器输入
    mm_processor_inputs = ProcessorInputs(
        prompt=prompt,
        mm_data_items=mm_data_items,
        mm_uuid_items=mm_uuid_items,
        hf_processor_mm_kwargs=mm_processor_kwargs or {},
    )

    # 4. ★ 调用多模态处理器
    mm_inputs = mm_processor.apply(mm_processor_inputs, timing_ctx)

    return mm_inputs
```

#### 3.4.3 `BaseMultiModalProcessor.apply()` 核心逻辑

```python
class BaseMultiModalProcessor:
    def apply(self, inputs, timing_ctx):
        # 1. 获取 HF Processor 实例
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # 2. 联合处理文本 + 多模态数据
        prompt_ids, processed_data, applied = self._apply_hf_processor_text_mm(
            prompt_text=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        # 3. 计算占位符位置
        mm_placeholders = self._find_mm_placeholders(prompt_ids, mm_prompt_updates)

        # 4. 组装多模态 kwargs
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, ...),
        )

        # 5. 返回完整的多模态输入
        return mm_inputs(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )
```

---

### 3.5 输出后处理

输出后处理发生在 `serving.py` 的流式/非流式生成器中：

| 处理类型 | 处理器 | 触发条件 |
|----------|--------|----------|
| **推理链解析** | `ReasoningParser.extract_reasoning_streaming()` | `reasoning_parser` 已配置 |
| **工具调用解析（auto）** | `ToolParser.extract_tool_calls_streaming()` | `tool_choice="auto"` + `enable_auto_tools` |
| **工具调用解析（required）** | `extract_tool_call_required_streaming()` | `tool_choice="required"` |
| **工具调用解析（named）** | 直接构建 `DeltaToolCall` | `tool_choice=ChatCompletionNamedToolChoiceParam` |
| **Harmony 输出解析** | `extract_harmony_streaming_delta()` | `use_harmony=True` (GPT-OSS) |
| **LogProbs 构建** | `_create_chat_logprobs()` | `request.logprobs=True` |

---

## 4. 完整调用链（精确版）

以下是 `/v1/chat/completions` 请求的精确调用链，基于源码逐行追踪：

```
HTTP POST /v1/chat/completions
    │
    ▼
OpenAIServingChat.create_chat_completion(request, raw_request)
    │
    │  ① 初始化 ReasoningParser
    │
    ├── self.render_chat_request(request)
    │   │
    │   ├── self._check_model(request)                    # 模型校验
    │   ├── self.engine_client.errored → raise dead_error  # 引擎健康检查
    │   │
    │   └── self.openai_serving_render.render_chat(request)   ← ★ 委托
    │       │
    │       │  [OpenAIServingRender.render_chat()]
    │       │
    │       ├── Mistral tokenizer 特殊处理
    │       │
    │       ├── tool_choice 校验
    │       │
    │       ├── validate_chat_template()
    │       │
    │       └── self.preprocess_chat(request, request.messages, ...)
    │           │
    │           │  [OpenAIServingRender.preprocess_chat()]
    │           │
    │           ├── 构建 tok_params = request.build_tok_params(model_config)
    │           ├── 构建 chat_params = request.build_chat_params(...).with_defaults(...)
    │           │
    │           └── renderer.render_chat_async([messages], chat_params, tok_params, ...)
    │               │
    │               │  [BaseRenderer.render_chat_async()]  ← ★★ 四步流水线
    │               │
    │               ├── Step 1: render_messages_async(messages, chat_params)
    │               │   │
    │               │   │  [HfRenderer.render_messages()]
    │               │   │
    │               │   ├── parse_chat_messages(messages, model_config, ...)
    │               │   │   → 提取 mm_data (PIL.Image 等), mm_uuids, conversation
    │               │   │
    │               │   ├── apply_chat_template(tokenizer, conversation, ...)
    │               │   │   → Jinja2 模板渲染 → prompt 文本
    │               │   │
    │               │   └── 返回 (conversation, DictPrompt)
    │               │
    │               ├── Step 2: tokenize_prompts_async(dict_prompts, tok_params)
    │               │   │
    │               │   └── _tokenize_singleton_prompt_async(prompt, params)
    │               │       ├── params.apply_pre_tokenization(tokenizer, prompt)
    │               │       ├── tokenizer.encode(prompt["prompt"])  → prompt_token_ids
    │               │       └── params.apply_post_tokenization(tokenizer, prompt)
    │               │       (mm_data 原样保留在 prompt dict 中)
    │               │
    │               ├── Step 3: _apply_prompt_extras(tok_prompts, prompt_extras)
    │               │   └── 将 mm_processor_kwargs, cache_salt 写入 prompt dict
    │               │
    │               └── Step 4: process_for_engine_async(prompt, arrival_time)
    │                   │
    │                   └── _process_tokens(prompt)
    │                       │
    │                       ├── [无 mm_data] → token_inputs(prompt_token_ids)
    │                       │
    │                       └── [有 mm_data] → _process_multimodal(...)
    │                           │
    │                           ├── mm_processor.info.parse_mm_data(mm_data)
    │                           ├── self._process_mm_uuids(...)
    │                           │
    │                           └── mm_processor.apply(processor_inputs, timing_ctx)
    │                               │
    │                               │  [BaseMultiModalProcessor.apply()]
    │                               │
    │                               ├── self._apply_hf_processor_text_mm(...)
    │                               │   └── ctx.call_hf_processor(hf_processor, data, kwargs)
    │                               │       └── hf_processor(**data, return_tensors="pt")
    │                               │           ★ transformers.ProcessorMixin.__call__()
    │                               │
    │                               ├── _find_mm_placeholders(prompt_ids, ...)
    │                               └── 返回 MultiModalInputs
    │
    │  返回 (conversation, [engine_prompt])
    │
    ├── ② 构建 request_id, request_metadata
    ├── ③ self._maybe_get_adapters(request)
    ├── ④ get_max_tokens(...) → sampling_params
    ├── ⑤ reasoning_parser.is_reasoning_end(prompt_token_ids)
    │
    └── ⑥ self.engine_client.generate(engine_prompt, sampling_params, ...)
        │
        └── 流式/非流式后处理 → HTTP Response
```

---

## 5. 关键数据结构流转

```
ChatCompletionRequest.messages
    │  [{role: "user", content: [{type: "text", text: "..."}, {type: "image_url", ...}]}]
    │
    ▼ (Step 1: parse_chat_messages + apply_chat_template)
DictPrompt
    {"prompt": "<|im_start|>user\n<image>\nWhat is this?\n...",
     "multi_modal_data": {"image": [PIL.Image]},
     "multi_modal_uuids": {"image": ["hash1"]}}
    │
    ▼ (Step 2: tokenize_prompts_async)
TokPrompt (TokensPrompt)
    {"prompt_token_ids": [151644, 872, ...],
     "multi_modal_data": {"image": [PIL.Image]},    ← mm_data 原样传递
     "multi_modal_uuids": {"image": ["hash1"]}}
    │
    ▼ (Step 3: _apply_prompt_extras)
TokPrompt + extras
    {"prompt_token_ids": [...], "multi_modal_data": {...},
     "mm_processor_kwargs": {...}, "cache_salt": "..."}
    │
    ▼ (Step 4: process_for_engine_async → _process_multimodal)
ProcessorInputs (MultiModalInputs)
    {"type": "multimodal",
     "prompt_token_ids": [int],
     "mm_kwargs": MultiModalKwargs {
         "pixel_values": Tensor,
         "image_grid_thw": Tensor, ...}
     "mm_placeholders": {"image": [PlaceholderRange(offset=5, length=256)]},
     "mm_hashes": {"image": ["hash1"]}}
    │
    ▼ (engine_client.generate)
引擎消费
```

---

## 6. 与 transformers 库的集成点

| 集成点 | transformers 组件 | vLLM 调用位置 | 用途 |
|--------|-------------------|---------------|------|
| **Processor 加载** | `AutoProcessor.from_pretrained()` | `vllm/transformers_utils/processor.py` | 加载模型对应的多模态处理器 |
| **Processor 调用** | `ProcessorMixin.__call__()` | `InputProcessingContext.call_hf_processor()` | 图片/音频预处理 → `pixel_values` 等张量 |
| **Chat Template** | `ProcessorMixin.chat_template` | `vllm/renderers/hf.py` | 获取处理器级别的 chat template |
| **Token 数计算** | `processor._get_num_multimodal_tokens()` | `MultiModalProcessingInfo.get_max_image_tokens()` | 计算每图最大 token 数 |
| **BatchFeature** | `transformers.BatchFeature` | `_call_hf_processor()` 返回值 | HF Processor 输出的标准容器 |
| **Vision Encoder** | `model.get_image_features()` | `MultiModalMixin.embed_multimodal()` | 模型执行阶段的视觉编码 |
| **RoPE Index** | `model.get_rope_index()` | `MultiModalMixin.get_mrope_input_positions()` | 多模态位置编码计算 |

---

## 7. ���块依赖关系图

```
vllm/entrypoints/openai/chat_completion/
├── serving.py                         # ① API 层入口
│   └── OpenAIServingChat
│       ├── __init__(): openai_serving_render  ← 持有 Render 服务引用
│       ├── render_chat_request()      # 模型校验 + 委托 render_chat()
│       ├── create_chat_completion()   # 主流程
│       ├── chat_completion_stream_generator()   # 流式后处理
│       └── chat_completion_full_generator()     # 非流式后处理
│
├── protocol.py                        # ChatCompletionRequest 等协议
└── stream_harmony.py                  # Harmony 流式处理

vllm/entrypoints/serve/render/
├── serving.py                         # ② Render 服务层
│   └── OpenAIServingRender
│       ├── __init__(): renderer       ← 持有 BaseRenderer 引用
│       ├── render_chat()              # 工具校验 + 模板校验 + 调度
│       ├── preprocess_chat()          # 参数构建 + 调用 renderer.render_chat_async()
│       └── _make_request_with_harmony()  # Harmony (GPT-OSS) 路径
└── api_router.py                      # /v1/chat/completions/render 路由

vllm/renderers/
├── base.py                            # ③ Renderer 渲染层核心
│   └── BaseRenderer
│       ├── render_chat_async()        # 四步处理流水线
│       ├── render_messages_async()    # Step 1: 消息渲染 (abstract)
│       ├── tokenize_prompts_async()   # Step 2: Tokenization
│       ├── _apply_prompt_extras()     # Step 3: 附加参数
│       ├── process_for_engine_async() # Step 4: 引擎输入构建
│       └── _process_multimodal()      # Step 4 内: 多模态处理调度
├── hf.py                              # HfRenderer — 标准 HF 路径
├── mistral.py                         # MistralRenderer
├── grok2.py                           # Grok2Renderer
└── params.py                          # ChatParams, TokenizeParams

vllm/entrypoints/chat_utils.py         # parse_chat_messages() — 消息解析

vllm/multimodal/                       # ④ 多模态处理层
├── registry.py                        # MULTIMODAL_REGISTRY — 处理器注册表
├── processing/
│   ├── processor.py                   # BaseMultiModalProcessor
│   │   ├── apply()                    # 处理主入口
│   │   ├── _call_hf_processor()       # 调用 HF Processor
│   │   └── _apply_hf_processor_text_mm()
│   ���── context.py                     # InputProcessingContext
│   │   └── call_hf_processor()        # 实际执行 hf_processor(...)
│   └── inputs.py                      # ProcessorInputs 数据结构
├── parse.py                           # 多模态数据解析
├── inputs.py                          # MultiModalInputs 等数据结构
├── cache.py                           # 处理结果缓存
└── hasher.py                          # 多模态数据哈希

vllm/model_executor/models/transformers/
└── multimodal.py                      # Transformers 通用多模态路径
    ├── MultiModalProcessingInfo       # 处理信息 (max tokens 等)
    ├── MultiModalProcessor            # 通用处理器 (mm_token_type_ids)
    ├── MultiModalMixin                # 模型 Mixin (embed_multimodal)
    └── MultiModalDummyInputsBuilder   # Dummy 输入构建 (profiling)

vllm/transformers_utils/
└── processor.py                       # get_processor() / cached_get_processor()

vllm/parser.py                         # ParserManager — 推理/工具解析器管理
vllm/reasoning.py                      # ReasoningParser — 推理链解析
vllm/tool_parsers/                     # ToolParser — 工具调用解析
vllm/sampling_params.py                # SamplingParams / BeamSearchParams
```

---

## 8. 关键设计决策

### 8.1 三层委托架构

vLLM 的数据预处理采用三层委托设计：

| 层次 | 类 | 职责 |
|------|-----|------|
| **API 层** | `OpenAIServingChat` | 模型校验、引擎健康检查、LoRA 适配、采样参数构建、引擎提交、输出后处理 |
| **Render 服务层** | `OpenAIServingRender` | 工具调用校验、chat template 校验、参数构建（ChatParams/TokenizeParams）、工具解析器调整 |
| **Renderer 渲染层** | `BaseRenderer` | 消息渲染、tokenization、多模态处理、引擎输入组装 |

这种设计使得 `OpenAIServingRender` 可以**独立于引擎运行**（GPU-less render server），只需要 tokenizer 和模型配置即可完成数据预处理，从而支持 prefill/decode 分离的 disaggregated serving 架构。

### 8.2 多模态处理在 API 进程中完成

多模态数据的预处理（图片 resize、归一化等）发生在 API 进程中（`BaseRenderer._process_multimodal()`），而非引擎 worker 中。这带来以下优势：
- **减轻引擎负担**：GPU worker 只需处理已准备好的张量
- **支持缓存**：通过 `mm_processor_cache` 避免重复处理相同的多模态输入
- **并行处理**：API 进程可在引擎忙时预处���下一个请求

### 8.3 Tokenizer 深拷贝

```python
mm_tokenizer = copy.deepcopy(tokenizer)
```

为多模态处理器单独深拷贝一个 tokenizer，避免 Rust tokenizer 后端的并发冲突（`RuntimeError: Already borrowed`）。

### 8.4 两种多模态处理路径

| 路径 | 适用模型 | 占位符定位方式 |
|------|----------|---------------|
| **模型专用 Processor** | 大多数原生 vLLM 支持的模型 | 通过 `_get_prompt_updates()` 和正则匹配 |
| **Transformers 通用路径** | `model_impl="transformers"` | 通过 `mm_token_type_ids` |

### 8.5 异步流水线设计

`BaseRenderer.render_chat_async()` 中 Step 1 和 Step 4 都使用了 `asyncio.gather()` 进行并行处理，当有多个 conversation 时可以并发渲染和处理，提高吞吐量。
