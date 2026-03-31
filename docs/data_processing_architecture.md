# vLLM 数据处理架构调研文档

## 1. 概述

vLLM 的数据处理架构负责将用户通过 OpenAI 兼容 API 发送的请求（包含文本、图片、音频等多模态内容），转换为引擎可直接消费的 token 化输入。整个流程分为 **API 层预处理**、**渲染层模板处理**、**多模态处理** 和 **引擎提交** 四大阶段，涉及 vLLM 内部多个子系统以及 HuggingFace `transformers` 库的深度集成。

---

## 2. 整体架构概览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           用户请求 (ChatCompletionRequest)                    │
│        messages: [{role, content: [text, image_url, ...]}], tools, ...       │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  API 层 (serving.py)                                                         │
│  OpenAIServingChat.create_chat_completion()                                  │
│  ├── 模型校验、引擎健康检查                                                     │
│  ├── 推理解析器初始化 (ReasoningParser)                                        │
│  └── 委托渲染 ──────────────────────────────────────┐                         │
└─────────────────────────────────────────────────────┼────────────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  渲染层 (renderers/)                                                         │
│  BaseRenderer.render_chat()                                                  │
│  ├── Step 1: render_messages()  → 消息解析 + chat template + mm_data 提取     │
│  ├── Step 2: tokenize_prompts() → 文本 tokenization                          │
│  ├── Step 3: _apply_prompt_extras()                                          │
│  └── Step 4: process_for_engine() → 多模态处理 + 组装引擎输入                   │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  多模态处理层 (multimodal/)                                                   │
│  BaseMultiModalProcessor.apply()                                             │
│  ├── _call_hf_processor()  → 调用 transformers ProcessorMixin                │
│  ├── 计算 mm_placeholders（占位符位置）                                        │
│  └── 组装 MultiModalInputs                                                   │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  引擎层 (engine/)                                                            │
│  EngineClient.generate(engine_prompt, sampling_params, ...)                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 各阶段详细流程

### 3.1 API 层：请求接收与校验

**入口文件**：`vllm/entrypoints/openai/chat_completion/serving.py`

**核心类**：`OpenAIServingChat`（继承自 `OpenAIServing`）

#### 3.1.1 初始化阶段预配置

`__init__` 中预先配置好生成过程所需的所有解析器和参数：

| 配置项 | 来源 | 作用 |
|--------|------|------|
| `reasoning_parser_cls` | `ParserManager.get_reasoning_parser()` | 推理/思考链解析（如 QwQ、DeepSeek-R1） |
| `tool_parser` | `ParserManager.get_tool_parser()` | 工具调用解析（Function Calling） |
| `default_sampling_params` | `model_config.get_diff_sampling_param()` | 从 generation_config 获取的默认采样参数 |
| `override_max_tokens` | generation_config 或 override 配置 | max_tokens 覆盖值 |
| `use_harmony` | `model_config.hf_config.model_type == "gpt_oss"` | GPT-OSS 模型特殊处理标志 |
| `tool_call_id_type` | `get_tool_call_id_type(model_config)` | tool_call_id 生成策略（kimi_k2 / random） |

#### 3.1.2 请求处理主流程

```python
async def create_chat_completion(self, request, raw_request):
    # 1. 初始化推理解析器
    reasoning_parser = self.reasoning_parser_cls(tokenizer, ...)

    # 2. 渲染请求（模型校验 + 消息处理）
    conversation, engine_prompts = await self.render_chat_request(request)

    # 3. 构建请求元数据
    request_id = f"chatcmpl-{self._base_request_id(...)}"
    request_metadata = RequestResponseMetadata(request_id=request_id)

    # 4. LoRA 适配器 & 模型名称
    lora_request = self._maybe_get_adapters(request)
    model_name = self.models.model_name(lora_request)

    # 5. 数据并行 rank（从 header 提取）
    data_parallel_rank = self._get_data_parallel_rank(raw_request)

    # 6. 构建采样参数
    max_tokens = get_max_tokens(max_model_len, request.max_tokens, ...)
    sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)

    # 7. 推理状态判断
    reasoning_ended = reasoning_parser.is_reasoning_end(prompt_token_ids)

    # 8. 提交引擎
    generator = self.engine_client.generate(
        engine_prompt, sampling_params, request_id,
        reasoning_ended=reasoning_ended, ...
    )
```

#### 3.1.3 涉及模块

| 模块路径 | 作用 |
|----------|------|
| `vllm.entrypoints.openai.engine.serving.OpenAIServing` | 父类，提供模型校验、适配器等基础方法 |
| `vllm.entrypoints.openai.chat_completion.protocol` | 请求/响应协议定义，含 `to_sampling_params()` |
| `vllm.entrypoints.openai.models.serving.OpenAIServingModels` | 模型列表与名称管理 |
| `vllm.entrypoints.utils` | `get_max_tokens()`、`should_include_usage()` |
| `vllm.entrypoints.chat_utils` | `ConversationMessage`、`make_tool_call_id()` 等 |
| `vllm.parser.ParserManager` | 统一管理推理解析器和工具解析器的注册/获取 |
| `vllm.reasoning.ReasoningParser` | 推理链解析 |
| `vllm.tool_parsers.ToolParser` | 工具调用解析 |
| `vllm.sampling_params` | `SamplingParams` / `BeamSearchParams` |
| `vllm.engine.protocol.EngineClient` | 引擎客户端协议 |

---

### 3.2 渲染层：消息解析、模板应用与 Tokenization

**入口文件**：`vllm/renderers/base.py`

**核心类**：`BaseRenderer`（抽象基类），具体实现包括 `HfRenderer`、`MistralRenderer`、`Grok2Renderer` 等

#### 3.2.1 四步处理流程

`BaseRenderer.render_chat()` 内部按四个步骤依次处理：

```python
def render_chat(self, conversations, chat_params, tok_params=None):
    # Step 1: 消息渲染 — 解析 messages，应用 chat template，提取多模态数据
    rendered = [self.render_messages(conversation, chat_params)
                for conversation in conversations]
    # 返回 (conversation, DictPrompt)
    # DictPrompt 中包含: prompt(文本), multi_modal_data, multi_modal_uuids

    # Step 2: Tokenization — 文本转 token IDs
    tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

    # Step 3: 附加额外参数
    self._apply_prompt_extras(tok_prompts, prompt_extras)

    # Step 4: 引擎输入构建 — 多模态处理在此触发
    eng_prompts = [self.process_for_engine(prompt, arrival_time)
                   for prompt in tok_prompts]
```

#### 3.2.2 Step 1 详解：消息渲染 (`render_messages`)

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

#### 3.2.3 Step 2 详解：Tokenization

```python
def _tokenize_singleton_prompt(self, prompt, params):
    if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
        # 应用 pre-tokenization（如 truncation）
        prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
        # 调用 tokenizer.encode()
        prompt = self._tokenize_prompt(prompt, params)

    # 需要时反向 detokenize
    if params.needs_detokenization and "prompt" not in prompt:
        prompt = self._detokenize_prompt(prompt)

    return params.apply_post_tokenization(self.tokenizer, prompt)
```

> **注意**：此阶段 `multi_modal_data` 原封不动地附着在 prompt dict 上传递，不做任何处理。

#### 3.2.4 Step 4 详解：引擎输入构建 (`process_for_engine`)

```
process_for_engine(prompt, arrival_time)
    │
    ├── _process_singleton(prompt)
    │   │
    │   └── _process_tokens(prompt)
    │       │
    │       ├── 检查 prompt.get("multi_modal_data")
    │       │
    │       ├── [有 mm_data] → _process_multimodal()  ← 触发多模态处理
    │       │
    │       └── [无 mm_data] → token_inputs(prompt_token_ids)  ← 纯文本
    │
    └── engine_prompt["arrival_time"] = arrival_time
```

---

### 3.3 多模态处理层：HF Processor 集成

**核心文件**：
- `vllm/renderers/base.py` — `_process_multimodal()` 调度入口
- `vllm/multimodal/processing/processor.py` — `BaseMultiModalProcessor` 抽象处理器
- `vllm/multimodal/processing/context.py` — `InputProcessingContext` 上下文，封装 HF Processor 调用
- `vllm/model_executor/models/transformers/multimodal.py` — Transformers 通用多模态处理器

#### 3.3.1 多模态处理器的创建

在 `BaseRenderer.__init__()` 中：

```python
if config.model_config.is_multimodal_model:
    from vllm.multimodal import MULTIMODAL_REGISTRY as mm_registry

    # 创建处理结果缓存
    mm_processor_cache = mm_registry.processor_cache_from_config(config)

    # 深拷贝 tokenizer 避免 Rust tokenizer 并发冲突
    mm_tokenizer = copy.deepcopy(tokenizer)

    # 通过注册表创建对应模型的多模态处理器
    self.mm_processor = mm_registry.create_processor(
        config.model_config,
        tokenizer=mm_tokenizer,
        cache=mm_processor_cache,
    )
```

`MULTIMODAL_REGISTRY` 根据模型类型分发到不同的 Processor 实现：
- 多数 HF 模型 → 各自注册的 `BaseMultiModalProcessor` 子类
- `model_impl="transformers"` 的通用路径 → `MultiModalProcessor`（`transformers/multimodal.py`）

#### 3.3.2 `_process_multimodal()` 调度流程

```python
def _process_multimodal(self, prompt, mm_data, mm_uuids, mm_processor_kwargs, ...):
    # 1. 解析原始多模态数据为结构化 items
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    # 例如: {"image": ImageProcessorItems([PIL.Image, ...])}

    # 2. 处理 UUIDs（用于缓存去重）
    mm_uuid_items = parse_mm_uuids(mm_uuids)
    mm_uuid_items = self._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, ...)

    # 3. 构建处理器输入
    mm_processor_inputs = MMProcessorInputs(
        prompt,              # token IDs 或文本
        mm_data_items,       # 结构化多模态数据
        mm_uuid_items,       # 缓存标识
        hf_processor_mm_kwargs=mm_processor_kwargs or {},
    )

    # 4. 调用多模态处理器
    mm_inputs = mm_processor.apply(mm_processor_inputs, timing_ctx)
    # → 这里最终会调用 HuggingFace Processor

    return mm_inputs
```

#### 3.3.3 `BaseMultiModalProcessor.apply()` 核心逻辑

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
            mm_kwargs=mm_kwargs,        # 包含 pixel_values 等张量
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )
```

#### 3.3.4 HuggingFace Processor 的实际调用

```python
# vllm/multimodal/processing/context.py
class InputProcessingContext:
    def call_hf_processor(self, hf_processor, data, kwargs):
        """
        data = {"text": "...", "images": [PIL.Image, ...]}
        kwargs = {"return_mm_token_type_ids": True, ...}
        """
        merged_kwargs = self.get_merged_mm_kwargs(kwargs)
        allowed_kwargs = get_allowed_kwarg_only_overrides(hf_processor, merged_kwargs)

        # ★ 这就是对 transformers.ProcessorMixin.__call__() 的直接调用
        output = hf_processor(**data, **allowed_kwargs, return_tensors="pt")
        # 例如: Qwen2VLProcessor(text="...", images=[img], return_tensors="pt")
        # 返回 BatchFeature: {
        #   "input_ids": tensor([...]),
        #   "pixel_values": tensor([...]),
        #   "image_grid_thw": tensor([...]),
        #   ...
        # }

        # 后处理：浮点张量转换为模型 dtype
        return BatchFeature(self._postprocess_output(output.data))
```

#### 3.3.5 Transformers 通用多模态路径

对于使用 `model_impl="transformers"` 的模型（如通过 `TransformersForCausalLM` 加载的 Qwen2-VL 等），有一条专门的通用处理路径：

```python
# vllm/model_executor/models/transformers/multimodal.py
class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    def apply(self, inputs, timing_ctx):
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # 如果输入是 token IDs，先 decode 为文本（HF Processor 需要文本输入）
        if not isinstance(prompt, str):
            prompt = hf_processor.decode(prompt)

        # 调用 HF Processor
        prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(...)

        # 通过 mm_token_type_ids 推断占位符位置
        mm_token_type_ids = processed_data.get("mm_token_type_ids")
        mm_positions = torch.where(mm_token_type_ids == 1)[1]

        # 调用 HF Processor 的内部方法计算每图 token 数
        mm_tokens_per_modality = hf_processor._get_num_multimodal_tokens(
            image_sizes=image_sizes, ...
        )

        # 切分并构建 PlaceholderRange
        chunked_mm_positions = torch.split(mm_positions, split_sizes)
        ...
```

**该路径的特殊之处**：
- 使用 `mm_token_type_ids`（而非正则匹配占位符 token）来定位多模态区域
- 调用 `hf_processor._get_num_multimodal_tokens()` 获取精确的 token 数量
- 通过 `return_mm_token_type_ids=True` 强制 HF Processor 返回类型标记

#### 3.3.6 多模态处理器的模型端集成

在模型执行阶段（`MultiModalMixin.embed_multimodal()`）：

```python
class MultiModalMixin(SupportsMultiModal, SupportsMRoPE):
    def embed_multimodal(self, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        num_image_patches = kwargs.pop("num_image_patches")

        if image_embeds is not None:
            return image_embeds  # 直接使用预计算的嵌入

        # 调用 transformers 模型的 vision encoder
        vision_embeddings = self.model.get_image_features(pixel_values, **kwargs)
        # → 这里调用的是 transformers 模型自身的视觉编码器

        # 按 num_image_patches 切分
        return list(torch.split(vision_embeddings, token_split_sizes, dim=0))
```

---

### 3.4 输出后处理

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

## 4. 关键数据结构流转

```
ChatCompletionRequest.messages
    │
    ▼ (parse_chat_messages)
ConversationMessage[] + MultiModalDataDict
    │                    {"image": [PIL.Image], "audio": [...]}
    ▼ (render_messages + apply_chat_template)
DictPrompt
    {"prompt": str, "multi_modal_data": {...}, "multi_modal_uuids": {...}}
    │
    ▼ (tokenize_prompts)
TokPrompt (TokensPrompt)
    {"prompt_token_ids": [int], "multi_modal_data": {...}, ...}
    │
    ▼ (process_for_engine → _process_multimodal)
MultiModalInputs
    {"prompt_token_ids": [int],
     "mm_kwargs": {
         "pixel_values": Tensor,
         "image_grid_thw": Tensor, ...
     },
     "mm_placeholders": {"image": [PlaceholderRange(...)]},
     "mm_hashes": {"image": ["hash1", ...]}}
    │
    ▼ (engine_client.generate)
ProcessorInputs → 引擎
```

---

## 5. 与 transformers 库的集成点

### 5.1 集成总览

| 集成点 | transformers 组件 | vLLM 调用位置 | 用途 |
|--------|-------------------|---------------|------|
| **Processor 加载** | `AutoProcessor.from_pretrained()` | `vllm/transformers_utils/processor.py` | 加载模型对应的多模态处理器 |
| **Processor 调用** | `ProcessorMixin.__call__()` | `InputProcessingContext.call_hf_processor()` | 图片/音频预处理 → `pixel_values` 等张量 |
| **Chat Template** | `ProcessorMixin.chat_template` | `vllm/renderers/hf.py` | 获取处理器级别的 chat template |
| **Token 数计算** | `processor._get_num_multimodal_tokens()` | `MultiModalProcessingInfo.get_max_image_tokens()` | 计算每图最大 token 数 |
| **BatchFeature** | `transformers.BatchFeature` | `_call_hf_processor()` 返回值 | HF Processor 输出的标准容器 |
| **Vision Encoder** | `model.get_image_features()` | `MultiModalMixin.embed_multimodal()` | 模型执行阶段的视觉编码 |
| **RoPE Index** | `model.get_rope_index()` | `MultiModalMixin.get_mrope_input_positions()` | 多模态位置编码计算 |

### 5.2 Processor 使用方式

```python
# 加载（带 LRU 缓存）
from vllm.transformers_utils.processor import cached_get_processor
processor = cached_get_processor(model_name, trust_remote_code=True)

# 调用（通过 InputProcessingContext 封装）
output = processor(
    text="<|im_start|>user\n<image>\nDescribe this image\n...",
    images=[PIL.Image.open("cat.jpg")],
    return_tensors="pt",
    return_mm_token_type_ids=True,  # transformers 通用路径额外需要
)
# output: BatchFeature {
#     "input_ids": tensor([[151644, 872, ...]]),
#     "pixel_values": tensor([[[[0.485, ...]]]]),
#     "image_grid_thw": tensor([[1, 28, 28]]),
#     "mm_token_type_ids": tensor([[0, 0, ..., 1, 1, ..., 0, 0]]),
# }
```

---

## 6. 模块依赖关系图

```
vllm/entrypoints/openai/chat_completion/
├── serving.py                         # API 层入口
│   └── OpenAIServingChat
│       ├── render_chat_request()      # 委托渲染
│       ├── create_chat_completion()   # 主流程
│       ├── chat_completion_stream_generator()   # 流式后处理
│       └── chat_completion_full_generator()     # 非流式后处理
│
├── protocol.py                        # ChatCompletionRequest 等协议
└── stream_harmony.py                  # Harmony 流式处理

vllm/entrypoints/serve/render/
└── serving.py                         # OpenAIServingRender
    └── render_chat()                  # 委托给 BaseRenderer

vllm/renderers/
├── base.py                            # BaseRenderer — 渲染层核心
│   ├── render_chat()                  # 四步处理流程
│   ├── _process_multimodal()          # 多模态处理调度
│   └── process_for_engine()           # 引擎输入构建
├── hf.py                              # HfRenderer — 标准 HF 路径
├── mistral.py                         # MistralRenderer
├── grok2.py                           # Grok2Renderer
└── params.py                          # ChatParams, TokenizeParams

vllm/entrypoints/chat_utils.py         # parse_chat_messages() — 消息解析

vllm/multimodal/
├── registry.py                        # MULTIMODAL_REGISTRY — 处理器注册表
├── processing/
│   ├── processor.py                   # BaseMultiModalProcessor — 处理器抽象
│   │   ├── apply()                    # 处理主入口
│   │   ├── _call_hf_processor()       # 调用 HF Processor
│   │   └── _apply_hf_processor_text_mm()
│   ├── context.py                     # InputProcessingContext
│   │   └── call_hf_processor()        # 实际执行 hf_processor(...)
│   └── inputs.py                      # ProcessorInputs 数据结构
├── parse.py                           # 多模态数据解析
├── inputs.py                          # MultiModalInputs 等数据结构
├── cache.py                           # 处理结果缓存
└── hasher.py                          # 多模态数据哈希

vllm/model_executor/models/transformers/
└── multimodal.py                      # Transformers 通用多模态路径
    ├── MultiModalProcessingInfo       # 处理信息（max tokens 等）
    ├── MultiModalProcessor            # 通用处理器（用 mm_token_type_ids）
    ├── MultiModalMixin                # 模型 Mixin
    │   ├── embed_multimodal()         # 视觉编码
    │   └── get_mrope_input_positions()# MRoPE 位置编码
    └── MultiModalDummyInputsBuilder   # Dummy 输入构建（profiling 用）

vllm/transformers_utils/
└── processor.py                       # get_processor() / cached_get_processor()

vllm/parser.py                         # ParserManager — 推理/工具解析器管理
vllm/reasoning.py                      # ReasoningParser — 推理链解析
vllm/tool_parsers/                     # ToolParser — 工具调用解析
vllm/sampling_params.py                # SamplingParams / BeamSearchParams
```

---

## 7. 关键设计决策

### 7.1 多模态处理在 API 进程中完成

多模态数据的预处理（图片 resize、归一化等）发生在 API 进程中（`BaseRenderer._process_multimodal()`），而非引擎 worker 中。这带来以下优势：
- **减轻引擎负担**：GPU worker 只需处理已准备好的张量
- **支持缓存**：通过 `mm_processor_cache` 避免重复处理相同的多模态输入
- **并行处理**：API 进程可在引擎忙时预处理下一个请求

### 7.2 Tokenizer 深拷贝

```python
mm_tokenizer = copy.deepcopy(tokenizer)
```

为多模态处理器单独深拷贝一个 tokenizer，避免 Rust tokenizer 后端的并发冲突（`RuntimeError: Already borrowed`）。

### 7.3 两种多模态处理路径

| 路径 | 适用模型 | 占位符定位方式 |
|------|----------|---------------|
| **模型专用 Processor** | 大多数原生 vLLM 支持的模型 | 通过 `_get_prompt_updates()` 和正则匹配 |
| **Transformers 通用路径** | `model_impl="transformers"` | 通过 `mm_token_type_ids` |

### 7.4 渲染与处理的分离

vLLM 将"消息到 prompt 文本"的渲染与"多模态数据处理"明确分离：
- **渲染层**（Step 1-2）只负责文本处理，`mm_data` 原样传递
- **处理层**（Step 4）在 `process_for_engine()` 中才触发多模态数据的实际处理
- 这使得渲染层可以被独立使用（如 `/v1/chat/completions/render` 端点）

---

## 8. 数据处理时序图

```
User Request
    │
    │  ① HTTP POST /v1/chat/completions
    ▼
OpenAIServingChat.create_chat_completion()
    │
    │  ② 模型校验 + 引擎健康检查
    │  ③ 初始化 ReasoningParser
    ▼
OpenAIServingRender.render_chat()
    │
    ▼
BaseRenderer.render_chat()
    │
    │  ④ render_messages():
    │     - parse_chat_messages() → 提取 mm_data (PIL.Image 等)
    │     - apply_chat_template() → 生成带占位符的 prompt
    │
    │  ⑤ tokenize_prompts():
    │     - tokenizer.encode(prompt) → token IDs
    │     - mm_data 原样传递
    │
    │  ⑥ process_for_engine():
    │     - _process_tokens() 检测到 multi_modal_data
    │     - _process_multimodal() 调度
    │         │
    │         ▼
    │     BaseMultiModalProcessor.apply()
    │         │
    │         │  ⑦ _call_hf_processor():
    │         │     hf_processor(text=prompt, images=[img], return_tensors="pt")
    │         │     → BatchFeature {pixel_values, input_ids, ...}
    │         │
    │         │  ⑧ 计算 mm_placeholders
    │         │  ⑨ 组装 MultiModalInputs
    │         ▼
    │     返回 engine_prompt (ProcessorInputs)
    ▼
回到 create_chat_completion()
    │
    │  ⑩ 构建 SamplingParams
    │  ⑪ 判断 reasoning_ended
    │  ⑫ engine_client.generate(engine_prompt, sampling_params, ...)
    ▼
Engine 生成 → 流式/非流式后处理 → 返回响应
```
