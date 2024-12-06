### Task 2: Inference Agent (plus bonus🔥)


#### TODO

You are required to implement two pytorch modules, one is named `PromptTemplate` in `src/modeling/prompt.py` and the other is named `InferenceAgent` in `src/inference/agent.py`.


#### Explanation

* To better support inference, we first of all implement an auxiliary `PromptTemplate` module in `src/modeling/prompt.py` (*See the langchain prompt template module in [Reference](#references) for more conceptual examples*), which provides the APIs described as follows:
    * `__init__(template_str: str = "")`: initialize the `PromptTemplate` module with the template f-string, with the format like: `"....{key1}...{key2}..."` (*NOTE: a normal string without any key is also a valid template, even like the default empty string `""`*), for example:
       ```python
       template_str = "You are a {profession} named {name}."
       ```
    * `set_default(**kwargs: Optional[Dict[str, str]])`: set the default values of the prompt template keys with the given keyword arguments like: `{key1: value1, key2: value2, ...}`, for example:
       ```python
       kwargs = {"profession": "teacher"}
       ```
    * `keys()`: get the keys with its default values (*if any key has not been set with default value, then use `None` as a placeholder*) of the prompt template as a dictionary like: `{key1: default_value1, key2: None, ...}`, for example:
       ```python
       keys = {"profession": "teacher", "name": None}
       ```
    * `forward(**kwargs: Optional[Dict[str, str]])`: set the prompt template keys with the given keyword arguments and get the formatted prompt string like: `"....value1...value2..."` (*NOTE: if certain key in the kwargs is not found in the keys of the prompt template, just ignore it; if certain prompt template key has not been set with its default value, then its corresponding keyword argument should be provided, otherwise a ValueError will be raised*), for example:
       ```python
       kwargs = {"name": "John"}
       return_prompt = "You are a teacher named John."
       ```
* Besides, we also need the `tokenizer` to encode the raw-string prompt into vocabulary token ids and decode the generated ones by the model back to the raw-string response. However, the tokenizer is often attached the specific llm and pretrained together, using the algorithm such as `BPE` (*See the `Byte-Pair Encoding tokenization` in the [Reference](#references) for more details*). Therefore, to support inference with `LlamaModel` we've built in the [task1](./task1.md), we have to implement its corresponding `LlamaTokenizer` module with its pretrained tokenization maps loaded. As the details for tokenization are intricated, to focus more on the main module `InferenceAgent`, we've already built the `LlamaTokenizer` module in `src/modeling/models/llama.py` upon the `HF Tokenizers` library (*See the `HF Tokenizers Tokenizer` in the [Reference](#references) for more details*), which requires two file paths: `tokenizer.json` and `tokenizer_config.json` from the model repo to initialize the pretrained tokenization map, and provides two APIs: `encode` and `decode` with several read-only attributes like `vocab_size`, `pad_token_id`, `bos_token_id`, `eos_token_id`, etc. Please refer to the source code and doc string for the specific usage instructions.
* Now, we are well-prepared to finally implement the `InferenceAgent` module, which provides the APIs described as follows:
    * `__init__(config: InferenceConfig, model: BaseModel, tokenizer: BaseTokenizer)`: initialize the `InferenceAgent` module with the specific inference configurations (*The details about the inference configurations are listed in the table below*), tokenizer and model
    * `set_prompt(prompt_template: PromptTemplate, prompt_type: PromptType = PromptType.SYSTEM)`: set the prompt template with the given prompt type, where the prompt type can be either `PromptType.SYSTEM` or `PromptType.CONTEXT`, as the system prompt or context prompt respectively.
    * `get_prompt(prompt_type: PromptType = PromptType.SYSTEM)`: get the prompt template with the given prompt type.
    * `forward(query: Union[str, List[str]], **kwargs: Optional[Dict[str, str]])`: the forward pass of the `InferenceAgent` module, which takes a single or a batch of user query prompt(s) as the core distinct instructions to ask the model to respond, appended to the end of the complete prompt with the same system prompt and context prompt (*set the template with `set_prompt` and formatted with the given keyword arguments*), and returns a list of dictionaries, each of which should contain every prompt type in `PromptType` enum (*key*) and the corresponding prompt (*value*), which look like a json list as below:
       ```plain
       [
           {
               <PromptType.SYSTEM: "system">: "You are a wise man.\n",
               <PromptType.CONTEXT: "context">: "When someone is in trouble, they think of:\n",
               <PromptType.QUERY: "query">: "The key to life is",
               <PromptType.RESPONSE: "response">: " not to be afraid to take risks and try new things.",
               <PromptType.PROMPT: "prompt">: "You are a wise man.\nWhen someone is in trouble, they think of:\nThe key to life is",
               <PromptType.ALL: "all">: "You are a wise man.\nWhen someone is in trouble, they think of:\nThe key to life is not to be afraid to take risks and try new things."},
       
           {
               <PromptType.SYSTEM: "system">: "You are a programmer.\n",
               <PromptType.CONTEXT: "context">: "One day you were working on a project and something happened:\n",
               <PromptType.QUERY: "query">: "The cat jumped on the keyboard and accidentally",
               <PromptType.RESPONSE: "response">: " pressed the "delete" key. The cat\'s owner was shocked.",
               <PromptType.PROMPT: "prompt">: "You are a programmer.\nOne day you were working on a project and something happened:\nThe cat jumped on the keyboard and accidentally",
               <PromptType.ALL: "all">: "You are a programmer.\nOne day you were working on a project and something happened:\nThe cat jumped on the keyboard and accidentally pressed the "delete" key. The cat\'s owner was shocked."
           }
       ]
       ```
    * `load_generation_config(config_file: str, **extra_configs: Optional[Dict[str, str]])`: a static method to instantiate the `InferenceConfig` module loading the generation configurations from the given file oftenly named `generation_config.json` in the model repo, and just like the `load_config` method for `LlamaModel` you'd have implemented in [task1](./task1.md), you can also provide optional extra configurations using the `**extra_configs` keyword arguments.
* Here's a full table that informs you the details about each configuration in the `InferenceConfig` dataclass as follows (*Note: "Required" means the argument must be provided with non-`None` values during initializationm and "Fixed" means the argument cannot be set during initialization and retain their default values.*):

   | **Config Name**           | **Type**                      | **Default**                    | **Required** | **Fixed** | **Description**                                                                                                                                      |
   |---------------------------|-------------------------------|--------------------------------|--------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
   | `decode_strategy`          | `DecodeStrategy`              | `DecodeStrategy.GREEDY`        | `False`      | `False`   | The strategy used for decoding during inference, such as greedy decoding or sampling.                                                                |
   | `temperature`              | `float`                       | `1.0`                          | `False`      | `False`   | The scaling factor for softmax, passed to the model's forward pass to controls the randomness of the generation; higher values (e.g., >1) lead to more random outputs, lower values (e.g., <1) make the model more deterministic.|
   | `max_new_tokens`           | `int`                         | `None`                         | `True`       | `False`   | The maximum number of new tokens to generate. It is a required field to avoid infinite generation.                                                   |
   | `top_p`                    | `float`                       | `1.0`                          | `False`      | `False`   | The cumulative probability for nucleus sampling, used when `decode_strategy` is set to sampling (*See the nucleus sampling or top-p sampling in the [Reference](#references) for more details*).                                                     |
   | `top_k`                    | `int`                         | `50`                           | `False`      | `False`   | The number of highest probability tokens to consider during sampling. Only used when `decode_strategy` is set to sampling (*See the top-k sampling in the [Reference](#references) for more details*).                            |
   | `streaming`                | `bool`                        | `False`                        | `False`      | `False`    | Whether streaming mode is enabled during inference, used when only one single user query is requested at a time, i.e. `inferred_batch_size == 1`                                                                  |
   | `sampling_seed`            | `Optional[int]`               | `None`                         | `False`      | `False`   | The seed for random number generation in sampling. If `None`, no seed is set.                                                                        |
   | `batch_layout`             | `BatchLayout`                 | `BatchLayout.STACK`            | `False`      | `True`    | Specifies how the input batch is organized. Only stacking is allowed for simplicity.                                                                  |
   | `padding_side`             | `PaddingSide`                 | `PaddingSide.LEFT`             | `False`      | `False`   | Specifies the side (left or right) on which padding is added.                                                                                       |
   | `pad_to_multiple_of`       | `int`                         | `1`                            | `False`      | `False`   | Ensures the padded sequence length of the input token ids is a multiple of this value, helping with efficient batching.                                                     |
   | `truncate_length`          | `Optional[int]`               | `None`                         | `False`      | `False`   | The maximum length to truncate any too-long input token ids to. If `None`, no truncation occurs.                                                                   |
   | `truncate_side`            | `TruncateSide`                | `TruncateSide.RIGHT`           | `False`      | `False`   | Specifies which side to truncate the input token ids from, either left or right.                                                                           |
   | `device`                   | `str`                         | `"cpu"`                        | `False`      | `False`   | The device where the inference (including the input token ids and the model's forward pass) will be executed.                                                                                  |
   

#### Summary

In summary, you should mainly implement this `InferenceAgent` module, which takes a single or a batch of user query prompt(s) with the pre-defined shared system prompt and context prompt, encodes them into input token ids using the tokenizer with the optional `truncation` and `padding` processings, and then passes the input token ids to the model to generate the output token ids, which are finally decoded back to the response prompt(s) and returned in a format of a list of dictionaries.


#### Notice

* If the `inferred_batch_size` is `1`, all of the lists occurring in the whole inference process, no matter the returned tensor-like / string-like list of the `encode` / `decode` method for `BaseTokenizer`, or the returned dict-like list of the `forward` method for `InferenceAgent`, should be of length 1 to keep the consistency, while the input of them can be either a single element or a list to be flexible.
* The input processing order should be `truncation` first and `padding` second. And the `truncation` is appiled to each too-long sequence in the batch individually, while the `padding` is applied to the entire batch, i.e. the padded length (*before considering `pad_to_multiple_of`*) is decided by the longest sequence in the batch.
* The token id used to pad the sequence should be the `tokenizer.pad_id`, but since the tokenizer for `Llama 3.2` model family has not set the `pad_id` explicitly, you can instead use the `tokenizer.eos_id` to pad the sequence if the `padding_side` is `PaddingSide.RIGHT`, or the `tokenizer.bos_id` if the `padding_side` is `PaddingSide.LEFT` (**NOTE: this padding logic will influence the final output to some extent, since we do not support `attention_mask` passing downto the attention module to mask the padded tokens in the previous assignments**).
* The concept of the `top_p` argument is different from the one of the functional `matmul_with_importance` we've implemented in the assignment0, where the latter is simply used as a threshold to mask out the "unimportant" elements with lower probabilities.
* We provide some helpful functions in `./src/utils.py`, including the load/save functions for `json`, `jsonl` and `safetensors` files. You can feel free to use them in your implementation.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to llm inference:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [LangChain Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/)
* [Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
* [Llama Tokenizer Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/tokenization_llama.py#L54)
* [HF Tokenizers Tokenizer](https://huggingface.co/docs/tokenizers/api/tokenizer)
* [HF Transformers GenerationConfig Class](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/text_generation#transformers.GenerationConfig)
* [Nucleus Sampling or Top-p Sampling](https://nn.labml.ai/sampling/nucleus.html)
* [Top-k Sampling](https://nn.labml.ai/sampling/top_k.html)


#### Bonus 🔥

* The system prompt and the context prompt now is just set manually with very simple formatting. However:
    * As for the system prompt, you can build a much more complex system prompt with even hundreds of words as a detailed guide to control the model's output behavior, a.k.a. [Meta Prompting](https://www.prompthub.us/blog/a-complete-guide-to-meta-prompting), either for some very specific tasks or for some highly-staked scenarios requiring ~100% precision.
    * As for the context prompt, sometimes it should be awared of the query prompt itself in practice, like retrieving the highly-related information w.r.t. the query from a specific knowledge base, a.k.a. [Retrieval-Augmented Generation](https://arxiv.org/pdf/2312.10997). RAG is nowadays a very popular and useful approach to extend the knowledge boundary of the llm beyond the one it's learned from the pretrained corpus, which might be out-dated, incomplete or too universal to be domain-specific.
* Therefore, **as for the bonus for task2**, you can tap into your imagination and design your own inference agent for a brainstormed application, such as `smart calculator`, `email assistant`, `football news reporter`, etc, which might need a elaborate and complicated system prompt to describe the task details and control the output format, with a RAG system (*either built by yourself or built upon some awesome existing ones such as [LangChain](https://python.langchain.com/docs/tutorials/rag/), [LightRAG](https://github.com/HKUDS/LightRAG) and [GraphRAG](https://github.com/microsoft/graphrag)*) to retrieve the specific context prompt from the knowledge base of the application for each query.
* **NOTE**: you might get into a lot of trouble like the response crush or the out-of-memory error, and you can feel free to pull out all the stops to tackle these problems, including but not limited to modifying the inner modules, switching to better base model, adjusting the inference configurations and using some external optimization tools, **as long as you still build this application upon the same code base in `src/`**.