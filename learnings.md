# AI Engineering Learnings

## Reliable JSON Extraction with Ollama and Distilabel

### The Problem
When generating synthetic data (e.g., using `UltraFeedback` for preference datasets), LLMs often add unnecessary formatting like Markdown bolding (`**Rating: 5**`) or conversational filler. Standard framework parsers (like those in `distilabel`) often use strict Regex that fails when these extra characters are present, resulting in `None` values in your dataset.

### The Learning: Structured Outputs
Ollama supports **Structured Outputs**, which allows you to enforce a specific JSON schema during the model's generation process. This forces the model to follow the schema mathematically, making it impossible for it to return invalid text or extra formatting.

### Key Implementation Steps
1.  **Extract the Schema**: Most `distilabel` tasks have a `.get_structured_output()` method that returns the JSON schema they expect.
2.  **Pass Schema to Ollama**: The `format` field in Ollama's API can accept a raw JSON schema dictionary, not just the string `"json"`.
3.  **Subclassing for Flexibility**: Standard library implementations (like `OllamaLLM` in older `distilabel` versions) might have strict type hints (e.g., `Literal["", "json"]`) that block passing a full dictionary. Subclassing the LLM class to bypass these checks allows for native schema enforcement.

### Example Pattern
```python
# 1. Get the expected schema from the task
task = UltraFeedback(aspect="overall-rating", llm=base_model)
schema = task.get_structured_output()

# 2. Configure the LLM to use this schema natively
judge_model = OllamaLLM(
    model="llama3.1",
    generation_kwargs={"format": schema}
)
```

### Result
By enforcing the schema at the **generation layer** rather than trying to fix it at the **parsing layer**, we achieved 100% data completeness with zero null values in the `ratings` and `rationales` columns.

---
*Reference: [Ollama Structured Outputs Documentation](https://docs.ollama.com/capabilities/structured-outputs)*

