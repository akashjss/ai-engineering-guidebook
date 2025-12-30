import pandas as pd
from distilabel.models.llms import OllamaLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, GroupColumns
from distilabel.steps.tasks import TextGeneration, UltraFeedback

model1 = OllamaLLM(model="llama3.1:latest", timeout=1000)
model2 = OllamaLLM(model="ministral-3:8b", timeout=1000)

# Custom Ollama LLM to support native JSON schema in 'format' field
# This allows us to use Ollama's native structured output as suggested
class OllamaSchemaLLM(OllamaLLM):
    async def agenerate(self, input, format=None, options=None, keep_alive=None):
        # We call private methods to bypass the strict Literal["", "json"] 
        # validation on the standard agenerate method.
        if self.tokenizer_id is None:
            completion = await self._generate_chat_completion(
                input, format, options, keep_alive
            )
            text = completion["message"]["content"]
        else:
            completion = await self._generate_with_text_generation(
                input, format, options, keep_alive
            )
            text = completion.response
        
        from distilabel.models.llms.utils import prepare_output
        return prepare_output([text], **self._get_llm_statistics(completion))

# Get the schema required by UltraFeedback for "overall-rating"
ultra_schema = UltraFeedback(aspect="overall-rating", llm=model1).get_structured_output()

# Judge model with native Ollama structured output
judge_model = OllamaSchemaLLM(
    model="llama3.1:latest", 
    timeout=1000,
    generation_kwargs={
        "format": ultra_schema 
    }
)

with Pipeline(name="preference-datagen-llama3-v10") as pipeline:

    # 1. Load the dataset with prompts
    load_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings = {"prompt": "instruction"},
        )

    # Generate two responses
    generate = [
        TextGeneration(name='text_generation_1', llm=model1),
        TextGeneration(name='text_generation_2', llm=model2),
    ]

    # Combine the responses
    combine = GroupColumns(
        name="combine_columns",
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"]
    )

    # 3. Evaluate the responses
    evaluate = UltraFeedback(
        name="evaluate_responses",
        aspect="overall-rating",
        llm=judge_model,
        use_default_structured_output=True,
        )

    # Run the pipeline
    load_dataset >> generate >> combine >> evaluate

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            }
        }
    )

    # Convert to pandas and show the data
    # We need to iterate through the Distiset AND the DatasetDict
    for name, dataset_dict in distiset.items():
        for split_name, dataset in dataset_dict.items():
            print(f"\n--- Data for: {name} (split: {split_name}) ---")
            df = dataset.to_pandas()
            print(df.head())
            
            # Save to CSV
            filename = f"synthetic_data_{name}_{split_name}.csv"
            df.to_csv(filename, index=False)
            print(f"\nSaved full data to {filename}")
