import torch
from vllm import LLM, SamplingParams

class VllmModel:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """
        Initializes the VllmModel.

        Args:
            model_path (str): The path to the model.
            tensor_parallel_size (int): The tensor parallel size.
        """
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts, max_tokens, stop, temperature):
        """
        Generates text based on the given prompts.
        """
        sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, temperature=temperature)
        outputs = self.llm.generate(prompts, sampling_params)
        generated_texts = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        return generated_texts