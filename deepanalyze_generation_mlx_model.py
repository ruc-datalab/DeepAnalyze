# 1. Importer les bonnes librairies
import mlx.core as mx
from mlx_lm import load, generate

class MlxModel:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """
        Initializes the MlxModel for Apple Silicon.

        Args:
            model_path (str): The path or HF repo ID for the MLX-compatible model.
            tensor_parallel_size (int): Ignored, MLX handles parallelism differently.
        """
        # 2. Charger le modèle et le tokenizer avec mlx-lm
        # mlx-lm est intelligent. Il va chercher un modèle compatible MLX.
        # Si tu as un modèle local, model_path doit être le chemin vers son dossier.
        print(f"Loading MLX model from: {model_path}")
        self.model, self.tokenizer = load(model_path)

    def generate(self, prompts, max_tokens, stop, temperature):
        """
        Generates text based on the given prompts using MLX.
        
        Args:
            prompts (list[str]): A list of prompts to generate from.
            max_tokens (int): The maximum number of tokens to generate.
            stop (list[str]): List of stop sequences. MLX-LM uses the tokenizer's stop tokens.
                               We'll pass the first one to the tokenizer if needed.
            temperature (float): The sampling temperature.
        """
        # 3. Adapter la logique de génération
        # mlx_lm.generate est plus direct. Il prend un seul prompt à la fois.
        # On doit donc faire une boucle sur les prompts.
        generated_texts = []
        for prompt in prompts:
            # Note: mlx-lm's `generate` n'a pas de paramètre `stop` direct comme vLLM.
            # On peut ajouter des stop words au tokenizer si c'est crucial.
            # Pour l'instant, on se concentre sur la génération simple.
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                max_tokens=max_tokens, 
                temp=temperature
            )
            generated_texts.append(response)
        
        return generated_texts