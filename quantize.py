"""
Quantize bf16 models to 4bit and 8bit using bitsandbytes
Supports quantization formats loadable by vLLM
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantize_model_4bit(model_path: str, output_path: str, use_double_quant: bool = True):
    """
    Quantize model to 4bit
    
    Args:
        model_path: Path to the original model
        output_path: Path to save the quantized model
        use_double_quant: Whether to use double quantization (further compression)
    """
    print(f"Starting 4bit quantization: {model_path}")
    
    # Configure 4bit quantization parameters
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization type
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bf16 for computation
        bnb_4bit_use_double_quant=use_double_quant,  # Double quantization
    )
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Save quantized model
    print(f"Saving 4bit quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("4bit quantization completed!")
    return model, tokenizer


def quantize_model_8bit(model_path: str, output_path: str):
    """
    Quantize model to 8bit
    
    Args:
        model_path: Path to the original model
        output_path: Path to save the quantized model
    """
    print(f"Starting 8bit quantization: {model_path}")
    
    # Configure 8bit quantization parameters
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # Outlier threshold
    )
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Save quantized model
    print(f"Saving 8bit quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("8bit quantization completed!")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Quantize models using bitsandbytes")
    parser.add_argument(
        "--model_path",
        type=str,
        default='/path/to/origin/model',
        help="Path to original model (bf16 or fp32)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/path/to/output/dir',
        help="Output directory for quantized models"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=["4bit", "8bit", "both"],
        default="both",
        help="Quantization type: 4bit, 8bit, or both"
    )
    parser.add_argument(
        "--no_double_quant",
        action="store_true",
        help="Disable double quantization (only effective for 4bit)"
    )
    
    args = parser.parse_args()
    
    # Execute quantization
    if args.quant_type in ["4bit", "both"]:
        output_4bit = os.path.join(args.output_dir, "4bit")
        quantize_model_4bit(
            args.model_path,
            output_4bit,
            use_double_quant=not args.no_double_quant
        )
    
    if args.quant_type in ["8bit", "both"]:
        output_8bit = os.path.join(args.output_dir, "8bit")
        quantize_model_8bit(args.model_path, output_8bit)
    
    print("\n" + "="*50)
    print("Quantization completed!")
    print("="*50)


if __name__ == "__main__":
    main()