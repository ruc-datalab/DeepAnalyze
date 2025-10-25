#!/usr/bin/env python3
"""
DeepAnalyze with OpenRouter - Complete Example
"""
import os
from deepanalyze import DeepAnalyzeOpenRouter

# ==============================================================================
# Configuration
# ==============================================================================
# Method 1: Use environment variables (recommended)
# Set in .env file or export:
#   export OPENROUTER_API_KEY=sk-or-v1-xxxxx
#   export OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Method 2: Pass directly to constructor
API_KEY = "your-openrouter-api-key-here"  # Replace with your key
MODEL = "anthropic/claude-3.5-sonnet"  # Or any OpenRouter model

# ==============================================================================
# Example 1: Simple Data Analysis
# ==============================================================================
def example_simple_analysis():
    """Simple CSV analysis example"""
    print("=" * 80)
    print("Example 1: Simple Data Analysis")
    print("=" * 80)

    # Initialize DeepAnalyze with OpenRouter
    deepanalyze = DeepAnalyzeOpenRouter(
        model_name=MODEL,
        api_key=API_KEY,  # or omit to use OPENROUTER_API_KEY env var
    )

    # Prepare prompt
    prompt = """# Instruction
Analyze this sales dataset and calculate total revenue by product category.

# Data
File 1: {"name": "sales.csv", "size": "5.2KB"}

The file contains columns: date, product, category, revenue, quantity
"""

    # Set workspace (create a test directory)
    workspace = "./workspace/example1"
    os.makedirs(workspace, exist_ok=True)

    # Create sample data
    import csv
    sample_data = [
        ["date", "product", "category", "revenue", "quantity"],
        ["2024-01-01", "Laptop", "Electronics", "1200", "2"],
        ["2024-01-02", "Mouse", "Electronics", "25", "10"],
        ["2024-01-03", "Desk", "Furniture", "300", "1"],
        ["2024-01-04", "Chair", "Furniture", "150", "2"],
    ]
    with open(f"{workspace}/sales.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)

    # Generate analysis
    print("\n🚀 Generating analysis...")
    result = deepanalyze.generate(prompt, workspace=workspace)

    # Print result
    print("\n📊 Result:")
    print(result["reasoning"])
    print("\n" + "=" * 80)


# ==============================================================================
# Example 2: Multi-file Data Research
# ==============================================================================
def example_multi_file_research():
    """Multi-file comprehensive analysis"""
    print("\n" + "=" * 80)
    print("Example 2: Multi-file Data Research")
    print("=" * 80)

    deepanalyze = DeepAnalyzeOpenRouter(
        model_name=MODEL,
        api_key=API_KEY,
    )

    prompt = """# Instruction
Generate a comprehensive data science report analyzing customer behavior across multiple datasets.

# Data
File 1: {"name": "customers.csv", "size": "8.5KB"}
File 2: {"name": "orders.csv", "size": "12.3KB"}
File 3: {"name": "products.csv", "size": "4.7KB"}

Requirements:
1. Analyze customer purchase patterns
2. Identify top-selling products
3. Create visualizations
4. Provide actionable insights
"""

    workspace = "./workspace/example2"
    os.makedirs(workspace, exist_ok=True)

    # Create sample datasets
    import csv

    # customers.csv
    with open(f"{workspace}/customers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["customer_id", "name", "age", "city"],
            ["1", "Alice", "28", "New York"],
            ["2", "Bob", "35", "Los Angeles"],
            ["3", "Charlie", "42", "Chicago"],
        ])

    # orders.csv
    with open(f"{workspace}/orders.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["order_id", "customer_id", "product_id", "amount", "date"],
            ["1", "1", "101", "1200", "2024-01-01"],
            ["2", "1", "102", "25", "2024-01-02"],
            ["3", "2", "103", "300", "2024-01-03"],
            ["4", "3", "101", "1200", "2024-01-04"],
        ])

    # products.csv
    with open(f"{workspace}/products.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["product_id", "name", "category", "price"],
            ["101", "Laptop", "Electronics", "1200"],
            ["102", "Mouse", "Electronics", "25"],
            ["103", "Desk", "Furniture", "300"],
        ])

    print("\n🚀 Generating comprehensive report...")
    result = deepanalyze.generate(prompt, workspace=workspace)

    print("\n📊 Result:")
    print(result["reasoning"])
    print("\n" + "=" * 80)


# ==============================================================================
# Example 3: Comparing Different Models
# ==============================================================================
def example_model_comparison():
    """Compare results from different models"""
    print("\n" + "=" * 80)
    print("Example 3: Comparing Different Models")
    print("=" * 80)

    prompt = """# Instruction
Calculate the sum of numbers from 1 to 10 and explain the formula.

Show your code in <Code> tags.
"""

    workspace = "./workspace/example3"
    os.makedirs(workspace, exist_ok=True)

    # Test different models
    models = [
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-r1",
        "openai/gpt-4o",
    ]

    for model in models:
        print(f"\n🤖 Testing model: {model}")
        print("-" * 80)

        try:
            deepanalyze = DeepAnalyzeOpenRouter(
                model_name=model,
                api_key=API_KEY,
            )

            result = deepanalyze.generate(prompt, workspace=workspace)
            print(f"✅ Success!")
            print(f"Response length: {len(result['reasoning'])} chars")
            print(f"Preview: {result['reasoning'][:200]}...")

        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n" + "=" * 80)


# ==============================================================================
# Example 4: Custom Configuration
# ==============================================================================
def example_custom_config():
    """Advanced configuration example"""
    print("\n" + "=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80)

    # Custom configuration
    deepanalyze = DeepAnalyzeOpenRouter(
        model_name="anthropic/claude-3.5-sonnet",
        api_key=API_KEY,
        max_rounds=10,  # Limit reasoning rounds
        site_url="https://your-app.com",
        app_name="MyDataApp",
    )

    prompt = """# Instruction
Create a simple Python function to calculate factorial and test it with n=5.
"""

    workspace = "./workspace/example4"
    os.makedirs(workspace, exist_ok=True)

    # Custom generation parameters
    result = deepanalyze.generate(
        prompt,
        workspace=workspace,
        temperature=0.3,  # Lower temperature for more deterministic output
        max_tokens=2000,  # Limit response length
        top_p=0.9,
    )

    print("\n📊 Result:")
    print(result["reasoning"])
    print("\n" + "=" * 80)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    print("\n🌟 DeepAnalyze with OpenRouter - Examples\n")

    # Check if API key is set
    if API_KEY == "your-openrouter-api-key-here" and not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set your OpenRouter API key!")
        print("\nOption 1: Edit this file and replace API_KEY")
        print("Option 2: Set environment variable:")
        print("  export OPENROUTER_API_KEY=sk-or-v1-xxxxx")
        print("\nGet your API key from: https://openrouter.ai/keys\n")
        exit(1)

    # Run examples
    try:
        example_simple_analysis()
        # example_multi_file_research()
        # example_model_comparison()
        # example_custom_config()

        print("\n✅ All examples completed!")
        print("\nNext steps:")
        print("  - Uncomment other examples in main()")
        print("  - Try different models")
        print("  - Check the workspace/ directory for generated files")
        print("  - Read OPENROUTER_GUIDE.md for more details")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
