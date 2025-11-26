import os
import dspy
from dotenv import load_dotenv
from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
from agent.reach_agent import ReachEstimatorAgent
from agent.brand_lift_agent import BrandLiftAgent
import json


load_dotenv()

dspy.configure(
    lm=dspy.LM(
        model = "ollama/phi3",
        # model="ollama/phi4-reasoning",
        max_tokens=4096,
        temperature=0.2,
    )
)

prompts_path = os.path.join("prompts", "sample_prompts.json")

with open(prompts_path, "r") as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} sample prompts.")

def main():
    brand_agent = BrandAgent()
    category_agent = CategoryAgent()

    for prompt in prompts:
        # Extract brands (returns dict with 'brands' and 'confidence')
        brand_result = brand_agent.extract_brand(prompt)
        brands = brand_result.get("brands", []) if isinstance(brand_result, dict) else brand_result
        
        # Extract category (returns dict with 'category' and 'confidence')
        category_result = category_agent.extract_category(prompt)
        category = category_result.get("category", "") if isinstance(category_result, dict) else category_result
        
        print(f"\nPrompt: {prompt}")
        print(f"→ Brands: {brands}")
        if isinstance(brand_result, dict) and "confidence" in brand_result:
            print(f"  Confidence: {brand_result['confidence'] * 100:.1f}%")
        print(f"→ Category: {category}")
        if isinstance(category_result, dict) and "confidence" in category_result:
            print(f"  Confidence: {category_result['confidence'] * 100:.1f}%")

if __name__ == "__main__":
    main()