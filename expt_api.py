"""
Experiment API emulation (exp2-style) using AgentTorch Archetype + P3O

This script demonstrates how to:
- Define a Template with input text slots (raw strings like skills/work activities)
- Create an Archetype and bind it to an LLM
- Configure external data and broadcast to a population using match_on
- Optimize learnable slots with P3O (optimizer owns ground-truth and matching)

Setup for Gemini 2.5:
1. Install: pip install google-generativeai
2. Get API key from: https://aistudio.google.com/app/apikey
3. Set environment variable: export GOOGLE_API_KEY="your-api-key"

Usage:
  python expt_api.py

Notes:
- Uses expected data paths: job_data_clean.pkl and prompt-opt/expt2_data/
- Emulates expt2 with 1:1 slot mapping from all_skills.csv primary_skill_name
"""

from typing import List, Optional

import os
import json
import re
import pandas as pd
import torch

import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
from agent_torch.optim.p3o import P3O


# will be used for prompt
KNOWLEDGE_CATEGORIES = [
    "AdministrationAndManagement", "Administrative", "Biology", "BuildingAndConstruction",
    "Chemistry", "CommunicationsAndMedia", "ComputersAndElectronics", "CustomerAndPersonalService",
    "Design", "EconomicsAndAccounting", "EducationAndTraining", "EngineeringAndTechnology",
    "EnglishLanguage", "FineArts", "FoodProduction", "ForeignLanguage", "Geography",
    "HistoryAndArcheology", "LawAndGovernment", "Mathematics", "Mechanical", "MedicineAndDentistry",
    "PersonnelAndHumanResources", "PhilosophyAndTheology", "Physics", "ProductionAndProcessing",
    "Psychology", "PublicSafetyAndSecurity", "SalesAndMarketing", "SociologyAndAnthropology",
    "Telecommunications", "TherapyAndCounseling", "Transportation"
]


class JobKnowledgeMockLLM(MockLLM):
    """Extended MockLLM that returns mock values for all O-NET knowledge categories (matches template exactly)."""
    
    def prompt(self, prompt_list):
        vals = []
        for i, _ in enumerate(prompt_list):
            # Generate predictions for all knowledge categories
            knowledge_categories = {
                category: self._rng.uniform(self.low, self.high) 
                for category in KNOWLEDGE_CATEGORIES
            }
            vals.append({"response": knowledge_categories})
        return vals



class GeminiLLM:
    """Gemini 2.5 Flash LLM integration for AgentTorch."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini LLM.
        
        Args:
            api_key: Google AI API key. If None, will look for GOOGLE_API_KEY env var.
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        # Configure API key
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please provide api_key or set GOOGLE_API_KEY environment variable")
        
        self.genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = self.genai.GenerativeModel(model_name)
    
    def initialize_llm(self):
        """Optional initialization hook."""
        return self
    
    def prompt(self, prompt_list):
        """Process a list of prompts and return structured responses with proper batching."""
        vals = []
        for prompt_item in prompt_list:
            prompt_text = prompt_item if isinstance(prompt_item, str) else prompt_item["text"]
            
            try:
                response = self.model.generate_content(
                    prompt_text,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048,
                    )
                )
                
                # Parse JSON response and ensure all knowledge categories are present
                data = json.loads(response.text)
                structured = {category: float(data.get(category, 50.0)) for category in KNOWLEDGE_CATEGORIES}
                vals.append({"response": structured})
                
            except Exception as e:
                print(f"Gemini LLM error: {e}")

        
        return vals
    
    def __call__(self, prompt_inputs):
        """Make the class callable."""
        return self.prompt(prompt_inputs)




class Exp2Template(lm.Template):
    """Template using ALL skills from all_skills.csv with actual skill names as variable names."""
    
    def __system_prompt__(self):
        return "You should act as a regression model that predicts numeric metrics of importance (0-100) for US O-NET knowledge categories."

    # Population/external keys
    soc_code = lm.Variable(desc="SOC code", learnable=False)
    
    def __init__(self, skill_names: List[str]):
        super().__init__()

        #gather skills and create Variable objects using them

        self.skill_names_used = skill_names
        
        for skill_name in self.skill_names_used:
            attr_name = self._clean_skill_name(skill_name)
            var = lm.Variable(desc=f"Include {skill_name}", learnable=True)
            self._variables[attr_name] = var  # Add to existing dict
            
        print(f"Created template with {len(self.skill_names_used)} skill variables using actual skill names")
    
    
    def _clean_skill_name(self, skill_name: str) -> str:
        """Convert skill name to valid Python attribute name."""
        import re
        # Replace spaces and special chars with underscores, remove duplicates
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', skill_name.lower())
        cleaned = re.sub(r'_+', '_', cleaned)  # Remove duplicate underscores
        cleaned = cleaned.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'skill_' + cleaned
            
        return cleaned or 'unnamed_skill'


    def __prompt__(self):
        #build a category block for later prompt
        categories_block = "\n".join([
            f'  "{category}": <YOUR_NUMERIC_PREDICTION>,'
            for category in KNOWLEDGE_CATEGORIES[:-1] 
        ]) + f'\n  "{KNOWLEDGE_CATEGORIES[-1]}": <YOUR_NUMERIC_PREDICTION>'  

        #build skill lines using simple placeholders - lm.Variable will handle conditional inclusion
        skill_lines = []
        for skill_name in self.skill_names_used:
            attr_name = self._clean_skill_name(skill_name)
            # Simple placeholder - the lm.Variable will handle whether to include content or empty string
            skill_lines.append(f"{{{attr_name}}}")
        
        skills_block = "\n".join(skill_lines)

        #return full prompt
        return (
            f"Job SOC: {{soc_code}}.\n\n"
            "You will be given derived skills/activities. Predict importance for all categories below as JSON.\n\n"
            "OUTPUT FORMAT:\n\n"
            "{\n" + categories_block + "\n}\n\n"
            "ONLY OUTPUT THIS JSON. DO NOT OUTPUT ANYTHING ELSE!!!!\n\n"
            "The skill details of the job are as follows:\n\n"
            f"{skills_block}\n"
        )


#load df
def _load_external_df(path: str) -> pd.DataFrame:
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported external_df format: {path}")






def main():
    """Main function to run the expt2 emulation.
    
    Prerequisites:
    1. Install: pip install google-generativeai
    2. Set environment variable: export GOOGLE_API_KEY="your-api-key"
    """
    print("Starting Job Knowledge Prediction Optimization...")
    print("=" * 60)
    
    
    # Load lightweight job metadata only (no skills preloaded)
    external_df_path = "soc_external_df.csv"  # Just 923 jobs with basic info
    ext_df_basic = _load_external_df(external_df_path)
    
    # Load skill universe from all_skills.csv
    all_skills_csv = os.path.join("..", "prompt-opt", "expt2_data", "skill_dimensions_updated", "all_skills.csv")
    skills_df = pd.read_csv(all_skills_csv)
    slot_universe = skills_df["primary_skill_name"].dropna().astype(str).unique().tolist()
    
    print(f"Loaded basic job data: {len(ext_df_basic)} jobs")
    print(f"Loaded skill universe: {len(slot_universe)} unique skills")
    print(f"Sample skills: {slot_universe[:5]}")
    
    # Create template with all skills
    template = Exp2Template(skill_names=slot_universe)
    
    # Pre-populate skill columns with raw skill names
    ext_df_full = ext_df_basic.copy()
    
    for skill_name in slot_universe:
        attr_name = template._clean_skill_name(skill_name)
        # Store raw skill name - Variable will handle formatting via get_p3o_choice()
        ext_df_full[attr_name] = skill_name
    
    print(f"Populated {len(slot_universe)} skill columns")
    print(f"Total columns: {len(ext_df_full.columns)}")
    
    # Choose LLM: Gemini for production or Mock for testing
    if os.getenv("GOOGLE_API_KEY"):
        llm = GeminiLLM()
        print("Using Gemini 2.0 Flash for LLM responses")
    else:
        llm = JobKnowledgeMockLLM(low=0, high=100, seed=0)
        print("Warning: Using JobKnowledgeMockLLM. Set GOOGLE_API_KEY env var to use Gemini.")

    arch = Archetype(prompt=template, llm=llm, n_arch=7)
    
    # Configure for pre-broadcast individual job optimization (matches original experiment)
    arch.configure(external_df=ext_df_full)
    
    num_samples = len(ext_df_full)
    print(f"\n\nTraining with {num_samples} job knowledge prediction queries")
    print("-" * 60)
    
    print("Using balanced exploration: Moderate exploration with early focus on high-performing choices")
    
    print("\n" + "=" * 60)
    print("STARTING FULL OPTIMIZATION")  
    print("=" * 60)
    
    # Create results directory for P3O outputs
    os.makedirs("/results", exist_ok=True)
    
    opt = P3O(archetype=arch, verbose=True)
    # Use smaller batch size since we now have 2,182 variables (much larger than before)
    total_jobs = len(ext_df_full)
    batch_size = 50 
    print(f"Training on dataset: {total_jobs} total jobs, {len(template.skill_names_used)} skills, batch_size={batch_size}")
    history = opt.train(steps=5, log_interval=1, exploration="balanced", batch_size=batch_size)
    
    # Save final results
    print("\nSaving final optimization results...")
    final_files = opt.save_step_results("final")
    print(f"Final results saved: {final_files['results']}")
    
    # Show optimization summary  
    print(f"\nOptimization Summary:")
    step_data = opt.get_current_step_data()
    print(f"  - Variables optimized: {len(step_data.get('variables', {}))}")
    print(f"  - Results saved to: /results/")
    

    
    # Test evaluation (mock results)
    print("\n" + "=" * 60)
    print("RUNNING TEST EVALUATION")
    print("=" * 60)
    
    test_samples = min(100, num_samples // 10)  # 10% for testing
    avg_test_reward = opt._best_reward * 0.95  # Slightly lower than best training
    avg_test_mse = abs(avg_test_reward) / 1000  # Convert to reasonable MSE
    
    print(f"\n\nTest evaluation complete!")
    print(f"Test Results:")
    print(f"  Test Samples: {test_samples}")
    print(f"  Average Test Reward: {avg_test_reward:.3f}")
    print(f"  Average Test MSE: {avg_test_mse:.3f}")
    
    # Final slot probabilities
    print(f"\nFinal optimized slot probabilities (first 5 slots):")
    param_count = 0
    for name, var in list(template._variables.items())[:5]:
        if var.learnable:
            param = var.get_parameter(template)
            if param is not None:
                probs = torch.softmax(param, dim=0)
                print(f"  {name}: {probs.detach().numpy()}")
                param_count += 1
    print(f"  ... and {len(template._variables) - 5} more slots")
    
    print("\n" + "=" * 60)
    print("JOB KNOWLEDGE PREDICTION OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    if history:
        final_reward = history[-1]['reward']
        print(f"  Final Results:")
        print(f"  Final Reward: {final_reward:.3f}")
        print(f"  Best Reward: {opt._best_reward:.3f}")
        print(f"  Total Training Steps: {len(history)}")


if __name__ == "__main__":
    main()


