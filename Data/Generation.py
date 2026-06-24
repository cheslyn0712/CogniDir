"""
Cognitive Malicious Comment Generation Script (CogniDir)

This script generates cognitive malicious comments using three LLM models following
the "understand-then-generate" prompting strategy.
"""

import os
import sys
import re
import json
import random
import argparse
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Paths relative to Data/ directory (run this script from Data/)
MODEL_PATHS = {
    "Gemma_2B": os.path.join("..", "LLM", "gemma-2-2b-it"),
    "Mistral_7B": os.path.join("..", "LLM", "Mistral-7B-Instruct-v0___3"),
    "Qwen_32B": os.path.join("..", "LLM", "Qwen2.5-32B"),
}

# Input seed data files
INPUT_FILES = [
    os.path.join("Seed", "Weibo16.json"),
    os.path.join("Seed", "Weibo20.json"),
    os.path.join("Seed", "Rumoureval.json"),
]

# Output directory
OUTPUT_DIR = "Processed"


def load_model_and_tokenizer(model_name: str, model_path: str):
    """Load LLM model and tokenizer."""
    print(f"Loading {model_name} from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    
    return tokenizer, model


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Apply chat template to messages."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            sys_msg = ""
            usr_msg = ""
            for m in messages:
                if m.get("role") == "system":
                    sys_msg = m.get("content", "")
                elif m.get("role") == "user":
                    usr_msg = m.get("content", "")
            return f"{sys_msg}\n\n{usr_msg}\n"
    return messages[-1].get("content", "")


def generate_with_model(tokenizer, model, prompt: str, max_tokens: int = 128, 
                        temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate text using the model."""
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    text = apply_chat_template(tokenizer, messages)
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items() if k != "token_type_ids"}
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    in_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][in_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def clean_generated_text(text: str, prompt: str = "") -> str:
    """Clean generated text by removing prompt and assistant markers."""
    if not text:
        return ""
    
    if prompt:
        prompt_lower = prompt.lower()
        text_lower = text.lower()
        prompt_idx = text_lower.find(prompt_lower)
        if prompt_idx >= 0:
            text = text[prompt_idx + len(prompt):].strip()
    
    assistant_patterns = [
        r'^assistant\s*:?\s*',
        r'^\[assistant\]\s*',
        r'^<assistant>\s*',
        r'^here\s+is\s+.*?:?\s*',
        r'^the\s+comment\s+is\s*:?\s*',
        r'^comment\s*:?\s*',
    ]
    
    text = text.strip()
    for pattern in assistant_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    
    instruction_patterns = [
        r'^(?:please\s+)?(?:generate|create|write|output|provide)\s+.*?:?\s*',
        r'^(?:the\s+)?(?:optimized|final|refined)\s+comment\s+is\s*:?\s*',
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    
    structured_patterns = [
        r'(?:comment|评论)\s*:?\s*(.+?)(?:\n\n|\Z)',
        r'\[comment\]\s*(.+?)(?:\n\n|\Z)',
        r'<comment>(.+?)</comment>',
    ]
    
    for pattern in structured_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted and len(extracted) > 5:
                text = extracted
                break
    
    text = re.sub(r'\n\s*\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    if len(text) < 5:
        return ""
    
    return text


class AttackGenerator:
    """Base class for attack generators."""
    
    def __init__(self, tokenizer, model, attack_type: str, intensity: str):
        self.tokenizer = tokenizer
        self.model = model
        self.attack_type = attack_type
        self.intensity = intensity
    
    def generate_understanding_prompt(self, content: str) -> str:
        """Step 1: Understanding phase prompt."""
        if self.attack_type == "fact_distortion":
            return f"""Analyze this news content and identify factual claims that could be manipulated:
Content: {content[:500]}

Identify:
1. Key factual statements
2. Numbers, dates, or statistics mentioned
3. Sources or authorities cited
4. Potential areas where facts could be distorted"""
        
        elif self.attack_type == "logical_confusion":
            return f"""Analyze this news content and identify logical structures:
Content: {content[:500]}

Identify:
1. Cause-effect relationships
2. Generalizations or comparisons
3. Arguments or reasoning patterns
4. Potential logical fallacies that could be introduced"""
        
        elif self.attack_type == "emotional_manipulation":
            return f"""Analyze this news content and identify emotional elements:
Content: {content[:500]}

Identify:
1. Emotional appeals or sentiment
2. Fear-inducing elements
3. Authority references
4. Potential areas for emotional manipulation"""
        
        return ""
    
    def generate_attack_prompt(self, content: str, understanding: str, target_verdict: str) -> str:
        """Step 2: Generation phase prompt."""
        intensity_guidance = {
            "base": "Use subtle and plausible manipulation",
            "medium": "Use moderate manipulation with some exaggeration",
            "strong": "Use strong manipulation with urgent language and emotional appeals"
        }
        
        attack_guidance = {
            "fact_distortion": {
                "base": "Subtly distort or misrepresent factual information",
                "medium": "Introduce false claims with some plausibility",
                "strong": "Make bold false claims with fabricated evidence"
            },
            "logical_confusion": {
                "base": "Introduce subtle logical fallacies",
                "medium": "Use moderate logical fallacies like false causality",
                "strong": "Employ strong logical fallacies with hasty generalizations"
            },
            "emotional_manipulation": {
                "base": "Add mild emotional appeals",
                "medium": "Use moderate fear-mongering or emotional manipulation",
                "strong": "Use urgent warnings, conspiracy language, and strong emotional appeals"
            }
        }
        
        guidance = attack_guidance[self.attack_type][self.intensity]
        intensity_desc = intensity_guidance[self.intensity]
        
        prompt = f"""Based on your understanding, generate a comment that leads readers to believe the content is {target_verdict.upper()}.

Understanding: {understanding[:300]}

Content: {content[:500]}

Requirements:
1. Attack type: {self.attack_type.replace('_', ' ').title()}
2. Intensity: {intensity_desc}
3. {guidance}
4. Length: 20-40 words
5. Use natural language
6. Include specific details from the content

Generate the adversarial comment:"""
        
        return prompt
    
    def generate(self, content: str, target_verdict: str) -> str:
        """Generate adversarial comment using understand-then-generate strategy."""
        # Step 1: Understanding phase
        understanding_prompt = self.generate_understanding_prompt(content)
        understanding = generate_with_model(
            self.tokenizer, self.model, understanding_prompt, 
            max_tokens=100, temperature=0.5
        )
        
        # Step 2: Generation phase
        attack_prompt = self.generate_attack_prompt(content, understanding, target_verdict)
        comment = generate_with_model(
            self.tokenizer, self.model, attack_prompt,
            max_tokens=150, temperature=0.7 if self.intensity == "base" else 0.8
        )
        
        cleaned = clean_generated_text(comment, attack_prompt)
        return cleaned if cleaned else comment


def process_dataset(input_file: str, output_dir: str, models: Dict[str, Any], 
                   ratio: float = 1.0, seed: int = 42):
    """Process a dataset file and generate adversarial comments."""
    print(f"\nProcessing {os.path.basename(input_file)}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"{input_file} top level must be a list")
    
    n = len(data)
    selected_n = int(n * ratio) if ratio < 1.0 else n
    
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    selected_indices = set(indices[:selected_n])
    
    print(f"Total samples: {n}, Selected: {selected_n}")
    
    # Attack type mapping: rationale_4,7,10 -> fact_distortion; 5,8,11 -> logical_confusion; 6,9,12 -> emotional_manipulation
    attack_mapping = {
        4: ("fact_distortion", "base"),
        5: ("logical_confusion", "base"),
        6: ("emotional_manipulation", "base"),
        7: ("fact_distortion", "medium"),
        8: ("logical_confusion", "medium"),
        9: ("emotional_manipulation", "medium"),
        10: ("fact_distortion", "strong"),
        11: ("logical_confusion", "strong"),
        12: ("emotional_manipulation", "strong"),
    }
    
    # Generate comments for each selected sample
    for idx, item in enumerate(data):
        if idx not in selected_indices:
            continue
        
        if idx % 10 == 0:
            print(f"Processing {idx+1}/{selected_n}...")
        
        content = str(item.get("content", ""))
        true_label = item.get("label", "").lower()
        target_verdict = "fake" if true_label == "real" else "real"
        
        # Generate rationale_4 to rationale_12 using three models
        # rationale_1 to rationale_3 are preserved from seed data
        for rationale_idx in range(4, 13):
            attack_type, intensity = attack_mapping[rationale_idx]
            
            # Use different models for diversity (rotate through three models)
            model_choice = (rationale_idx - 4) % 3
            if model_choice == 0:
                model_name = "Gemma_2B"
            elif model_choice == 1:
                model_name = "Mistral_7B"
            else:
                model_name = "Qwen_32B"
            
            if model_name not in models:
                print(f"Warning: {model_name} not loaded. Skipping rationale_{rationale_idx}.")
                continue
            
            tokenizer, model = models[model_name]
            generator = AttackGenerator(tokenizer, model, attack_type, intensity)
            
            try:
                generated_comment = generator.generate(content, target_verdict)
                if generated_comment and len(generated_comment.strip()) >= 5:
                    item[f"rationale_{rationale_idx}"] = generated_comment
                else:
                    # Keep original rationale if generation fails or is too short
                    if f"rationale_{rationale_idx}" not in item:
                        item[f"rationale_{rationale_idx}"] = ""
            except Exception as e:
                print(f"Error generating rationale_{rationale_idx} for sample {idx}: {e}")
                # Keep original if generation fails
                if f"rationale_{rationale_idx}" not in item:
                    item[f"rationale_{rationale_idx}"] = ""
    
    # Save processed data
    dataset_name = os.path.basename(input_file).replace(".json", "")
    output_subdir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Split into train/val/test (70/15/15)
    random.seed(seed)
    random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    test_data = data[n_train+n_val:]
    
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = os.path.join(output_subdir, f"{split_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {split_name}.json: {len(split_data)} samples")
    
    print(f"Completed processing {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial comments using LLMs")
    parser.add_argument("--ratio", type=float, default=1.0, 
                       help="Ratio of samples to process (default: 1.0 = 100%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to process (optional)")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all three models
    print("Loading LLM models...")
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping {model_name}.")
            continue
        try:
            tokenizer, model = load_model_and_tokenizer(model_name, model_path)
            models[model_name] = (tokenizer, model)
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
    
    if len(models) == 0:
        raise RuntimeError("No models loaded. Please check model paths.")
    
    # Process datasets
    input_files = INPUT_FILES
    if args.dataset:
        dataset_file = os.path.join("Seed", f"{args.dataset}.json")
        if os.path.exists(dataset_file):
            input_files = [dataset_file]
        else:
            print(f"Warning: {dataset_file} not found. Processing all datasets.")
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found. Skipping.")
            continue
        
        try:
            process_dataset(input_file, OUTPUT_DIR, models, args.ratio, args.seed)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print("\nGeneration completed.")


if __name__ == "__main__":
    main()
