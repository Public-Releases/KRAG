import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from authentication import hf_token
from typing import Optional, List, Dict

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class LocalLLM:
    """Local LLM inference using Hugging Face transformers"""
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "auto",
        load_in_8bit: bool = True,
        max_memory: Optional[Dict] = None
    ):
        
        local_model_paths = {
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "./models/llama-3.1-8b-instruct",
            "aaditya/Llama3-OpenBioLLM-8B": "./models/llama3-openbiollm-8b",
            "meta-llama/Meta-Llama-3.1-70B-Instruct": "./models/llama-3.1-70b-instruct",
        }
        
        # Check if this model has a local version
        local_model_dir = local_model_paths.get(model_name)
        if local_model_dir and os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            print(f"Found local model at: {os.path.abspath(local_model_dir)}")
            model_name = local_model_dir  # Use local path instead
            print(f"  Loading from disk (no download needed)")
        elif local_model_dir:
            print(f"Local model not found at {local_model_dir}")
            print(f"  Will download from: {model_name}")
            print(f"  Tip: Run 'python download_openbiollm.py' to download once and reuse")
        else:
            print(f"Model: {model_name}")
            print(f"  Will download from HuggingFace (no local cache configured)")
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\nInitializing {model_name}...")
        print(f"Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if load_in_8bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                token=hf_token
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                low_cpu_mem_usage=True,
                token=hf_token
            )
            if self.device == "cuda":
                self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully!")
        
    def format_llama_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompt for Llama 3.1 instruct format"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        full_prompt = self.format_llama_prompt(system_prompt, user_prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        del inputs, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return generated_text.strip()
    
    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_llm_instance = None

def get_local_llm(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> LocalLLM:
    """Get or create singleton LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM(model_name=model_name)
    return _llm_instance


def generate_diagnosis_with_local_llm(
    system_prompt: str,
    user_prompt: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 512
) -> str:
    llm = get_local_llm(model_name)
    return llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )