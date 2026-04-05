import json
import logging
import asyncio
import os
import re
import time
from typing import Dict, Any, List
# We'll use try-except for imports to not fail hard if packages aren't installed correctly yet
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

logger = logging.getLogger("inframind.llm_provider")

class LLMProviderService:
    """Manages concurrent API calls to ChatGPT, DeepSeek, and Google Gemini."""

    def __init__(self, keys_path: str = "api_keys.json"):
        self.keys = self._load_keys(keys_path)
        
        self.openai_client = None
        self.deepseek_client = None
        self.gemini_client = None

        if self.keys.get("openai_api_key") and AsyncOpenAI:
            self.openai_client = AsyncOpenAI(api_key=self.keys["openai_api_key"])
            
        if self.keys.get("deepseek_api_key") and AsyncOpenAI:
            self.deepseek_client = AsyncOpenAI(
                api_key=self.keys["deepseek_api_key"],
                base_url="https://api.deepseek.com"
            )
            
        if self.keys.get("gemini_api_key") and genai:
            self.gemini_client = genai.Client(api_key=self.keys["gemini_api_key"])

    def _load_keys(self, path: str) -> Dict[str, str]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return {}
        
    def _extract_terraform(self, response_text: str) -> str:
        # Strip markdown ticks
        if "```terraform" in response_text:
            return response_text.split("```terraform")[1].split("```")[0].strip()
        elif "```" in response_text:
            return response_text.split("```")[1].strip()
        return response_text.strip()

    def _extract_config(self, tf_code: str) -> Dict[str, Any]:
        """A simple heuristic tool to guess config from tf for simulation."""
        config = {
            "api_gateway": {"replicas": 2, "cpu_millicores": 500, "memory_mb": 512},
            "auth": {"replicas": 2, "cpu_millicores": 1000, "memory_mb": 1024},
            "catalog": {"replicas": 3, "cpu_millicores": 1000, "memory_mb": 2048},
        }
        # Very basic regex scanning
        # If we see many Fargate profiles or high desired_count, increase reps
        if "desired_count" in tf_code:
            matches = re.findall(r"desired_count\s*=\s*(\d+)", tf_code)
            if matches:
                 try:
                     val = int(matches[0])
                     config["api_gateway"]["replicas"] = max(1, val)
                     # distribute a bit
                     if len(matches) > 1:
                         config["auth"]["replicas"] = max(1, int(matches[1]))
                 except: pass
                 
        if "cpu" in tf_code:
            matches = re.findall(r"cpu\s*=\s*[\"']?(\d+)[\"']?", tf_code)
            if matches:
                 try:
                     config["api_gateway"]["cpu_millicores"] = int(matches[0])
                 except: pass
                 
        if "memory" in tf_code:
            matches = re.findall(r"memory\s*=\s*[\"']?(\d+)[\"']?", tf_code)
            if matches:
                 try:
                     config["api_gateway"]["memory_mb"] = int(matches[0])
                 except: pass
        return config

    async def _call_openai(self, prompt: str, target: str) -> Dict[str, Any]:
        if not self.openai_client:
            return self._mock_openai(prompt)
            
        system = f"You are an infrastructure expert. Generate raw {target} code for AWS. No markdown wraps, no explanations."
        try:
            rs = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                ),
                timeout=25.0
            )
            code = self._extract_terraform(rs.choices[0].message.content)
            return {"provider": "ChatGPT (GPT-4o)", "code": code, "config": self._extract_config(code)}
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return self._mock_openai(prompt)

    async def _call_deepseek(self, prompt: str, target: str) -> Dict[str, Any]:
        if not self.deepseek_client:
            return self._mock_deepseek(prompt)
            
        system = f"You are an infrastructure expert. Generate raw {target} code for AWS. No markdown wraps, no explanations."
        try:
            rs = await asyncio.wait_for(
                self.deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                ),
                timeout=25.0
            )
            code = self._extract_terraform(rs.choices[0].message.content)
            return {"provider": "DeepSeek Chat", "code": code, "config": self._extract_config(code)}
        except Exception as e:
            logger.error(f"DeepSeek call failed: {e}")
            return self._mock_deepseek(prompt)

    async def _call_gemini(self, prompt: str, target: str) -> Dict[str, Any]:
        if not self.gemini_client:
            return self._mock_gemini(prompt)
            
        instruction = f"You are an infrastructure expert. Generate raw {target} code for AWS. No markdown wraps, no explanations. Request: {prompt}"
        try:
            # Gemini python sdk does not currently support true asyncio well, we'll wrap blocking call in asyncio.to_thread
            def blocking_call():
                return self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=instruction
                )
            rs = await asyncio.wait_for(
                asyncio.to_thread(blocking_call),
                timeout=25.0
            )
            code = self._extract_terraform(rs.text)
            return {"provider": "Gemini 2.5 Flash", "code": code, "config": self._extract_config(code)}
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            return self._mock_gemini(prompt)

    async def generate_all(self, prompt: str, target: str) -> List[Dict[str, Any]]:
        # Run all three concurrently
        results = await asyncio.gather(
            self._call_openai(prompt, target),
            self._call_deepseek(prompt, target),
            self._call_gemini(prompt, target)
        )
        return list(results)
        
    # --- Mocks ---
    def _mock_openai(self, prompt: str):
        code = f"""# ChatGPT (Fallback) - Standard ECS Cluster
provider "aws" {{ region = "us-east-1" }}
resource "aws_vpc" "main" {{ cidr_block = "10.0.0.0/16" }}
resource "aws_security_group" "open" {{
  vpc_id = aws_vpc.main.id
  ingress {{ from_port = 80; to_port = 80; protocol = "tcp"; cidr_blocks = ["0.0.0.0/0"] }}
}}
resource "aws_ecs_service" "app" {{ desired_count = 5 }} # moderate replicas
"""
        return {"provider": "ChatGPT (GPT-4o)", "code": code, "config": {"api_gateway": {"replicas": 5, "cpu_millicores": 500, "memory_mb": 512}}}

    def _mock_deepseek(self, prompt: str):
        code = f"""# DeepSeek (Fallback) - Cost-Optimized Fargate
provider "aws" {{ region = "us-east-1" }}
resource "aws_vpc" "main" {{ cidr_block = "10.0.0.0/16" }}
resource "aws_security_group" "strict" {{
  vpc_id = aws_vpc.main.id
  ingress {{ from_port = 443; to_port = 443; protocol = "tcp"; cidr_blocks = ["10.0.0.0/8"] }}
}}
# Fargate uses less CPU per replica but runs leaner
resource "aws_ecs_service" "app" {{ desired_count = 2; launch_type = "FARGATE" }}
"""
        return {"provider": "DeepSeek Chat", "code": code, "config": {"api_gateway": {"replicas": 2, "cpu_millicores": 256, "memory_mb": 256}}}

    def _mock_gemini(self, prompt: str):
        code = f"""# Gemini 2.5 (Fallback) - High Availability Multi-AZ
provider "aws" {{ region = "us-west-2" }}
resource "aws_vpc" "ha" {{ cidr_block = "10.1.0.0/16" }}
resource "aws_subnet" "sub1" {{ vpc_id = aws_vpc.ha.id }}
resource "aws_ebs_volume" "data" {{ encrypted = true; size = 100 }}
# Overprovisioned for HA
resource "aws_ecs_service" "app" {{ desired_count = 8 }}
"""
        return {"provider": "Gemini 2.5 Flash", "code": code, "config": {"api_gateway": {"replicas": 8, "cpu_millicores": 1000, "memory_mb": 2048}}}
