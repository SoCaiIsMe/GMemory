import os

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import load_config


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

URL = os.environ["OPENAI_API_BASE"]
KEY = os.environ["OPENAI_API_KEY"]
print('# api url: ', URL)


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens
        
        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5  
        wait_time = 1 

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs
                )

                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                
                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer  

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break 

        return "" 


class VLLMChat(LLM):
    """vLLM本地部署支持"""
    
    def __init__(self, model_name: str, vllm_server_url: str = None):
        super().__init__(model_name=model_name)
        self.vllm_server_url = vllm_server_url or os.getenv("VLLM_SERVER_URL", "http://localhost:8000")
        
        # 确保URL格式正确
        if not self.vllm_server_url.startswith('http'):
            self.vllm_server_url = f'http://{self.vllm_server_url}'
        
        # 从环境变量获取API密钥，如果服务器配置了密钥，客户端必须匹配
        api_key = os.getenv("VLLM_API_KEY", "sk-test-123")  # 默认使用服务器配置的密钥
        
        self.client = OpenAI(
            base_url=f"{self.vllm_server_url}/v1",
            api_key=api_key
        )
        
        print(f"vLLM客户端初始化: 服务器={self.vllm_server_url}, 模型={model_name}, API密钥={api_key}")

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens
        
        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5  
        wait_time = 1 

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs
                )

                answer = response.choices[0].message.content
                
                # vLLM可能不返回usage信息，需要处理
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens += response.usage.prompt_tokens
                    completion_tokens += response.usage.completion_tokens
                else:
                    # 估算token使用量
                    estimated_prompt_tokens = len(str(messages)) // 4
                    estimated_completion_tokens = len(answer) // 4
                    prompt_tokens += estimated_prompt_tokens
                    completion_tokens += estimated_completion_tokens
                
                if answer is None:
                    print("Error: vLLM returned None")
                    continue
                return answer  

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during vLLM API call: {error_message}")
                    break 

        return ""


def get_llm_model(model_name: str, llm_type: str = None, **kwargs) -> LLMCallable:
    """
    LLM 工厂函数，根据类型返回相应的 LLM 实例
    
    Args:
        model_name: 模型名称
        llm_type: LLM 类型 (openai, vllm, huggingface)
        **kwargs: 额外参数
    
    Returns:
        LLMCallable: LLM 实例
    """
    llm_type = llm_type or CONFIG.get("llm_config", {}).get("llm_type", "openai")
    
    if llm_type == "openai":
        return GPTChat(model_name=model_name)
    elif llm_type == "vllm":
        vllm_server_url = kwargs.get("vllm_server_url") or CONFIG.get("vllm_config", {}).get("server_url", "http://localhost:8000")
        return VLLMChat(model_name=model_name, vllm_server_url=vllm_server_url)
    elif llm_type == "huggingface":
        # 可以添加 HuggingFaceChat 实现
        return GPTChat(model_name=model_name)  # 暂时回退到 OpenAI 格式
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000