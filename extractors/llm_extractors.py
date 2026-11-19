from transformers import AutoTokenizer, AutoModelForCausalLM
#from utils import llm_preprocessing
import ast
import json
from ollama import Client

class BaseExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Initializing {self.model_name}...")

    def _build_chat(self, text, question):
        chat = [
            {"role": "system", "content": "You are an assistant for QA tasks. Use only provided context."},
            {"role": "user", "content": f"Context chunk: {text}"}
        ]
        
        prompt = (
            f"Given the following question: {question}\n"
            "Return the answer only in a Python list format, i.e. ['A','B']. "
            "You must return an empty list if there is no answer."
        )
        chat.append({"role": "user", "content": prompt})
        
        return chat

    def extract(self, text, question):
        raise NotImplementedError("Las clases hijas deben implementar este método")


class QwenExtractor(BaseExtractor):
    def __init__(self, qwen_model = "Qwen/Qwen3-1.7B"):
        super().__init__(qwen_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto"
        )
        self.max_context_tokens = 32768 - 2048

        self.model.generation_config.max_new_tokens = 128
        self.model.generation_config.repetition_penalty = 1.1
        self.model.generation_config.do_sample = False
        
        print(f"{self.model_name} ready for extraction.")

    def extract(self, input_text, question):
        chat = self._build_chat(input_text, question)

        predictions = llm_preprocessing.query_model_return_list(
            self.model, 
            chat, 
            self.tokenizer, 
            local=True
        )
        return predictions



class LlamaExtractor(BaseExtractor):
    def __init__(self, llama_model = "llama3.1:8b"):
        super().__init__(llama_model)
        self.client = Client(timeout=600.0)
        
        print(f"{self.model_name} ready for extraction.")

        
        

    def _build_chat(self, text, question):
        prompt = (
            f"Context chunk: {text}\n\n"
            f"Given the following question: {question}\n"
            "Return the answer STRICTLY in JSON format. The JSON must contain a single key."
            "with a list of strings. Example: {\"key\": [\"A\", \"B\"]}. "
            "If there is no answer, return {\"key\": []}. Do not output any other text or markdown."
        )
        
        chat = [
            {"role": "system", "content": "You are a precise data extraction assistant."},
            {"role": "user", "content": prompt}
        ]
        
        return chat

    def extract(self, text, question):
        chat = self._build_chat(text, question)
        predictions = []
        
        try:
            response = self.client.chat(
                model=self.model_name, 
                messages=chat,
                format='json',
                options={
                    "num_ctx": 16384,     # <-- NUEVO: Ampliamos la memoria de lectura
                    "num_predict": 256,  # <-- Ampliamos un poco el margen de respuesta
                    "temperature": 0.0,  
                    "repeat_penalty": 1.0 # <-- NUEVO: Lo bajamos a 1.0 (neutral) para no romper URLs
                }
            )
            
            str_predictions = response['message']['content'].strip()
            
            parsed_dict = json.loads(str_predictions)
            
            # Extraemos la lista usando la clave que le pedimos ('datasets')
            parsed_list = parsed_dict.get('key', [])

            if isinstance(parsed_list, list):
                if parsed_list == ['']:
                    predictions = []
                else:
                    predictions = parsed_list
            else:
                print(f"Llama has not returned a list but {type(parsed_list).__name__}")
                
        except json.JSONDecodeError:
            print(f"Llama has not returned a legible JSON. It returned: {str_predictions}")
            
        except Exception as e:
            print(f"Error when connecting with Llama: {e}")

        return predictions

class OllamaModelCardAdapter:
    def __init__(self, model_name="qwen2.5:1.5b"):
        self.model_name = model_name
        self.client = Client(timeout=180.0)
        print(f"Ollama Modular CBR Adapter ready with: {self.model_name}")

    def _query_field(self, prompt, system_instruction="You are a precise data extractor."):
        """Función auxiliar rápida para extraer un solo campo en formato JSON."""
        chat = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=chat,
                format='json',
                options={
                    "num_ctx": 8192,
                    "num_predict": 128,  # Respuestas cortas -> Inferencia ultrarrápida
                    "temperature": 0.0,
                    "repeat_penalty": 1.0
                }
            )
            raw = response['message']['content'].strip()
            return json.loads(raw)
        except Exception as e:
            return {}

    def extract_model_name(self, title, abstract, similar_name):
        prompt = (
            f"Example Model Name from similar paper: {similar_name}\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            "Question: What is the main Machine Learning model name proposed in this target paper?\n"
            'Return strictly valid JSON: {"model_name": "Name"}'
        )
        res = self._query_field(prompt)
        return res.get("model_name", "")

    def extract_tasks(self, abstract, similar_tasks):
        prompt = (
            f"Example Tasks from similar paper: {json.dumps(similar_tasks)}\n\n"
            f"Abstract: {abstract}\n\n"
            "Question: What specific Machine Learning tasks are addressed (e.g. Link Prediction, Entity Alignment)?\n"
            'Return strictly valid JSON: {"tasks": ["Task 1", "Task 2"]}'
        )
        res = self._query_field(prompt)
        tasks = res.get("tasks", [])
        return tasks if isinstance(tasks, list) else [str(tasks)]

    def extract_category(self, abstract, similar_category):
        prompt = (
            f"Example Category from similar paper: {similar_category}\n\n"
            f"Abstract: {abstract}\n\n"
            "Question: What is the general research category of the model?\n"
            'Return strictly valid JSON: {"category": "Category Name"}'
        )
        res = self._query_field(prompt)
        return res.get("category", "")

    def extract_repo(self, abstract, full_text, similar_repo):
        # Buscamos en el abstract y las primeras 2000 letras del full_text (donde van las notas al pie)
        snippet = f"{abstract}\n\n{full_text[:2000]}"
        prompt = (
            f"Example Repository from similar paper: {json.dumps(similar_repo)}\n\n"
            f"Text: {snippet}\n\n"
            "Question: What is the official code repository or GitHub URL mentioned in the paper?\n"
            'Return strictly valid JSON: {"repository": ["https://github.com/..."]}. If none found, return {"repository": []}.'
        )
        res = self._query_field(prompt)
        repo = res.get("repository", [])
        return repo if isinstance(repo, list) else ([repo] if repo else [])

    def extract_datasets_and_metrics(self, sections, similar_datasets, similar_metrics):
        # Las secciones de experimentos contienen las tablas con datasets y métricas
        prompt = (
            f"Example Datasets: {json.dumps(similar_datasets)}\n"
            f"Example Metrics: {json.dumps(similar_metrics)}\n\n"
            f"Experimental Sections:\n{sections[:3500]}\n\n"
            "Question: Extract the benchmark datasets and evaluation metrics used in these experiments.\n"
            'Return strictly valid JSON: {"datasets": ["Dataset1", "Dataset2"], "metrics": ["Metric1", "Metric2"]}'
        )
        res = self._query_field(prompt)
        datasets = res.get("datasets", [])
        metrics = res.get("metrics", [])
        
        return (
            datasets if isinstance(datasets, list) else [str(datasets)],
            metrics if isinstance(metrics, list) else [str(metrics)]
        )