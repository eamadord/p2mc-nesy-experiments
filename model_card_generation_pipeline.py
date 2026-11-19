from pathlib import Path
from datetime import datetime
import joblib
import json
import time
import copy
import chromadb
from chromadb.config import Settings


import os
from extractors.gliner_dataset_extractor import GlinerDatasetExtractor
from extractors.gliner_metric_extractor import GlinerMetricExtractor
from extractors.qwen_metric_extractor import QwenMetricExtractor
from extractors.llama_dataset_extractor import LlamaDatasetExtractor
from extractors.qwen_dataset_extractor import QwenDatasetExtractor
from extractors.gliner_extractor import GlinerExtractor
from extractors.llm_extractors import LlamaExtractor, QwenExtractor, OllamaModelCardAdapter
from extractors.gemma_summarizer import GemmaSummarizer

from utils.uri_fetcher import UriFetcher
from utils.uri_builder import UriBuilder
from utils.task_matcher import TaskMatcher
from utils.category_mapper import CategoryMapper


MODEL_TEMPLATE_FILE = "template_1.jsonld"
IMPLEMENTATION_TEMPLATE_FILE = "implementation_template_1.jsonld"

TAXONOMY_CLASSIFIER_FILE = "taxonomy_classifier.joblib"



class ModelCardGenerator:
    def __init__(self,cbr_threshold=50):
        #---COSITAS DE CHROMADB---
        self.cbr_threshold=cbr_threshold
        self.chroma_client=chromadb.PersistentClient(path="./data/cbr_storage")
        self.case_collection=self.chroma_client.get_or_create_collection(name="model_cards_history")
        print(f"Number of current cases: {self.case_collection.count()}")
        # First we load both templates, starting with the base model template.
        model_template_path = Path("templates") / MODEL_TEMPLATE_FILE
        with open(model_template_path, "r", encoding="utf-8") as f:
            self._model_template = json.load(f)

        implementation_template_path = Path("templates") / IMPLEMENTATION_TEMPLATE_FILE
        with open(implementation_template_path, "r", encoding="utf-8") as f:
            self._implementation_template = json.load(f)

        # Load the classifier (TF-IDF + SVC)
        pipeline_path = Path(os.getcwd()) / "utils" / TAXONOMY_CLASSIFIER_FILE
        self._classification_pipeline = joblib.load(pipeline_path)

        # Load the URI fetcher which will help us identify different entities in SemOpenAlex and LinkedPapersWithCode by providing us with URIs.
        # On the other hand the URI builder will help creating URIs for those elements we failed to identify or which need our identification.
        self._uri_fetcher = UriFetcher()
        self._uri_builder = UriBuilder()

        # Load llama, qwen and gliner
        self._gliner = GlinerExtractor()
        self._qwen = QwenExtractor()
        self._llama = LlamaExtractor()
        self._gliner_dataset_extractor = GlinerDatasetExtractor()
        self._gliner_metric_extractor = GlinerMetricExtractor()
        self._qwen_metric_extractor = QwenMetricExtractor()
        self._llama_dataset_extractor = LlamaDatasetExtractor()
        self._qwen_dataset_extractor = QwenDatasetExtractor()
        self._modular_cbr_adapter = OllamaModelCardAdapter(model_name="qwen2.5:1.5b")
        
        self._summarizer = GemmaSummarizer()
        self._task_matcher = TaskMatcher()
        self._category_mapper = CategoryMapper()

        print("Generator ready!")
        

    def _get_timestamp(self):
        timestamp = datetime.now()
        timestamp_string = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        return timestamp_string
        
    def _get_html_tables(self, raw_tables):
        html_tables = []
        for page_data in raw_tables.get("results", []):
            for table_dict in page_data.get("tables", []):
                if "html" in table_dict:
                    html_tables.append(table_dict["html"])
        return html_tables

    def _match_tasks(self, tasks):
        clean_tasks_dict = {}
        for task in tasks:
            lpwc_task = self._task_matcher.match_task(task)
            if(lpwc_task):
                task_data = {
                    "@id": lpwc_task["uri"],
                    "name": lpwc_task["name"],
                }
                clean_tasks_dict[lpwc_task["uri"]] = task_data
        clean_extracted_tasks = list(clean_tasks_dict.values())
        return clean_extracted_tasks
    


    
    def _extract_reference_publication(self, arxiv_id):
        soa_data = self._uri_fetcher.get_soa_data_from_arxiv_id(arxiv_id)
        if not soa_data:
            return None
        publication_authors = []
        for author in soa_data.get("authors"):
            publication_author = {
                "@type": "Person",
                "@id": author,
            }
            publication_authors.append(publication_author)
        
        reference_publication = {
            "@type": "ScholarlyArticle",
            "@id": arxiv_id,
            "author": publication_authors,
            "schema:sameAs": soa_data.get("soa_uri"),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }

        return reference_publication
        
    def _extract_classification(self, text):
        predicted_class = self._classification_pipeline.predict([text]).tolist()
        return predicted_class[0]

    def _order_publication_authors(self, reference_authors, publication_authors):
        right_order = {author['@id']: i for i, author in enumerate(reference_authors)}
        publication_authors.sort(key=lambda x: right_order.get(x['@id'], float('inf')))
        return publication_authors
        
    def _clean_empty_fields(self, d):
        if not isinstance(d, (dict, list)):
            return d
            
        if isinstance(d, list):
            # Limpia los elementos de la lista y quita los que se queden vacíos
            cleaned_list = [self._clean_empty_fields(v) for v in d]
            return [v for v in cleaned_list if v not in (None, "", [], {})]
            
        if isinstance(d, dict):
            # Limpia los valores del diccionario
            cleaned_dict = {k: self._clean_empty_fields(v) for k, v in d.items()}
            # Devuelve el diccionario sin las claves cuyos valores acabaron vacíos
            return {k: v for k, v in cleaned_dict.items() if v not in (None, "", [], {})}
        
    def _generate_modelcard_llms(self, extracted_data):
        jsonld = copy.deepcopy(self._model_template)
        
        jsonld["dateCreated"] = self._get_timestamp()

        abstract = extracted_data.get("abstract")
        full_text = extracted_data.get("full_text")
        html_tables = self._get_html_tables(extracted_data.get("tables"))
        arxiv_id = extracted_data.get("arxiv_id")

        jsonld["@id"] = self._uri_builder.build_modelcard_uri(arxiv_id)
        
        extracted_names = self._llama.extract(full_text, question = "What is the name of the model presented in this paper?")
        if extracted_names:
            jsonld["name"] = extracted_names[0]
        else:
            jsonld["name"] = "Unknown Model"

            
        jsonld["description"] = self._summarizer.summarize(abstract)
        jsonld["keywords"] = self._summarizer.get_keywords(abstract)

        
        # Dataset extraction and identification
        extracted_datasets = self._qwen_dataset_extractor.extract("", html_tables)
        for dataset in extracted_datasets:
            
            dataset_uri = self._uri_fetcher.guess_dataset_uri(dataset)
            if not dataset_uri:
                dataset_uri = self._uri_builder.build_dataset_uri(dataset)

            dataset_data = {
                "@id": dataset_uri,
                "name": dataset,
            }
            jsonld["evaluatedOn"].append(dataset_data)

        extracted_category = self._extract_classification(abstract)
        category_object = self._category_mapper.get_category_object(extracted_category)
        jsonld["modelCategory"] = category_object
        
        # Author identification
        for author in extracted_data.get("authors"):
            author_uri = self._uri_fetcher.extract_author_uri(author, arxiv_id)
            if not author_uri:
                author_uri = self._uri_builder.build_author_uri(author)
            author_data = {
                "@id": author_uri,
                "name": author,
            }
            jsonld["author"].append(author_data)

        # Task extraction and identification
        tasks_prediction = self._llama.extract(abstract, question =  "What are the tasks addressed in this paper?")
        jsonld["mlTask"] = self._match_tasks(tasks_prediction)
        
        extracted_repo = self._llama.extract(full_text, question = "Return the URL link for the paper implementation (like GitHub). Return an empty list if not found. DO NOT  invent links")
        jsonld["codeRepository"] = list(set(extracted_repo))

        reference_publication = self._extract_reference_publication(arxiv_id)
        if(reference_publication):
            jsonld["referencePublication"] = reference_publication

        jsonld["referencePublication"]["author"] = self._order_publication_authors(jsonld["author"], jsonld["referencePublication"]["author"])

        clean_jsonld = self._clean_empty_fields(jsonld)
        return clean_jsonld
        
    def _generate_modelcard_cbr(self, extracted_data):
        abstract = extracted_data.get("abstract", "")
        arxiv_id = extracted_data.get("arxiv_id", "")
        title = extracted_data.get("title", "")
        sections = extracted_data.get("sections", "")
        full_text = extracted_data.get("full_text", "")
        
        # 1. RETRIEVE
        results = self.case_collection.query(
            query_texts=[abstract],
            n_results=1
        )
        
        if not results.get('metadatas') or not results['metadatas'][0]:
            return self._generate_modelcard_llms(extracted_data)
            
        sim_case = json.loads(results['metadatas'][0][0]['model_card_json'])
        
        # Extraemos pistas del caso similar para guiar al modelo
        sim_name = sim_case.get("name", "")
        sim_tasks = [t.get("name") if isinstance(t, dict) else t for t in sim_case.get("mlTask", [])]
        sim_cat = sim_case.get("modelCategory", {}).get("name", "") if isinstance(sim_case.get("modelCategory"), dict) else ""
        sim_repo = sim_case.get("codeRepository", [])
        sim_datasets = [d.get("name") if isinstance(d, dict) else d for d in sim_case.get("evaluatedOn", [])]
        sim_metrics = [m.get("name") if isinstance(m, dict) else m for m in sim_case.get("metrics", [])]
    
        # 2. REVISE / EXTRACT MODULAR
        pred_model = self._modular_cbr_adapter.extract_model_name(title, abstract, sim_name)
        pred_tasks = self._modular_cbr_adapter.extract_tasks(abstract, sim_tasks)
        pred_cat = self._modular_cbr_adapter.extract_category(abstract, sim_cat)
        pred_repo = self._modular_cbr_adapter.extract_repo(abstract, full_text, sim_repo)
        pred_datasets, pred_metrics = self._modular_cbr_adapter.extract_datasets_and_metrics(sections, sim_datasets, sim_metrics)
    
        # 3. ENSAMBLAJE DE LA MODEL CARD (JSON-LD)
        jsonld = copy.deepcopy(self._model_template)
        jsonld["dateCreated"] = self._get_timestamp()
        jsonld["@id"] = self._uri_builder.build_modelcard_uri(arxiv_id)
        jsonld["name"] = pred_model
        
        # Tasks
        jsonld["mlTask"] = self._match_tasks(pred_tasks) if hasattr(self, '_match_tasks') else [{"name": t} for t in pred_tasks]
        
        # Category
        jsonld["modelCategory"] = self._category_mapper.get_category_object(pred_cat) if hasattr(self, '_category_mapper') else {"name": pred_cat}
        
        # Code Repository
        jsonld["codeRepository"] = pred_repo

        # Datasets
        jsonld["evaluatedOn"] = []
        for d in pred_datasets:
            uri = self._uri_fetcher.guess_dataset_uri(d) or self._uri_builder.build_dataset_uri(d) if hasattr(self, '_uri_fetcher') else ""
            jsonld["evaluatedOn"].append({"@id": uri, "name": d})
            
        # Metrics
        jsonld["metrics"] = [{"name": m} for m in pred_metrics]
        
        # Autores
        for author in extracted_data.get("authors", []):
            author_uri = self._uri_fetcher.extract_author_uri(author, arxiv_id) or self._uri_builder.build_author_uri(author) if hasattr(self, '_uri_fetcher') else ""
            jsonld["author"].append({"@id": author_uri, "name": author})
    
        return self._clean_empty_fields(jsonld)
        

    def generate_modelcard(self, extracted_data):
            """Router Dinámico: Decide si usar LLMs o CBR basado en el histórico"""
            current_cases = self.case_collection.count()
            
            if current_cases < self.cbr_threshold:
                print("Ruta: LLM (Construyendo base de conocimiento...)")
                # 1. Generamos con tu código original
                final_jsonld = self._generate_modelcard_llms(extracted_data)
                
                # 2. Retain: Guardamos el caso para el futuro
                abstract = extracted_data.get("abstract", "")
                arxiv_id = extracted_data.get("arxiv_id", str(current_cases))
                
                # Guardamos el abstract como texto de búsqueda, y el JSON como metadata
                self.case_collection.add(
                    documents=[abstract],
                    metadatas=[{"model_card_json": json.dumps(final_jsonld)}],
                    ids=[f"paper_{arxiv_id}"]
                )
                return final_jsonld
            else:
                print("Ruta: CBR + qwen1B (Masa crítica alcanzada!)")
                return self._generate_modelcard_cbr(extracted_data)

  


