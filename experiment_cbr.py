import os
import json
import time
import pandas as pd
from pathlib import Path
import re
import warnings

# Silenciar advertencias
warnings.filterwarnings('ignore')

# Importamos tu parser, el generador y TUS herramientas
from utils.XMLParser import XMLParser
from model_card_generation_pipeline import ModelCardGenerator  
from utils.evaluation_tools import calculate_bertscore_complete, aplanar

def to_list(val):
    """Convierte cualquier valor en una lista plana de strings, extrayendo 'name' si hay diccionarios."""
    if val is None:
        return []
        
    if isinstance(val, dict):
        if len(val) == 0:
            return []
        # Manejo específico para JSON-LD: extraemos el nombre real
        if 'name' in val:
            return [str(val['name'])]
        return aplanar(list(val.values()))
        
    if isinstance(val, (list, tuple, set)):
        if len(val) == 0:
            return []
        res = []
        for item in val:
            if isinstance(item, dict):
                if 'name' in item:
                    res.append(str(item['name']))
                else:
                    res.extend(aplanar(list(item.values())))
            elif isinstance(item, list):
                res.extend(aplanar(item))
            else:
                res.append(str(item))
        return res
        
    if hasattr(val, 'tolist'):
        val_list = val.tolist()
        if len(val_list) == 0:
            return []
        return aplanar(val_list)
        
    if isinstance(val, float) and pd.isna(val):
        return []
        
    if isinstance(val, str):
        if not val.strip() or val.lower() == 'nan':
            return []
        return [val]
        
    return [str(val)]

def extract_field_recursive(data, possible_keys):
    """Busca recursivamente una clave dentro de diccionarios anidados limpiando espacios."""
    if not isinstance(data, dict):
        return ""

    # .strip() elimina espacios fantasma generados por el LLM en las claves
    data_lower = {str(k).lower().strip(): v for k, v in data.items()}
    
    for key in possible_keys:
        clean_key = key.lower().strip()
        if clean_key in data_lower:
            val = data_lower[clean_key]
            if val: 
                return val

    for val in data.values():
        if isinstance(val, dict):
            res = extract_field_recursive(val, possible_keys)
            if res: return res
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    res = extract_field_recursive(item, possible_keys)
                    if res: return res
    return ""

def extract_field(model_card, possible_keys):
    """Extrae un campo de manera hiper-robusta, a prueba de Markdown y listas crudas."""
    if not model_card:
        return []
        
    extracted_val = ""
    
    if isinstance(model_card, dict):
        extracted_val = extract_field_recursive(model_card, possible_keys)

    elif isinstance(model_card, str):
        # 1. Limpieza de bloques Markdown (```json ... ```)
        clean_str = re.sub(r'^```(?:json)?\s*', '', model_card, flags=re.IGNORECASE | re.MULTILINE)
        clean_str = re.sub(r'```\s*$', '', clean_str, flags=re.MULTILINE).strip()
        
        # 2. Intentamos parsear el JSON limpio
        try:
            parsed = json.loads(clean_str)
            if isinstance(parsed, dict):
                extracted_val = extract_field_recursive(parsed, possible_keys)
        except json.JSONDecodeError:
            pass 
            
        # 3. Plan C: Regex mejorado (ahora captura listas enteras [...])
        if not extracted_val:
            for key in possible_keys:
                # Busca listas [...] o strings "..." o valores sueltos
                pattern = rf'"{key}"\s*:\s*(\[[^\]]*\]|"[^"]*"|[^,}}]+)'
                match = re.search(pattern, clean_str, re.IGNORECASE)
                if match:
                    val = match.group(1).strip(' \n\r\t"\'')
                    if val and val.lower() != "null":
                        # Si era una lista en formato texto (ej. '["Knowledge Graph Completion"]')
                        # la limpiamos un poco antes de pasarla
                        val = re.sub(r'^\[\vert{}\]$', '', val).replace('"', '')
                        extracted_val = val
                        break
                        
    return to_list(extracted_val)

def evaluate_with_your_function(prediction, ground_truth):
    """
    Envuelve tu función asegurando que reciba listas correctas.
    Devuelve (None, None, None) si el Ground Truth no existe para no penalizar.
    """
    gt_list = to_list(ground_truth)
    
    if not gt_list:
        return None, None, None
        
    pred_list = to_list(prediction)
    
    try:
        p, r, f1 = calculate_bertscore_complete(pred_list, gt_list)
        return p, r, f1
    except Exception as e:
        print(f"Error evaluando BERTScore: {e}")
        return 0.0, 0.0, 0.0

def load_clean_dataset():
    print("Iniciando carga y parseo de TODOS los datos...")
    base_path = Path(os.getcwd())
    data_dir = base_path / "testing_data" 
    
    with open(data_dir / "pwc_final.json", 'r', encoding='utf-8') as f:
        pwc_json = json.load(f)

    print(f"{len(pwc_json)} papers encontrados en el JSON. Procesando todo...")
    clean_dataset = []

    for paper in pwc_json:
        xml_path_str = paper.get("local_xml_path")
        if not xml_path_str:
            continue
        
        filename = xml_path_str.replace("\\", "/").split("/")[-1]
        xml_path = data_dir / "xml_outputs" / filename 

        try:
            parser = XMLParser(xml_path)
            full_text = parser.get_full_text()
            abstract = parser.get_abstract()
            section_dict = parser.get_sections(target_sections=['Experiments','Evaluation','Results'])
            sections = "\n\n".join(section_dict.values())
        except Exception:
            continue

        json_filename = filename.replace(".tei.xml", "_lightonocr.json")
        lightocr_json_path = data_dir / "lightocr_json" / json_filename 

        tables = {}
        if lightocr_json_path.is_file():
            try:
                with open(lightocr_json_path, 'r', encoding='utf-8') as f:
                    tables = json.load(f)
            except Exception:
                pass
        
        clean_arxiv_id = filename.replace(".tei.xml", "")
        
        clean_paper = {
            "gt_model_name": paper.get("model_name"),
            "gt_category": paper.get("model_category"), 
            "gt_tasks": paper.get("tasks"), 
            "gt_implementation": paper.get("paper_repo"),
            "gt_datasets": paper.get("Datasets"),
            "gt_metrics": paper.get("Metrics"),
            
            "arxiv_id": clean_arxiv_id,
            "title": paper.get("title"),
            "authors": paper.get("authors", []), 
            "local_xml_path": str(xml_path), 
            "pwc_abstract": paper.get("abstract"),
            "abstract": abstract,
            "full_text": full_text,
            "sections": sections,
            "tables": tables,
        }
        clean_dataset.append(clean_paper)
        
    print(f"Carga completa. {len(clean_dataset)} papers listos para el experimento.")
    return clean_dataset

def run_threshold_experiment():
    dataset = load_clean_dataset()
    if not dataset:
        print("El dataset está vacío.")
        return

    thresholds = [100, 500, 1000, 3000] 
    csv_file = "resultados_detallados_cbr.csv"
    
    processed_records = set()
    
    columnas = [
        "threshold", "arxiv_id", "generation_time_seconds", "used_cbr",
        "model_f1", "model_precision", "model_recall",
        "cat_f1", "cat_precision", "cat_recall",
        "tasks_f1", "tasks_precision", "tasks_recall",
        "datasets_f1", "datasets_precision", "datasets_recall",
        "metrics_f1", "metrics_precision", "metrics_recall",
        "repo_f1", "repo_precision", "repo_recall",
        "paper_avg_f1", "paper_avg_precision", "paper_avg_recall"
    ]

    if os.path.exists(csv_file):
        df_existente = pd.read_csv(csv_file)
        for _, row in df_existente.iterrows():
            processed_records.add((row['threshold'], str(row['arxiv_id'])))
    else:
        pd.DataFrame(columns=columnas).to_csv(csv_file, index=False)

    for threshold in thresholds:
        print(f"\n{'='*50}")
        print(f"INICIANDO EXPERIMENTO CON UMBRAL: {threshold}")
        print(f"{'='*50}")
        
        records_for_this_threshold = sum(1 for t, _ in processed_records if t == threshold)
        if records_for_this_threshold == len(dataset):
            continue
            
        generator = ModelCardGenerator(cbr_threshold=threshold)
        
        if records_for_this_threshold == 0:
            try:
                generator.chroma_client.delete_collection("model_cards_history")
                generator.case_collection = generator.chroma_client.create_collection("model_cards_history")
            except Exception:
                pass 
            
        threshold_time = 0
        
        for i, paper_data in enumerate(dataset, 1):
            if (threshold, str(paper_data["arxiv_id"])) in processed_records:
                continue
                
            start_time = time.time()
            
            try:
                model_card = generator.generate_modelcard(paper_data)
            except Exception as e:
                print(f"Error generando {paper_data['arxiv_id']}: {e}")
                continue
                
            end_time = time.time()
            doc_time = end_time - start_time
            threshold_time += doc_time
            used_cbr = (i > threshold) 
            
            # --- EXTRACCIÓN Y CONVERSIÓN A LISTAS ---
            pred_model = extract_field(model_card, ["name", "model_name", "model_name_prediction"])
            
            pred_cat = extract_field(model_card, ["modelcategory", "category", "category_prediction"])
            
            # Añadido "mltask" y "mltasks"
            pred_tasks = extract_field(model_card, ["mltask", "mltasks", "tasks", "task", "tasks_prediction", "primary_tasks", "intended_task"]) 
            
            pred_datasets = extract_field(model_card, ["datasets", "dataset", "datasets_prediction", "trainingdata"])
            
            pred_metrics = extract_field(model_card, ["metrics", "metric", "metrics_prediction", "evaluation"])
            
            # Añadido "coderepository"
            pred_repo = extract_field(model_card, ["coderepository", "repository", "repo", "paper_repo", "implementation_prediction", "url"])

            # --- CÁLCULO CON TU EVALUADOR ---
            model_p, model_r, model_f1 = evaluate_with_your_function(pred_model, paper_data["gt_model_name"])
            cat_p, cat_r, cat_f1 = evaluate_with_your_function(pred_cat, paper_data["gt_category"])
            tasks_p, tasks_r, tasks_f1 = evaluate_with_your_function(pred_tasks, paper_data["gt_tasks"])
            ds_p, ds_r, ds_f1 = evaluate_with_your_function(pred_datasets, paper_data["gt_datasets"])
            met_p, met_r, met_f1 = evaluate_with_your_function(pred_metrics, paper_data["gt_metrics"])
            repo_p, repo_r, repo_f1 = evaluate_with_your_function(pred_repo, paper_data["gt_implementation"])

            
            valid_f1 = [f for f in [model_f1, cat_f1, tasks_f1, ds_f1, met_f1, repo_f1] if f is not None]
            valid_prec = [p for p in [model_p, cat_p, tasks_p, ds_p, met_p, repo_p] if p is not None]
            valid_rec = [r for r in [model_r, cat_r, tasks_r, ds_r, met_r, repo_r] if r is not None]

            paper_avg_f1 = sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0
            paper_avg_prec = sum(valid_prec) / len(valid_prec) if valid_prec else 0.0
            paper_avg_rec = sum(valid_rec) / len(valid_rec) if valid_rec else 0.0

            row_data = {
                "threshold": threshold,
                "arxiv_id": paper_data["arxiv_id"],
                "generation_time_seconds": doc_time,
                "used_cbr": used_cbr,
                
                "model_f1": model_f1, "model_precision": model_p, "model_recall": model_r,
                "cat_f1": cat_f1, "cat_precision": cat_p, "cat_recall": cat_r,
                "tasks_f1": tasks_f1, "tasks_precision": tasks_p, "tasks_recall": tasks_r,
                "datasets_f1": ds_f1, "datasets_precision": ds_p, "datasets_recall": ds_r,
                "metrics_f1": met_f1, "metrics_precision": met_p, "metrics_recall": met_r,
                "repo_f1": repo_f1, "repo_precision": repo_p, "repo_recall": repo_r,
                
                "paper_avg_f1": paper_avg_f1,
                "paper_avg_precision": paper_avg_prec,
                "paper_avg_recall": paper_avg_rec
            }
            
            pd.DataFrame([row_data]).to_csv(csv_file, mode='a', header=False, index=False)
            
            if i % 10 == 0:
                print(f"Progreso umbral {threshold}: {i}/{len(dataset)} | F1 medio del paper actual: {paper_avg_f1:.2f}")
                
        print(f"Fin umbral {threshold}. Tiempo total procesado hoy: {threshold_time / 60:.2f} minutos.")

    print("\nGenerando resumen de métricas...")
    df_detallado = pd.read_csv(csv_file)
    columnas_numericas = [
        "generation_time_seconds", 
        "model_f1", "cat_f1", "tasks_f1", 
        "datasets_f1", "metrics_f1", "repo_f1",
        "paper_avg_f1", "paper_avg_precision", "paper_avg_recall"
    ]
    df_resumen = df_detallado.groupby("threshold")[columnas_numericas].mean().reset_index()
    
    df_resumen = df_resumen.rename(columns={
        "generation_time_seconds": "avg_time_seconds",
        "paper_avg_f1": "GLOBAL_F1",
        "paper_avg_precision": "GLOBAL_Precision",
        "paper_avg_recall": "GLOBAL_Recall"
    })
    
    df_resumen.to_csv("resumen_metricas_cbr_bertscore.csv", index=False)
    print("\n¡Experimento completado y exportado!")

if __name__ == "__main__":
    run_threshold_experiment()