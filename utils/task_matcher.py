"""
Alinea las tareas extraídas con un vocabulario de tareas (LPWC / PwC).

Cambios respecto a la versión anterior:
  * Preparado para vocabularios grandes (miles de tareas del volcado completo de PwC).
  * Antes se usaba fuzz.token_set_ratio > 85, que con vocabularios grandes da
    falsos positivos: "classification" puntúa 100 frente a "Image Classification"
    porque un conjunto de tokens contiene al otro. Ahora se usa
    token_sort_ratio, que exige que ambas cadenas tengan los mismos tokens,
    con umbral 90.
  * Admite sinónimos ("synonyms") si el vocabulario los trae.
"""
import json
import re

from rapidfuzz import fuzz, process


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


class TaskMatcher:
    def __init__(self, json_path="testing_data/tasks.json", threshold=90):
        with open(json_path, encoding="utf-8") as f:
            raw_tasks = json.load(f)
        self.threshold = threshold
        self.master_tasks = {}
        self._lookup = {}
        for task in raw_tasks:
            entry = {"name": task["name"], "uri": task.get("uri", "")}
            key = _norm(task["name"])
            self.master_tasks[key] = entry
            self._lookup[key] = entry
            for syn in task.get("synonyms") or []:
                self._lookup.setdefault(_norm(syn), entry)
        self._keys = list(self._lookup.keys())

    def __len__(self):
        return len(self.master_tasks)

    def match_task(self, extracted_task):
        if not extracted_task:
            return None
        key = _norm(extracted_task)
        if not key:
            return None
        if key in self._lookup:
            return self._lookup[key]
        best = process.extractOne(key, self._keys, scorer=fuzz.token_sort_ratio,
                                  score_cutoff=self.threshold)
        if best:
            return self._lookup[best[0]]
        return None
