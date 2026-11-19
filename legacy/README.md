# Legado

Scripts y resultados del experimento CBR anterior. **Los números de estos CSV y
gráficas no son válidos**: se calcularon con una evaluación que puntuaba siempre
0 en datasets, no extraía métricas en la ruta LLM y usaba BERTScore sin reescalar
(ver el historial de git). Se conservan solo como referencia.

`model_card_generation_pipeline_basic.py` (pipeline solo-LLM) tiene un error de
sintaxis en la línea 8 y está sustituido por la ruta `llm` de
`model_card_generation_pipeline.FieldPipeline`.
