from src.app import run_analysis_pipeline

tests = [
    "SCS-CN Method The SCS-CN method, introduced by the Soil Conservation Service of the United States in 1969, is a widely used conceptual hydrological model for estimating direct runoff generated from daily rainfall depth.",
    "First developed by the US Soil Conservation Service in 1969, the SCS-CN approach is a standard hydrological technique for predicting rainfall runoff.",
    "Researchers often use the SCS-CN approach—created in 1969 by the US Soil Conservation Service—to calculate surface runoff from daily rainfall amounts.",
    "The SCS-CN framework, created by the U.S. Soil Conservation Service in 1969, offers a practical way to predict daily rainfall runoff."
]

print("=== ACADEMIC DEFINITION RESCORING TEST ===")
for i, t in enumerate(tests):
    res = run_analysis_pipeline(doc_text=t, filename=f"test_{i}.txt", page_count=1)
    score = res["overall_ai_percentage"]
    print(f"[{i}] Score: {score:5.1f}%  | Text: {t}")
