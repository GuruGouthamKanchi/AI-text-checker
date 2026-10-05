from robustness_eval.humanizer import humanize_text

sample = "SCS-CN Method The SCS-CN method, introduced by the Soil Conservation Service of the United States in 1969, is a widely used conceptual hydrological model for estimating direct runoff generated from daily rainfall depth."

print("=== INPUT ===")
print(sample)
print()

res_resume = humanize_text(sample, tone="resume")
print("=== RESUME TONE OUTPUT ===")
print(res_resume["humanized_text"])
print(f"AI Score: {res_resume['humanized_ai_percentage']}%\n")

res_academic = humanize_text(sample, tone="academic")
print("=== ACADEMIC TONE OUTPUT ===")
print(res_academic["humanized_text"])
print(f"AI Score: {res_academic['humanized_ai_percentage']}%\n")
