from robustness_eval.humanizer import humanize_text

samples = [
    "I am Goutham from, Tirupati district and I final year student from SISTK. Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features.",
    "Furthermore, it is crucial to delve into the multifaceted challenges of AI alignment to foster responsible development.",
    "Leveraging comprehensive transformer architectures is paramount for achieving seamless language understanding at scale.",
]

print("=== Humanizer Test Results ===\n")
for i, s in enumerate(samples):
    res = humanize_text(s)
    print(f"[Sample {i+1}]")
    print(f"  ORIGINAL  : {s}")
    print(f"  HUMANIZED : {res['humanized_text']}")
    print(f"  AI SCORE  : {res['original_ai_percentage']}% -> {res['humanized_ai_percentage']}%  ({res['resilience_verdict']})")
    print()
