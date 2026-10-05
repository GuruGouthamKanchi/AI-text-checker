from robustness_eval.humanizer import humanize_text

samples = {
    "resume": "Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features.",
    "academic": "Furthermore, it is crucial to delve into the multifaceted challenges of AI alignment to foster responsible development.",
    "casual": "Moreover, it is paramount to utilize state-of-the-art techniques to seamlessly enhance system throughput.",
    "creative": "The comprehensive platform leverages innovative algorithms to foster seamless user engagement."
}

print("=================================================================")
print("TESTING ALL 4 TARGET TONE PROFILES DIRECTLY VIA ENGINE")
print("=================================================================\n")

for tone, text in samples.items():
    print(f"--- TONE: {tone.upper()} ---")
    print(f"INPUT : {text}")
    
    try:
        data = humanize_text(text, tone=tone)
        print(f"OUTPUT: {data['humanized_text']}")
        print(f"SCORES: {data['original_ai_percentage']}% AI -> {data['humanized_ai_percentage']}% AI  ({data['resilience_verdict']})")
    except Exception as e:
        print(f"ERROR : {e}")
    
    print()

print("=================================================================")
