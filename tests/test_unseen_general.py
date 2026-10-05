from robustness_eval.humanizer import humanize_text

test_sentences = [
    "Implemented a scalable microservice architecture using Go and Redis to handle 50,000 requests per second with high availability.",
    "Formulated a deep reinforcement learning policy to optimize traffic signal timing in dense urban networks.",
    "Streamlined CI/CD pipeline deployments using Docker and Kubernetes to reduce deployment times by 40%."
]

print("=== TESTING ZERO-SHOT GENERAL HUMANIZATION ON UNSEEN SENTENCES ===")
for sent in test_sentences:
    res = humanize_text(sent, tone="resume")
    print(f"\nINPUT : {sent}")
    print(f"OUTPUT: {res['humanized_text']}")
    print(f"SCORE : {res['original_ai_percentage']}% -> {res['humanized_ai_percentage']}% AI")
