import requests

url = 'http://localhost:8000/sample?type=human'
r = requests.get(url)
data = r.json()

print("Status code:", r.status_code)
print("Overall AI Percentage for Human Sample:", data.get("overall_ai_percentage"), "%")
print("Summary Breakdown:", data.get("summary"))
