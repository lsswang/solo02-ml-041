import json
from pathlib import Path
from profile import analyze
from coding import design

if __name__ == "__main__":
    fixture = json.loads((Path(__file__).parent / "fixtures/cases.json").read_text())
    result = analyze(fixture["model"], fixture["query_before"])
    print(json.dumps({"profile": result,
                      "coding": design(result["residue_posterior"], fixture["ordinary_policy"])}, indent=2))
