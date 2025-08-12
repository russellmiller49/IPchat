
import json, glob, sys
from jsonschema import validate

schema_path = sys.argv[1] if len(sys.argv) > 1 else "textbook_chapter.schema.json"
schema = json.load(open(schema_path, "r", encoding="utf-8"))

for path in glob.glob("*.chapter.json"):
    data = json.load(open(path,"r",encoding="utf-8"))
    validate(instance=data, schema=schema)
    print("OK:", path)
