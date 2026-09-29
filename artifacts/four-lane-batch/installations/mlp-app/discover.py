import hashlib, json
from pathlib import Path
import paralyn
import sys, paralyn.tensor; tensor = sys.modules["paralyn.tensor"]
lib = paralyn._library().path
package = Path(paralyn.__file__).parent
build = json.loads((package / "_build.json").read_text())
print(json.dumps({"package": paralyn.__file__, "tensor_module": tensor.__file__, "library": lib,
                  "library_sha256": hashlib.sha256(Path(lib).read_bytes()).hexdigest(),
                  "packaged_build": build}, indent=1))
