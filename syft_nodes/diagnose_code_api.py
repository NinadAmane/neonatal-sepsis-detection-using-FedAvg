# syft_nodes/diagnose_code_api.py
import syft as sy
import pprint
import importlib
import sys

print("syft version:", getattr(sy, "__version__", "unknown"))
print()

# Top-level attributes that might contain Code
candidates = [
    "Code",
    "code",
    "service.code.Code",
    "service.code.code.Code",
    "service.code.user_code.Code",
    "service.code.user_code.SubmitUserCode",
    "service.code",
    "service",
]

print("=== Inspecting potential Code locations ===")
for cand in candidates:
    try:
        # attempt to import the attribute/path
        parts = cand.split(".")
        mod = sy
        for p in parts:
            mod = getattr(mod, p)
        print(f"FOUND: syft.{cand} -> {mod}")
    except Exception as e:
        print(f"NOT FOUND: syft.{cand}  ({e.__class__.__name__})")

print("\n=== Inspect service.code module (if present) ===")
try:
    svc_code = sy.service.code
    print("service.code:", svc_code)
    print("dir(service.code):")
    pprint.pprint([a for a in dir(svc_code) if not a.startswith("_")])
except Exception as e:
    print("service.code not importable:", e)

print("\n=== Attempt to connect to datasite and inspect client.code ===")
try:
    client = sy.login(port=8080, email="info@openmined.org", password="changethis")
    print("Logged into:", client.name)
    print("\nclient.code attributes:")
    try:
        attrs = [a for a in dir(client.code) if not a.startswith("_")]
        pprint.pprint(attrs)
    except Exception as e:
        print("Could not dir(client.code):", e)
    # show callables in client.code
    print("\nclient.code callables:")
    try:
        callables = [a for a in attrs if callable(getattr(client.code, a, None))]
        pprint.pprint(callables)
    except Exception as e:
        print("Could not list callables:", e)
except Exception as e:
    print("Could not login or inspect client:", e)
    print("Make sure hospital_server.py is running and uses dev-mode credentials.")
