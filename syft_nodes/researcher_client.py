import syft as sy
import sys

try:
    # 1️⃣ Connect to the Datasite (Hospital)
    client = sy.login(
        port=8080,
        email="info@openmined.org",
        password="changethis"
    )

    print(f"✅ Successfully logged in to: {client.name}")

    # 2️⃣ List Datasets (This part works!)
    print("\n📊 Registered Datasets:")
    all_datasets = client.datasets 
    
    if not all_datasets:
        print("No datasets registered yet.")
    else:
        for ds in all_datasets:
            print(f" • {ds.name}") 

    # 3️⃣ List Models (Code Assets) - FIXED LOOP
    print("\n🧠 Registered Models (Code Assets):")
    
    all_models = client.code
    model_count = 0
    
    # We iterate and count, this is safer
    for model_code in all_models:
        print(f" • {model_code.name}")
        model_count += 1
        
    if model_count == 0:
        print("No models registered yet.")

except Exception as e:
    print(f"❌ Error connecting: {e}", file=sys.stderr)
    print("Did you start the 'syft_hospital_server.py' script first?", file=sys.stderr)