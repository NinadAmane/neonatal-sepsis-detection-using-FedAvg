import syft as sy
import pandas as pd
import os
import sys
from syft.service.dataset.dataset import CreateDataset, CreateAsset 
# We also need the success message class to check it
from syft.service.response import SyftSuccess 

def upload_dataset(csv_path: str):
    try:
        client = sy.login(
            port=8080,
            email="info@openmined.org",
            password="changethis"
        )
        print(f"✅ Logged into: {client.name}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"📂 Loaded dataset: {csv_path} | shape = {df.shape}")

        asset_name = os.path.basename(csv_path).replace(".csv", "")
        
        syft_asset = CreateAsset(
            name=f"{asset_name}_asset",
            data=df,
            mock=df.head(1) 
        )

        dataset_to_create = CreateDataset(
            name=asset_name,
            asset_list=[syft_asset],
            description="Neonatal sepsis dataset for federated training."
        )

        # 6. Upload the 'CreateDataset' object
        result = client.upload_dataset(dataset=dataset_to_create)

        # 7. NEW: Check if the result is a success message
        if isinstance(result, SyftSuccess):
            print(f"\n🎉 ✅ Successfully uploaded dataset: '{asset_name}'")
            print("You can now run 'researcher_client.py' to see it!")
        else:
            # If it's not a success, print the error
            print(f"\n❌ Upload failed with: {result}")

    except Exception as e:
        print(f"❌ Error during upload: {e}", file=sys.stderr)
        print("Is the server running?", file=sys.stderr)


if __name__ == "__main__":
    csv_path = "data/federated/hospital_1.csv"
    upload_dataset(csv_path)