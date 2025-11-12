import syft as sy
from multiprocessing import freeze_support # <-- 1. Import this

def start_server():
    # 1️⃣ Launch a Syft "Datasite" (acts as a hospital server)
    domain = sy.orchestra.launch(
        name="hospital-1-datasite",
        port=8080,          # Port for this hospital
        dev_mode=True,      # dev_mode gives auto admin + local DB
    )

    print(f"🏥 Syft Datasite '{domain.name}' running at: http://localhost:{domain.port}")
    print("🔑 Login with email: info@openmined.org | password: changethis")
    print("\n🎉 --- SERVER IS RUNNING ---")
    print("Keep this terminal running! Open a new terminal to run the client.")


# 2. Add this block at the bottom
if __name__ == "__main__":
    freeze_support()  # <-- 3. Add this line for Windows
    start_server()    # <-- 4. Call your main function