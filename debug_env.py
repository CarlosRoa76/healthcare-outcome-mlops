import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_diagnostic():
    print("="*60)
    print("🔍 MLOPS ENVIRONMENT DIAGNOSTIC TOOL (v2)")
    print("="*60)

    # 1. Check Current Working Directory
    cwd = Path.cwd()
    print(f"📂 Current Directory: {cwd}")

    # 2. Check for .env file existence
    env_file = cwd / ".env"
    
    if not env_file.exists():
        print(f"❌ ERROR: .env file NOT FOUND at {env_file}")
        print("\n🔎 Searching for common naming mistakes...")
        all_files = os.listdir(cwd)
        for f in all_files:
            if "env" in f.lower():
                print(f"   ⚠️ Found suspicious file: '{f}' (Should be exactly '.env')")
        return # Stop here if file is missing

    print(f"✅ SUCCESS: .env file found at: {env_file}")

    # 3. Try to read the file raw (to check for key existence)
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
            has_uri = any("MONGODB_URI" in line for line in lines)
            if has_uri:
                print("✅ SUCCESS: 'MONGODB_URI' key exists inside the file.")
            else:
                print("❌ ERROR: 'MONGODB_URI' key NOT found inside the file.")
                print("   Current file content summary:")
                for line in lines:
                    if line.strip(): print(f"   -> {line.split('=')[0]}...")
    except Exception as e:
        print(f"❌ ERROR: Could not read file: {e}")

    # 4. Attempt to load with dotenv
    print("\nAttempting load_dotenv()...")
    load_dotenv(dotenv_path=env_file)
    
    # 5. Check the environment variable
    url = os.getenv("MONGODB_URL")
    
    if url:
        print(f"✅ SUCCESS: Environment variable loaded!")
        # Safety mask: show first few chars and last few
        masked_url = f"{url[:15]}...{url[-10:]}"
        print(f"🔗 Loaded URL: {masked_url}")
        
        # 6. Test PyMongo Connection
        print("\n📡 Testing MongoDB Connection (5s timeout)...")
        try:
            import pymongo
            client = pymongo.MongoClient(url, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            print("✅ CONNECTION SUCCESS: MongoDB Atlas is reachable!")
        except ImportError:
            print("⚠️ SKIPPED: pymongo not installed in this venv.")
        except Exception as e:
            print(f"❌ CONNECTION FAILED: {e}")
    else:
        print("❌ ERROR: os.getenv('MONGODB_URL') returned None.")
        print("   💡 Check for spaces around the '=' in your .env file.")

    print("="*60)

if __name__ == "__main__":
    test_diagnostic()