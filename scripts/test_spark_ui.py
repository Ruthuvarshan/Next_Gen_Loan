"""
Simple Spark UI test - diagnose connection issues
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("TESTING SPARK UI CONNECTIVITY")
print("="*80)
print()

print("Step 1: Importing PySpark...")
try:
    from pyspark.sql import SparkSession
    print("✅ PySpark imported successfully")
except Exception as e:
    print(f"❌ Failed to import PySpark: {e}")
    sys.exit(1)

print("\nStep 2: Creating Spark Session with UI enabled...")
try:
    spark = SparkSession.builder \
        .appName("SparkUI-Test") \
        .master("local[*]") \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .getOrCreate()
    
    print("✅ Spark Session created successfully")
except Exception as e:
    print(f"❌ Failed to create Spark Session: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Checking Spark UI status...")
try:
    sc = spark.sparkContext
    ui_web_url = sc.uiWebUrl
    print(f"✅ Spark UI should be available at: {ui_web_url}")
    print(f"   Application ID: {sc.applicationId}")
    print(f"   Master: {sc.master}")
except Exception as e:
    print(f"❌ Failed to get UI URL: {e}")

print("\n" + "="*80)
print("INSTRUCTIONS:")
print("="*80)
print()
print("1. Open your web browser")
print("2. Navigate to the URL shown above (usually http://localhost:4040)")
print("3. You should see the Spark UI dashboard")
print()
print("If you still can't access it:")
print("  - Check if port 4040 is blocked by firewall")
print("  - Try http://127.0.0.1:4040 instead")
print("  - Check if another application is using port 4040")
print()
print("="*80)

input("\nPress ENTER to create a simple job and check UI again...")

print("\nStep 4: Running a simple Spark job...")
try:
    # Create a simple DataFrame to trigger UI activity
    data = [(1, "test1"), (2, "test2"), (3, "test3")]
    df = spark.createDataFrame(data, ["id", "name"])
    
    print("Created DataFrame:")
    df.show()
    
    print(f"\n✅ Job completed! Check the Jobs tab in Spark UI at: {ui_web_url}")
except Exception as e:
    print(f"❌ Error running job: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Spark UI URL: {ui_web_url}")
print("Try these alternatives if localhost doesn't work:")
print("  - http://127.0.0.1:4040")
print("  - http://0.0.0.0:4040")
print()

input("\nPress ENTER to stop Spark and exit...")

spark.stop()
print("\n✅ Spark session stopped")
