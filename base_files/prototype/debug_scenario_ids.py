import json

def debug_scenario_ids(file_path):
    """
    Debug what scenario IDs are actually in the JSON file
    """
    print(f"Analyzing file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            logs = data
            print(f"File contains a direct list of {len(logs)} entries")
        elif isinstance(data, dict) and "logs" in data:
            logs = data["logs"]
            print(f"File contains structured data with {len(logs)} logs")
        else:
            # Single log entry
            logs = [data]
            print("File contains a single log entry")
        
        # Extract scenario IDs
        scenario_ids = []
        for i, log in enumerate(logs):
            if isinstance(log, dict):
                scenario_id = log.get("scenario_id")
                scenario_ids.append(scenario_id)
                if i < 5:  # Show first 5 entries
                    print(f"Entry {i}: scenario_id = {scenario_id}")
        
        print(f"\nTotal entries: {len(logs)}")
        print(f"Scenario IDs found: {len([sid for sid in scenario_ids if sid is not None])}")
        print(f"Missing scenario_ids: {len([sid for sid in scenario_ids if sid is None])}")
        
        # Show range and unique IDs
        valid_ids = [sid for sid in scenario_ids if sid is not None]
        if valid_ids:
            print(f"Min scenario_id: {min(valid_ids)}")
            print(f"Max scenario_id: {max(valid_ids)}")
            print(f"Unique scenario_ids: {len(set(valid_ids))}")
            
            # Check if 60 exists
            if 60 in valid_ids:
                print("✓ Scenario ID 60 IS present in the file")
                # Find its position
                for i, log in enumerate(logs):
                    if log.get("scenario_id") == 60:
                        print(f"  Found at index {i}")
                        break
            else:
                print("✗ Scenario ID 60 is NOT present in the file")
            
            # Show some example IDs around 60
            nearby_ids = [sid for sid in valid_ids if 55 <= sid <= 65]
            if nearby_ids:
                print(f"IDs near 60: {sorted(nearby_ids)}")
            
            # Check for duplicates
            duplicates = []
            seen = set()
            for sid in valid_ids:
                if sid in seen:
                    duplicates.append(sid)
                seen.add(sid)
            
            if duplicates:
                print(f"Duplicate scenario_ids found: {set(duplicates)}")
            else:
                print("No duplicate scenario_ids found")
        else:
            print("No valid scenario_ids found in file")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Error analyzing file: {e}")

# Run the debug
if __name__ == "__main__":
    file_path = "base_files/logs/MedQA_Ext_none_bias_corrected_20250804_211039.json"
    debug_scenario_ids(file_path)
    
    # Also check if there are any other log files that might contain scenario 60
    import glob
    all_log_files = glob.glob("base_files/logs/*.json")
    print(f"\nAll JSON files in logs directory: {len(all_log_files)}")
    for file in all_log_files:
        print(f"  {file}")