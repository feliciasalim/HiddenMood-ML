def load_csv(data):
    try:
        data.to_csv("mental_health_dataset.csv", index=False)
        print("Data Loaded to CSV Successfully!")
    except Exception as e:
        print(f"Error: {e}")