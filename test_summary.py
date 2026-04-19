from src.utils.forecast_summary import generate_summary

# dummy predictions (example)
predictions = [20, 35, 50, 80, 90, 60, 30]

result = generate_summary(predictions)

print("=== SUMMARY ===")
print(result)