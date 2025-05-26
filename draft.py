import pandas as pd

# Define the data
data = {
    "Date": [
        "2025-05-09", "2025-05-12", "2025-05-13", "2025-05-14", "2025-05-15",
        "2025-05-16", "2025-05-19", "2025-05-20", "2025-05-21", "2025-05-22"
    ],
    "Actual VN-30": [
        1352.25, 1372.04, 1382.78, 1397.87, 1401.49,
        1384.44, 1379.75, 1407.52, 1419.36, 1409.56
    ],
    "Predicted VN-30": [
        1346.701310, 1347.415171, 1369.428847, 1371.215098, 1399.137710,
        1393.034173, 1385.732631, 1396.628894, 1410.550078, 1419.489072
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel("results.xlsx", index=False)

print("Excel file 'VN_304.xlsx' created successfully.")
