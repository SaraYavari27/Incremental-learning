import pandas as pd
import os



data_path = os.getcwd()
csv_path = os.path.join(data_path, "marksheet.csv")
clinical_information = pd.read_csv(csv_path)

# Group the rows based on the "case_ISUP" column
groups = clinical_information.groupby("case_ISUP")

# Loop through the groups and save each one to a separate CSV file
for group_name, group_df in groups:
    output_filename = f"group_{group_name}.csv"
    group_df.to_csv(output_filename, index=False)
