# Re-import needed modules after reset
import pandas as pd
folder = 'ready_data'

def create_change_df(file_name):
    # Reload the uploaded file
    if file_name == 'vn_index':
        file_path = f'{folder}/{file_name}_data/cleaned_{file_name}_data.csv'
    else:
        file_path = f'{folder}/vn_30_data/cleaned_{file_name}_data.csv'

    df = pd.read_csv(file_path)
    # Extract Date and Change columns
    if 'Change' not in df.columns:
        df['Change'] = df[df.columns[1]].pct_change()
    df_change = df[['Date', 'Change']].copy()

    # Convert 'Date' to datetime type
    df_change['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    df_change = df_change.sort_values('Date')

    # Save the cleaned DataFrame to a new CSV
    df_change.to_csv(f'raw_data/change_data/change_{file_name}.csv', index=False)

for fn in ['vn_index', 'vn_30', 'vn_30f1']:
    create_change_df(fn)
    print(f'Successful for {fn}')


