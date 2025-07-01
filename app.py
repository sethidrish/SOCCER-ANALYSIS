import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb
import pandas as pd
from scipy.spatial import ConvexHull

st.set_page_config(layout="wide")
st.title("⚽️ Soccer Tactical Analyzer")

xt_data = [[0.00788321,0.00803695,0.00819069,0.00834443,0.00849817,0.00865191,0.00880565,0.00895939,0.00911313,0.00926687,0.00942061,0.00957435],[0.0088734,0.0090333,0.0091932,0.0093531,0.009513,0.0096729,0.0098328,0.0099927,0.0101526,0.0103125,0.0104724,0.0106323],[0.0100782,0.01024346,0.01040872,0.01057398,0.01073924,0.0109045,0.01106976,0.01123502,0.01140028,0.01156554,0.0117308,0.01189606],[0.01160358,0.01176214,0.0119207,0.01207926,0.01223782,0.01239638,0.01255494,0.0127135,0.01287206,0.01303062,0.01318918,0.01334774],[0.0135894,0.01377488,0.01396036,0.01414584,0.01433132,0.0145168,0.01470228,0.01488776,0.01507324,0.01525872,0.0154442,0.01562968],[0.0163351,0.01662998,0.01692486,0.01721974,0.01751462,0.0178095,0.01810438,0.01839926,0.01869414,0.01898902,0.0192839,0.01957878],[0.0205843,0.02118331,0.02178232,0.02238133,0.02298034,0.02357935,0.02417836,0.02477737,0.02537638,0.02597539,0.0265744,0.02717341],[0.0287163,0.03043633,0.03215636,0.03387639,0.03559642,0.03731645,0.03903648,0.04075651,0.04247654,0.04419657,0.0459166,0.04763663]]

@st.cache_data
def load_data(match_id):
    df = sb.events(match_id=match_id)
    xt_grid = pd.DataFrame(xt_data)
    return df, xt_grid

df, xt_grid = load_data(match_id=7580)

st.sidebar.header("Filters")
teams = df['team'].unique()
selected_team = st.sidebar.selectbox("Select a Team", options=teams)
periods = sorted(df['period'].unique())
selected_period = st.sidebar.selectbox("Select a Period", options=periods)

df_filtered = df[(df['team'] == selected_team) & (df['period'] == selected_period)].copy()
passes = df_filtered[df_filtered['type'] == 'Pass'].copy()

# --- Data Preparation for Network (Simplified) ---
passes['passer'] = passes['player']
passes['recipient'] = passes['pass_recipient']
passes.dropna(subset=['location', 'pass_end_location', 'passer', 'recipient'], inplace=True)
passes['location_x'] = passes['location'].apply(lambda loc: loc[0])
passes['location_y'] = passes['location'].apply(lambda loc: loc[1])

avg_locations = passes.groupby('passer').agg({'location_x': ['mean'], 'location_y': ['mean']})
avg_locations.columns = ['x', 'y']

pass_combinations = passes.groupby(['passer', 'recipient']).size().reset_index(name='pass_count')

pass_combinations = pass_combinations.merge(avg_locations, left_on='passer', right_index=True)
pass_combinations = pass_combinations.merge(avg_locations, left_on='recipient', right_index=True, suffixes=('_start', '_end'))

st.header(f"Passing Network & Team Shape for {selected_team}")
st.subheader(f"Period: {selected_period}")

pitch = Pitch(pitch_type='statsbomb', pitch_color='#011627', line_color='#FDFFFC', stripe=False)
fig, ax = pitch.draw(figsize=(16, 11))
fig.set_facecolor('#011627')

if not avg_locations.empty and len(avg_locations) >= 3:
    hull_df = avg_locations[avg_locations.index.isin(pass_combinations['passer']) | avg_locations.index.isin(pass_combinations['recipient'])]
    if len(hull_df) >= 3:
        hull = ConvexHull(hull_df[['x', 'y']])
        polygon = pitch.polygon([hull_df.iloc[hull.vertices]], ax=ax, edgecolor='#E4572E', facecolor='#E4572E', alpha=0.2, zorder=0)

lines = pitch.lines(
    xstart=pass_combinations['x_start'], ystart=pass_combinations['y_start'],
    xend=pass_combinations['x_end'], yend=pass_combinations['y_end'],
    lw=pass_combinations['pass_count'] * 0.4,
    color='#FDFFFC', ax=ax, zorder=1, alpha=0.6
)

nodes = pitch.scatter(
    x=avg_locations['x'], y=avg_locations['y'],
    s=500, color='#12947F', edgecolors='#FDFFFC', ax=ax, zorder=2
)
for i, row in avg_locations.iterrows():
    pitch.annotate(row.name.split()[-1], xy=(row['x'], row['y']), ax=ax, ha='center', va='center', color='white', fontsize=10, zorder=3)

st.pyplot(fig)

st.header(f"Key Threatening Passes for {selected_team}")

successful_passes = df_filtered[(df_filtered['type'] == 'Pass') & (df_filtered['pass_outcome'].isna())].copy()
successful_passes.dropna(subset=['location', 'pass_end_location'], inplace=True)

if not successful_passes.empty:
    xt_grid_rows, xt_grid_cols = xt_grid.shape
    successful_passes['x_start_bin'] = pd.cut(successful_passes['location'].apply(lambda x: x[0]), bins=xt_grid_cols, labels=False, right=False)
    successful_passes['y_start_bin'] = pd.cut(successful_passes['location'].apply(lambda x: x[1]), bins=xt_grid_rows, labels=False, right=False)
    successful_passes['x_end_bin'] = pd.cut(successful_passes['pass_end_location'].apply(lambda x: x[0]), bins=xt_grid_cols, labels=False, right=False)
    successful_passes['y_end_bin'] = pd.cut(successful_passes['pass_end_location'].apply(lambda x: x[1]), bins=xt_grid_rows, labels=False, right=False)

    successful_passes.dropna(subset=['x_start_bin', 'y_start_bin', 'x_end_bin', 'y_end_bin'], inplace=True)

    successful_passes['xt_start'] = successful_passes.apply(lambda row: xt_grid.iloc[int(row['y_start_bin']), int(row['x_start_bin'])], axis=1)
    successful_passes['xt_end'] = successful_passes.apply(lambda row: xt_grid.iloc[int(row['y_end_bin']), int(row['x_end_bin'])], axis=1)

    successful_passes['xt_added'] = successful_passes['xt_end'] - successful_passes['xt_start']

    top_xt_passes = successful_passes.sort_values(by='xt_added', ascending=False).head(5)
    st.dataframe(top_xt_passes[['player', 'minute', 'second', 'xt_added']])
else:
    st.warning("Not enough pass data to calculate Expected Threat (xT).")