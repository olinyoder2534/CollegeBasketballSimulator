# %% [markdown]
# ## Exploratory Data Analysis

# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv(r"cbb1.csv") #cbb1 csv file -- has some edits I made in R from cbb20.csv file

# %% [markdown]
# The data set contains offensive and defensive information about each Division 1 baseketball team, as well as their rank and conference.

# %%
df.head()

# %%
df.shape

# %%
df.columns

# %%
kenPom = pd.read_csv(r"kenPom.csv") #kenPom csv file
kenPom = kenPom.iloc[:, 0:5]

# %%
kenPom.head()

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# Since teams did not play the same number of games before the season was cancelled, I created a win percentage column rather than using total wins. Additionally, I brought in data from KenPom such as Net Rating, Luck, and Strength of Schedule.

# %%
# Create win % column
df['WinPercentage'] = (df['W'] / df['G']) * 100

# %%
# merge df and KenPom
df['TEAM'] = df['TEAM'].str.upper()
kenPom['Team'] = kenPom['Team'].str.upper()
df['TEAM'] = df['TEAM'].str.strip()
kenPom['Team'] = kenPom['Team'].str.strip()
df = df.merge(kenPom, how='left', left_on='TEAM', right_on='Team')

# %%
df.head()

# %%
df.drop(columns=['Team'], inplace=True)

# %%
df['NetRTG'].describe()

# %% [markdown]
# ## Tournament Prep

# %% [markdown]
# Creating the bracket. Each team with the highest KenPom Net Rating receives an automatic bid (as if they won their conference tournament), and the remaining at-large bids are given to teams with the highest Wins Above the Bubble.

# %%
## Calculate the Automatic Bids
Automatic = df.loc[df.groupby('CONF')['KenPomRk'].idxmin()]

# Highest ranking team from each conference
print(Automatic[['TEAM', 'CONF', 'KenPomRk']].sort_values(by='KenPomRk', ascending=True))
print(Automatic.shape)

# %%
At_large_prep = df[~df['TEAM'].isin(Automatic['TEAM'])]

# Get the 32 teams with the highest "Wins Above the Bubble" not who did not get an automatic bid
At_large = At_large_prep.nlargest(32, 'WAB')
print(At_large[['TEAM', 'CONF', 'RK']])

# %%
At_large['Bid_Type'] = 'At Large'
Automatic['Bid_Type'] = 'Automatic'

tournament_teams = pd.concat([At_large, Automatic])

tournament_teams = tournament_teams.sort_values(by='KenPomRk', ascending=True)
tournament_teams['Tournament_Rank'] = range(1, len(tournament_teams) + 1)
tournament_teams['Seed'] = ((tournament_teams['Tournament_Rank'] - 1) // 4 + 1)

tournament_teams

# %%
tournament_teams[tournament_teams['Bid_Type'] == 'At Large'][['RK', 'KenPomRk', 'TEAM', 'Bid_Type', 'WAB']]

# %%
df[(df['RK'] >= 60) & (df['RK'] <= 73)]

# %% [markdown]
# ## Creating Winning Simulation

# %% [markdown]
# First, the simulation calculates z-scores for each team's KenPom Net Rating. It then uses the difference in z-scores between the two teams to determine projected win probabilities based on a random normal distribution with shifting bounds. For example, if Team 1 has a z-score of 2.0 (indicating a very good team) and Team 2 has a z-score of 1.0 (indicating an above-average team, but not quite as good as Team 1), Team 1’s projected win probability will be higher than 50%, but not so high that it would be very unexpected for Team 2 to win. However, if Team 2 had a z-score of -2.0 (indicating a very bad team), Team 1's projected win probability would increase such that it would be rare for Team 2 to win.

# %%
from itertools import combinations

# %%
import math

# %%
from matplotlib import pyplot as plt

# %%
plt.hist(kenPom['NetRTG'], bins=10, edgecolor='black')

plt.xlabel('NetRTG')
plt.ylabel('Frequency')
plt.title('Histogram of NetRTG')

plt.show()

# %%
#calculate z score for each teams KenPom rating
mean_netRTG = df['NetRTG'].mean()
std_netRTG = df['NetRTG'].std()

# Calculate the z-score for each team's NetRTG
df['NetRTG_ZScore'] = (df['NetRTG'] - mean_netRTG) / std_netRTG

# %%
df

# %%
df['NetRTG_ZScore'].describe() 

# %%
def simulate_win1(NetRTGZDiff):
    """
    Simulate a win probability based on NetRTGZDiff by splitting the range 
    into 0.1‑wide segments. The base (neutral) case is when -0.05 <= NetRTGZDiff <= 0.05,
    in which case win probability is drawn uniformly from (0.475, 0.525).

    For positive values (an advantage), we define segments as follows:
      • 0.05 < NetRTGZDiff <= 0.15  -> seg 1: Uniform(0.50, 0.55)
      • 0.15 < NetRTGZDiff <= 0.25  -> seg 2: Uniform(0.5125, 0.5625)
      • 0.25 < NetRTGZDiff <= 0.35  -> seg 3: Uniform(0.525, 0.575)
      • 0.35 < NetRTGZDiff <= 0.45  -> seg 4: Uniform(0.5375, 0.5875)
      • 0.45 < NetRTGZDiff <= 0.55  -> seg 5: Uniform(0.55, 0.60)
      • 0.55 < NetRTGZDiff <= 0.65  -> seg 6: Uniform(0.5625, 0.6125)
      • 0.65 < NetRTGZDiff <= 0.75  -> seg 7: Uniform(0.575, 0.625)
      • 0.75 < NetRTGZDiff <= 0.85  -> seg 8: Uniform(0.5875, 0.6375)
    and so on—each additional segment adds 0.0125 to both the lower and
    upper bounds—until for high values the lower bound would reach 0.95; then
    the probability is capped and drawn uniformly from (0.95, 1.0).

    For negative values (a disadvantage), the logic is mirrored:
      • -0.15 <= NetRTGZDiff < -0.05  -> seg 1: Uniform(0.45, 0.50)
      • -0.25 <= NetRTGZDiff < -0.15  -> seg 2: Uniform(0.4375, 0.4875)
      • -0.35 <= NetRTGZDiff < -0.25  -> seg 3: Uniform(0.425, 0.475)
      • -0.45 <= NetRTGZDiff < -0.35  -> seg 4: Uniform(0.4125, 0.4625)
      • -0.55 <= NetRTGZDiff < -0.45  -> seg 5: Uniform(0.40, 0.45)
      • -0.65 <= NetRTGZDiff < -0.55  -> seg 6: Uniform(0.3875, 0.4375)
      • -0.75 <= NetRTGZDiff < -0.65  -> seg 7: Uniform(0.375, 0.425)
      • -0.85 <= NetRTGZDiff < -0.75  -> seg 8: Uniform(0.3625, 0.4125)
    and so on—subtracting 0.0125 per segment—until the lower bound is capped at 0.00,
    at which point win probability is drawn uniformly from (0.00, 0.05).
    """
    # teams with nearly identical KenPom rankings
    if -0.05 <= NetRTGZDiff <= 0.05:
        return np.random.uniform(0.475, 0.525)
    
    # For positive values (teams with higher kenpom - teams with lower kenpom)
    if NetRTGZDiff > 0.05:
        seg = math.ceil((NetRTGZDiff - 0.05) / 0.1)
        # For seg==1 (i.e. 0.05 < diff <= 0.15) the interval is (0.50, 0.55).
        lower = 0.50 + (seg - 1) * 0.0125
        upper = 0.55 + (seg - 1) * 0.0125
        # Cap the interval when the lower bound reaches 0.95.
        if lower >= 0.95:
            lower, upper = 0.95, 1.0
        return np.random.uniform(lower, upper)
    
    # For negative values (inverse of previous case)
    else:  # NetRTGZDiff < -0.05
        seg = math.ceil((abs(NetRTGZDiff) - 0.05) / 0.1)
        lower = 0.45 - (seg - 1) * 0.0125
        upper = 0.50 - (seg - 1) * 0.0125
        # Cap the interval when the lower bound falls to 0.00.
        if lower <= 0.00:
            lower, upper = 0.00, 0.05
        return np.random.uniform(lower, upper)

# %%
matchups = []
teams = df['TEAM'].unique()

for team1, team2 in combinations(teams, 2):
    team1_stats = df[df['TEAM'] == team1].iloc[0]
    team2_stats = df[df['TEAM'] == team2].iloc[0]
    
    matchup = {
        'team1': team1,
        'team2': team2,
        'team1_NetRTG': team1_stats['NetRTG_ZScore'], 
        'team2_NetRTG': team2_stats['NetRTG_ZScore'], 
        'NetRTG_ZScore_diff': team1_stats['NetRTG_ZScore'] - team2_stats['NetRTG_ZScore'], 
        'conf_match': 1 if team1_stats['CONF'] == team2_stats['CONF'] else 0  #1 = same conference, 0 = diff
    }
    
    numeric_stats = ['ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 
                     'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', 'X2P_O', 'X2P_D', 'X3P_O', 
                     'X3P_D', 'ADJ_T', 'WAB', 'WinPercentage', 'KenPomRk', 'NetRTG', 'Luck', 'SoS', 'NetRTG_ZScore']
    
    for stat in numeric_stats:
        matchup[f'{stat}_diff'] = team1_stats[stat] - team2_stats[stat]
    
    matchup['WinProb'] = simulate_win1(matchup['NetRTG_ZScore_diff'])
    
    matchups.append(matchup)

matchup_df1 = pd.DataFrame(matchups)

# %%
matchup

# %%
matchup_df1[matchup_df1['team1'] == "KANSAS"]

# %%
matchup_df1[matchup_df1['team1'] == "KENNESAW ST."]

# %%
matchup_df1[matchup_df1['team1'] == "NORTH TEXAS"]

# %% [markdown]
# ## Modeling

# %% [markdown]
# Since we already have the KenPom Net Rating for the entire population, modeling to predict it is unnecessary and would only introduce noise into the simulation. While this approach contradicts best practices, in the spirit of March Madness, a little extra chaos doesn’t hurt.

# %%
from sklearn.model_selection import train_test_split

# %%
df.columns

# %%
feature_columns = [
    'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 
    'EFG_O_diff', 'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
    'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
    'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 'Luck_diff', 'SoS_diff', 'conf_match'
]

X = matchup_df1[feature_columns]  
y = matchup_df1['WinProb']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

# Assumptions of a linear model are not met

# %%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# %%
# Win probs for KU -- num. 1 team in the country at the end of the season
kansas_stats = df[df['TEAM'] == "KANSAS"].iloc[0]

# Create matchups where Kansas is team1
matchups = []
for team in df['TEAM'].unique():
    if team != "KANSAS":
        team_stats = df[df['TEAM'] == team].iloc[0]
        
        matchup = {
            'team1': "KANSAS",
            'team2': team,
            'rank_diff': kansas_stats['RK'] - team_stats['RK'],
            'ADJOE_diff': kansas_stats['ADJOE'] - team_stats['ADJOE'],
            'ADJDE_diff': kansas_stats['ADJDE'] - team_stats['ADJDE'],
            'BARTHAG_diff': kansas_stats['BARTHAG'] - team_stats['BARTHAG'],
            'EFG_O_diff': kansas_stats['EFG_O'] - team_stats['EFG_O'],
            'EFG_D_diff': kansas_stats['EFG_D'] - team_stats['EFG_D'],
            'TOR_diff': kansas_stats['TOR'] - team_stats['TOR'],
            'TORD_diff': kansas_stats['TORD'] - team_stats['TORD'],
            'ORB_diff': kansas_stats['ORB'] - team_stats['ORB'],
            'DRB_diff': kansas_stats['DRB'] - team_stats['DRB'],
            'FTR_diff': kansas_stats['FTR'] - team_stats['FTR'],
            'FTRD_diff': kansas_stats['FTRD'] - team_stats['FTRD'],
            'X2P_O_diff': kansas_stats['X2P_O'] - team_stats['X2P_O'],
            'X2P_D_diff': kansas_stats['X2P_D'] - team_stats['X2P_D'],
            'X3P_O_diff': kansas_stats['X3P_O'] - team_stats['X3P_O'],
            'X3P_D_diff': kansas_stats['X3P_D'] - team_stats['X3P_D'],
            'ADJ_T_diff': kansas_stats['ADJ_T'] - team_stats['ADJ_T'],
            'WAB_diff': kansas_stats['WAB'] - team_stats['WAB'],
            'WinPercentage_diff': kansas_stats['WinPercentage'] - team_stats['WinPercentage'],
            'Luck_diff': kansas_stats['Luck'] - team_stats['Luck'],
            'SoS_diff': kansas_stats['SoS'] - team_stats['SoS'],
            'conf_match': 1 if kansas_stats['CONF'] == team_stats['CONF'] else 0
        }
        matchups.append(matchup)

kansas_matchups = pd.DataFrame(matchups)

feature_columns = [
    'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 
    'EFG_O_diff', 'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
    'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
    'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 'Luck_diff', 'SoS_diff', 'conf_match'
]
X_kansas = kansas_matchups[feature_columns]

win_probabilities = model.predict(X_kansas)

kansas_matchups['win_probability'] = win_probabilities

kansas_matchups[['team2', 'win_probability']]
kansas_matchups[['team2', 'win_probability']].iloc[0:25]
kansas_matchups[['team2', 'win_probability']].iloc[340:]

# %% [markdown]
# ## Simulation

# %% [markdown]
# Simulating the tournament structure.

# %%
# Generate the bracket by pairing teams based on Tournament_Rank
num_teams = len(tournament_teams)
bracket = []

for i in range(num_teams // 2):
    team1 = tournament_teams.iloc[i]  # Higher-ranked team
    team2 = tournament_teams.iloc[num_teams - i - 1]  # Lower-ranked team
    matchup = {
        'Team1': team1['TEAM'],
        'Team1_Seed': team1['Seed'],
        'Team1_Rank': team1['Tournament_Rank'],
        'Team2': team2['TEAM'],
        'Team2_Seed': team2['Seed'],
        'Team2_Rank': team2['Tournament_Rank']
    }
    bracket.append(matchup)

# Convert the bracket into a DataFrame for display
bracket_df = pd.DataFrame(bracket)

# %%
df = df.merge(tournament_teams[['TEAM', 'Seed', 'Tournament_Rank']], on='TEAM', how='left')

# %%
df.shape

# %%
df

# %%
bracket_df

# %%
def simulate_tournament(tournament_teams, model, df, verbose=False):
    tournament_teams = tournament_teams.sort_values(by='Tournament_Rank')
    current_round_teams = tournament_teams['TEAM'].tolist()
    round_number = 1

    feature_columns = [
        #'rank_diff', 
        'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 'EFG_O_diff', 
        'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
        'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
        'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 
        'Luck_diff', 'SoS_diff', 'conf_match'
    ]

    while len(current_round_teams) > 1:
        print(f"\nSimulating Round {round_number}...")
        next_round_teams = []

        # Build and display the matchups based on the fixed bracket order.
        print(f"Matchups for Round {round_number}:")
        matchups = []
        num_games = len(current_round_teams) // 2
        for i in range(num_games):
            team1 = current_round_teams[i]
            team2 = current_round_teams[-(i + 1)]  # Pair first with last, etc.
            matchups.append((team1, team2))
            
            team1_info = df[df['TEAM'] == team1].iloc[0]
            team2_info = df[df['TEAM'] == team2].iloc[0]
            print(f"  {team1} (Seed: {team1_info['Seed']}, Tournament Rank: {team1_info['Tournament_Rank']}) vs "
                  f"{team2} (Seed: {team2_info['Seed']}, Tournament Rank: {team2_info['Tournament_Rank']})")

        # Simulate each matchup
        for team1, team2 in matchups:
            # Retrieve team info (needed later for seed adjustments)
            team1_info = df[df['TEAM'] == team1].iloc[0]
            team2_info = df[df['TEAM'] == team2].iloc[0]
            
            # Calculate feature differences for the matchup.
            feature_vector = {
                #'rank_diff': df[df['TEAM'] == team1]['RK'].values[0] - df[df['TEAM'] == team2]['RK'].values[0],
                'ADJOE_diff': df[df['TEAM'] == team1]['ADJOE'].values[0] - df[df['TEAM'] == team2]['ADJOE'].values[0],
                'ADJDE_diff': df[df['TEAM'] == team1]['ADJDE'].values[0] - df[df['TEAM'] == team2]['ADJDE'].values[0],
                'BARTHAG_diff': df[df['TEAM'] == team1]['BARTHAG'].values[0] - df[df['TEAM'] == team2]['BARTHAG'].values[0],
                'EFG_O_diff': df[df['TEAM'] == team1]['EFG_O'].values[0] - df[df['TEAM'] == team2]['EFG_O'].values[0],
                'EFG_D_diff': df[df['TEAM'] == team1]['EFG_D'].values[0] - df[df['TEAM'] == team2]['EFG_D'].values[0],
                'TOR_diff': df[df['TEAM'] == team1]['TOR'].values[0] - df[df['TEAM'] == team2]['TOR'].values[0],
                'TORD_diff': df[df['TEAM'] == team1]['TORD'].values[0] - df[df['TEAM'] == team2]['TORD'].values[0],
                'ORB_diff': df[df['TEAM'] == team1]['ORB'].values[0] - df[df['TEAM'] == team2]['ORB'].values[0],
                'DRB_diff': df[df['TEAM'] == team1]['DRB'].values[0] - df[df['TEAM'] == team2]['DRB'].values[0],
                'FTR_diff': df[df['TEAM'] == team1]['FTR'].values[0] - df[df['TEAM'] == team2]['FTR'].values[0],
                'FTRD_diff': df[df['TEAM'] == team1]['FTRD'].values[0] - df[df['TEAM'] == team2]['FTRD'].values[0],
                'X2P_O_diff': df[df['TEAM'] == team1]['X2P_O'].values[0] - df[df['TEAM'] == team2]['X2P_O'].values[0],
                'X2P_D_diff': df[df['TEAM'] == team1]['X2P_D'].values[0] - df[df['TEAM'] == team2]['X2P_D'].values[0],
                'X3P_O_diff': df[df['TEAM'] == team1]['X3P_O'].values[0] - df[df['TEAM'] == team2]['X3P_O'].values[0],
                'X3P_D_diff': df[df['TEAM'] == team1]['X3P_D'].values[0] - df[df['TEAM'] == team2]['X3P_D'].values[0],
                'ADJ_T_diff': df[df['TEAM'] == team1]['ADJ_T'].values[0] - df[df['TEAM'] == team2]['ADJ_T'].values[0],
                'WAB_diff': df[df['TEAM'] == team1]['WAB'].values[0] - df[df['TEAM'] == team2]['WAB'].values[0],
                'WinPercentage_diff': df[df['TEAM'] == team1]['WinPercentage'].values[0] - df[df['TEAM'] == team2]['WinPercentage'].values[0],
                'Luck_diff': df[df['TEAM'] == team1]['Luck'].values[0] - df[df['TEAM'] == team2]['Luck'].values[0],
                'SoS_diff': df[df['TEAM'] == team1]['SoS'].values[0] - df[df['TEAM'] == team2]['SoS'].values[0],
                'conf_match': 1 if team1_info['CONF'] == team2_info['CONF'] else 0
            }

            # Ensure feature_df matches the training features
            feature_df = pd.DataFrame([feature_vector])[feature_columns]

            # Predict the win probability for team1.
            win_probability = model.predict(feature_df)[0]

            # Adjust win probability based on seed differences.
            # Retrieve seeds (assumed to be numeric, with lower numbers being the higher seed)
            team1_seed = team1_info['Seed']
            team2_seed = team2_info['Seed']
            
            # Calculate the absolute difference in seeds.
            seed_diff = abs(team1_seed - team2_seed)
            
            # Apply 5% reduction per seed difference. For example, if team1 is a 1 seed and team2 is a 16 seed, the difference is 15 seeds, so the reduction is 0.05 * 15 = 0.75. If team1 is a 2 seed and team2 is a 3 seed, the difference is 1 seed, so the reduction is 0.05 * 1 = 0.05.
            reduction = seed_diff * 0.05

            # Adjust win probability based on the seeding advantage.
            if team1_seed < team2_seed:
                # team1 is the favorite, so add the reduction fraction of the underdog's remaining chance.
                adjusted_win_probability = win_probability + (1 - win_probability) * reduction
            elif team1_seed > team2_seed:
                # team1 is the underdog, so decrease its win probability.
                adjusted_win_probability = win_probability * (1 - reduction)
            else:
                # In the unlikely event that both seeds are the same, no adjustment is applied.
                adjusted_win_probability = win_probability

            # Print the adjusted win probabilities.
            print(f"Projected win probability: {team1} vs {team2}")
            print(f"  {team1}: {adjusted_win_probability:.2f}")
            print(f"  {team2}: {1 - adjusted_win_probability:.2f}")

            # Simulate the winner using the adjusted win probability.
            if np.random.rand() < adjusted_win_probability:
                next_round_teams.append(team1)  # team1 wins
            else:
                next_round_teams.append(team2)  # team2 wins

        # Display advancing teams without reordering them.
        print(f"\nTeams advancing to Round {round_number + 1}:")
        for team in next_round_teams:
            team_info = df[df['TEAM'] == team].iloc[0]
            print(f"  {team} (Seed: {team_info['Seed']}, Tournament Rank: {team_info['Tournament_Rank']})")

        # Update the teams for the next round.
        current_round_teams = next_round_teams
        round_number += 1

    return current_round_teams[0]


# %%
# Simulate tourney
winner = simulate_tournament(tournament_teams, model, df)
print(f"\nThe winner of the tournament is: {winner}")

# %%
from collections import Counter

# %%
#simulate tournament n times
num_simulations = 1000

champion_counts = Counter()

for _ in range(num_simulations):
    champion = simulate_tournament(tournament_teams.copy(), model, df, verbose=False)
    champion_counts[champion] += 1

print("\nChampionship counts over", num_simulations, "simulations:")
for team, wins in champion_counts.items():
    print(f"{team}: {wins}")

# %% [markdown]
# ## Score Predictions

# %%
df['PPG'] = df['ADJOE']/100*df['ADJ_T']
df['PAPG'] = df['ADJDE']/100*df['ADJ_T']

# %%
df.head()

# %%
def simulate_tournament1(tournament_teams, model, df, verbose=True):
    # Ensure teams are sorted by Tournament_Rank.
    tournament_teams = tournament_teams.sort_values(by='Tournament_Rank')
    current_round_teams = tournament_teams['TEAM'].tolist()
    round_number = 1

    #features from the ml model
    feature_columns = [
        #'rank_diff', 
        'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 'EFG_O_diff', 
        'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
        'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
        'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 
        'Luck_diff', 'SoS_diff', 'conf_match'
    ]
    
    margin_factor = np.random.uniform(0,1) #how much the score disparity between higher and lower seeds should be (.5 works better well)
    std_dev = np.random.uniform(2,6) #just kinda a guess and check -- tried doing this statistically and was getting scores like RUTGERS 143 - USC 139 | Winner: RUTGERS
    
    while len(current_round_teams) > 1:
        print(f"\n----------------------------------------------------------\nSimulating Round {round_number}...")
        next_round_teams = []
        round_results = []
    
        # Build and display the matchups based on a fixed bracket order.
        print(f"Matchups for Round {round_number}:")
        matchups = []
        num_games = len(current_round_teams) // 2
        for i in range(num_games):
            team1 = current_round_teams[i]
            team2 = current_round_teams[-(i + 1)]  # Pair first with last, etc.
            matchups.append((team1, team2))
            
            team1_info = df[df['TEAM'] == team1].iloc[0]
            team2_info = df[df['TEAM'] == team2].iloc[0]
            print(f"  {team1} (Seed: {team1_info['Seed']}, Tournament Rank: {team1_info['Tournament_Rank']}) vs "
                  f"{team2} (Seed: {team2_info['Seed']}, Tournament Rank: {team2_info['Tournament_Rank']})")
    
        # Simulate each matchup.
        for team1, team2 in matchups:
            # Retrieve information for both teams.
            team1_info = df[df['TEAM'] == team1].iloc[0]
            team2_info = df[df['TEAM'] == team2].iloc[0]
            
            # Calculate feature differences for the matchup.
            feature_vector = {
                #'rank_diff': df[df['TEAM'] == team1]['RK'].values[0] - df[df['TEAM'] == team2]['RK'].values[0],
                'ADJOE_diff': team1_info['ADJOE'] - team2_info['ADJOE'],
                'ADJDE_diff': team1_info['ADJDE'] - team2_info['ADJDE'],
                'BARTHAG_diff': team1_info['BARTHAG'] - team2_info['BARTHAG'],
                'EFG_O_diff': team1_info['EFG_O'] - team2_info['EFG_O'],
                'EFG_D_diff': team1_info['EFG_D'] - team2_info['EFG_D'],
                'TOR_diff': team1_info['TOR'] - team2_info['TOR'],
                'TORD_diff': team1_info['TORD'] - team2_info['TORD'],
                'ORB_diff': team1_info['ORB'] - team2_info['ORB'],
                'DRB_diff': team1_info['DRB'] - team2_info['DRB'],
                'FTR_diff': team1_info['FTR'] - team2_info['FTR'],
                'FTRD_diff': team1_info['FTRD'] - team2_info['FTRD'],
                'X2P_O_diff': team1_info['X2P_O'] - team2_info['X2P_O'],
                'X2P_D_diff': team1_info['X2P_D'] - team2_info['X2P_D'],
                'X3P_O_diff': team1_info['X3P_O'] - team2_info['X3P_O'],
                'X3P_D_diff': team1_info['X3P_D'] - team2_info['X3P_D'],
                'ADJ_T_diff': team1_info['ADJ_T'] - team2_info['ADJ_T'],
                'WAB_diff': team1_info['WAB'] - team2_info['WAB'],
                'WinPercentage_diff': team1_info['WinPercentage'] - team2_info['WinPercentage'],
                'Luck_diff': team1_info['Luck'] - team2_info['Luck'],
                'SoS_diff': team1_info['SoS'] - team2_info['SoS'],
                'conf_match': 1 if team1_info['CONF'] == team2_info['CONF'] else 0
            }
    
            feature_df = pd.DataFrame([feature_vector])[feature_columns]
    
            #win prob for team1.
            win_probability = model.predict(feature_df)[0]
    
            #Adjust win probability based on seed differences. A 1-16 seed matchup is a 15 seed difference, so the reduction in win % for the 16 seed is 0.05 * 15 = 0.75.
            team1_seed = team1_info['Seed']
            team2_seed = team2_info['Seed']
            seed_diff = abs(team1_seed - team2_seed)
            reduction = seed_diff * 0.05
            
            if team1_seed < team2_seed:
                # Team1 is favored.
                adjusted_win_probability = win_probability + (1 - win_probability) * reduction
            elif team1_seed > team2_seed:
                # Team1 is the underdog.
                adjusted_win_probability = win_probability * (1 - reduction)
            else:
                adjusted_win_probability = win_probability
            
            print(f"\nProjected win probabilities for {team1} vs {team2}:")
            print(f"  {team1}: {adjusted_win_probability:.2f}")
            print(f"  {team2}: {1 - adjusted_win_probability:.2f}")
    
            # Base expected scores from team offensive and opponent defensive metrics.
            base_team1_expected = (team1_info['PPG'] + team2_info['PAPG']) / 2
            base_team2_expected = (team2_info['PPG'] + team1_info['PAPG']) / 2
    
            # Adjust expected scores based on the seeding differential.
            if team1_seed < team2_seed:
                margin_adjust = seed_diff * margin_factor
                team1_expected = base_team1_expected + margin_adjust
                team2_expected = base_team2_expected - margin_adjust
            elif team1_seed > team2_seed:
                margin_adjust = seed_diff * margin_factor
                team1_expected = base_team1_expected - margin_adjust
                team2_expected = base_team2_expected + margin_adjust
            else:
                team1_expected = base_team1_expected
                team2_expected = base_team2_expected
    
            # Generate simulated scores using a normal distribution, ensuring non-negative results.
            team1_score = max(0, int(np.rint(np.random.normal(loc=team1_expected, scale=std_dev))))
            team2_score = max(0, int(np.rint(np.random.normal(loc=team2_expected, scale=std_dev))))
    
            # Use a single random draw to decide the winner and force consistency with the scores.
            rand_val = np.random.rand()
            if rand_val < adjusted_win_probability:
                winner = team1
                # Ensure team1 wins in the scoreline.
                if team1_score <= team2_score:
                    team1_score = team2_score + np.random.randint(1, 16) #makes team 1 win by 1-15 points 
                    #thought about using np.random.poisson(lam=4) to make games closer, but just doing a random int is better
            else:
                winner = team2
                if team2_score <= team1_score:
                    team2_score = team1_score + np.random.randint(1, 16)
    
            #print(f"Final Score: {team1} {team1_score} - {team2} {team2_score}\n")
    
            # Record the outcome for this game.
            next_round_teams.append(winner)
            round_results.append({
                "team1": team1,
                "team1_score": team1_score,
                "team2": team2,
                "team2_score": team2_score,
                "winner": winner
            })
    
        #Game scores
        print(f"\nResults for Round {round_number}:")
        for result in round_results:
            print(f"  {result['team1']} {result['team1_score']} - {result['team2']} {result['team2_score']} | Winner: {result['winner']}")
    
        # Display the teams advancing to the next round with team info.
        print(f"\nTeams advancing to Round {round_number + 1}:")
        for team in next_round_teams:
            team_info = df[df['TEAM'] == team].iloc[0]
            print(f"  {team} (Seed: {team_info['Seed']}, Tournament Rank: {team_info['Tournament_Rank']})")
    
        current_round_teams = next_round_teams
        round_number += 1
    
    return current_round_teams[0]


# %%
# Simulate tourney
winner = simulate_tournament1(tournament_teams, model, df)
print(f"\nThe winner of the tournament is: {winner}")

# %% [markdown]
# ## 2025 Simulation

# %%
df_2025 = pd.read_csv(r"cbb25.csv")

# %%
df_2025.head()

# %%
print(df_2025.columns)
print(df.columns)

# %%
df_2025 = df_2025.rename(columns={"Team": "TEAM",
                                  "2P_O": "X2P_O",
                                  "3P_O": "X3P_O",
                                  "2P_D": "X2P_D",
                                  "3P_D": "X3P_D",
                                  "KenPomRank": "KenPomRk"})
cols = ['TEAM'] + [col for col in df_2025.columns if col != 'TEAM']
df_2025 = df_2025[cols]

# %%
mean_netRTG1 = df_2025['NetRTG'].mean()
std_netRTG1 = df_2025['NetRTG'].std()

# Calculate the z-score for each team's NetRTG
df_2025['NetRTG_ZScore'] = (df_2025['NetRTG'] - mean_netRTG1) / std_netRTG1

# %%
df_sorted = df_2025.sort_values(by="NetRTG_ZScore", ascending=False)
df_sorted.head()

# %%
tournament_teams_2025 = df_2025[df_2025["Seed"].notnull()]

# %%
tournament_teams_2025.shape

# %%
tournament_teams_2025.head()

# %%
winner = simulate_tournament1(tournament_teams_2025, model, df_2025)
print(f"\nThe winner of the tournament is: {winner}")

# %%
#simulate tournament n times
num_simulations = 1000

champion_counts = Counter()

for _ in range(num_simulations):
    champion = simulate_tournament1(tournament_teams_2025.copy(), model, df_2025, verbose=False)
    champion_counts[champion] += 1

print("Championship counts over", num_simulations, "simulations:")
for team, wins in champion_counts.items():
    print(f"{team}: {wins}")

# %%
# Win probs for Duke -- num. 1 team in the country at the end of the season
kansas_stats = df_2025[df_2025['TEAM'] == "Duke"].iloc[0]

# Create matchups where Duke is team1
matchups = []
for team in df_2025['TEAM'].unique():
    if team != "Duke":
        team_stats = df_2025[df_2025['TEAM'] == team].iloc[0]
        
        matchup = {
            'team1': "Duke",
            'team2': team,
            'rank_diff': kansas_stats['RK'] - team_stats['RK'],
            'ADJOE_diff': kansas_stats['ADJOE'] - team_stats['ADJOE'],
            'ADJDE_diff': kansas_stats['ADJDE'] - team_stats['ADJDE'],
            'BARTHAG_diff': kansas_stats['BARTHAG'] - team_stats['BARTHAG'],
            'EFG_O_diff': kansas_stats['EFG_O'] - team_stats['EFG_O'],
            'EFG_D_diff': kansas_stats['EFG_D'] - team_stats['EFG_D'],
            'TOR_diff': kansas_stats['TOR'] - team_stats['TOR'],
            'TORD_diff': kansas_stats['TORD'] - team_stats['TORD'],
            'ORB_diff': kansas_stats['ORB'] - team_stats['ORB'],
            'DRB_diff': kansas_stats['DRB'] - team_stats['DRB'],
            'FTR_diff': kansas_stats['FTR'] - team_stats['FTR'],
            'FTRD_diff': kansas_stats['FTRD'] - team_stats['FTRD'],
            'X2P_O_diff': kansas_stats['X2P_O'] - team_stats['X2P_O'],
            'X2P_D_diff': kansas_stats['X2P_D'] - team_stats['X2P_D'],
            'X3P_O_diff': kansas_stats['X3P_O'] - team_stats['X3P_O'],
            'X3P_D_diff': kansas_stats['X3P_D'] - team_stats['X3P_D'],
            'ADJ_T_diff': kansas_stats['ADJ_T'] - team_stats['ADJ_T'],
            'WAB_diff': kansas_stats['WAB'] - team_stats['WAB'],
            'WinPercentage_diff': kansas_stats['WinPercentage'] - team_stats['WinPercentage'],
            'Luck_diff': kansas_stats['Luck'] - team_stats['Luck'],
            'SoS_diff': kansas_stats['SoS'] - team_stats['SoS'],
            'conf_match': 1 if kansas_stats['CONF'] == team_stats['CONF'] else 0
        }
        matchups.append(matchup)

kansas_matchups = pd.DataFrame(matchups)

feature_columns = [
    'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 
    'EFG_O_diff', 'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
    'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
    'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 'Luck_diff', 'SoS_diff', 'conf_match'
]
X_kansas = kansas_matchups[feature_columns]

win_probabilities = model.predict(X_kansas)

kansas_matchups['win_probability'] = win_probabilities

kansas_matchups[['team2', 'win_probability']]
kansas_matchups[['team2', 'win_probability']].iloc[0:25]
#kansas_matchups[['team2', 'win_probability']].iloc[340:]

# %% [markdown]
# Training things on 2025 data

# %%
df_2025.head()
#3.036701-2.708994

# %%
matchups = []
teams = df_2025['TEAM'].unique()

for team1, team2 in combinations(teams, 2):
    team1_stats = df_2025[df_2025['TEAM'] == team1].iloc[0]
    team2_stats = df_2025[df_2025['TEAM'] == team2].iloc[0]
    
    matchup = {
        'team1': team1,
        'team2': team2,
        'team1_NetRTG': team1_stats['NetRTG_ZScore'], 
        'team2_NetRTG': team2_stats['NetRTG_ZScore'], 
        'NetRTG_ZScore_diff': team1_stats['NetRTG_ZScore'] - team2_stats['NetRTG_ZScore'], 
        'conf_match': 1 if team1_stats['CONF'] == team2_stats['CONF'] else 0  #1 = same conference, 0 = diff
    }
    
    numeric_stats = ['ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 
                     'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', 'X2P_O', 'X2P_D', 'X3P_O', 
                     'X3P_D', 'ADJ_T', 'WAB', 'WinPercentage', 'KenPomRk', 'NetRTG', 'Luck', 'SoS', 'NetRTG_ZScore']
    
    for stat in numeric_stats:
        matchup[f'{stat}_diff'] = team1_stats[stat] - team2_stats[stat]
    
    matchup['WinProb'] = simulate_win1(matchup['NetRTG_ZScore_diff'])
    
    matchups.append(matchup)

matchup_df1 = pd.DataFrame(matchups)

feature_columns = [
    'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 
    'EFG_O_diff', 'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
    'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
    'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 'Luck_diff', 'SoS_diff', 'conf_match'
]

# %%
matchup_df1[matchup_df1['team1'] == 'Duke']

# %%
X = matchup_df1[feature_columns]  
y = matchup_df1['WinProb']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestRegressor(random_state=42)
model1.fit(X_train, y_train)

# %%
from xgboost import XGBRegressor

# %%
model2 = XGBRegressor(random_state=42)
model2.fit(X_train, y_train)

print(model2)

# %%
from sklearn.metrics import mean_squared_error

y_pred1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
print(mse1)

y_pred2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)
print(mse2)

# %%
# Win probs for Duke -- num. 1 team in the country at the end of the season
kansas_stats = df_2025[df_2025['TEAM'] == "Duke"].iloc[0]

# Create matchups where Duke is team1
matchups = []
for team in df_2025['TEAM'].unique():
    if team != "Duke":
        team_stats = df_2025[df_2025['TEAM'] == team].iloc[0]
        
        matchup = {
            'team1': "Duke",
            'team2': team,
            'rank_diff': kansas_stats['RK'] - team_stats['RK'],
            'ADJOE_diff': kansas_stats['ADJOE'] - team_stats['ADJOE'],
            'ADJDE_diff': kansas_stats['ADJDE'] - team_stats['ADJDE'],
            'BARTHAG_diff': kansas_stats['BARTHAG'] - team_stats['BARTHAG'],
            'EFG_O_diff': kansas_stats['EFG_O'] - team_stats['EFG_O'],
            'EFG_D_diff': kansas_stats['EFG_D'] - team_stats['EFG_D'],
            'TOR_diff': kansas_stats['TOR'] - team_stats['TOR'],
            'TORD_diff': kansas_stats['TORD'] - team_stats['TORD'],
            'ORB_diff': kansas_stats['ORB'] - team_stats['ORB'],
            'DRB_diff': kansas_stats['DRB'] - team_stats['DRB'],
            'FTR_diff': kansas_stats['FTR'] - team_stats['FTR'],
            'FTRD_diff': kansas_stats['FTRD'] - team_stats['FTRD'],
            'X2P_O_diff': kansas_stats['X2P_O'] - team_stats['X2P_O'],
            'X2P_D_diff': kansas_stats['X2P_D'] - team_stats['X2P_D'],
            'X3P_O_diff': kansas_stats['X3P_O'] - team_stats['X3P_O'],
            'X3P_D_diff': kansas_stats['X3P_D'] - team_stats['X3P_D'],
            'ADJ_T_diff': kansas_stats['ADJ_T'] - team_stats['ADJ_T'],
            'WAB_diff': kansas_stats['WAB'] - team_stats['WAB'],
            'WinPercentage_diff': kansas_stats['WinPercentage'] - team_stats['WinPercentage'],
            'Luck_diff': kansas_stats['Luck'] - team_stats['Luck'],
            'SoS_diff': kansas_stats['SoS'] - team_stats['SoS'],
            'conf_match': 1 if kansas_stats['CONF'] == team_stats['CONF'] else 0
        }
        matchups.append(matchup)

kansas_matchups = pd.DataFrame(matchups)

feature_columns = [
    'ADJOE_diff', 'ADJDE_diff', 'BARTHAG_diff', 
    'EFG_O_diff', 'EFG_D_diff', 'TOR_diff', 'TORD_diff', 'ORB_diff', 'DRB_diff', 
    'FTR_diff', 'FTRD_diff', 'X2P_O_diff', 'X2P_D_diff', 'X3P_O_diff', 
    'X3P_D_diff', 'ADJ_T_diff', 'WAB_diff', 'WinPercentage_diff', 'Luck_diff', 'SoS_diff', 'conf_match'
]
X_kansas = kansas_matchups[feature_columns]

win_probabilitiesRF = model1.predict(X_kansas)
win_probabilitiesXG = model2.predict(X_kansas)

kansas_matchups['win_probabilityRF'] = win_probabilitiesRF
kansas_matchups['win_probabilityXG'] = win_probabilitiesXG

kansas_matchups[['team2', 'win_probabilityRF', 'win_probabilityXG']]
#kansas_matchups[['team2', 'win_probability']].iloc[0:25]
#kansas_matchups[['team2', 'win_probability']].iloc[340:]



#Win probs jump for Duke when the model is trained on the current year's data

# %% [markdown]
# Defining tournament simulating for current year, using XGBoost this year instead of a RF

# %%
winner = simulate_tournament1(tournament_teams_2025, model2, df_2025)
print(f"\nThe winner of the tournament is: {winner}")

# %%
#simulate tournament n times
num_simulations = 1000

champion_counts = Counter()

for _ in range(num_simulations):
    champion = simulate_tournament1(tournament_teams_2025.copy(), model2, df_2025, verbose=False)
    champion_counts[champion] += 1

print("\nChampionship counts over", num_simulations, "simulations:")
for team, wins in champion_counts.items():
    print(f"{team}: {wins}")


