"""
Build a violin training manifest from Medley-solos-DB.

Keeps only ciolin clips (this being instrument_id 7), reconstructs each clip's
on-disk path, and assigns whole songs to a K=5 clip-balanced folds bia greedy
longest-first bin-packing. Deterministic (non-random seed), folds are 
song disjoint to ensure there is no/to prevent train/val/test data leakage
"""

import pandas
from pathlib import Path

DATA_ROOT = Path("/mnt/e/Text_to_Audio/data/Medley-solos-DB")
CSV_PATH = "/mnt/e/Text_to_Audio/data/Medley-solos-DB_metadata.csv"
INSTRUMENT_ID = 7 #violin
K = 5
OUTPUT_PATH = 'data/manifest.csv'

df = pandas.read_csv(CSV_PATH)
# REMOVED FILTER: df = df[df["instrument_id"] == INSTRUMENT_ID]

#checks

#print(len(df))
#print(df["song_id"].nunique())

def build_path(row):
    name = f"Medley-solos-DB_{row['subset']}-{row['instrument_id']}_{row['uuid4']}.wav"
    return DATA_ROOT / name

df["filepath"] = df.apply(build_path, axis = 1)
df["exists"] = df["filepath"].apply(lambda p: p.exists())
 

#Checks
#print(df["exists"].sum()) #how many paths resolved
#print(df["filepath"].iloc[0]) #check one full path

song_counts = (
    df["song_id"].value_counts() #clips per song
    .sort_index() #order by song_id first
    .sort_values(ascending=False, kind="stable") #then the largest
)
print(song_counts.head())

fold_totals = [0] * K #running clip count in each fold
fold_of_song = {} #song_id => fold index

for song_id, count in song_counts.items():
    smallest = fold_totals.index(min(fold_totals)) #fold with fewest clips
    fold_of_song[song_id] = smallest #assign this song there
    fold_totals[smallest] += count #update that fold's total

df["fold"] = df["song_id"].map(fold_of_song) #maps fold onto every clip

print(df["fold"].value_counts().sort_index())      # clips per fold
print(df.groupby("fold")["song_id"].nunique())     # songs per fold

#song dis-joint: No song may appear in more then one fold
assert df.groupby("song_id")["fold"].nunique().max() == 1, "LEAKAGE: a song spans multiple folds!!!"

#Check if every referenced file actually exists on disk
assert df["exists"].all(), "some audio files are missing!!!"

#Sanity check on the count we expect
# assert len(df) == 5971, f"expecred 5971 violin clips, got {len(df)}"

#ACTUAL MANIFEST
columns = ["filepath", 'uuid4', "song_id", "instrument", "instrument_id", "subset", "fold"]
df[columns].to_csv(OUTPUT_PATH, index = False)
print(f"wrote {len(df)} rows to {OUTPUT_PATH}")


