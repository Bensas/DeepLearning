import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

command = input("Select the desired excercise:")

if command == "1a":
  print("Loading data...")
  raw_data = pd.read_csv("europe.csv")
  country_names = raw_data["Country"]
  raw_data = raw_data.drop(columns="Country")
  data_normalized = (raw_data - raw_data.mean()) / raw_data.std()
  data = data_normalized.to_numpy()

  print("Exiting.")

elif command == "1b":
  print("Loading data...")

  print("Exiting.")

elif command == "2":
  print("Loading data...")

  print("Exiting.")