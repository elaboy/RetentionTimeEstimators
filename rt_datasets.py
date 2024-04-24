import pandas as pd 
import matplotlib.pyplot as plt
from functools import reduce
import dask.dataframe as dd

def sortPSMTSV(path : str):
    df = pd.read_csv(path, sep='\t', low_memory=True)
    sorted_df = df[["Base Sequence", "Scan Retention Time"]].sort_values("Scan Retention Time").reset_index()
    return sorted_df[["Base Sequence", "Scan Retention Time"]]

# df = pd.read_csv(r"/mnt/f/RetentionTimeProject/OtherPeptideResultsForTraining/PXD005573/RFig1HeLa11ppm_Report.xls", sep = "\t",
#                    usecols=['Stripped Sequence', "RT"], low_memory=True).rename(
#                         columns={'Stripped Sequence' : 'Base Sequence', 'RT' : 'Scan Retention Time'}).sort_values(
#                             'Scan Retention Time').reset_index()

# df = df[["Base Sequence", "Scan Retention Time"]]

# df.to_csv(r"/mnt/f/RetentionTimeProject/OtherPeptideResultsForTraining/PXD005573/RFig1HeLa11ppm_Report_sorted.csv")

a549_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/A549_AllPeptides.psmtsv")
# a549_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/A549_AllPeptides_sorted.csv")
gamg_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/GAMG_AllPeptides.psmtsv")
# gamg_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/GAMG_AllPeptides_sorted.csv")
hek293_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/HEK293_AllPeptides.psmtsv")
# hek293_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/HEK293_AllPeptides_sorted.csv")
hela_df = sortPSMTSV(r'/mnt/f/RetentionTimeProject/MannPeptideResults/Hela_AllPeptides.psmtsv')
# hela_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/Hela_AllPeptides_sorted.csv")
hepG2a_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/HepG2AllPeptides.psmtsv")
# hepG2a_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/HepG2AllPeptides_sorted.csv")
jurkat_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/Jurkat_AllPeptides.psmtsv")
# jurkat_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/Jurkat_AllPeptides_sorted.csv")
k562_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/K562_AllPeptides.psmtsv")
# k562_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/K562_AllPeptides_sorted.csv")
lanCap_df = sortPSMTSV(r"/mnt/f/RetentionTimeProject/MannPeptideResults/LanCap_AllPeptides.psmtsv")
# lanCap_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/LanCap_AllPeptides_sorted.csv")
mcf7_df = sortPSMTSV(r'/mnt/f/RetentionTimeProject/MannPeptideResults/MCF7_AllPeptides.psmtsv')
# mcf7_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/MCF7_AllPeptides_sorted.csv")
rko_df = sortPSMTSV(r'/mnt/f/RetentionTimeProject/MannPeptideResults/RKO_AllPeptides.psmtsv')
# rko_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/RKO_AllPeptides_sorted.csv")
u2os_df = sortPSMTSV(r'/mnt/f/RetentionTimeProject/MannPeptideResults/U2OS_AllPeptides.psmtsv')
# u2os_df.to_csv(r"/mnt/f/RetentionTimeProject/MannPeptideResults/U2OS_AllPeptides_sorted.csv")

dataframes = [a549_df, gamg_df, hek293_df, hela_df, hepG2a_df, jurkat_df, k562_df, lanCap_df, mcf7_df, rko_df, u2os_df]

#read them back in 

df = a549_df

for index, i in enumerate(dataframes):
    df = pd.merge(df, i, on="Base Sequence", suffixes = (str(index-1),str(index)))

df.to_csv("/mnt/f/RetentionTimeProject/overlapping_sequences.csv")

df = pd.read_csv("/mnt/f/RetentionTimeProject/overlapping_sequences.csv")

#average of retention times 
df["average"] = df[["Scan Retention Time-1", "Scan Retention Time0"]].mean(axis=1)
df["std"] = df[["Scan Retention Time-1", "Scan Retention Time0"]].std(axis=1)

#scatter plot of retention times
plt.scatter(df["Scan Retention Time-1"], df["Scan Retention Time0"], s=0.2)
plt.savefig("/mnt/f/RetentionTimeProject/overlapping_sequences.png")

# #visualize the data as scatter plot
# fig, ax = plt.subplots()
# ax.errorbar(df.index[0:1000], df["average"][0:1000], yerr=df["std"][0:1000], fmt='o')
# ax.set_xlabel("Index")
# ax.set_ylabel("Retention Time")
# #change size
# fig.set_size_inches(18.5, 10.5)
# fig.savefig("/mnt/f/RetentionTimeProject/overlapping_sequences.png")

# overlapping_sequences.plot()