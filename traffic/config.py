from pathlib import Path

numApps = 3  # number of categories of Apps with different delay budgets
fileSize = 3 << 20  # file size in bits
delayBudgets = [.05, .15, .30]  # in seconds
appNames = ['Delay Stringent', 'Delay Sensitive', 'Delay Tolerant']
# peakSumRate = 1 << 30  # maximum downlink sum rate in bps
dpiSampleRates = [1/200, 1/200, 1/200, 1/200, 1/200]  # probability of inspecting a traffic session in DPI data
profilesPath = Path(__file__).parent.parent/'data/cluster_traffic_profiles.csv'
