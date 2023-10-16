import csv
import sys

# Function to read .csv files based on program arguments
def readCSVfile(filePath):
    try:
        with open(filePath, 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filePath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{filePath}': {e}")
        sys.exit(1)

# Check for correct amount of arguments
if len(sys.argv) != 3:
    print("Usage: python3 main.py <training_csv> <testing_csv>")
    print("Example: python3 main.py train.csv UnseenData.csv")
    sys.exit(1)

trainingCSVfile = sys.argv[1]
testingCSVfile = sys.argv[2]

# Check if the last two arguments are CSV files
if not trainingCSVfile.endswith(".csv") or not testingCSVfile.endswith(".csv"):
    print("Error: Both arguments should be .csv files.")
    sys.exit(1)

# Read Training CSV File
try:
    trainData = readCSVfile(trainingCSVfile)
except SystemExit:
    sys.exit(1)

# Read Testing CSV File
try:
    testData = readCSVfile(testingCSVfile)
except SystemExit:
    sys.exit(1)

# Start here
