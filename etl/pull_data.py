# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "/Users/lennoxanderson/Documents/School/DukeUniversity/DeepLearning/DL-Stock-Picker/data/data.csv"

# Load the latest version using the Pandas adapter
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "iveeaten3223times/massive-yahoo-finance-dataset",  # Ensure this identifier is correct.
    file_path,
    # Provide additional arguments if needed
)

print("First 5 records:", df.head())
