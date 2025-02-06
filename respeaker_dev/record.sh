# Get the RESPEAKER_INDEX from the command-line argument
RESPEAKER_INDEX=$1
START_TIME=$2

DIRPATH='data'

# Change directory to the script's location
cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev

# Activate the conda environment
source /Applications/anaconda3/bin/activate newenv


#source venv/bin/activate
python record_DOA_ID_chunks_pi.py -d $DIRPATH -s 300 -i $RESPEAKER_INDEX -t $START_TIME