# Get the RESPEAKER_INDEX from the command-line argument
RESPEAKER_INDEX=$1
OTHER_INDEX=$2
START_TIME=$3

DIRPATH='data'

# Change directory to the project folder
cd C:/Users/bhpar/Desktop/CS/Sound-Project/respeaker_dev

# Launch PowerShell and activate the conda environment within it, ensuring it stays open
start powershell -NoExit -Command "conda activate newenv; python record_DOA_ID_chunks_pi.py -d $DIRPATH -s 300 -i $RESPEAKER_INDEX -t $START_TIME -o $OTHER_INDEX"
