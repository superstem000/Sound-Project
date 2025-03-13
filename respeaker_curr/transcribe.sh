DIRPATH='data'

# Change directory to the script's location
cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev

# Activate the conda environment
source /Applications/anaconda3/bin/activate newenv


#source venv/bin/activate
python transcribe_chunk_pi.py -d $DIRPATH