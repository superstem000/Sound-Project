DIRPATH='data'

# Change directory to the script's location
cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev

# Activate the conda environment
source /Applications/anaconda3/bin/activate newenv

#export KMP_DUPLICATE_LIB_OK=TRUE

#source venv/bin/activate
python flask_prep_pi_dynamoDB.py -d $DIRPATH