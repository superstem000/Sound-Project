#gnome-terminal --tab -- bash -ic "source transcribe.sh; exec bash"
#gnome-terminal --tab -- bash -ic "source flask.sh; exec bash"
#gnome-terminal --tab -- bash -ic "source record.sh; exec bash"

#osascript -e 'tell application "Terminal" to do script "cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev; bash transcribe.sh"'
#osascript -e 'tell application "Terminal" to do script "cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev; bash flask.sh"'
#osascript -e 'tell application "Terminal" to do script "cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev; bash record.sh"'

# Calculate the shared start time (current Unix timestamp)
#START_TIME=$(python -c 'import time; print(time.time())')
# Run two instances with the same start_time
#osascript -e "tell application \"Terminal\" to do script \"cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev; bash record.sh 2 $START_TIME\"" &
#osascript -e "tell application \"Terminal\" to do script \"cd /Users/park90286/Documents/CS/Sound_Project/respeaker_dev; bash record.sh 3 $START_TIME\"" &


# Calculate the shared start time (current Unix timestamp)
START_TIME=$(python -c "import time; print(time.time())")

# Run two instances with the same start_time in PowerShell windows
start powershell -Command "cd 'C:\Users\bhpar\Desktop\CS\Sound-Project\respeaker_dev'; bash -i record.sh 2 3 $START_TIME; bash -i record.sh 3 2 $START_TIME; pause"


