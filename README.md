# muse_overlay
Creating an overlay that runs with the MUSE 2 headset allowing the program to respond to brain data in real time.

The goal is to have a "tilt score" that will accurately(or sort of) determine a change in emotional, especially a negative one using metrics such as lowered HRV, changes in frontal alpha asymmetry, and the NASA engagement index. 

When the user runs the overlay with the command `python overlay.py` there will be a 33 second period for initial data collection to get baseline numbers for the user. Then the user can simply play a game or do anything they want with the 
overlay on. If the algorithm recognizes a change in their emotional state, the screen with flash red at first then shake aggressively if they are extrememly stressed. 

Please try this, I have used it with The Binding of Isaac and found it was a lot of fun. Press `shift+esc` to exit and `ctrl+shift+r` to recalibrate. Here are some examples of me playing with it.

https://drive.google.com/file/d/1L9Pe69w-HZRkwQv5vo42wpd6HqRLQJ5H/view?usp=sharing 
https://drive.google.com/file/d/1M344sY3oYxKxAS96JJOIsJVSmfXXVoL_/view?usp=sharing

You may have to change games out of fullscreen and into windowed mode to see the overlay. 
