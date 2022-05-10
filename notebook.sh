#!/bin/bash

pw=IhD999777
server=david@192.168.1.242

sshpass -p $pw ssh $server screen -d -m -S test "jupyter-notebook --no-browser --port=1234"
#screen -d -m xdg-open https://loalhost:8000
sshpass -p $pw ssh $server -L 1234:localhost:1234
