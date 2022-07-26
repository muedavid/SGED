#!/bin/bash

pw=IhD999777
ip=192.168.1.242
client=/home/david/SemesterProject/SGED
server=/home/david/SemesterProject/
data_server=/output

sshpass -p $pw scp -r $client david@$ip:$server


if [ "$1" == data ]
then
sshpass -p $pw scp -r david@$ip:$server$data_server $client$data_client
fi
