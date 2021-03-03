#!/bin/bash

DB="TPCDS10G"

db2 connect to $DB
db2 set current schema $DB
db2 set current explain mode explain

for FILE in *; do 
    echo $FILE;
    db2 -tvf $FILE
    db2exfmt -d $DB -1 -o $FILE.exfmt
done;

db2 set current explain mode no

