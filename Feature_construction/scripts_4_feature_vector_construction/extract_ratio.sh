#!bin/bash
#for file in deadloc_results/*;
for file in deadloc_res/*;
do
  fname=$(basename "$file")
  sname=$(basename "$file" | cut -f 1 -d '_')
  echo $sname"_deadloc.txt"
  grep "ratio" $file  >> deadloc_ratio/$fname
  sed -i 's/overall ratio: //g' deadloc_ratio/$fname 
done	
