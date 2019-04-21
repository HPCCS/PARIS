#!bin/bash
for file in 4instrype_results/small_programs/*;
do
  fname=$(basename "$file")
  sname=$(basename "$file" | cut -f 1 -d '_')
  #mkdir combinedd_features_hpc
  paste -d " " 4instrype_results_counter/small_programs/$sname* deadloc_ratio/small_programs/$sname* repeatadd_results/small_programs/$sname* > combined_features_new/$sname".txt"
  #paste -d " " 4instrype_results/$sname* deadloc_ratio/$sname* > combinedd_features_hpc/$sname".txt"

  echo $sname
done	
